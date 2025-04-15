import json
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import nibabel as nib
import numpyro
import numpyro.distributions as dist
from jax import lax
from jaxtyping import Array, ArrayLike, Complex, Float
from loguru import logger
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
from numpyrotils import run_svi

from mriutils_in_jax.loader import Loaded
from mriutils_in_jax.utils import grid_basis


def downsample_mean(
    array,
    factor: int = 1,
    axes: tuple[int, ...] | None = None,
    fill_value: float = jnp.nan,
):
    if axes is None:
        axes = tuple(range(array.ndim))

    _shape = tuple(
        sz / factor if ax in axes else sz for ax, sz in enumerate(array.shape)
    )
    if jnp.any(jnp.array(_shape) - jnp.array(_shape).astype(int)).item():
        raise ValueError(
            f"Rounding error when applying {factor=} to {array.shape}: {_shape}"
        )
    window_shape = tuple(factor if ax in axes else 1 for ax in range(array.ndim))

    # window_shape should be (n, m, k)
    # strides should match window_shape for non-overlapping windows
    counts = lax.reduce_window(
        jnp.isfinite(array).astype(int),
        init_value=0,
        computation=lax.add,
        window_dimensions=window_shape,
        window_strides=window_shape,
        padding="VALID",
    )
    sums = lax.reduce_window(
        jnp.where(jnp.isfinite(array), array, 0),
        init_value=jnp.array(0, dtype=array.dtype),
        computation=lax.add,
        window_dimensions=window_shape,
        window_strides=window_shape,
        padding="VALID",
    )
    return jnp.where(counts > 0, sums / counts, fill_value)


def model(te, basis, weights=1.0):
    phi0 = numpyro.sample("phi0", dist.TruncatedNormal(0, 1, low=-jnp.pi, high=jnp.pi))
    freq0 = numpyro.sample("freq0", dist.Normal(0, 1))
    with numpyro.plate("grad_comp", basis.shape[-1]):
        dfreq = numpyro.sample("dfreq", dist.Normal(0, 1))
    global_conc = numpyro.sample(
        "global-conc", dist.TruncatedNormal(10.0, 3.0, low=1.0)
    )

    phase_offset_predicted = phi0 + (freq0 + basis @ dfreq)[..., None] * te
    with numpyro.plate("echo", te.size):
        with numpyro.plate_stack("spatial", basis.shape[:-1], rightmost_dim=-2):
            with numpyro.handlers.mask(mask=weights > 0):
                numpyro.sample(
                    "offset",
                    dist.VonMises(phase_offset_predicted, global_conc * weights),
                )


def plot_comparison(
    arrays: list[Float[Array, "*spatial echo"]],
    filename_png: Path | None = None,
    phase_cmap=True,
):
    _shapes = jnp.unique(jnp.stack([jnp.array(a.shape) for a in arrays]), axis=0)
    if _shapes.shape[0] != 1:
        raise ValueError(f"Inconsistent sizes of specified arrays: {_shapes}")
    centre: list[int] = (_shapes[0, :-1] / 2).astype(int).tolist()

    necho = _shapes[0, -1].item()
    if phase_cmap:
        norm = TwoSlopeNorm(0, -jnp.pi, jnp.pi)
        cmap = "RdBu_r"
    else:
        norm = Normalize()
        cmap = "viridis"

    _, axes = plt.subplots(
        nrows=len(centre) * len(arrays),
        ncols=necho,
        figsize=(3 * necho, 1.2 * len(centre) * len(arrays)),
    )
    for idx_axis, idx_centre in enumerate(centre):
        for idx_te in range(necho):
            for idx_array, array in enumerate(arrays):
                im = axes[len(arrays) * idx_axis + idx_array, idx_te].imshow(
                    array[..., idx_te].take(idx_centre, idx_axis).T,
                    norm=norm,
                    cmap=cmap,
                )
    for idx_te, ax in enumerate(axes[0]):
        ax.set_title(f"Echo #{idx_te}")
    for ax in axes.flat:
        ax.axis("off")

    for offset, label in enumerate(["Phase offset", "Corrected"]):
        for ax in axes[offset::2, 0]:
            ax.text(
                -0.05,
                0.5,
                label,
                va="center",
                ha="right",
                transform=ax.transAxes,
                rotation=90,
            )
    cbar = plt.colorbar(im, ax=axes)
    cbar.ax.set_ylabel("Phase (rad)")
    if filename_png is not None:
        plt.savefig(filename_png, dpi=300, bbox_inches="tight")
    plt.close()


def correct_repetition_phase(
    moving_magn: Path,
    moving_phase: Path,
    reference_magn: Path,
    reference_phase: Path,
    te: list[float],
    output_basename: Path | None = None,
    axis_echo: int = -1,
    mask_fg_threshold: float | None = 0.3,
    sel: str = "",
    factor: int = 5,
    check_phase: bool = True,
    plot_hist: bool = True,
):
    nifti_suffix = "".join(moving_phase.suffixes)  # can be .nii or .nii.gz
    if output_basename is None:
        output_basename = moving_phase.parent / moving_phase.name.replace(
            nifti_suffix, "-corrected"
        )
    output_basename.parent.mkdir(exist_ok=True, parents=True)
    output_phase = output_basename.with_suffix(nifti_suffix)
    output_coeff = output_basename.with_suffix(".coefs")

    te = jnp.array(te)
    logger.debug("Loading the reference images")
    ref = Loaded(
        reference_magn,
        reference_phase,
        axis_echo,
        sel,
        check_phase,
        magn_scale="percentile",
    )
    logger.debug("Loading the moving images")
    moving = Loaded(
        moving_magn,
        moving_phase,
        axis_echo,
        sel,
        check_phase,
        magn_scale="percentile",
    )
    if ref.shape != moving.shape:
        raise ValueError(
            f"Incompatible shapes for the reference {ref.shape} "
            f"and the moving {moving.shape}"
        )
    if ref.shape[-1] != te.size:
        raise ValueError(
            f"Provided number of TE ({te.size}) does not match the data {ref.shape[-1]}"
        )
    logger.debug("Downsampling the phase offset")
    magn_downsampled = downsample_mean(
        ref.magn + moving.magn,
        factor=factor,
        axes=tuple(range(ref.magn.ndim - 1)),
        fill_value=0.0,
    )
    complex_downsampled = downsample_mean(
        moving.complex / ref.complex,
        factor=factor,
        axes=tuple(range(ref.magn.ndim - 1)),
        fill_value=0.0,
    )

    logger.debug("Defining the foreground mask")
    if mask_fg_threshold is not None:
        mask_fg = magn_downsampled.mean(-1)[..., None] > mask_fg_threshold
    else:
        mask_fg = True

    if plot_hist:
        plt.hist(magn_downsampled.mean(-1).flatten(), bins=50)
        if mask_fg_threshold:
            plt.axvline(mask_fg_threshold, c="k")
        plt.savefig(
            output_basename.parent / f"{output_basename.name}-hist.png",
        )
        plt.close()

    phase_offset = jnp.where(mask_fg, jnp.angle(complex_downsampled), 0)
    basis_downsampled = grid_basis(phase_offset.shape[:-1]) / 2
    logger.debug("Plot the weights")
    weights = jnp.where(mask_fg, magn_downsampled, 0)
    plot_comparison(
        [weights],
        filename_png=output_basename.parent / f"{output_basename.name}-weights.png",
        phase_cmap=False,
    )
    del ref.magn, moving.magn

    logger.debug("Running the optimisation")
    result, svi = run_svi(
        numpyro.handlers.condition(model, {"offset": phase_offset}),
        te=te,
        basis=basis_downsampled,
        weights=weights,
        learning_rate=1e-2,
    )

    logger.debug("Sampling prior predictive")
    phase_offset_prior_pred = jnp.moveaxis(
        numpyro.infer.Predictive(model, num_samples=20)(
            jr.PRNGKey(0), te=te, basis=basis_downsampled, weights=weights
        )["offset"],
        0,  # sample axis (size is 20)
        -1,
    )

    logger.debug("Sampling posterior predictive")
    phase_offset_post_pred = numpyro.infer.Predictive(
        model, num_samples=1, guide=svi.guide, params=result.params
    )(jr.PRNGKey(0), te=te, basis=basis_downsampled, weights=weights)["offset"][0]

    logger.debug("Plot offset over echoes with prior and posterior predictives")
    ncol = 5
    whsz = 10
    indices = jnp.linspace(0, phase_offset.shape[0], ncol + 2).astype(int)[1:-1]
    _lims = tuple(
        (int(sz / 2) - whsz, int(sz / 2) + whsz) for sz in phase_offset.shape[1:-1]
    )
    slices = tuple(slice(*sl) for sl in _lims)
    _sel_phase_avg = {}
    for key, array in zip(
        ["obs", "prior_pred", "posterior_pred"],
        [
            jnp.where(mask_fg, phase_offset, jnp.nan),
            phase_offset_prior_pred,
            phase_offset_post_pred,
        ],
    ):
        _sel_phase_offset = array[indices][(slice(None),) + slices]
        _sel_phase_avg[key] = jnp.angle(
            jnp.nanmean(jnp.exp(1j * _sel_phase_offset), (1, 2))
        )

    _, axes = plt.subplots(
        nrows=3, ncols=ncol, sharex="row", sharey="row", figsize=(20, 8)
    )
    for idx_col, (ax_col, idx_in_orig_space) in enumerate(
        zip(axes.T, indices.tolist())
    ):
        ax_col[0].imshow(
            phase_offset[idx_in_orig_space, ..., -1],
            norm=TwoSlopeNorm(0, vmin=-jnp.pi, vmax=jnp.pi),
            cmap="RdBu_r",
        )
        p = Rectangle(
            (_lims[0][0], _lims[1][0]), 2 * whsz, 2 * whsz, fc="none", ec="C2"
        )
        ax_col[0].add_patch(p)
        ax_col[0].axis("off")
        ax_col[0].set_title(f"Slice along readout: {idx_in_orig_space}")
        ax_col[1].plot(
            te, _sel_phase_avg["obs"][idx_col], marker=".", c="k", label="observed"
        )
        ax_col[1].plot(
            te,
            _sel_phase_avg["prior_pred"][idx_col][:, 0],
            c="grey",
            alpha=0.5,
            label="prior pred",
        )
        ax_col[1].plot(te, _sel_phase_avg["prior_pred"][idx_col], c="grey", alpha=0.5)
        ax_col[1].plot(
            te,
            _sel_phase_avg["posterior_pred"][idx_col],
            c="C1",
            label="posterior pred",
        )
        ax_col[1].grid()

        ax_col[2].plot(
            te, jnp.unwrap(_sel_phase_avg["obs"][idx_col]), marker=".", c="k"
        )
        ax_col[2].plot(
            te, jnp.unwrap(_sel_phase_avg["posterior_pred"][idx_col]), c="C1"
        )
        ax_col[2].grid()
        ax_col[2].set_xlabel("TE (ms)")

    axes[1, 0].legend()
    axes[1, 0].set_ylabel("Phase, wrapped (rad)")
    axes[2, 0].set_ylabel("Phase, unwrapped (rad)")

    plt.savefig(
        output_basename.parent / f"{output_basename.name}-offset_over_te.png",
        bbox_inches="tight",
        dpi=150,
    )
    plt.close()

    def predict_from_result(params: dict, te, basis):
        return (
            params["phi0_auto_loc"]
            + te
            * (params["freq0_auto_loc"] + basis @ params["dfreq_auto_loc"])[..., None]
        )

    del weights
    logger.debug("Applying the optimal correction")
    phase_offset_corrected = jnp.angle(
        jnp.exp(
            1j * (phase_offset - predict_from_result(result[0], te, basis_downsampled))
        )
    )
    phase_offset_predicted_upsampled = predict_from_result(
        result[0], te, grid_basis(moving.shape[:-1]) / 2
    )
    phase_corrected = jnp.angle(
        jnp.exp(1j * (moving.phase - phase_offset_predicted_upsampled))
    )

    logger.debug("Plotting the results")
    plot_comparison(
        [phase_offset, phase_offset_corrected],
        filename_png=output_basename.parent
        / f"{output_basename.name}-phase_offset.png",
    )
    plot_comparison(
        [moving.phase, phase_corrected, ref.phase],
        filename_png=output_basename.parent / f"{output_basename.name}-phases.png",
    )

    logger.debug("Outputting the corrected phase to {}", output_phase)
    nib.nifti1.Nifti1Image(
        jnp.zeros(moving.img.shape).at[moving.sel].set(phase_corrected),
        moving.img.affine,
        moving.img.header,
    ).to_filename(output_phase)
    output_coeff.write_text(
        json.dumps(
            {k.replace("_auto_loc", ""): v.tolist() for k, v in result[0].items()}
        )
    )
    plt.figure()
    plt.plot(result.losses)
    plt.xlabel("Iteration number")
    plt.ylabel("Loss")
    plt.savefig(
        output_basename.parent / f"{output_basename.name}-loss.png",
    )
    plt.close()


def estimate_freq(
    echo0: Complex[Array, "..."],
    echo1: Complex[Array, "..."],
    dte: Float[ArrayLike, ""],
    sum_axis: int | None = None,
):
    """Estimate frequency from two first echoes

    Parameters
    ----------
    echo0, echo1 : ndarray,
        Two arrays representing corresponding echoes. Can be 3- or 4D. In the
        case of 4D sum_axis must be specified to perform summation across.
    dte : float
        Difference between the echoes.
    sum_axis : int, optional
        Axis to sum 4D array across. Usually corresponds to channel dimension.

    Returns
    -------
    freq : ndarray, 3D
        Estimate to the frequency in radians per unit of time, used by dte

    """
    combined = echo1 * jnp.conj(echo0)
    if jnp.isnan(combined).any():
        raise ValueError("Data must not contain NaNs")
    if sum_axis is not None:
        combined = combined.sum(axis=sum_axis)

    try:
        from skimage.restoration import unwrap_phase

        unwrapped = jnp.array(unwrap_phase(jnp.angle(combined)))
    except ModuleNotFoundError:
        if combined.ndim == 1:
            unwrapped = jnp.unwrap(jnp.angle(combined))
        else:
            raise
    freq = unwrapped / dte
    return freq


def estimate_offset(
    echo: Complex[Array, "..."],
    te: Float[ArrayLike, ""],
    freq: Float[Array, "..."],
    unwrap=True,
) -> Float[Array, "..."]:
    """Estimate phase offset from two first echoes
    Parameters
    ----------
    echo : ndarray, 3D
        Single echo complex data.
    te : float
        Corresponding echo time in seconds.
    freq : ndarray, 3D
        Frequency estimate, radians per unit of te
    unwrap : bool, optional
        If to unwrap the result spatially. Default: True
    Returns
    -------
    freq : ndarray, 3D
        Estimate to the frequency in Hz
    """
    phi0 = jnp.angle(echo * jnp.exp(-1j * freq * te))
    if unwrap:
        if jnp.isnan(phi0).any():
            raise ValueError("If unwrap, data must not contain NaNs")
        try:
            from skimage.restoration import unwrap_phase

            phi0 = jnp.array(unwrap_phase(phi0))
        except ModuleNotFoundError:
            if phi0.ndim == 1:
                phi0 = jnp.unwrap(phi0)

    return phi0
