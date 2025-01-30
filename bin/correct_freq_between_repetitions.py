import json
from pathlib import Path

import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
import nibabel as nib
import typer
from jax import jit, lax
from jaxtyping import Array, Float
from loguru import logger
from matplotlib.colors import Normalize, TwoSlopeNorm

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
            f"Rounding error when applying {factor=} to {array.shape}: " f"{_shape}"
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


def predict_freq(
    freq0: Float[Array, ""],
    grads: Float[Array, " ndim"],
    basis: Float[Array, "*spatial ndim"],
) -> Float[Array, " *spatial"]:
    return freq0 + basis @ grads


def predict_phase_offset(
    coefs: Float[Array, " 2+ndim"],
    te: Float[Array, " echo"],
    basis: Float[Array, "*spatial ndim"],
) -> Float[Array, " *spatial echo"]:
    return coefs[0] + predict_freq(coefs[1], coefs[2:], basis)[..., None] * te


def predict_constant_phase_offset(
    coefs: Float[Array, " 2"], te: Float[Array, " echo"]
) -> Float[Array, " echo"]:
    return coefs[0] + coefs[1] * te


@jit
def loss(
    coefs,
    te: Float[Array, " echo"],
    basis: Float[Array, "*spatial ndim"],
    phase: Float[Array, "*spatial echo"],
    magn: Float[Array, "*spatial echo"] | None = None,
):
    if magn is None:
        magn = jnp.ones_like(phase)
    return jnp.linalg.norm(
        # magn * jnp.sin(0.5 * (predict_constant_phase_offset(coefs, te) - phase))
        magn * jnp.sin(0.5 * (predict_phase_offset(coefs, te, basis) - phase))
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
        figsize=(5 * necho, 2 * len(centre) * len(arrays)),
    )
    for idx_axis, idx_centre in enumerate(centre):
        for idx_te in range(necho):
            for idx_array, array in enumerate(arrays):
                im = axes[len(arrays) * idx_axis + idx_array, idx_te].imshow(
                    array[..., idx_te].take(idx_centre, idx_axis).T,
                    norm=norm,
                    cmap=cmap,
                )
    for ax in axes.flat:
        ax.axis("off")
    cbar = plt.colorbar(im, ax=axes)
    cbar.ax.set_ylabel("Phase (rad)")
    if filename_png is not None:
        plt.savefig(filename_png, dpi=300, bbox_inches="tight")
    plt.close()


def main(
    moving_magn: Path,
    moving_phase: Path,
    reference_magn: Path,
    reference_phase: Path,
    header: Path,
    output_basename: Path | None = None,
    axis_echo: int = -1,
    mask_fg_threshold: float | None = 0.3,
    sel: str = "",
    factor: int = 5,
    check_phase: bool = True,
):
    nifti_suffix = "".join(moving_phase.suffixes)  # can be .nii or .nii.gz
    if output_basename is None:
        output_basename = moving_phase.parent / moving_phase.name.replace(
            nifti_suffix, "-corrected"
        )
    output_basename.parent.mkdir(exist_ok=True, parents=True)
    output_phase = output_basename.with_suffix(nifti_suffix)
    output_coeff = output_basename.with_suffix(".coefs")

    with open(header) as f:
        te = jnp.array(json.load(f)["echoTime"])
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
            f"Provided number of TE ({te.size}) does not match "
            f"the data {ref.shape[-1]}"
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
    phase_offset = jnp.where(mask_fg, jnp.angle(complex_downsampled), 0)
    weights = jnp.where(mask_fg, magn_downsampled, 0)
    plot_comparison(
        [weights],
        filename_png=output_basename.parent / f"{output_basename.name}-weights.png",
        phase_cmap=False,
    )
    del ref.magn, moving.magn, mask_fg

    # param_init = np.polyfit(te, resid_mean["phase_unwrpd"].sel(rep=rep_idx).data, 1)
    init_params = jnp.array([0.0, 0.0] + [0.0] * (phase_offset.ndim - 1))

    logger.debug("Running the optimisation")
    opt = jaxopt.LBFGS(loss, maxiter=25)
    result = opt.run(
        init_params, te, grid_basis(phase_offset.shape[:-1]), phase_offset, weights
    )
    logger.debug("Completed after {} iterations", result[1].iter_num.item())
    del weights
    logger.debug("Applying the optimal correction")
    # predicted = predict_constant_phase_offset(result[0], te)
    phase_offset_corrected = jnp.angle(
        jnp.exp(
            1j
            * (
                phase_offset
                - predict_phase_offset(
                    result[0], te, grid_basis(phase_offset.shape[:-1])
                )
            )
        )
    )
    phase_corrected = jnp.angle(
        jnp.exp(
            1j
            * (
                moving.phase
                - predict_phase_offset(result[0], te, grid_basis(moving.shape[:-1]))
            )
        )
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
    output_coeff.write_text(json.dumps(result[0].tolist()))


if __name__ == "__main__":
    typer.run(main)
