import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
import pytest
from jaxtyping import Array, Float
from scipy.ndimage import fourier_shift
from yaslp.phantom import shepp_logan
from yaslp.utils import grid_basis

from mriutils_in_jax.register import register

PLOT = False
NVOL = 5
NREP = 4
NSOS = 2
NSPAT_MAX = 3
SHIFT_MAX = 6
SPAT_SHAPE = (30, 20, 10)


@pytest.fixture
def shifts():
    shift = onp.linspace(SHIFT_MAX, 0, NVOL)
    # _scaling_factor = onp.array([20, -5, 1])
    _scaling_factor = onp.array([1, -5, 20])
    return shift[:, None] / _scaling_factor


def test_gen(shifts):
    plt.plot(shifts)
    plt.show()


def fourier_shift_jax(
    array: Float[Array, "..."],
    shift: Float[Array, " ndim"],
    axes: tuple[int, ...] | None = None,
):
    array = jnp.asarray(array)
    shift = jnp.asarray(shift)
    if axes is None:
        axes = tuple(range(array.ndim))
    batch_axes = tuple(a for a in range(array.ndim) if a not in axes)
    assert len(axes) == len(shift)
    shape = tuple(array.shape[a] for a in axes)

    # Compute the frequency grid
    phase_shift = grid_basis(shape, reciprocal=True, rfft=True) @ shift
    # Apply Fourier shift theorem
    return jnp.fft.irfftn(
        jnp.fft.rfftn(array, axes=axes)
        * jnp.expand_dims(jnp.exp(-2j * jnp.pi * phase_shift), batch_axes),
        axes=axes,
    )


def shift_array(array, shift, axes=(0, 1, 2)):
    input_ = onp.fft.rfftn(array, axes=axes)
    result = fourier_shift(input_, shift=shift, n=array.shape[-1])
    return onp.fft.irfftn(result, axes=axes)


@pytest.mark.parametrize("axis", [0, -1])
@pytest.mark.parametrize("parallel,jit", [(False, False), (False, True), (True, True)])
# @pytest.mark.parametrize("parallel,jit", [(False, False)])
def test_simple(shifts, axis, parallel, jit):
    """Dims: spatial + subvolume."""

    spat_data = shepp_logan(SPAT_SHAPE)
    shifted = jnp.stack([shift_array(spat_data, shift) for shift in shifts], axis=axis)
    shifts_recovered = register(
        onp.array(shifted),
        axis=axis,
        parallel=parallel,
        jit=jit,
    )
    assert jnp.allclose(shifts_recovered, -shifts, atol=0.1, rtol=0.1)
    unshifted = jax.vmap(
        fourier_shift_jax,
        in_axes=(axis, 0),
        out_axes=axis,
    )(shifted, shifts_recovered)

    fig, axes = plt.subplots(ncols=len(shifts) + 1, nrows=2, sharex=True, sharey=True)
    vmin, vmax = spat_data.min(), spat_data.max()

    def show(ax, array):
        return ax.imshow(array[..., 5].T, origin="lower", vmin=vmin, vmax=vmax)

    im = show(axes[0, -1], spat_data)
    for ax_row, arrays in zip(axes, [shifted, unshifted]):
        for idx, ax in enumerate(ax_row[:-1]):
            show(ax, arrays.take(idx, axis))
    plt.colorbar(im, ax=axes)
    axes[1, -1].axis("off")
    plt.show()
    resid_norm_new = jax.vmap(lambda x: jnp.linalg.norm(x - spat_data), in_axes=axis)(
        unshifted
    )
    resid_norm_old = jax.vmap(lambda x: jnp.linalg.norm(x - spat_data), in_axes=axis)(
        shifted
    )
    resid_norm_ratio = (
        (resid_norm_new / resid_norm_old).at[-1].set(resid_norm_new[-1])
    )  # otherwise it is not fair
    assert jnp.allclose(resid_norm_ratio, 0, atol=0.1, rtol=0.1)


@pytest.mark.parametrize("axis,coil_axis", [(0, -1), (0, -2), (-1, -2)])
@pytest.mark.parametrize("parallel,jit", [(False, False), (False, True), (True, True)])
def test_multicoil(shifts, axis, parallel, jit, coil_axis):
    """Dims: spatial + subvolume + coil."""

    spat_data = shepp_logan(SPAT_SHAPE)

    assert coil_axis != axis
    ndim = spat_data.ndim + 2
    axis = axis % ndim

    coil_axis_ = coil_axis % (ndim - 1)  # wo axis, used for data generation
    coil_axis = coil_axis % ndim
    reduction_axes = (coil_axis,)

    basis = grid_basis(SPAT_SHAPE)
    centre_ = jnp.array([0.5] * len(SPAT_SHAPE))
    centres = centre_[:, None] * jnp.array([-1, 1])
    assert centres.shape[-1] == NSOS
    coil_profile = jnp.linalg.norm(basis[..., None] - centres, axis=-2)
    coil_profile = (coil_profile.max() - coil_profile) / coil_profile.max()

    assert coil_profile.shape == SPAT_SHAPE + (NSOS,)

    # coils were and remained along axis=-1 -> roll 'em
    coil_profile = jnp.swapaxes(coil_profile, -1, coil_axis_)

    spat_data = jnp.expand_dims(spat_data, coil_axis_) * coil_profile
    if PLOT:
        fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True, sharex=True)
        for ax_row, arr in zip(axes, [coil_profile, spat_data]):
            for coil_idx, ax in enumerate(ax_row):
                im = ax.imshow(
                    arr.take(coil_idx, coil_axis_)[..., 5].T,
                    origin="lower",
                    vmin=-0.1,
                    vmax=1.1,
                )
                plt.colorbar(im, ax=ax)
        plt.show()

    shifted = jax.vmap(
        jax.vmap(fourier_shift_jax, in_axes=(coil_axis_, None), out_axes=coil_axis_),
        in_axes=(None, 0),
        out_axes=axis,
    )(spat_data, shifts)

    shifts_recovered = register(
        onp.array(shifted),
        axis=axis,
        parallel=parallel,
        jit=jit,
        reduction_axes=reduction_axes,
    )
    assert jnp.allclose(shifts_recovered, -shifts, atol=0.5, rtol=0.1)

    # if axis is to the left from coil_axis, external vmap will squeeze
    # axis out and coil_axis will not be correct anymore, thus I need
    # to use coil_axis_. Madness. Good luck generalising it
    ca = coil_axis_ if axis < coil_axis else coil_axis
    unshifted = jax.vmap(
        jax.vmap(fourier_shift_jax, in_axes=(ca, None), out_axes=ca),
        in_axes=(axis, 0),
        out_axes=axis,
    )(shifted, shifts_recovered)

    resid_norm_new = jax.vmap(
        lambda x: jnp.linalg.norm(x - shifted.take(-1, axis)), in_axes=axis
    )(unshifted)
    resid_norm_old = jax.vmap(
        lambda x: jnp.linalg.norm(x - shifted.take(-1, axis)), in_axes=axis
    )(shifted)
    resid_norm_ratio = (
        (resid_norm_new / resid_norm_old).at[-1].set(resid_norm_new[-1])
    )  # otherwise it is not fair
    # assert jnp.allclose(resid_norm_ratio, 0, atol=0.2, rtol=0.2)

    fig, axes = plt.subplots(
        ncols=len(shifts), nrows=2 * NSOS, sharex=True, sharey=True
    )

    def show(ax, array):
        return ax.imshow(array[..., 5].T, origin="lower")

    for arr_idx, arrays in enumerate([shifted, unshifted]):
        for coil_idx in range(NSOS):
            for idx, ax in enumerate(axes[arr_idx * NSOS + coil_idx]):
                # mind the use of coil_axis_, i.e. one wo accounting for axis,
                # as it is squeezed out by the preceeding .take
                im = show(ax, arrays.take(idx, axis).take(coil_idx, coil_axis_))
    plt.colorbar(im, ax=axes)
    plt.show()
