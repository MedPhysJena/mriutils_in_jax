import tempfile
from dataclasses import dataclass

import jax
import jax.random as jr
import jax.numpy as jnp
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as onp
import pytest
from jaxtyping import Array, Float
from scipy.ndimage import fourier_shift
from yaslp.phantom import shepp_logan
from yaslp.utils import grid_basis

from mriutils_in_jax.register import (
    register_complex_data,
    identify_necessary_shifts,
    fourier_shift as fourier_shift_in_jax,
)

INTERACTIVE = False
NVOL = 5
NREP = 4
NSOS = 2
NSPAT_MAX = 3
SHIFT_MAX = 6
SPAT_SHAPE = (10, 20, 30)

interactive = pytest.mark.skipif(
    not INTERACTIVE, reason="interactive plotting switched off"
)


def gen_phantom(nd):
    if nd < 1 or nd > 3:
        raise ValueError("gen_phantom supports nd of 1, 2, or 3.")
    data = shepp_logan((10, 30, 20)).swapaxes(1, 2)
    centre = tuple(int(sz / 2) for sz in data.shape)
    return data[*centre[:-nd]]


def _show(arr, ax=None):
    if ax is None:
        ax = plt.subplot()
    if arr.ndim == 1:
        ax.plot(arr)
    elif arr.ndim == 2:
        ax.imshow(arr)
    elif arr.ndim == 3:
        ax.imshow(arr[5])
    return ax


@interactive
@pytest.mark.parametrize("nd", [1, 2, 3])
def test_phantom(nd):
    """Quick visual check for the fixture it depends on."""
    phantom = gen_phantom(nd)
    assert phantom.ndim == nd
    _show(phantom)
    plt.title("Phantom")
    plt.show()


def fourier_shift_in_scipy(array, shift, axes: tuple[int, ...] | None = None):
    if axes is None:
        axes = tuple(range(min(array.ndim, 3)))
    input_ = onp.fft.rfftn(array, axes=axes)
    result = fourier_shift(input_, shift=shift, n=array.shape[-1])
    return onp.fft.irfftn(result, axes=axes)


@pytest.mark.parametrize(
    "nd,shift",
    [
        (1, jnp.array([2])),
        (1, jnp.array([3.5])),
        (2, jnp.array([2, 4])),
        (2, jnp.array([3.5, 4.9])),
        (3, jnp.array([2, 4, 6])),
        (3, jnp.array([3.5, 4.9, 7.3])),
    ],
)
def test_fourier_shift_in_jax(nd, shift):
    """Most basic test against the scipy implementation."""
    data = gen_phantom(nd)
    data_expected = fourier_shift_in_scipy(data, shift)
    data_actual = fourier_shift_in_jax(data, shift)
    assert jnp.allclose(data_expected, data_actual, atol=1e-5, rtol=1e-5)


## test actual registration


@pytest.fixture
def shifts():
    shift = onp.linspace(SHIFT_MAX, 0, NVOL)
    _scaling_factor = onp.array([6, -3, 1])
    return shift[:, None] / _scaling_factor


@dataclass
class DataDef:
    orig: Float[Array, "..."]
    observed: Float[Array, "..."]
    axis: int
    spatial_axes: tuple[int, ...]
    batch_axes: tuple[int, ...]
    reduce_axes: tuple[int, ...]


@pytest.fixture(params=["jax", "numpy", "arrayproxy"])
def cast_to(request):
    """Generate a callback that will cast the test data to either of the types."""
    if request.param == "arrayproxy":

        def cb(observed):
            with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
                filename = tmp.name
            nib.nifti1.Nifti1Image(observed, onp.eye(4)).to_filename(filename)
            img = nib.nifti1.load(filename)
            return img.dataobj
    elif request.param == "numpy":

        def cb(observed):
            return onp.array(observed)
    elif request.param == "jax":

        def cb(observed):
            return jnp.array(observed)
    else:
        raise ValueError(f"Unexpected {request.param=}")
    return cb


@pytest.fixture(
    params=[
        (1, 0),
        (1, -1),
        (2, 0),
        (2, -1),
        (3, 0),
        (3, -1),
    ]
)
def data__spatial_echo(shifts, request):
    nd, axis = request.param
    data = gen_phantom(nd)
    shifted = jax.vmap(fourier_shift_in_jax, in_axes=(None, 0), out_axes=axis)(
        data, shifts[:, :nd]
    )
    # same as this:
    # shifted = jnp.stack(
    #     [fourier_shift_in_jax(data, shift) for shift in shifts[:, :nd]], axis=axis
    # )
    spatial_axes = tuple(a for a in range(nd) if a != axis)
    return DataDef(
        orig=data,
        observed=shifted,
        axis=axis,
        spatial_axes=spatial_axes,
        batch_axes=(),
        reduce_axes=(),
    )


@interactive
def test_data__spatial_echo_only(data__spatial_echo):
    """Just a check how the data are generated."""
    dat = data__spatial_echo
    _, axes = plt.subplots(nrows=NVOL + 1, sharex=True)
    _show(dat.orig, axes[-1])
    axes[-1].set_ylabel("Orig")
    for idx, ax in enumerate(axes[:-1]):
        _show(dat.observed.take(idx, dat.axis), ax)
        ax.set_ylabel(f"Shifted echo = {idx}")
    axes[-1].set_xlabel("x")
    plt.show()


def test_registration__spatial_echo_only(data__spatial_echo, shifts, cast_to):
    """Just a check how the data are generated."""
    dat = data__spatial_echo
    shifts_recovered = identify_necessary_shifts(cast_to(dat.observed), axis=dat.axis)
    assert jnp.allclose(
        shifts[:, : dat.orig.ndim] + shifts_recovered, 0, atol=0.1, rtol=0.1
    )


@pytest.fixture(params=[1, 2, 3])
def data__spatial_echo_coil(shifts, request):
    nd = request.param
    coil_axis = -1
    axis = -2

    data = gen_phantom(nd)
    basis = grid_basis(data.shape)
    centre_ = jnp.array([0.5] * nd)
    centres = centre_[:, None] * jnp.array([-1, 1])
    assert centres.shape[-1] == NSOS
    coil_profile = jnp.linalg.norm(basis[..., None] - centres, axis=-2)
    coil_profile = (coil_profile.max() - coil_profile) / coil_profile.max()
    assert coil_profile.shape == data.shape + (NSOS,)

    data = jnp.expand_dims(data, coil_axis) * coil_profile
    # same as data[..., None] * coil_profile
    if INTERACTIVE:
        _, axes = plt.subplots(nrows=2, ncols=2, sharey=True, sharex=True)
        for ax_row, arr in zip(axes, [coil_profile, data]):
            for coil_idx, ax in enumerate(ax_row):
                _show(arr.take(coil_idx, coil_axis), ax)
        plt.show()

    shifted = jnp.stack(
        [
            # excluding coil_axis = -1
            fourier_shift_in_jax(data, shift, axes=tuple(range(nd)))
            for shift in shifts[:, :nd]
        ],
        axis=axis,
    )
    # HACK: implies that coil and echo are the last
    # FIX: just skip it, current implementation does not need it
    spatial_axes = tuple(a for a in range(shifted.ndim - 2))
    return DataDef(
        orig=data,
        observed=shifted,
        axis=axis,
        spatial_axes=spatial_axes,
        batch_axes=(),
        reduce_axes=(coil_axis,),
    )


@interactive
def test_data__spatial_echo_coil(data__spatial_echo_coil):
    """Just a check how the data are generated."""
    dat = data__spatial_echo_coil
    _, axes = plt.subplots(nrows=NVOL + 1, ncols=NSOS, sharex=True, sharey=True)

    for coil_idx, ax_col in enumerate(axes.T):
        for echo_idx, ax in enumerate(ax_col[:-1]):
            _show(dat.observed.take(echo_idx, dat.axis).take(coil_idx, -1), ax)

        _show(dat.orig.take(coil_idx, -1), ax_col[-1])

    for idx, ax in enumerate(axes[0]):
        ax.set_title(f"Coil = {idx}")
    for idx, ax in enumerate(axes[:-1, 0]):
        ax.set_ylabel(f"Echo = {idx}")
    axes[-1, 0].set_ylabel("Orig")

    plt.show()


def test_registration__spatial_echo_coil(data__spatial_echo_coil, shifts, cast_to):
    dat = data__spatial_echo_coil
    phase = jr.uniform(
        jr.PRNGKey(0), shape=dat.observed.shape, minval=-jnp.pi, maxval=jnp.pi
    )
    shifted_complex, shifts_recovered = register_complex_data(
        magn=cast_to(dat.observed),
        phase=cast_to(phase),
        axis_echo=dat.axis,
        axis_coil=dat.reduce_axes[0],
    )

    assert jnp.allclose(
        shifts[:, : dat.orig.ndim - 1] + shifts_recovered, 0, atol=0.1, rtol=0.1
    )


@pytest.fixture(params=[1, 2, 3])
def data__spatial_echo_batch(shifts, request):
    nd = request.param
    axis = 0
    batch_axis = 1
    batch_size = 3
    batch_factor = 0.8 + 0.2 * jnp.arange(batch_size)
    # in the absence of axis (echo), batch_axis is not 1 but 0
    data = jnp.expand_dims(batch_factor, tuple(range(1, nd + 1))) * gen_phantom(nd)
    shifts = jnp.expand_dims(batch_factor, (1, 2)) * shifts

    shifted = jax.vmap(
        jax.vmap(fourier_shift_in_jax, in_axes=(None, axis), out_axes=axis),
        in_axes=(0, 0),  # batch_axis - 1
        out_axes=batch_axis,  # batch AFTER axis = 0
    )(data, shifts[..., :nd])
    assert shifted.shape == ((shifts.shape[1],) + data.shape)

    spatial_axes = tuple(a for a in range(nd) if a != axis)
    return DataDef(
        orig=data,
        observed=shifted,
        axis=axis,
        spatial_axes=spatial_axes,
        batch_axes=(batch_axis,),
        reduce_axes=(),
    )


@interactive
def test_data__spatial_echo_batch(data__spatial_echo_batch):
    """Just a check how the data are generated."""
    dat = data__spatial_echo_batch
    _, axes = plt.subplots(
        nrows=NVOL + 1, ncols=dat.orig.shape[0], sharex=True, sharey=True
    )
    for rep_idx, ax_col in enumerate(axes.T):
        _show(dat.orig[rep_idx], ax_col[-1])
        for idx, ax in enumerate(ax_col[:-1]):
            _show(dat.observed[idx, rep_idx], ax)

    for idx, ax in enumerate(axes[0]):
        ax.set_title(f"Batch = {idx}")
    for idx, ax in enumerate(axes[:-1, 0]):
        ax.set_ylabel(f"Echo = {idx}")
    axes[-1, 0].set_ylabel("Orig")

    plt.show()


@pytest.fixture(params=[1, 2, 3])
def data__spatial_echo_coil_batch(shifts, request):
    nd = request.param
    axis = 0
    coil_axis = -1
    batch_axis = 1
    batch_size = 3
    data = gen_phantom(nd)
    basis = grid_basis(data.shape)
    centre_ = jnp.array([0.5] * nd)
    centres = centre_[:, None] * jnp.array([-1, 1])
    assert centres.shape[-1] == NSOS
    coil_profile = jnp.linalg.norm(basis[..., None] - centres, axis=-2)
    coil_profile = (coil_profile.max() - coil_profile) / coil_profile.max()
    assert coil_profile.shape == data.shape + (NSOS,)

    batch_factor = 0.8 + 0.2 * jnp.arange(batch_size)
    # in the absence of axis (echo), batch_axis is not 1 but 0
    data = jnp.expand_dims(batch_factor, tuple(range(1, nd + 1))) * data
    shifts = jnp.expand_dims(batch_factor, (1, 2)) * shifts

    data = jnp.expand_dims(data, coil_axis) * coil_profile
    # same as data[..., None] * coil_profile
    if INTERACTIVE:
        _, axes = plt.subplots(nrows=2, ncols=2, sharey=True, sharex=True)
        for ax_row, arr in zip(axes, [coil_profile, data[1]]):
            for coil_idx, ax in enumerate(ax_row):
                _show(arr[..., coil_idx], ax)
        plt.show()

    shifted = jax.vmap(
        jax.vmap(fourier_shift_in_jax, in_axes=(None, axis, None), out_axes=axis),
        in_axes=(0, 0, None),  # batch_axis - 1
        out_axes=batch_axis,  # batch AFTER axis = 0
    )(data, shifts[..., :nd], tuple(range(nd)))
    assert shifted.shape == ((shifts.shape[1],) + data.shape)

    spatial_axes = tuple(a for a in range(nd) if a != axis)
    return DataDef(
        orig=data,
        observed=shifted,
        axis=axis,
        spatial_axes=spatial_axes,
        batch_axes=(batch_axis,),
        reduce_axes=(coil_axis,),
    )


@interactive
def test_data__spatial_echo_coil_batch(data__spatial_echo_coil_batch):
    """Just a check how the data are generated."""
    dat = data__spatial_echo_coil_batch
    _, axes = plt.subplots(nrows=NVOL + 1, ncols=NSOS, sharex=True, sharey=True)
    for coil_idx, ax_col in enumerate(axes.T):
        for echo_idx, ax in enumerate(ax_col[:-1]):
            _show(dat.observed[echo_idx, 1, ..., coil_idx], ax)

        _show(dat.orig[1, ..., coil_idx], ax_col[-1])

    for idx, ax in enumerate(axes[0]):
        ax.set_title(f"Coil = {idx}")
    for idx, ax in enumerate(axes[:-1, 0]):
        ax.set_ylabel(f"Echo = {idx}")
    axes[-1, 0].set_ylabel("Orig")

    plt.show()
