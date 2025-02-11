import jax
import jax.numpy as jnp
import numpy as onp
from jaxtyping import Array, Float
from skimage_in_jax.registration import phase_cross_correlation
from mriutils_in_jax.utils import grid_basis


def fourier_shift(
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


def sos(
    array: onp.ndarray | Float[Array, "..."],
    batch_axes: tuple[int, ...] = (),
    keepdims=False,
) -> onp.ndarray | Float[Array, "..."]:
    return (array**2).sum(batch_axes, keepdims=keepdims) ** 0.5


def register(
    array: Float[Array, "axis ..."],
    ref: int = -1,
    axis: int = -1,
    upsample_factor: int = 100,
    try_in_parallel: bool = True,
) -> Float[Array, "..."]:
    """Register subvolumes along specified axis.

    Parameters:
    array : ArrayProxy
        Multidimensional array which will be registered along axis=0.
        Loaded in memory.
    ref : int, default = -1
        Index of the reference volume along the subvolume axis
    axis : int, default = -1
        Axis which will be subdivided and each subvolume along this axis
        will be registered against array.take(ref, axis)
    try_in_parallel: bool, default is True
        If True and the number of available devices is greater than the number of
        subvolumes to register, will attempt to register each on a separate device.
        Otherwise vectorises the computation over a single available device.
        By default JAX consider a CPU as a singe device. In order to parallelise over
        CPU cores, set 'XLA_FLAGS="--xla_force_host_platform_device_count=1"'
    """
    ref = array.take(ref, axis)
    axis = axis % array.ndim  # pmap requires non-negative integers 🤷

    def register_single_vol(moving: Float[Array, "..."]) -> Float[Array, " ndim"]:
        return phase_cross_correlation(ref, moving, upsample_factor=upsample_factor)[0]

    jmap = (
        jax.pmap
        if try_in_parallel and len(jax.devices()) > array.shape[axis]
        else jax.vmap
    )
    return jmap(register_single_vol, in_axes=axis, out_axes=0)(array)
