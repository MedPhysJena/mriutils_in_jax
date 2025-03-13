from enum import Enum

import jax
import jax.numpy as jnp
import numpy as onp
from jaxtyping import Array, Complex, Float
from skimage_in_jax.registration import phase_cross_correlation
from tqdm import trange

from mriutils_in_jax.utils import grid_basis, take, update_axis_after_indexing


class ExecutionMode(str, Enum):
    low_memory = "low_memory"
    vectorized = "vectorized"
    threaded = "threaded"


def fourier_shift(
    array: Float[Array, "..."] | Complex[Array, "..."],
    shift: Float[Array, " ndim"],
    axes: tuple[int, ...] | None = None,
) -> Float[Array, "..."] | Complex[Array, "..."]:
    """
    Applies a sub-pixel spatial shift to an input array using the Fourier shift theorem.

    This function computes the Fourier transform of the input array, applies a phase
    shift corresponding to the desired translation along the specified axes, and then
    computes the inverse Fourier transform to obtain the shifted array. The appropriate
    FFT routines are chosen based on whether the input array is real-valued or
    complex-valued.

    Parameters
    ----------
    array : A real- or complex-valued input array.
        The array to be shifted.
    shift : A 1D array of shift values.
        Each element corresponds to the desired shift along the respective axis.
        Its length must equal the number of axes along which the shift is applied.
        (By default, all axes of the input.)
    axes : A tuple of integers, optional.
        The axes along which to apply the shift. If None, the shift is applied to
        all axes of the array.

    Returns
    -------
    shifted : A real- or complex values array
        The shifted array, having the same shape and data type as the input array.

    Raises
    ------
        AssertionError: If the length of the shift array does not match the number
        of axes specified.

    Notes
    -----
    - When the input array is real-valued, the function uses the real FFT
        (rfftn and irfftn) to optimize performance. For complex arrays, the standard
        FFT (fftn and ifftn) is used.
    - The phase shift is computed as:
            exp(-2πj * (frequency_grid @ shift))
        and is applied in the Fourier domain.
    """
    array = jnp.asarray(array)
    shift = jnp.asarray(shift)
    if use_rfft := not jnp.iscomplexobj(array):
        fft, ifft = jnp.fft.rfftn, jnp.fft.irfftn
    else:
        fft, ifft = jnp.fft.fftn, jnp.fft.ifftn
    if axes is None:
        axes = tuple(range(array.ndim))
    batch_axes = tuple(a for a in range(array.ndim) if a not in axes)
    assert len(axes) == len(shift)
    shape = tuple(array.shape[a] for a in axes)

    # Compute the frequency grid
    phase_shift = grid_basis(shape, reciprocal=True, rfft=use_rfft) @ shift
    # Apply Fourier shift theorem
    return ifft(
        fft(array, axes=axes)
        * jnp.expand_dims(jnp.exp(-2j * jnp.pi * phase_shift), batch_axes),
        axes=axes,
    )


def sos(
    array: onp.ndarray | Float[Array, "..."],
    batch_axes: tuple[int, ...] = (),
    keepdims=False,
) -> onp.ndarray | Float[Array, "..."]:
    return (array**2).sum(batch_axes, keepdims=keepdims) ** 0.5


def identify_necessary_shifts(
    array,
    ref: int = -1,
    axis: int = -1,
    upsample_factor: int = 100,
    execution_mode: ExecutionMode = ExecutionMode.low_memory,
    combine_fn=lambda x: x,
) -> Float[Array, "nvol ndim"]:
    """
    Identify shifts that register subvolumes along a specified axis.

    The computation can be executed in one of three modes:
      - "low_memory": Loads one volume at a time (minimal memory footprint, default).
        This is possible and helpful because of the way nibabel's ArrayProxy works.
      - "vectorized": Loads all volumes in memory and uses JAX vectorization (vmap).
      - "threaded": Loads all volumes in memory and tries to threaded across
        available devices (GPUs or CPU threads, depending on jax config, see below).
        If enough threads are available, it assigns one subvolume to each otherwise
        falling back to vmap.

    Additionally, once the data is loaded in the memory, an optional `combine_fn`
    is applied over `combine_axes` which may further reduce the data size
    (e.g. by applying SoS channel combination)

    Parameters
    ----------
    array : np.ndarray, jnp.ndarray, or nib.arrayproxy.ArrayProxy
        Array which will be registered along `axis`.
    ref : int, default = -1
        Index of the reference volume along the subvolume axis.
    axis : int, default = -1
        Axis which will be subdivided; each subvolume along this axis will be
        registered against `array.take(ref, axis)`.
    upsample_factor : int, default = 100
        Upsampling factor for subpixel registration accuracy.
        Images will be registered to within `1 / upsample_factor` of a pixel.
        For example `upsample_factor == 20` means the images will be registered
        within 1/20th of a pixel. The default (100) may be overly zealous.
    execution_mode : str, default = "vectorized"
        Execution mode: one of "low_memory", "vectorized", or "threaded".
    combine_fn : Callable[[Array], Array]
        Optional function, mapping array to an array, to reduce dimensions that are
        irrelevant for registration, e.g. by applying a channel combination.
        Note, that it must operate on _indexed_ arrays, i.e. with ndim of array.ndim - 1

    Returns
    -------
    shifts : Float[Array, "nvol ndim"]
        Array of necessary shifts, stacked along the leftmost dim over all volumes.
        Each shift has a length equal to the volume's dimensionality.

    Notes
    -----
    - If installed with a GPU support, jax will try to use GPUs if available.
      If no GPU is found, it will default to use the CPU. (This can also be controlled
      by setting `JAX_PLATFORM_NAME` environment variable to `gpu` or `cpu`.)
    - By default, JAX sees a CPU, however many threads there is, as a single device.
      In this case execution_mode="vectorized" will attempt to use all available threads
      _when possible_, e.g. when performing FFT due to its underlying vectorisation.
    - If you have sufficient number of threads, you may want to force jax to treat each
      threat as a separate device and `pmap` over them (running the entire subvolume
      processing on a single thread). To do this simply set execution_mode="threaded"
      and the code will set the number of devices to the number of threads automatically.
      In my limited testing, I observed that forcing the device count to the max number
      proves marginally more beneficial than setting it to the exact number of echoes.
    """
    if execution_mode not in ExecutionMode:
        raise ValueError(
            f"Invalid execution_mode {execution_mode!r}. Choose one of {ExecutionMode}."
        )
    if execution_mode == "threaded":
        import multiprocessing
        import os

        os.environ["XLA_FLAGS"] = (
            f"--xla_force_host_platform_device_count={multiprocessing.cpu_count()}"
        )

    axis = axis % array.ndim  # pmap requires non-negative integers
    # combine_fn must be able to operate on a single echo array / subvolume
    reference = combine_fn(take(array, indices=ref, axis=axis))

    @jax.jit
    def identify_necessary_shifts_in_single_vol(
        moving: Float[Array, "..."],
    ) -> Float[Array, " ndim"]:
        """Compute the shift required to align a given volume with the reference.

        Compile combine_fn into this function directly.
        """
        return phase_cross_correlation(
            reference, combine_fn(moving), upsample_factor=upsample_factor
        )[0]

    # Choose the processing strategy based on the execution_mode.
    if execution_mode == "low_memory":
        # Process one volume at a time using a comprehension (thanks to `take`)
        shifts = [
            identify_necessary_shifts_in_single_vol(take(array, indices=idx, axis=axis))
            for idx in trange(
                array.shape[axis], position=1, leave=False, desc="Processing volumes"
            )
        ]
        return jnp.stack(shifts, axis=0)

    # In parallel execution the entire data is loaded first.
    # The innate mechanism of {v,p}map squeezes `axis` out and combine_fn inside
    # identify_necessary_shifts_in_single_vol will operate on each subvolume
    if execution_mode == "threaded" and len(jax.devices()) > array.shape[axis]:
        jmap = jax.pmap
    else:
        jmap = jax.vmap
    return jmap(identify_necessary_shifts_in_single_vol, in_axes=axis, out_axes=0)(
        jnp.array(array)
    )


def register_complex_data(
    magn,
    phase,
    axis_echo: int = -2,
    axis_coil: int | None = -1,
    execution_mode: ExecutionMode = ExecutionMode.low_memory,
) -> tuple[Complex[Array, "..."], Float[Array, "necho ndim"]]:
    """Register (and shift) subvolumes along specified axis.

    The computation can be executed in one of three modes:
      - "low_memory": Loads one volume at a time (minimal memory footprint, default).
        This is possible and helpful because of the way nibabel's ArrayProxy works.
      - "vectorized": Loads all volumes in memory and uses JAX vectorization (vmap).
      - "threaded": Loads all volumes in memory and tries to threaded across
        available devices (GPUs or CPU threads, depending on jax config, see below).
        If enough threads are available, it assigns one subvolume to each otherwise
        falling back to vmap.

    Parameters
    ----------
    magn, phase : np.ndarray, jnp.ndarray, or nib.arrayproxy.ArrayProxy
        Pair of input arrays
    axis_echo: int, default = -2
        Axis which will be subdivided and each subvolume along this axis
        will be registered against array.take(ref, axis).
        In this project, second from the right.
    axis_coil : int, default = -1
        In general, when set to None, the data assumed to not include multiple channels.
        Otherwise, this axis will be reduced using sum-of-squares before identifying
        the shifts, and then each channel will be shifted by the same amount.
        In this project, the channels are assumed to be stacked along the rightmost dim.
    execution_mode : str, default = "vectorized"
        Execution mode: one of "low_memory", "vectorized", or "threaded".

    Notes
    -----
    - If installed with a GPU support, jax will try to use GPUs if available.
      If no GPU is found, it will default to use the CPU. (This can also be controlled
      by setting `JAX_PLATFORM_NAME` environment variable to `gpu` or `cpu`.)
    - By default, a CPU, however many threads there is, is seen as a single device.
      In this case "vectorized" will attempt to use the multiple threads at times, e.g.
      due to the underlying vectorised FFT implementation.
    - If you have sufficient number of threads, you may want to force jax to treat each
      threat as a separate device and `pmap` over them (running the entire subvolume
      processing on a single thread). For this export the following environment variable
      `XLA_FLAGS="--xla_force_host_platform_device_count=16"` (or whatever number of
      threads. You can use `os.environ` but it must be done before jax is imported)
    """
    if execution_mode not in ExecutionMode:
        raise ValueError(
            f"Invalid execution_mode {execution_mode!r}. Choose one of {ExecutionMode}."
        )

    if magn.shape != phase.shape:
        raise ValueError(
            "Specified niftis have incompatible shapes: "
            f"{magn.shape=} != {phase.shape=}"
        )
    _n_spatial_dims = magn.ndim-1 -(0 if axis_coil is None else 1 )
    if _n_spatial_dims>3:
        raise ValueError(
            f"Unexpected number of dimensions: {magn.shape=}, {axis_echo=}, {axis_coil=}. " 
            "Currently supports only up to 3 spatial dims"
        )


    # convert numpy array to jax immediately as it is already in memory, but
    # _certainly_ don't do that with ArrayProxy, as it would load it all in memory
    if isinstance(magn, onp.ndarray):
        magn = jnp.array(magn)
    if isinstance(phase, onp.ndarray):
        phase = jnp.array(phase)

    # The following should error if there are NaNs, rather than FFT them silently
    phase_min, phase_max = onp.min(phase).item(), onp.max(phase).item()
    # Should I bother masking NaNs?
    if not (phase_min >= -onp.pi) and (phase_max <= onp.pi):
        raise ValueError(
            f"phase must be scaled to [-π,π], got [{phase_min, phase_max}]"
        )

    # cast negative axis value to a positive (pmap requires)
    axis_echo = axis_echo % magn.ndim

    ## Step 1: identify how much to shift
    # using dataobj allows to read directly into jax, skipping nibabel's cache

    if axis_coil is None:

        def combine_fn(x):
            return x
    else:
        axis_coil_after_echo_is_indexed = update_axis_after_indexing(
            ndim=magn.ndim, target=axis_coil, removed=axis_echo
        )

        def combine_fn(x):
            return jnp.sum(x**2, axis=axis_coil_after_echo_is_indexed) ** 0.5

    shifts = identify_necessary_shifts(
        magn,
        axis=axis_echo,
        execution_mode=execution_mode,
        combine_fn=combine_fn,
    )

    ## Step 2: shift in Fourier space

    ## Step 2.1: prepare the shifting callback function
    # It will be called on each complex echo with the echo axis being squeezed out*.
    # Because of this, if coil is specified, it should, in general, be updated to
    # reflect that echo is indexed (e.g. if axis_echo=3, axis_coil=4 -> axis_coil
    # must become 3)
    #
    # *Why not keep the singleton dim:
    # - FFTying over it, while being kinda unnecessary anyway, was also a no go, as:
    #   - in jax<0.5.0, jax.fft.fftn could not handle more than 3 dims
    #   - in jax==0.5.0, jax.fft.fftn naively attempting to feed this singleton to FFT
    #     I experienced odd OOM and never looked deeper into it
    # - vectorising over it may be an option, but at this point I felt it was easier
    #   to squeeze than to keep track of vmap-ping over 1 (singleton) or 2 (echo & coil)
    #   axes...
    # - I am not sure, at the time of writing this comment, if pmap dictates the squeeze
    if axis_coil is None:
        shift = fourier_shift  # will shift the entire array
    else:
        # nesting pmap may be ill-advised. Plus N_channel < N_echo most of the time,
        # so I chose to use vmap here
        shift = jax.vmap(
            # the following axes=... are trivial as both of the other two dims are
            # squeezed out, thus the remaining axes are ordered
            lambda x, s: fourier_shift(x, s, axes=tuple(range(magn.ndim - 2))),
            # in_axes capture one axis per input: axis_coil... for the array
            # and None for the shift (as shifts are not repeated for each coil,
            # but apply to all coils)
            in_axes=(axis_coil_after_echo_is_indexed, None),
            out_axes=axis_coil_after_echo_is_indexed,
        )

    ## Step 2.2: actually shift
    if execution_mode == "low_memory":
        shifted_complex = jnp.stack(
            [
                shift(
                    take(magn, indices=idx, axis=axis_echo)
                    * jnp.exp(1j * take(phase, indices=idx, axis=axis_echo)),
                    shifts[idx],
                )
                for idx in trange(
                    magn.shape[axis_echo], position=1, leave=False, desc="Echo"
                )
            ],
            axis=axis_echo,
        )
    else:
        if execution_mode == "threaded" and len(jax.devices()) >= shifts.shape[0]:
            jmap = jax.pmap
        else:
            jmap = jax.vmap
        # in_axes shows which axes to zip in both inputs: axis_echo in the complex array
        # and 0 (leftmost) in shift
        shifted_complex = jmap(shift, in_axes=(axis_echo, 0), out_axes=axis_echo)(
            jnp.array(magn) * jnp.exp(1j * jnp.array(phase)),
            shifts,
        )

    return shifted_complex, shifts
