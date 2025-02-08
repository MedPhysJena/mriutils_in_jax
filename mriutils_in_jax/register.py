import jax
import jax.numpy as jnp
import nibabel as nib
import numpy as onp
from jaxtyping import Array, Float
from loguru import logger
from skimage_in_jax.registration import phase_cross_correlation
from tqdm import trange


def sos(
    array: onp.ndarray | Float[Array, "..."],
    batch_axes: tuple[int, ...] = (),
    keepdims=False,
) -> onp.ndarray | Float[Array, "..."]:
    return (array**2).sum(batch_axes, keepdims=keepdims) ** 0.5


def register(
    array: nib.arrayproxy.ArrayProxy,
    axis: int = -1,
    ref: int = -1,
    translation_axes: tuple[int, ...] | None = None,
    reduction_axes: tuple[int, ...] | None = None,
    parallel=False,
    jit=False,
    upsample_factor: int = 100,
) -> Float[Array, "..."]:
    """Register subvolumes along specified axis.

    Parameters:
    array : ArrayProxy
        Multidimensional array which will be registered along axis.
        Loaded in memory or not
    axis : int, default = -1
        Axis which will be subdivided and each subvolume along this axis
        will be registered against array.take(ref, axis)
    ref : int, default = -1
        Index of the reference volume along the subvolume axis
    translation_axes : tuple[int, ...], optional
        Axes along which the subvolumes will be translated to perform registration.
        By default, all but axis.
    reduction_axes : tuple[int, ...], optional
        Axes along which the subvolumes will NOT be translated to perform registration.
        Instead, these axes will be collapsed using a sum-of-squares.
        By default, all that is neither translation_axes, nor the subvolume axis.
    """
    if parallel and not jit:
        logger.warning("parallel implies jit, jit=False is ignored")
    ndim = array.ndim
    axis = axis % ndim  # cast negative axes to positive values
    axes_ = set(range(ndim)).difference(set([axis]))
    if translation_axes is not None:
        translation_axes = tuple(a % ndim for a in translation_axes)
        if diff_ := set(translation_axes).difference(axes_):
            raise ValueError(
                f"Unexpected axes in translation_axes: {translation_axes=} != "
                f"{diff_} == {tuple(range(array.ndim))} \\ {(axis,)}"
            )
    if reduction_axes is not None:
        reduction_axes = tuple(a % ndim for a in reduction_axes)
        if diff_ := set(reduction_axes).difference(axes_):
            raise ValueError(
                f"Unexpected axes in reduction_axes: {reduction_axes=} != "
                f"{diff_} == {tuple(range(array.ndim))} \\ {(axis,)}"
            )

    if translation_axes is None and reduction_axes is None:
        translation_axes = tuple(axes_)
        logger.info("Inferred translation_axes = {}", translation_axes)
        reduction_axes = ()
        batch_axes = ()
    elif translation_axes is not None and reduction_axes is None:
        reduction_axes = ()
        batch_axes = tuple(axes_.difference(translation_axes))
        logger.info("Inferred batch_axes = {}", batch_axes)
    elif translation_axes is None and reduction_axes is not None:
        batch_axes = ()
        translation_axes = tuple(axes_.difference(reduction_axes))
        logger.info("Inferred translation_axes = {}", translation_axes)
    elif translation_axes is not None and reduction_axes is not None:
        batch_axes = tuple(
            axes_.difference(translation_axes).difference(reduction_axes)
        )
    else:
        raise NotImplementedError("cannot be reached, pasifying my static checker")

    if batch_axes:
        raise NotImplementedError

    def collapse_reduction(array):
        return sos(array, reduction_axes)

    # do not keep axes after all is reduced / indexed, also squeeze `axis`
    ref = jnp.array(collapse_reduction(onp.take(array, onp.array([ref]), axis=axis)))

    def register_single_vol(moving: Float[Array, "..."]) -> Float[Array, " ndim"]:
        return phase_cross_correlation(
            # squeeze moving is critical in pmap
            ref.squeeze(),
            moving.squeeze(),
            upsample_factor=upsample_factor,
        )[0]

    if jit:
        register_single_vol = jax.jit(register_single_vol)

    if parallel:
        logger.info("Loading all data in memory")
        return jax.pmap(register_single_vol, in_axes=axis, out_axes=0)(
            jnp.array(
                sos(array, reduction_axes, keepdims=True)
            )  # keepdim so that pmap knows what to do
        )
    else:
        return jnp.stack(
            [
                register_single_vol(
                    jnp.array(
                        collapse_reduction(
                            onp.take(array, indices=onp.array([idx]), axis=axis)
                        ).squeeze()
                    )
                )
                for idx in trange(array.shape[axis])
            ]
        )
