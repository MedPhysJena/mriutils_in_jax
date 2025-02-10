import jax.numpy as jnp
from jaxtyping import Array, Float


def grid_basis(
    grid_shape: tuple[int, ...],
    reciprocal=False,
    normalise=False,
    stacking_axis=-1,
    rfft=False,
) -> Float[Array, "*batch ndim"]:
    if reciprocal:
        basis_vecs = [jnp.fft.fftfreq(size) for size in grid_shape]
        if rfft:
            basis_vecs[-1] = jnp.fft.rfftfreq(grid_shape[-1])
    else:
        if rfft:
            logger.warning("rfft=True is ignored if reciprocal=False")
        basis_vecs = [jnp.linspace(-1, 1, size) for size in grid_shape]

    grid = jnp.stack(jnp.meshgrid(*basis_vecs, indexing="ij"), axis=stacking_axis)

    if normalise:
        norm = jnp.linalg.norm(grid, axis=-1)
        grid /= jnp.where(norm == 0, jnp.inf, norm)[..., None]
    return grid


def parse_selection_from_string(sel: str = "") -> tuple[slice, ...]:
    """Parse inputs like '10:-10,:,40:150' into tuple of slices"""
    if not sel:
        return ()
    sel_axes = sel.split(",")
    if len(sel_axes) > 4:
        raise NotImplementedError(
            f"selection string with more than 4 elements is not supported, got {sel}"
        )
    selection = []
    for axis_sel in sel_axes:
        ranges = axis_sel.split(":")
        if len(ranges) != 2:
            raise ValueError(
                f"Each axis selection must contain a single :, got {axis_sel}"
            )
        ranges = [int(idx) if idx else None for idx in ranges]
        selection.append(slice(*ranges))
    return tuple(selection)
