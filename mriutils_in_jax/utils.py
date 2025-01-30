import jax.numpy as jnp
from jaxtyping import Array, Float


def grid_basis(
    grid_shape: tuple[int, ...], stacking_axis: int = -1
) -> Float[Array, "*batch ndim"]:
    basis_vecs = [jnp.linspace(-1, 1, size) for size in grid_shape]
    return jnp.stack(jnp.meshgrid(*basis_vecs, indexing="ij"), axis=stacking_axis)


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
