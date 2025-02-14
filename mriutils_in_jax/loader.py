from pathlib import Path
from typing import Literal

import jax.numpy as jnp
import nibabel as nib
from jaxtyping import Array, Complex
from loguru import logger

from mriutils_in_jax.utils import parse_selection_from_string


class Loaded:
    def __init__(
        self,
        path_magn: Path,
        path_phase: Path,
        axis_echo: int = -1,
        selection: str = "",
        check_phase: bool = True,
        magn_scale: float | Literal["percentile"] = 1.0,
    ):
        self.sel = parse_selection_from_string(selection)
        logger.debug("Loading magn from {}", path_magn)
        self.img = nib.nifti1.load(path_magn)
        img_phase = nib.nifti1.load(path_phase)
        if self.img.shape != img_phase.shape:
            raise ValueError(
                f"Incompatible shapes for magnitude {self.img.shape} "
                f"and phase {img_phase.shape}"
            )
        magn = jnp.array(self.img.dataobj[*self.sel])
        logger.info("Loaded magn.shape: {}", magn.shape)
        if magn.ndim > 4:
            raise NotImplementedError(
                f"Batch dimensions are not yet supported, magn.shape = {magn.shape}"
            )
        if magn_scale == "percentile":
            logger.debug("Computing magnitude's 99th percentile")
            self.scale = jnp.nanpercentile(magn, jnp.array(99))
        elif isinstance(magn_scale, float):
            self.scale = magn_scale
        else:
            raise ValueError(
                f"Unexpected value for {magn_scale=}, "
                "must be a float, 'percentile'"
            )

        logger.debug("Scaling the magnitude by {}", self.scale)
        self.magn = magn / self.scale

        logger.debug("Loading phase from {}", path_phase)
        phase = jnp.array(img_phase.dataobj[*self.sel])
        if check_phase:
            # this is an expensive operation for large arrays, so may be worth skipping
            logger.debug("Computing phase limits")
            # below is ~ ×5 times faster than nanpercentile(x, [0, 100])
            vlims = jnp.array([jnp.nanmin(phase), jnp.nanmax(phase)])
            if not jnp.allclose(vlims, jnp.array([-jnp.pi, jnp.pi])):
                raise ValueError(f"phase must be scaled to [-π,π], got {vlims}")
        self.phase = phase

        if axis_echo != -1:
            logger.info("Moving axis_echo = {} to the end", axis_echo)
            self.magn = jnp.swapaxes(self.magn, axis_echo, -1)
            self.phase = jnp.swapaxes(self.phase, axis_echo, -1)
        self.shape = magn.shape

    @property
    def complex(self) -> Complex[Array, "*spatial echo"]:
        return self.magn * jnp.exp(1j * self.phase)
