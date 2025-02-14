from pathlib import Path

import jax.numpy as jnp
import nibabel as nib
import typer
from jaxtyping import Array, Complex
from loguru import logger
from tqdm import tqdm

from mriutils_in_jax.loader import Loaded, braced_glob


def aggregate(filenames_magn, filenames_phase, **kws) -> Complex[Array, "..."]:
    if len(filenames_magn) != len(filenames_phase):
        raise ValueError(
            "Inconsistent number of filenames: "
            f"{len(filenames_magn)=} != {len(filenames_phase)=}"
        )

    n = len(filenames_magn)

    agg = Loaded(filenames_magn[0], filenames_phase[0], **kws).complex
    for fm, fp in tqdm(zip(filenames_magn[1:], filenames_phase[1:]), total=n):
        agg = agg + Loaded(fm, fp, **kws).complex
    return agg / n




def main(
    pattern_magn: str,
    pattern_phase: str,
    output_basename: Path,
    check_phase: bool = True,
    magn_scale: float = 1.0,
):
    output_basename.parent.mkdir(exist_ok=True, parents=True)

    filenames_magn = braced_glob(pattern_magn)
    filenames_phase = braced_glob(pattern_phase)
    print(filenames_magn)
    print(filenames_phase)

    logger.debug("Aggregating")
    mean = aggregate(
        filenames_magn, filenames_phase, check_phase=check_phase, magn_scale=magn_scale
    )

    img = nib.nifti1.load(filenames_magn[0])
    nib.nifti1.Nifti1Image(abs(mean), img.affine, img.header).to_filename(
        output_basename.parent / f"{output_basename.name}-part=magn.nii"
    )
    nib.nifti1.Nifti1Image(jnp.angle(mean), img.affine, img.header).to_filename(
        output_basename.parent / f"{output_basename.name}-part=phase.nii"
    )


if __name__ == "__main__":
    typer.run(main)
