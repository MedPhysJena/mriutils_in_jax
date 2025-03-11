from pathlib import Path

import numpy as onp
import matplotlib.pyplot as plt
import nibabel as nib
import typer

from jaxtyping import Array, Float
from mriutils_in_jax.loader import Loaded
from mriutils_in_jax.register import ExecutionMode, register_complex_data

app = typer.Typer()


def plot(shifts: Float[Array, "necho ndim"]):
    """Convenience function to plot the identified shifts."""
    ax = plt.subplot()
    ax.plot(shifts, marker=".")
    ax.grid()
    ax.set_ylabel("Shift (voxel)")
    ax.set_xlabel("Echo (index)")
    return ax


@app.command()
def register_echos_cli(
    magn: Path,
    phase: Path,
    output_base: Path,
    axis: int = -1,
    coil_axis: int | None = None,
    phase_lim_check: bool = True,
    mode: ExecutionMode = "low_memory",
    compress: bool = False,
    plot_shifts: bool = True,
):
    output_base.parent.mkdir(exist_ok=True, parents=True)

    data = Loaded(
        magn,
        phase,
        axis_echo=axis,
        magn_scale="percentile",
        check_phase=phase_lim_check,
    )
    data.mask_infinite_inplace()

    registered_complex, shifts = register_complex_data(
        data.magn, data.phase, axis_coil=coil_axis, axis_echo=axis, execution_mode=mode
    )

    if plot_shifts:
        plot(shifts)
        plt.savefig(output_base.with_suffix(".png"))
        plt.close()

    suffix = "nii.gz" if compress else "nii"
    for name, fn in zip(["phase", "magn"], [onp.angle, abs]):
        nib.nifti1.Nifti1Image(
            onp.where(data.mask_finite, fn(registered_complex), onp.nan),
            data.img.affine,
            data.img.header,
        ).to_filename(output_base.parent / (output_base.name + f"-{name}.{suffix}"))
