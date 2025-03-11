from enum import Enum
from pathlib import Path

import numpy as onp
import matplotlib.pyplot as plt
import nibabel as nib
import typer

from jaxtyping import Array, Float
from mriutils_in_jax.loader import Loaded
from mriutils_in_jax.register import ExecutionMode, register_complex_data

# don't show local variables as those contain large arrays
app = typer.Typer(pretty_exceptions_show_locals=False)

class ExecutionMode(str, Enum):
    low_memory = "low_memory"
    vectorized = "vectorized"
    threaded = "threaded"

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
    mode: ExecutionMode = "low_memory",
    compress: bool = False,
    plot_shifts: bool = True,
):
    output_base.parent.mkdir(exist_ok=True, parents=True)

    magn_ = img.dataobj
    phase_ = nib.nifti1.load(phase).dataobj

    registered_complex, shifts = register_complex_data(
        magn_, phase_, axis_coil=coil_axis, axis_echo=axis, execution_mode=mode
    )

    if plot_shifts:
        plot(shifts)
        plt.savefig(output_base.with_suffix(".png"))
        plt.close()

    suffix = "nii.gz" if compress else "nii"
    for name, fn in zip(["phase", "magn"], [onp.angle, abs]):
        nib.nifti1.Nifti1Image(
            fn(registered_complex), img.affine, img.header
        ).to_filename(output_base.parent / (output_base.name + f"-{name}.{suffix}"))
