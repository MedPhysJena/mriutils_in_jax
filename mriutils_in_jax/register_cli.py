from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as onp
import typer
from jaxtyping import Array, Float

from mriutils_in_jax.register import register_complex_data

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
    """Register subvolumes in MAGN and PHASE along an axis using the Fourier shift theorem.

    Shifted images will be written under OUTPUT_BASE-{magn,phase}.nii[.gz]
    (if --compress is passed).

    The values of the shifts and their plot (unless --no-plot-shifts is specified)
    will also be saved as OUTPUT_BASE.{png,txt}.

    If the data contains multiple channels, specify the dimension using --coil-axis
    1. to combine the magnitude over the channels using SoS prior to registration
    2. and to avoid attempting to shift the image along the channel dim

    The computation can be executed in one of three modes:
      - "low_memory": Loads one volume at a time (minimal memory footprint, default).
        This is possible and helpful because of the way nibabel's ArrayProxy works.
      - "vectorized": Loads all volumes in memory and uses JAX vectorization (vmap).
      - "threaded": Loads all volumes in memory and tries to threaded across
        available devices (GPUs or CPU threads, depending on jax config, see below).
        If enough threads are available, it assigns one subvolume to each otherwise
        falling back to vmap.

    - If installed with a GPU support, jax will try to use GPUs if available.
      If no GPU is found, it will default to use the CPU. (This can also be controlled
      by setting `JAX_PLATFORM_NAME` environment variable to `gpu` or `cpu`.)
    - By default, JAX sees a CPU, however many threads there is, as a single device.
      In this case --mode "vectorized" will attempt to use all available threads
      _when possible_, e.g. when performing FFT due to its underlying vectorisation.
    - If you have sufficient number of threads, you may want to force jax to treat each
      threat as a separate device and `pmap` over them (running the entire subvolume
      processing on a single thread). To do this simply pass --mode "threaded" and the
      code will set the number of devices to the number of threads automatically.
      In my limited testing, I observed that forcing the device count to the max number
      proves marginally more beneficial than setting it to the exact number of echoes.
    """
    output_base.parent.mkdir(exist_ok=True, parents=True)

    img = nib.nifti1.load(magn)
    magn_ = img.dataobj
    phase_ = nib.nifti1.load(phase).dataobj

    registered_complex, shifts = register_complex_data(
        magn_, phase_, axis_coil=coil_axis, axis_echo=axis, execution_mode=mode
    )

    if plot_shifts:
        plot(shifts)
        plt.savefig(output_base.with_suffix(".png"))
        plt.close()
    onp.savetxt(output_base.with_suffix(".shifts"), onp.array(shifts))

    suffix = "nii.gz" if compress else "nii"
    for name, fn in zip(["phase", "magn"], [onp.angle, abs]):
        nib.nifti1.Nifti1Image(
            fn(registered_complex), img.affine, img.header
        ).to_filename(output_base.parent / (output_base.name + f"-{name}.{suffix}"))
