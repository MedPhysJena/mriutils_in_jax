import json
from typing import Iterable
from pathlib import Path
from jaxtyping import Array, Float

import click
import matplotlib.pyplot as plt
import nibabel as nib
import jax.numpy as jnp
from cycler import cycler
from scipy.ndimage import fourier_shift
from skimage_in_jax.registration import phase_cross_correlation
from tqdm import trange


class Registration:
    def __init__(
        self,
        time: Float[Array, " nvol"] | None = None,
        axis: int = -1,
        upsample_factor: float = 100,
    ):
        if time is not None:
            assert time.ndim == 1, f"time must be a 1D array, got {time.shape=}"
        self.time = time

        self.axis = axis
        self.upsample_factor = upsample_factor

    def regiser(
        self, magn: Float[Array, "..."], ref: int = 0
    ) -> Float[Array, "nvol ndim"]:
        img_ref = magn.take(indices=ref, axis=self.axis)
        if self.time is not None:
            if self.time.size != magn.shape[self.axis]:
                raise ValueError(f"{self.time.size=} != {magn.shape[self.axis]=}")
        click.echo("Registration")

        return jnp.stack(
            [
                phase_cross_correlation(
                    img_ref,
                    magn.take(idx, axis=self.axis),
                    upsample_factor=self.upsample_factor,
                )[0]
                if idx != ref
                else jnp.zeros(img_ref.ndim)
                for idx in trange(magn.shape[self.axis])
            ]
        )

    def regress(
        self,
        shifts: Float[Array, "nvol ndim"],
        regress_axes: Iterable[int] = (2,),
        bipolar_axis: int | None = 0,
    ) -> Float[Array, "nvol ndim"]:
        x = self.time if self.time is not None else jnp.arange(shifts.shape[0])

        if spurrious_axes := set(regress_axes).difference(set(range(shifts.shape[1]))):
            raise ValueError(
                f"{regress_axes=} contains axes {spurrious_axes} "
                f"which are not present in the {shifts.shape[1]}D image"
            )
        shifts_predicted = jnp.zeros_like(shifts)

        # fit a linear trend
        for ax in regress_axes:
            slope, offset = jnp.polyfit(x, shifts[:, ax], deg=1)
            shifts_predicted = shifts_predicted.at[:, ax].set(slope * x + offset)

        if bipolar_axis is not None:
            # BIPOLAR READOUT: split in two and fit two linear trends
            for offs in [0, 1]:
                t = x[offs::2]
                slope, offset = jnp.polyfit(t, shifts[offs::2, bipolar_axis], deg=1)
                shifts_predicted = shifts_predicted.at[offs::2, bipolar_axis].set(
                    slope * x + offset
                )

        return shifts_predicted

    def plot(self, shifts, shifts_predicted=None, shifts_to_apply=None):
        x = self.time if self.time is not None else jnp.arange(shifts.shape[0])
        if shifts_predicted is None:
            fig, ax_shift = plt.subplots(nrows=1, figsize=(10, 5))
            axes = [ax_shift]
        else:
            fig, (ax_shift, ax_resid) = plt.subplots(
                nrows=2, sharex=True, figsize=(10, 8)
            )
            axes = [ax_shift, ax_resid]

        ax_shift.plot(x, shifts[:, :3], marker=".", alpha=0.5)
        ax_shift.legend(["readout", "phase", "slice"])

        if shifts_predicted is not None:
            ax_shift.set_prop_cycle(cycler(color=["#1f77b4", "#ff7f0e", "#2ca02c"]))
            ax_shift.plot(x, shifts_predicted[:, :3], ls="--")
            if shifts_to_apply is not None:
                ax_shift.plot(x, shifts_to_apply[:, :3])

            ax_resid.axhline(y=0, c="grey", ls="--")
            ax_resid.plot(x, (shifts - shifts_predicted)[:, :3], marker=".", alpha=0.5)
            if shifts_to_apply is not None:
                ax_resid.plot(x, (shifts - shifts_to_apply)[:, :3])

        for ax, label in zip(axes, ["Shift (voxel)", "Residual shift (voxel)"]):
            ax.grid()
            ax.set_ylabel(label)

        axes[-1].set_xlabel(f"Echo ({'ms' if self.time else 'index'})")

    def shift_real(
        self, array: Float[Array, "..."], shifts: Float[Array, "nvol ndim"]
    ) -> Float[Array, "..."]:
        shifted = []
        for idx in trange(array.shape[self.axis]):
            shifted.append(
                jnp.fft.ifftn(
                    fourier_shift(jnp.fft.fftn(array.take(idx, self.axis)), shifts[idx])
                ).real
            )
        return jnp.stack(shifted, axis=self.axis)  # TODO: check if it injects the axis

    def shift_complex(self, data, shifts):
        return self.shift_real(data.real, shifts) + 1j * self.shift_real(
            data.imag, shifts
        )


@click.command()
@click.argument("output_base", type=click.Path(path_type=Path))
@click.argument("input_magn", type=click.Path(path_type=Path))
@click.argument("input_phase", type=click.Path(path_type=Path))
@click.option("--input-header", type=click.Path(path_type=Path), default=None)
@click.option(
    "--axis",
    type=int,
    default=-1,
    help="Axis to perform registration over. By default, the last one. This axis will "
    " be split and each subvolume will be registered to the reference (default: -1)",
)
@click.option(
    "--ref", type=int, default=-1, help="Reference volume to register to. Default: 0"
)
@click.option("--upsample-factor", "-u", type=float, default=100)
@click.option(
    "--regress",
    is_flag=True,
    help="If to regress the shifts on indices of individual echoes. "
    "If not given, registration is performed as is",
)
@click.option(
    "--use-te",
    is_flag=True,
    help="If to use echos (or time shifts) instead of the indices.",
)
@click.option(
    "--compress", is_flag=True, help="If to compress the output niftis", default=False
)
@click.option(
    "--coil-axis",
    type=int,
    default=None,
    help="Axis to perform coil combination over. Only used if the data have more than "
    "4 axes. May be inferred if the header has description with chl in it",
)
@click.option(
    "--spatial-axes",
    "-s",
    type=str,
    default=None,
    help="Spatial axes along which to perform tranformation. Registration cannot be "
    "constrained to a subset of axes thus this yields no speedup. Default: all axes",
)
@click.option(
    "--shift-threshold",
    "-t",
    type=float,
    default=None,
    help="Minimum shift to be applied. Shifts lower than the threshold are ignored. "
    "Not used by default.",
)
@click.option(
    "--dry/--no-dry",
    default=False,
    help="If specified, no transformation will be applied, only registration plot will be produced",
)
def main(
    input_magn,
    input_phase,
    input_header,
    output_base,
    axis: int,
    ref,
    upsample_factor,
    regress,
    use_te,
    compress,
    coil_axis: int,
    spatial_axes,
    shift_threshold,
    dry,
):
    output_base.parent.mkdir(exist_ok=True, parents=True)

    if input_header:
        with open(input_header) as f:
            te_ms = jnp.array(json.load(f)["echoTime"])
    else:
        if use_te:
            raise ValueError("input_header must be specified if use_te is True")
        te_ms = None

    if spatial_axes is not None:
        spatial_axes = jnp.array([int(a) for a in spatial_axes.split(",")], dtype=int)

    click.echo("Loading data")
    img = nib.load(input_magn)

    magn = img.get_fdata()
    phase = nib.load(input_phase).get_fdata()

    mask_finite = jnp.isfinite(magn) & jnp.isfinite(phase)
    magn = jnp.where(mask_finite, magn, 0.0)
    phase = jnp.where(mask_finite, phase, 0.0)

    if img.ndim > 4:
        # assuming multiple channels
        if coil_axis is None:
            raise ValueError(
                f"Data is {img.ndim}D: assuming multiple coils. "
                "Dims cannot be inferred from the header. Specify --coil-axis"
            )

        magn_to_reg = jnp.sum(magn**2, axis=coil_axis, keepdims=True) ** 0.5
        nib.Nifti1Image(magn_to_reg, img.affine, img.header).to_filename(
            input_magn.parent / (input_magn.stem + "-sos.nii")
        )
    else:
        magn_to_reg = magn

    reg = Registration(
        time=te_ms if regress and use_te else None,
        axis=axis,
        upsample_factor=upsample_factor,
    )

    shifts = reg.regiser(magn_to_reg, ref=ref)

    if spatial_axes is None:
        shifts_to_apply = shifts
    else:
        shifts_to_apply = (
            jnp.zeros_like(shifts).at[:, spatial_axes].set(shifts[:, spatial_axes])
        )

    if shift_threshold is not None:
        shift_mask = abs(shifts_to_apply) > shift_threshold
        print(f"{shift_mask.mean():.1%} of shifts over threshold of {shift_threshold}")
        shifts_to_apply = jnp.where(shift_mask, shifts_to_apply, 0)

    if regress:
        shifts_predicted = reg.regress(shifts_to_apply)
        fig = reg.plot(
            shifts=shifts,
            shifts_predicted=shifts_predicted,
            shifts_to_apply=shifts_to_apply,
        )
    else:
        shifts_predicted = shifts_to_apply
        fig = reg.plot(
            shifts=shifts,
            shifts_to_apply=shifts_to_apply,
        )

    plt.savefig(
        output_base.parent
        / (output_base.name + f"-shifts-upsampling_factor={upsample_factor}.png"),
        bbox_inches="tight",
    )
    plt.close(fig)

    if not dry:
        data_shifted = reg.shift_complex(magn * jnp.exp(1j * phase), shifts_to_apply)

        suffix = "nii.gz" if compress else "nii"
        for name, fn in zip(["phase", "magn"], [jnp.angle, abs]):
            nib.Nifti1Image(
                jnp.where(mask_finite, fn(data_shifted), jnp.nan),
                img.affine,
                img.header,
            ).to_filename(output_base.parent / (output_base.name + f"-{name}.{suffix}"))


if __name__ == "__main__":
    main()
