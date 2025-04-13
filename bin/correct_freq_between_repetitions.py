import json
from pathlib import Path

import typer

from mriutils_in_jax.phase import correct_repetition_phase


def main(
    moving_magn: Path,
    moving_phase: Path,
    reference_magn: Path,
    reference_phase: Path,
    header: Path,
    output_basename: Path | None = None,
    output_average: Path | None = None,
    axis_echo: int = -1,
    mask_fg_threshold: float | None = 0.4,
    sel: str = "",
    factor: int = 5,
    check_phase: bool = True,
    plot_hist: bool = True,
):
    correct_repetition_phase(
        moving_magn=moving_magn,
        moving_phase=moving_phase,
        reference_magn=reference_magn,
        reference_phase=reference_phase,
        te=json.loads(header.read_text())["echoTime"],
        output_basename=output_basename,
        axis_echo=axis_echo,
        mask_fg_threshold=mask_fg_threshold,
        sel=sel,
        factor=factor,
        check_phase=check_phase,
        plot_hist=plot_hist,
    )
if __name__ == "__main__":
    typer.run(main)
