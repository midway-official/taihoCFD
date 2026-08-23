#!/usr/bin/env python3
"""Combine and visualize MPI output written by ``saveMeshData``.

The current C++ writer stores *owned* columns only.  Consequently rank files
must be concatenated directly; no ghost-column trimming is valid.  The VTK
writer uses ``STRUCTURED_GRID`` with the complete cell-center coordinates, so
non-uniform meshes retain their geometry instead of being approximated by
one-dimensional rectilinear axes.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np


FIELD_NAMES = ("u", "v", "p", "xc", "yc")
_RANK_PATTERN = re.compile(r"^(u|v|p|xc|yc)_(\d+)\.dat$")


def _rank_ids(data_dir: Path, ranks: int | Iterable[int] | None) -> list[int]:
    if ranks is None:
        found = {
            int(match.group(2))
            for path in data_dir.iterdir()
            if (match := _RANK_PATTERN.match(path.name)) and match.group(1) == "u"
        }
        if not found:
            raise FileNotFoundError(f"no u_<rank>.dat files in {data_dir}")
        ids = sorted(found)
    elif isinstance(ranks, int):
        if ranks < 1:
            raise ValueError("ranks must be positive")
        ids = list(range(ranks))
    else:
        ids = sorted(set(int(rank) for rank in ranks))
    if not ids or ids != list(range(ids[-1] + 1)):
        raise ValueError(f"rank files must be contiguous and start at zero: {ids}")
    return ids


def _load_array(path: Path, ny: int | None = None) -> np.ndarray:
    try:
        array = np.loadtxt(path, dtype=np.float64)
    except OSError as error:
        raise FileNotFoundError(path) from error
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        if ny is not None and array.size % ny == 0:
            array = array.reshape(ny, -1)
        else:
            array = array.reshape(1, -1)
    return array


def _mesh_shape(data_dir: Path) -> tuple[int | None, int | None]:
    params = data_dir.parent / "params.txt"
    if not params.exists():
        params = data_dir / "params.txt"
    if not params.exists():
        return None, None
    values = params.read_text(encoding="utf-8").split()
    if len(values) < 2:
        raise ValueError(f"invalid params.txt: {params}")
    return int(values[1]), int(values[0])


def load_and_combine_data(
    data_dir: str | Path = "result",
    ranks: int | Iterable[int] | None = None,
) -> dict[str, np.ndarray]:
    """Load rank files and concatenate their owned columns.

    ``ranks`` is a process count when it is an integer.  If omitted, the
    contiguous rank ids present in ``data_dir`` are discovered automatically.
    The returned dictionary contains ``u``, ``v``, ``p``, ``xc`` and ``yc``.
    """

    directory = Path(data_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"result directory does not exist: {directory}")
    ny, expected_nx = _mesh_shape(directory)
    rank_ids = _rank_ids(directory, ranks)
    per_field: dict[str, list[np.ndarray]] = {name: [] for name in FIELD_NAMES}

    for rank in rank_ids:
        arrays = {
            name: _load_array(directory / f"{name}_{rank}.dat", ny)
            for name in FIELD_NAMES
        }
        shape = arrays["u"].shape
        if any(array.shape != shape for array in arrays.values()):
            raise ValueError(f"rank {rank} files do not have identical shapes")
        if ny is not None and shape[0] != ny:
            raise ValueError(
                f"rank {rank} has {shape[0]} rows, expected {ny} from params.txt"
            )
        for name, array in arrays.items():
            per_field[name].append(array)

    combined = {
        name: np.concatenate(chunks, axis=1)
        for name, chunks in per_field.items()
    }
    shape = combined["u"].shape
    if expected_nx is not None and shape[1] != expected_nx:
        raise ValueError(
            f"owned-column widths sum to {shape[1]}, expected nx={expected_nx}; "
            "check --ranks and stale result files"
        )
    if not all(np.isfinite(array).all() for array in combined.values()):
        raise ValueError("result contains non-finite values")

    xc, yc = combined["xc"], combined["yc"]
    if shape[1] > 1 and not np.all(np.diff(xc, axis=1) > 0.0):
        raise ValueError("combined x cell centers are not strictly increasing")
    if shape[0] > 1 and not np.all(np.diff(yc, axis=0) > 0.0):
        raise ValueError("combined y cell centers are not strictly increasing")
    return combined


def save_combined_data(fields: dict[str, np.ndarray], output_dir: str | Path) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name in FIELD_NAMES:
        np.savetxt(directory / f"{name}_combined.dat", fields[name], fmt="%.17g")


def _flat(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=float).ravel(order="C")


def save_vtk(
    fields: dict[str, np.ndarray],
    output_dir: str | Path,
    filename: str = "result.vtk",
) -> Path:
    """Write a legacy VTK structured grid with cell-center point data."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    xc, yc = fields["xc"], fields["yc"]
    u, v, p = fields["u"], fields["v"], fields["p"]
    velocity_magnitude = np.hypot(u, v)
    ny, nx = xc.shape
    n_points = nx * ny

    with path.open("w", encoding="utf-8") as output:
        output.write("# vtk DataFile Version 3.0\n")
        output.write("Taiho-CFD cell-centered result\nASCII\n")
        output.write("DATASET STRUCTURED_GRID\n")
        output.write(f"DIMENSIONS {nx} {ny} 1\n")
        output.write(f"POINTS {n_points} double\n")
        for x_value, y_value in zip(_flat(xc), _flat(yc)):
            output.write(f"{x_value:.17g} {y_value:.17g} 0\n")
        output.write(f"POINT_DATA {n_points}\n")
        output.write("VECTORS Velocity double\n")
        for u_value, v_value in zip(_flat(u), _flat(v)):
            output.write(f"{u_value:.17g} {v_value:.17g} 0\n")
        for name, values in (
            ("Pressure", p),
            ("VelocityMagnitude", velocity_magnitude),
        ):
            output.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n")
            for value in _flat(values):
                output.write(f"{value:.17g}\n")
    return path


def save_tecplot(
    fields: dict[str, np.ndarray],
    output_dir: str | Path,
    filename: str = "result.plt",
) -> Path:
    """Write Tecplot ASCII POINT data without changing the y orientation."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    xc, yc = fields["xc"], fields["yc"]
    u, v, p = fields["u"], fields["v"], fields["p"]
    velocity_magnitude = np.hypot(u, v)
    ny, nx = xc.shape
    with path.open("w", encoding="utf-8") as output:
        output.write('TITLE = "Taiho-CFD result"\n')
        output.write('VARIABLES = "X", "Y", "U", "V", "P", "VelMag"\n')
        output.write(f'ZONE T="cells", I={nx}, J={ny}, F=POINT\n')
        for i in range(ny):
            for j in range(nx):
                output.write(
                    f"{xc[i, j]:.17g} {yc[i, j]:.17g} "
                    f"{u[i, j]:.17g} {v[i, j]:.17g} "
                    f"{p[i, j]:.17g} {velocity_magnitude[i, j]:.17g}\n"
                )
    return path


def _save_plots(fields: dict[str, np.ndarray], output_dir: Path, show: bool) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover - depends on local GUI stack
        raise RuntimeError(
            "matplotlib is unavailable; rerun with --no-plots or install it"
        ) from error

    xc, yc = fields["xc"], fields["yc"]
    u, v, p = fields["u"], fields["v"], fields["p"]
    velocity_magnitude = np.hypot(u, v)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    def contour(values: np.ndarray, title: str, name: str, cmap: str = "viridis") -> None:
        figure, axis = plt.subplots(figsize=(8, 5))
        image = axis.contourf(xc, yc, values, levels=40, cmap=cmap)
        figure.colorbar(image, ax=axis)
        axis.set(title=title, xlabel="x", ylabel="y", aspect="equal")
        figure.tight_layout()
        path = output_dir / name
        figure.savefig(path, dpi=150)
        saved.append(path)
        if not show:
            plt.close(figure)

    contour(velocity_magnitude, "Velocity magnitude", "velocity_magnitude.png", "magma")
    contour(p, "Pressure", "pressure.png", "coolwarm")

    # Matplotlib streamplot currently requires equally spaced axes.  The
    # solver intentionally supports stretched meshes, so interpolate only
    # for this display product; VTK/Tecplot retain the original coordinates.
    x_axis = xc[0, :]
    y_axis = yc[:, 0]
    nx = xc.shape[1]
    ny = yc.shape[0]
    x_uniform = np.linspace(x_axis[0], x_axis[-1], nx)
    y_uniform = np.linspace(y_axis[0], y_axis[-1], ny)

    def resample(values: np.ndarray) -> np.ndarray:
        along_x = np.vstack([
            np.interp(x_uniform, x_axis, row) for row in values
        ])
        return np.column_stack([
            np.interp(y_uniform, y_axis, along_x[:, column])
            for column in range(along_x.shape[1])
        ])

    u_stream = resample(u)
    v_stream = resample(v)
    magnitude_stream = np.hypot(u_stream, v_stream)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.streamplot(
        x_uniform,
        y_uniform,
        u_stream,
        v_stream,
        density=1.2,
        color=magnitude_stream,
    )
    axis.set(title="Streamlines", xlabel="x", ylabel="y", aspect="equal")
    figure.tight_layout()
    path = output_dir / "streamlines.png"
    figure.savefig(path, dpi=150)
    saved.append(path)
    if not show:
        plt.close(figure)
    if show:
        plt.show()
    return saved


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="result", help="directory containing rank files")
    parser.add_argument("--ranks", type=int, help="MPI process count; omit to auto-detect")
    parser.add_argument(
        "--output-dir",
        help="post-processing output directory (default: <data-dir>/postprocess)",
    )
    parser.add_argument("--no-plots", action="store_true", help="skip PNG generation")
    parser.add_argument("--show", action="store_true", help="display plots interactively")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "postprocess"
    fields = load_and_combine_data(data_dir, args.ranks)
    save_combined_data(fields, output_dir)
    vtk_path = save_vtk(fields, output_dir)
    tecplot_path = save_tecplot(fields, output_dir)
    plots = [] if args.no_plots else _save_plots(fields, output_dir, args.show)
    print(
        f"combined shape={fields['u'].shape} ranks={args.ranks or 'auto'} "
        f"output={output_dir}"
    )
    print(f"vtk={vtk_path}")
    print(f"tecplot={tecplot_path}")
    for path in plots:
        print(f"plot={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
