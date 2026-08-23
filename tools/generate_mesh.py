#!/usr/bin/env python3
"""Generate structured meshes understood by the Taiho-CFD solver.

The C++ reader currently consumes the legacy text format
(``params.txt``, ``x.dat``, ``y.dat``, ``bctype.dat``, ``zoneid.dat`` and
``zoneuv.txt``).  This module keeps that file format, but makes the geometry
and boundary convention explicit and validates it before writing:

* array rows are ordered from ``y=0`` to ``y=Ly`` (the last row is the top);
* columns are ordered from ``x=0`` to ``x=Lx``;
* ``bctype`` uses 0=interior, 1=wall, -2=velocity inlet, -1=pressure outlet;
* a positive ``zoneid`` selects the velocity value in ``zoneuv.txt``.

The functions are intentionally independent of Jupyter so that mesh creation
is reproducible from a command line and can be used by notebooks as a small
wrapper.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np


INTERIOR = 0
WALL = 1
PRESSURE_OUTLET = -1
VELOCITY_INLET = -2


def exp_stretch(n: int, length: float, alpha: float = 0.0) -> np.ndarray:
    """Return ``n + 1`` monotonically increasing grid-node coordinates.

    ``alpha=0`` is uniform.  For ``alpha>0`` both ends are refined while the
    midpoint remains at ``length/2``.  The explicit zero case also avoids the
    removable ``0/0`` in the exponential formula.
    """

    if n < 1:
        raise ValueError("n must be positive")
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("length must be finite and positive")
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    if alpha == 0.0:
        return np.linspace(0.0, length, n + 1, dtype=float)

    t = np.linspace(0.0, 1.0, n + 1, dtype=float)
    denominator = np.expm1(0.5 * alpha)
    coordinates = np.empty_like(t)
    left = t <= 0.5
    coordinates[left] = (
        0.5 * length * np.expm1(alpha * t[left]) / denominator
    )
    coordinates[~left] = length - (
        0.5 * length * np.expm1(alpha * (1.0 - t[~left])) / denominator
    )
    coordinates[0] = 0.0
    coordinates[-1] = length
    return coordinates


def _as_float_pair(value: Iterable[float], name: str) -> tuple[float, float]:
    pair = tuple(float(item) for item in value)
    if len(pair) != 2 or not all(np.isfinite(item) for item in pair):
        raise ValueError(f"{name} must contain two finite values")
    return pair


def validate_mesh(
    x: np.ndarray,
    y: np.ndarray,
    bctype: np.ndarray,
    zoneid: np.ndarray,
    zoneuv: np.ndarray,
    nx: int,
    ny: int,
) -> None:
    """Validate the exact assumptions made by ``readMesh`` and ``Mesh``."""

    if nx < 5 or ny < 5:
        raise ValueError("the solver requires at least 5 cells in each direction")
    expected_nodes = (ny + 1, nx + 1)
    expected_cells = (ny, nx)
    if x.shape != expected_nodes or y.shape != expected_nodes:
        raise ValueError(f"node arrays must have shape {expected_nodes}")
    if bctype.shape != expected_cells or zoneid.shape != expected_cells:
        raise ValueError(f"cell arrays must have shape {expected_cells}")
    if zoneuv.ndim != 2 or zoneuv.shape[1] != 2 or zoneuv.shape[0] == 0:
        raise ValueError("zoneuv must have shape (number_of_zones, 2)")
    if not all(np.isfinite(array).all() for array in (x, y, zoneuv)):
        raise ValueError("mesh coordinates and boundary values must be finite")

    # The current C++ geometry code expects an axis-aligned, monotonic grid.
    if not np.allclose(x, x[0:1, :]):
        raise ValueError("x coordinates must be constant along each row")
    if not np.allclose(y, y[:, 0:1]):
        raise ValueError("y coordinates must be constant along each column")
    if not np.all(np.diff(x[0, :]) > 0.0):
        raise ValueError("x coordinates must increase strictly from left to right")
    if not np.all(np.diff(y[:, 0]) > 0.0):
        raise ValueError("y coordinates must increase strictly from bottom to top")

    allowed = {INTERIOR, WALL, PRESSURE_OUTLET, VELOCITY_INLET}
    if not set(np.unique(bctype)).issubset(allowed):
        raise ValueError(f"bctype contains unsupported values; allowed={sorted(allowed)}")
    if np.any(zoneid < 0) or np.any(zoneid >= zoneuv.shape[0]):
        raise ValueError("zoneid contains an index outside zoneuv.txt")
    if (
        np.any(bctype[0, :] == INTERIOR)
        or np.any(bctype[-1, :] == INTERIOR)
        or np.any(bctype[:, 0] == INTERIOR)
        or np.any(bctype[:, -1] == INTERIOR)
    ):
        raise ValueError("all four physical outer boundaries must be explicit")


def _write_matrix(path: Path, matrix: np.ndarray, fmt: str) -> None:
    np.savetxt(path, matrix, fmt=fmt)


def write_mesh(
    output_dir: str | Path,
    x: np.ndarray,
    y: np.ndarray,
    bctype: np.ndarray,
    zoneid: np.ndarray,
    zoneuv: np.ndarray,
) -> Path:
    """Validate and write one solver mesh directory."""

    ny, nx = bctype.shape
    validate_mesh(x, y, bctype, zoneid, zoneuv, nx, ny)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    (directory / "params.txt").write_text(f"{nx} {ny}\n", encoding="utf-8")
    _write_matrix(directory / "x.dat", x, "%.17g")
    _write_matrix(directory / "y.dat", y, "%.17g")
    _write_matrix(directory / "bctype.dat", bctype, "%d")
    _write_matrix(directory / "zoneid.dat", zoneid, "%d")
    _write_matrix(directory / "zoneuv.txt", zoneuv, "%.17g")
    return directory


def _structured_coordinates(
    nx: int,
    ny: int,
    Lx: float,
    Ly: float,
    alpha_x: float,
    alpha_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    xs = exp_stretch(nx, Lx, alpha_x)
    ys = exp_stretch(ny, Ly, alpha_y)
    return np.meshgrid(xs, ys)


def generate_lid_driven_cavity(
    nx: int,
    ny: int,
    Lx: float = 1.0,
    Ly: float = 1.0,
    lid_u: float = 1.0,
    lid_v: float = 0.0,
    output_dir: str | Path = "ldc_mesh",
    alpha_x: float = 0.0,
    alpha_y: float = 0.0,
) -> Path:
    """Generate a cavity with a moving top wall.

    Rows are bottom-to-top, so ``zoneid[-1, :]`` is the moving lid.  Keeping
    all four sides as wall cells is important because the solver requires
    explicit physical boundary cells at the outer array boundary.
    """

    lid = _as_float_pair((lid_u, lid_v), "lid velocity")
    x, y = _structured_coordinates(nx, ny, Lx, Ly, alpha_x, alpha_y)
    bctype = np.full((ny, nx), WALL, dtype=np.int32)
    bctype[1:-1, 1:-1] = INTERIOR
    zoneid = np.zeros((ny, nx), dtype=np.int32)
    zoneid[-1, :] = 1
    zoneuv = np.asarray(((0.0, 0.0), lid), dtype=float)
    return write_mesh(output_dir, x, y, bctype, zoneid, zoneuv)


def generate_poiseuille_flow(
    nx: int,
    ny: int,
    Lx: float = 4.0,
    Ly: float = 1.0,
    inlet_u: float = 1.0,
    inlet_v: float = 0.0,
    output_dir: str | Path = "poiseuille_mesh",
    alpha_y: float = 0.0,
) -> Path:
    """Generate a channel with velocity inlet and zero-pressure outlet."""

    inlet = _as_float_pair((inlet_u, inlet_v), "inlet velocity")
    x, y = _structured_coordinates(nx, ny, Lx, Ly, 0.0, alpha_y)
    bctype = np.full((ny, nx), INTERIOR, dtype=np.int32)
    bctype[0, :] = WALL
    bctype[-1, :] = WALL
    bctype[:, 0] = VELOCITY_INLET
    bctype[:, -1] = PRESSURE_OUTLET
    zoneid = np.zeros((ny, nx), dtype=np.int32)
    zoneid[:, 0] = 1
    zoneuv = np.asarray(((0.0, 0.0), inlet), dtype=float)
    return write_mesh(output_dir, x, y, bctype, zoneid, zoneuv)


def _summary(directory: Path) -> None:
    params = (directory / "params.txt").read_text(encoding="utf-8").split()
    nx, ny = (int(params[0]), int(params[1]))
    bctype = np.loadtxt(directory / "bctype.dat", dtype=int)
    print(f"mesh={directory} cells={nx}x{ny} nodes={(ny + 1)}x{(nx + 1)}")
    for code, name in (
        (INTERIOR, "interior"),
        (WALL, "wall"),
        (VELOCITY_INLET, "velocity_inlet"),
        (PRESSURE_OUTLET, "pressure_outlet"),
    ):
        print(f"  {name}: {int(np.count_nonzero(bctype == code))}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="case", required=True)

    cavity = subparsers.add_parser("cavity", help="moving-lid cavity")
    cavity.add_argument("--nx", type=int, default=64)
    cavity.add_argument("--ny", type=int, default=64)
    cavity.add_argument("--lx", type=float, default=1.0)
    cavity.add_argument("--ly", type=float, default=1.0)
    cavity.add_argument("--lid-u", type=float, default=1.0)
    cavity.add_argument("--lid-v", type=float, default=0.0)
    cavity.add_argument("--alpha-x", type=float, default=0.0)
    cavity.add_argument("--alpha-y", type=float, default=0.0)
    cavity.add_argument("--output-dir", default="ldc_mesh")

    poiseuille = subparsers.add_parser("poiseuille", help="channel flow")
    poiseuille.add_argument("--nx", type=int, default=200)
    poiseuille.add_argument("--ny", type=int, default=80)
    poiseuille.add_argument("--lx", type=float, default=4.0)
    poiseuille.add_argument("--ly", type=float, default=1.0)
    poiseuille.add_argument("--inlet-u", type=float, default=1.0)
    poiseuille.add_argument("--inlet-v", type=float, default=0.0)
    poiseuille.add_argument("--alpha-y", type=float, default=5.0)
    poiseuille.add_argument("--output-dir", default="poiseuille_mesh")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.case == "cavity":
        directory = generate_lid_driven_cavity(
            args.nx,
            args.ny,
            args.lx,
            args.ly,
            args.lid_u,
            args.lid_v,
            args.output_dir,
            args.alpha_x,
            args.alpha_y,
        )
    else:
        directory = generate_poiseuille_flow(
            args.nx,
            args.ny,
            args.lx,
            args.ly,
            args.inlet_u,
            args.inlet_v,
            args.output_dir,
            args.alpha_y,
        )
    _summary(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
