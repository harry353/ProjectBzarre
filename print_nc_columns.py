from __future__ import annotations

import gzip
import sys
from pathlib import Path

try:
    import xarray as xr
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "xarray is required to read NetCDF files. "
        "Install it or run in an environment that includes it."
    ) from exc

try:
    import cdflib
except ImportError:
    cdflib = None


def _open_dataset(path: Path) -> xr.Dataset:
    with gzip.open(path, "rb") as handle:
        return xr.open_dataset(handle)


def _print_netcdf(path: Path) -> None:
    ds = _open_dataset(path)
    data_vars = list(ds.data_vars.keys())
    coords = list(ds.coords.keys())

    print(f"File: {path}")
    print("Data variables:")
    for name in data_vars:
        print(f"  - {name}")

    print("Coordinates:")
    for name in coords:
        print(f"  - {name}")


def _print_cdf(path: Path) -> None:
    if cdflib is None:
        raise SystemExit(
            "cdflib is required to read CDF files. Install it and re-run."
        )
    cdf = cdflib.CDF(str(path))
    info = cdf.cdf_info()
    variables = list(info.zVariables) + list(info.rVariables)

    print(f"File: {path}")
    print("Variables:")
    for name in variables:
        print(f"  - {name}")


def _handle_path(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".cdf":
        _print_cdf(path)
    elif suffix == ".gz" or suffix == ".nc":
        _print_netcdf(path)
    else:
        raise SystemExit(f"Unsupported file type: {path}")


DEFAULT_PATHS = [
    Path(
        "/home/haris/Downloads/oe_m1m_dscovr_s20251201000000_e20251201235959_p20251202020340_pub.nc.gz"
    ),
    Path("/home/haris/Downloads/ac_h1_mfi_20251204_v07.cdf"),
]


def main() -> None:
    if len(sys.argv) > 1:
        paths = [Path(arg) for arg in sys.argv[1:]]
    else:
        paths = [path for path in DEFAULT_PATHS if path.exists()]
        if not paths:
            raise SystemExit("Usage: python print_nc_columns.py <file.nc.gz> [file.cdf ...]")

    for path in paths:
        _handle_path(path)


if __name__ == "__main__":
    main()
