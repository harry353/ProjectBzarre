from __future__ import annotations

from pathlib import Path

import astropy.units as u
from astropy.time import Time, TimeDelta
import matplotlib.pyplot as plt
import numpy as np

import sunpy.map
from sunpy.net import attrs as a
from sunpy.net import hek
from sunpy.physics.differential_rotation import solar_rotate_coordinate

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_soho_map() -> sunpy.map.Map:
    from sunpy.net import Fido
    from sunpy.net import attrs as na

    query = Fido.search(
        na.Time(Time("2012-01-01T00:00:00"), Time("2012-01-01T23:59:59")),
        na.Instrument("EIT"),
        na.Wavelength(195 * u.angstrom),
    )
    if not query:
        raise RuntimeError("Unable to locate SOHO/EIT data for 2012-01-01.")

    downloaded = Fido.fetch(query[0, 0], progress=False)
    if not downloaded:
        raise RuntimeError("Download failed for SOHO/EIT FITS file.")
    return sunpy.map.Map(downloaded[0])


def plot_spoca_coronal_hole():
    m = _load_soho_map()

    hek_client = hek.HEKClient()
    start_time = m.date - TimeDelta(2 * u.hour)
    end_time = m.date + TimeDelta(2 * u.hour)
    responses = hek_client.search(
        a.Time(start_time, end_time),
        a.hek.CH,
        a.hek.FRM.Name == "SPoCA",
    )
    if not responses:
        raise RuntimeError("No HEK SPoCA coronal hole detections in the time window.")

    area = 0.0
    response_index = 0
    for i, response in enumerate(responses):
        if response["area_atdiskcenter"] > area and np.abs(response["hgc_y"]) < 80.0:
            area = response["area_atdiskcenter"]
            response_index = i

    ch = responses[response_index]
    boundary = ch["hpc_boundcc"]
    rotated = solar_rotate_coordinate(boundary, time=m.date)

    fig = plt.figure()
    ax = fig.add_subplot(projection=m)
    m.plot(axes=ax, clip_interval=(1, 99.99) * u.percent)
    ax.plot_coord(rotated, color="c")
    ax.set_title(f"{m.name}\n{ch['frm_specificid']}")
    plt.colorbar()
    plt.show()


def main() -> None:
    plot_spoca_coronal_hole()


if __name__ == "__main__":
    main()
