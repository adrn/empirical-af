import pathlib

import agama
import astropy.table as at
import astropy.units as u
import numpy as np
from config import R0, agama_pot, gala_pot, vc0
from gala.units import galactic

agama.setUnits(mass=u.Msun, length=u.kpc, time=u.Myr)


def make_toy_df(overwrite=False):
    this_path = pathlib.Path(__file__).absolute().parent
    data_path = this_path.parent / "data"
    filename = data_path / "toy-df.fits"

    if not overwrite and filename.exists():
        print(f"{filename!s} already exists - use --overwrite to re-make.")
        return

    vcirc_test = gala_pot.circular_velocity(R0 * [1.0, 0, 0])[0]
    assert u.isclose(vcirc_test, vc0, atol=0.1 * u.km / u.s)

    Jphi0 = (vc0 * R0).decompose(galactic).value
    dJphi = Jphi0 * 0.05  # spread = 5% at solar radius
    dJr = (10 * u.km / u.s * 150 * u.pc).decompose(galactic).value
    dJz = (40 * u.km / u.s * 0.5 * u.kpc).decompose(galactic).value

    def df(J):
        # Gaussian in Jphi - exp in JR, Jz
        Jr, Jz, Jphi = J.T
        return np.exp(
            -0.5 * ((Jphi - Jphi0) / dJphi) ** 2 - np.abs(Jr) / dJr - np.abs(Jz) / dJz
        )

    N = 50_000_000

    gm = agama.GalaxyModel(agama_pot, df)
    xv = gm.sample(N)[0]

    R = np.sqrt(xv[:, 0] ** 2 + xv[:, 1] ** 2)
    vR = (xv[:, 0] * xv[:, 3] + xv[:, 1] * xv[:, 4]) / R

    act_finder = agama.ActionFinder(agama_pot)
    act, ang, freq = act_finder(xv, angles=True)

    tbl = at.QTable()
    tbl["R"] = R * galactic["length"]
    tbl["v_R"] = vR * galactic["length"] / galactic["time"]
    tbl["z"] = xv[:, 2] * galactic["length"]
    tbl["v_z"] = xv[:, 5] * galactic["length"] / galactic["time"]
    tbl["J"] = act * galactic["length"] ** 2 / galactic["time"]
    tbl["Omega"] = freq * u.rad / galactic["time"]
    tbl["theta"] = ang * u.rad
    tbl.write(filename, overwrite=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    make_toy_df(args.overwrite)
