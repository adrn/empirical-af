import pathlib

import agama
import astropy.table as at
import astropy.units as u
import numpy as np
from gala.units import galactic
from helpers import R0, agama_pot, gala_pot

agama.setUnits(mass=u.Msun, length=u.kpc, time=u.Myr)


def main():
    this_path = pathlib.Path(__file__).absolute().parent
    data_path = this_path.parent / "data"

    vcirc = gala_pot.circular_velocity(R0 * [1.0, 0, 0])[0]

    Jphi0 = (vcirc * R0).to_value(u.kpc**2 / u.Myr)
    dJphi = 0.01
    dJr = 0.06
    dJz = 0.06

    def df(J):
        # Gaussian in JR, Jphi - exp in Jz
        Jr, Jz, Jphi = J.T
        return np.exp(
            -np.abs(Jr) / dJr - 0.5 * ((Jphi - Jphi0) / dJphi) ** 2 - np.abs(Jz) / dJz
        )

    N = 50_000_000

    gm = agama.GalaxyModel(agama_pot, df)
    xv_samples = gm.sample(N)[0]

    act_finder = agama.ActionFinder(agama_pot)
    act, ang, freq = act_finder(xv_samples, angles=True)

    # Impose some masks:
    thresh = 0.05  # ~15 km/s * 300 pc
    mask = (act[:, 0] < thresh) & (np.abs(act[:, 2] - np.median(act[:, 2])) < thresh)

    tbl = at.QTable()
    tbl["R"] = (
        np.sqrt(xv_samples[:, 0] ** 2 + xv_samples[:, 1] ** 2) * galactic["length"]
    )
    tbl["z"] = xv_samples[:, 2] * galactic["length"]
    tbl["vz"] = xv_samples[:, 5] * galactic["length"] / galactic["time"]
    tbl["J"] = act * galactic["length"] ** 2 / galactic["time"]
    tbl["Omega"] = freq * u.rad / galactic["time"]
    tbl["theta"] = ang * u.rad
    tbl[mask].write(data_path / "toy-df.fits", overwrite=True)


if __name__ == "__main__":
    main()
