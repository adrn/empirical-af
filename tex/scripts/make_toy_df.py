import pathlib

import agama
import astropy.table as at
import astropy.units as u
import gala.potential as gp
import numpy as np
from gala.units import galactic

agama.setUnits(mass=u.Msun, length=u.kpc, time=u.Myr)


def get_potentials():
    # Create a toy, two-component Galaxy model:
    gala_pot = gp.CCompositePotential(
        disk=gp.MiyamotoNagaiPotential(m=6.91e10, a=3, b=0.25, units=galactic),
        halo=gp.NFWPotential(m=5.4e11, r_s=15.0, units=galactic),
    )

    agama_pot = agama.Potential(
        dict(
            type="miyamotonagai",
            mass=gala_pot["disk"].parameters["m"].value,
            scaleradius=gala_pot["disk"].parameters["a"].value,
            scaleheight=gala_pot["disk"].parameters["b"].value,
        ),
        dict(
            type="nfw",
            mass=gala_pot["halo"].parameters["m"].value,
            scaleradius=gala_pot["halo"].parameters["r_s"].value,
        ),
    )

    return gala_pot, agama_pot


def main():
    this_path = pathlib.Path(__file__).absolute().parent
    data_path = this_path.parent / "data"

    gala_pot, agama_pot = get_potentials()

    R0 = 8.275 * u.kpc
    vcirc = gala_pot.circular_velocity(R0 * [1.0, 0, 0])[0]

    Jphi0 = (vcirc * R0).to_value(u.kpc**2 / u.Myr)
    dJphi = 0.01
    dJr = 0.06
    dJz = 0.06

    def df(J):
        # Gaussian in JR, Jphi - exp in Jz
        Jr, Jz, Jphi = J.T
        return np.exp(
            -0.5 * Jr**2 / dJr**2
            - 0.5 * ((Jphi - Jphi0) / dJphi) ** 2
            - np.abs(Jz) / dJz
        )

    N = 10_000_000

    gm = agama.GalaxyModel(agama_pot, df)
    xv_samples = gm.sample(N)[0]

    act_finder = agama.ActionFinder(agama_pot)
    act, ang, freq = act_finder(xv_samples, angles=True)

    # Impose some masks:
    thresh = 0.005  # ~15 km/s * 300 pc
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
