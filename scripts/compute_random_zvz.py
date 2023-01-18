import agama
import astropy.table as at
import astropy.units as u
import numpy as np
from schwimmbad.utils import batch_tasks

agama.setUnits(mass=u.Msun, length=u.kpc, time=u.Myr)


def get_random_z_vz(pot, Jzs, Jphi=None, rng=None):
    in_shape = Jzs.shape
    Jzs = np.atleast_1d(Jzs)

    if rng is None:
        rng = np.random.default_rng()

    if Jphi is None:
        Jphi = 220 * u.km / u.s * 8 * u.kpc

    JR = 0 * u.km / u.s * u.kpc

    thzs = rng.uniform(0, 2 * np.pi, size=len(Jzs))
    xvs = []
    for Jz, thz in zip(Jzs, thzs):
        act = u.Quantity([JR, Jz, Jphi]).to_value(u.kpc**2 / u.Myr)
        torus_mapper = agama.ActionMapper(pot, act)
        ang = [0.0, thz, 0.0]
        xv_torus = torus_mapper(ang)
        xvs.append(xv_torus)
    xvs = np.array(xvs)

    return xvs[:, [2, 5]].reshape(in_shape + (2,))


def worker(task):
    """
    TODO: Add an option to MN3 potential to get 3 MN potentials out - then can use
    MilkyWayPotential2022 instead here
    https://github.com/adrn/gala/blob/main/gala/potential/potential/builtin/core.py#L495
    """
    _, Jzs, rng = task

    # Same as gala MilkyWayPotential
    agama_pot = agama.Potential(
        dict(type="miyamotonagai", mass=6.8e10, scaleradius=3.0, scaleheight=0.28),
        dict(type="dehnen", mass=5.00e9, scaleradius=1.0),
        dict(type="dehnen", mass=1.71e9, scaleradius=0.07),
        dict(type="nfw", mass=5.4e11, scaleradius=15.62),
    )
    zvz = get_random_z_vz(agama_pot, Jzs, rng=rng)

    return zvz


def main(pool):
    Jz_scale = 15 * u.km / u.s * 0.25 * u.kpc

    rng = np.random.default_rng(42)
    Jzs = (rng.exponential(scale=1.0, size=1_000_000) + 1e-3) * Jz_scale

    batches = batch_tasks(8 * max(1, pool.size), arr=Jzs)

    ss = np.random.SeedSequence(42)
    child_seeds = ss.spawn(len(batches))
    rngs = [np.random.default_rng(s) for s in child_seeds]
    batches = [batch + (rng,) for batch, rng in zip(batches, rngs)]

    zvz_batches = []
    for res in pool.map(worker, batches):
        zvz_batches.append(res)

    all_zvz = np.row_stack(zvz_batches)

    tbl = at.Table()
    tbl["z"] = all_zvz[:, 0]
    tbl["vz"] = all_zvz[:, 1]
    tbl.write("zvz-random.fits")


if __name__ == "__main__":
    # from schwimmbad import SerialPool
    # with SerialPool() as pool:
    from schwimmbad import MPIPool

    with MPIPool() as pool:
        main(pool)
