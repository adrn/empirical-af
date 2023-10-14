import pathlib

import agama
import astropy.table as at
import astropy.units as u
import gala.potential as gp
import numpy as np
from gala.units import galactic

from .config import R0, agama_pot, gala_pot, vc0

agama.setUnits(mass=u.Msun, length=u.kpc, time=u.Myr)


def make_mgfe(zmax: u.Quantity[u.kpc], rng=None):
    if rng is None:
        rng = np.random.default_rng()
    sim_mgfe_std = 0.05
    mgfe = 0.064 * zmax.to_value(u.kpc) + 0.009
    mgfe = rng.normal(mgfe, sim_mgfe_std)
    mgfe_err = np.exp(rng.normal(-4.0, 0.5, size=len(zmax)))
    mgfe = rng.normal(mgfe, mgfe_err)
    return mgfe, mgfe_err


def make_sho_data(filename: str | pathlib.Path, N: int):
    filename = pathlib.Path(filename)

    Omega = 0.08 * u.rad / u.Myr

    gala_pot = gp.HarmonicOscillatorPotential(Omega / u.rad, units=galactic)
    gala_pot.save(filename.parent / "sho-gala-pot.yml")

    scale_vz = 50 * u.km / u.s
    sz = (scale_vz / np.sqrt(Omega)).decompose(galactic)  # .value

    rng = np.random.default_rng(42)
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        Jzs = (rng.exponential(scale=sz.value**2, size=N) * sz.unit**2).to(
            galactic["length"] ** 2 / galactic["time"]
        )
        thzs = rng.uniform(0, 2 * np.pi, size=N) * u.rad

        pdata = {
            "z": (np.sqrt(2 * Jzs / Omega) * np.sin(thzs)).to(galactic["length"]),
            "vz": (np.sqrt(2 * Jzs * Omega) * np.cos(thzs)).to(
                galactic["length"] / galactic["time"]
            ),
            "Jz": Jzs,
            "thetaz": thzs,
            "zmax": np.sqrt(2 * Jzs / Omega).to(galactic["length"]),
        }
        pdata["r_e"] = np.sqrt(pdata["z"] ** 2 * Omega + pdata["vz"] ** 2 / Omega)
        pdata["mgfe"], pdata["mgfe_err"] = make_mgfe(pdata["zmax"], rng=rng)

    pdata = at.QTable(pdata)
    pdata.write(filename, overwrite=True, serialize_meta=True)
    return pdata


def make_df_table(xv):
    R = np.sqrt(xv[:, 0] ** 2 + xv[:, 1] ** 2)
    vR = (xv[:, 0] * xv[:, 3] + xv[:, 1] * xv[:, 4]) / R

    act_finder = agama.ActionFinder(agama_pot)
    act, ang, freq = act_finder(xv, angles=True)

    idx = [0, 2, 1]  # reorder to be: JR, Jphi, Jz

    vc = gala_pot.circular_velocity(xv[:, :3].T)

    tbl = at.QTable()
    tbl["R"] = R * galactic["length"]
    tbl["v_R"] = vR * galactic["length"] / galactic["time"]
    tbl["z"] = xv[:, 2] * galactic["length"]
    tbl["v_z"] = xv[:, 5] * galactic["length"] / galactic["time"]
    tbl["J"] = act[:, idx] * galactic["length"] ** 2 / galactic["time"]
    tbl["Rg"] = (tbl["J"][:, 1] / vc).to(u.kpc)
    tbl["Omega"] = freq[:, idx] * u.rad / galactic["time"]
    tbl["theta"] = ang[:, idx] * u.rad

    return tbl


# Old toy DF:
# def make_toy_df(filename, overwrite=False):
#     filename = pathlib.Path(filename)
#     if not overwrite and filename.exists():
#         print(f"{filename!s} already exists - use --overwrite to re-make.")
#         return

#     vcirc_test = gala_pot.circular_velocity(R0 * [1.0, 0, 0])[0]
#     assert u.isclose(vcirc_test, vc0, atol=0.1 * u.km / u.s)

#     Jphi0 = (vc0 * R0).decompose(galactic).value
#     dJphi = Jphi0 * 0.1  # spread = 10% at solar radius
#     dJr = (25 * u.km / u.s * 350 * u.pc).decompose(galactic).value
#     dJz = (40 * u.km / u.s * 0.5 * u.kpc).decompose(galactic).value

#     def df(J):
#         # Gaussian in Jphi - exp in JR, Jz
#         Jr, Jz, Jphi = J.T
#         return np.exp(
#             -0.5 * ((Jphi - Jphi0) / dJphi) ** 2 - np.abs(Jr) / dJr - np.abs(Jz) / dJz
#         )

#     N = 50_000_000

#     gm = agama.GalaxyModel(agama_pot, df)
#     xv = gm.sample(N)[0]

#     tbl = make_df_table(xv)
#     tbl.write(filename, overwrite=True, serialize_meta=True)

#     return tbl


def make_qiso_df(filename, overwrite=False):
    filename = pathlib.Path(filename)
    if not overwrite and filename.exists():
        print(f"{filename!s} already exists - use --overwrite to re-make.")
        return

    vcirc_test = gala_pot.circular_velocity(R0 * [1.0, 0, 0])[0]
    assert u.isclose(vcirc_test, vc0, atol=0.1 * u.km / u.s)

    # Make a fast fitting function to get Rcirc (Rc) from Lz
    xyz = np.zeros((3, 128)) * u.kpc
    xyz[0] = np.linspace(4, 14, 128) * u.kpc
    Lz_circ = (xyz[0] * gala_pot.circular_velocity(xyz)).decompose(galactic).value
    coef = np.polyfit(Lz_circ, xyz[0].value, deg=5)
    poly_Lz_to_Rc = np.poly1d(coef)

    # Do the same for getting nu and kappa (frequencies) from Rc:
    dPhi2_dR2 = gala_pot.hessian(xyz)[0, 0].decompose(galactic).value
    dPhi2_dz2 = gala_pot.hessian(xyz)[2, 2].decompose(galactic).value
    kappa = np.sqrt(dPhi2_dR2 + 3 * Lz_circ**2 / xyz[0].value ** 4)
    nu = np.sqrt(dPhi2_dz2)

    coef = np.polyfit(xyz[0].value, kappa, deg=4)
    poly_Rc_to_kappa = np.poly1d(coef)

    coef = np.polyfit(xyz[0].value, nu, deg=4)
    poly_Rc_to_nu = np.poly1d(coef)

    _R0 = R0.decompose(galactic).value

    def df(J):
        # quasi-isothermal DF and parameters from Sanders et al. 2015
        Jr, Jz, Jphi = J.T

        R_c = poly_Lz_to_Rc(Jphi)
        Omega_c = Jphi / R_c**2
        kappa = poly_Rc_to_kappa(R_c)
        nu = poly_Rc_to_nu(R_c)

        Sigma_0 = 1.0  # normalization doesn't matter?
        R0 = _R0

        # Binney 2012
        # - Table 2, thick disk
        # R_d = 2.5
        # q = 0.705
        # R0 = 8.3
        # sigma_r0 = 25.2 / 1e3
        # sigma_z0 = 32.7 / 1e3
        # Sigma = Sigma_0 * np.exp(-R_c / R_d)
        # sigma_r = sigma_r0 * np.exp(q * (R0 - R_c) / R_d)
        # sigma_z = sigma_z0 * np.exp(q * (R0 - R_c) / R_d)

        # L0 = 0.01  # 10 km/s*kpc
        # A = Omega_c * Sigma / (np.pi * sigma_r**2 * kappa)
        # f_r = A * (1 + np.tanh(Jphi / L0)) * np.exp(-kappa * Jr / sigma_r**2)

        # f_z = nu / (2 * np.pi * sigma_z**2) * np.exp(-nu * Jz / sigma_z**2)

        # val = f_r * f_z

        # Sanders & Binney 2015
        # Thin disk, table 3
        kms_to_kpcMyr = 977.792
        R_d = 3.45
        R_sigma = 7.8
        sigma_r0 = 48.3 / kms_to_kpcMyr
        sigma_z0 = 30.7 / kms_to_kpcMyr
        L0 = 0.01  # ~10 km/s*kpc in kpc**2/Myr

        sigma_r = sigma_r0 * np.exp((R0 - R_c) / R_sigma)
        sigma_z = sigma_z0 * np.exp((R0 - R_c) / R_sigma)

        A = Omega_c * Sigma_0 / (R_d**2 * kappa)
        Sigma_term = A * np.exp(-R_c / R_d)
        R_term = kappa / sigma_r**2 * np.exp(-kappa * Jr / sigma_r**2)
        z_term = nu / sigma_z**2 * np.exp(-nu * Jz / sigma_z**2)
        Jphi_term = 1 + np.tanh(Jphi / L0)

        # Modify Jphi term to only generate around solar orbit
        # DISABLED: doesn't make sense to have a peak at solar value...
        # Jphi0 = (vc0 * R0).decompose(galactic).value
        # dJphi = Jphi0 * 0.05  # spread = 5% of solar Jphi
        # Jphi_term = np.exp(-0.5 * (Jphi - Jphi0) ** 2 / dJphi**2)

        val = Sigma_term * Jphi_term * R_term * z_term

        # Restrict to range of Jphi - roughly Rg in (5, 12) kpc
        val[~np.isfinite(val) | (val < 0.0) | (Jphi < 1.17) | (Jphi > 2.8)] = 0.0
        return val

    N = 400_000_000

    gm = agama.GalaxyModel(agama_pot, df)
    xv = gm.sample(N)[0]

    tbl = make_df_table(xv)

    mask = (np.abs(tbl["R"] - R0) < 1.0 * u.kpc) & (
        np.abs(tbl["R"] - tbl["Rg"]) < 1.0 * u.kpc
    )
    tbl = tbl[mask]

    rng = np.random.default_rng(42)
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        zmax = np.sqrt(2 * tbl["J"][:, 2] / tbl["Omega"][:, 2]).to(galactic["length"])

    tbl["mgfe"], tbl["mgfe_err"] = make_mgfe(zmax, rng=rng)

    tbl.write(filename, overwrite=True, serialize_meta=True)

    return tbl
