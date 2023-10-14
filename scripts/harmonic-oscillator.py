import pickle
import sys

import astropy.table as at
import astropy.units as u
import gala.potential as gp
import jax
import numpy as np
import torusimaging as oti
from config import cache_path
from gala.units import galactic

jax.config.update("jax_enable_x64", True)
short_name = "sho"


def make_data(N: int):
    Omega = 0.08 * u.rad / u.Myr

    gala_pot = gp.HarmonicOscillatorPotential(Omega / u.rad, units=galactic)
    gala_pot.save(cache_path / f"{short_name}-gala-pot.yml")

    scale_vz = 50 * u.km / u.s
    sz = (scale_vz / np.sqrt(Omega)).decompose(galactic)  # .value
    sim_mgfe_std = 0.05

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
            "zmax": np.sqrt(2 * Jzs / Omega).to_value(galactic["length"]),
        }
        pdata["r_e"] = np.sqrt(pdata["z"] ** 2 * Omega + pdata["vz"] ** 2 / Omega)
        pdata["mgfe"] = rng.normal(0.064 * pdata["zmax"] + 0.009, sim_mgfe_std)

        pdata["mgfe_err"] = np.exp(
            rng.uniform(np.log(0.005), np.log(0.2), size=len(Jzs))
        )
        pdata["mgfe"] = rng.normal(pdata["mgfe"], pdata["mgfe_err"])

    return at.QTable(pdata)


def main(overwrite=False):
    pdata_file = cache_path / f"{short_name}-pdata.hdf5"
    bdata_file = cache_path / f"{short_name}-bdata.hdf5"

    if not pdata_file.exists() or overwrite:
        print("Generating simulated particle data...")
        pdata = make_data(N=2**20)
        pdata.write(pdata_file, overwrite=True)
        print(f"Particle data generated and cached to file {pdata_file!s}")
    else:
        pdata = at.QTable.read(pdata_file)
        print(f"Particle data loaded from cache file {pdata_file!s}")

    if not bdata_file.exists() or overwrite:
        print("Generating binned data...")
        max_z = np.round(3 * np.nanpercentile(pdata["z"].to(u.kpc), 90), 1)
        max_vz = np.round(3 * np.nanpercentile(pdata["vz"].to(u.km / u.s), 90), 0)

        zvz_bins = {
            "pos": np.linspace(-max_z, max_z, 151),
            "vel": np.linspace(-max_vz, max_vz, 151),
        }

        bdata = oti.get_binned_label(
            pdata["z"],
            pdata["vz"],
            label=pdata["mgfe"],
            label_err=pdata["mgfe_err"],
            bins=zvz_bins,
            units=galactic,
        )
        print(f"Binned data generated and cached to file {bdata_file!s}")

    else:
        bdata = at.QTable.read(bdata_file)
        print(f"Binned data loaded from cache file {bdata_file!s}")

    model, bounds, init_params = oti.TorusImaging1DSpline.auto_init(
        bdata,
        label_knots=8,
        e_knots={2: 8},
        label_l2_sigma=1.0,
        label_smooth_sigma=0.5,
        e_l2_sigmas={2: 0.1},
        e_smooth_sigmas={2: 0.2},
    )
    with open(cache_path / f"{short_name}-model.pkl", "w") as f:
        pickle.dump(model, f)

    with open(cache_path / f"{short_name}-params-init.pkl", "w") as f:
        pickle.dump(model, init_params)

    # print(init_params)
    # print(bounds)

    data_kw = dict(
        pos=bdata["pos"],
        vel=bdata["vel"],
        label=bdata["label"],
        label_err=bdata["label_err"],
    )
    mask = (
        np.isfinite(bdata["label"])
        & np.isfinite(bdata["label_err"])
        & (bdata["label_err"] > 0)
    )
    data_kw = {k: v[mask] for k, v in data_kw.items()}

    test_val = model.objective_gaussian(init_params, **data_kw)
    print(f"Test evaluation of objective function: {test_val}")

    print("Running optimize...")
    res = model.optimize(init_params, objective="gaussian", bounds=bounds, **data_kw)
    if res.state.success:
        print(f"Optimize completed successfully in {res.state.iter_num} steps")
    else:
        print(f"Optimize failed: {res.state!r}")
        sys.exit(1)

    with open(cache_path / f"{short_name}-params-opt.pkl", "w") as f:
        pickle.dump(model, res.params)

    print("Running MCMC...")
    states, mcmc_samples = model.mcmc_run_label(
        bdata, p0=res.params, bounds=bounds, num_warmup=500, num_steps=200
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--overwrite", action="store_true")
    args = parser.parse_args()
    main(overwrite=args.overwrite)
