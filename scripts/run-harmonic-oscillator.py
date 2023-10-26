import pickle
import sys

import astropy.table as at
import astropy.units as u
import jax
import numpy as np
import torusimaging as oti
from gala.units import galactic
from torusimaging_helpers.config import cache_path
from torusimaging_helpers.make_sim_data import make_sho_data

jax.config.update("jax_enable_x64", True)
short_name = "sho"


def main(overwrite=False):
    pdata_file = cache_path / f"{short_name}-pdata.hdf5"
    bdata_file = cache_path / f"{short_name}-bdata.npz"

    if not pdata_file.exists() or overwrite:
        print("Generating simulated particle data...")
        pdata = make_sho_data(pdata_file, N=2**20)
        print(f"Particle data generated and cached to file {pdata_file!s}")
    else:
        pdata = at.QTable.read(pdata_file)
        print(f"Particle data loaded from cache file {pdata_file!s}")

    if not bdata_file.exists() or overwrite:
        print("Generating binned data...")
        max_z = np.round(2 * np.nanpercentile(pdata["z"].to(u.kpc), 99), 1)
        max_vz = np.round(2 * np.nanpercentile(pdata["v_z"].to(u.km / u.s), 99), 0)

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
        np.savez(bdata_file, **bdata)

    else:
        bdata = dict(np.load(bdata_file))

        # TODO: fix this!
        if not hasattr(bdata["pos"], "unit"):
            bdata["pos"] = bdata["pos"] * u.kpc
            bdata["vel"] = bdata["vel"] * u.kpc / u.Myr

        print(f"Binned data loaded from cache file {bdata_file!s}")

    model, bounds, init_params = oti.TorusImaging1DSpline.auto_init(
        bdata,
        label_knots=8,
        e_knots={2: 8},
        label_l2_sigma=1.0,
        label_smooth_sigma=0.5,
        e_l2_sigmas={2: 1.0},
        e_smooth_sigmas={2: 0.2},
        dacc_strength=0.0,
    )
    with open(cache_path / f"{short_name}-model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(cache_path / f"{short_name}-params-init.pkl", "wb") as f:
        pickle.dump(init_params, f)

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
        print(res.params)
    else:
        print(f"Optimize failed: {res.state!r}")
        sys.exit(1)

    with open(cache_path / f"{short_name}-params-opt.pkl", "wb") as f:
        pickle.dump(res.params, f)

    print("Running MCMC...")
    states, mcmc_samples = model.mcmc_run_label(
        bdata, p0=res.params, bounds=bounds, num_warmup=1000, num_steps=1000
    )
    with open(cache_path / f"{short_name}-mcmc-results.pkl", "wb") as f:
        pickle.dump((states, mcmc_samples), f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--overwrite", action="store_true")
    args = parser.parse_args()
    main(overwrite=args.overwrite)
