import pickle
import sys

import astropy.table as at
import astropy.units as u
import gala.dynamics as gd
import h5py
import jax
import numpy as np
import torusimaging as oti
from gala.units import galactic
from pyia import GaiaData
from torusimaging_helpers.config import cache_path, gala_pot, galcen_frame

jax.config.update("jax_enable_x64", True)
short_name = "gaia-spiral"


def make_gaia_data(pdata_file, bdata_file):
    # Note: fixed values!
    max_z = 1.8 * u.kpc
    max_vz = 90 * u.km / u.s

    # Note: arbitrary number of bins
    zvz_bins = {
        "pos": np.linspace(-max_z, max_z, 151),
        "vel": np.linspace(-max_vz, max_vz, 151),
    }

    tbl = at.QTable.read("/mnt/home/apricewhelan/data/Gaia/DR3/dr3-rv-good-plx.fits")
    for col in tbl.colnames:
        if tbl[col].dtype.char in np.typecodes["AllFloat"]:
            tbl[col] = tbl[col].astype(np.float64)
    g = GaiaData(tbl)

    c = g.get_skycoord()
    galcen = c.transform_to(galcen_frame)
    w = gd.PhaseSpacePosition(galcen.data)

    Lz = np.abs(w.angular_momentum()[2]).to(u.kpc * u.km / u.s)
    Rg = (Lz / gala_pot.circular_velocity(w)).to(u.kpc)
    Rg_bins = np.arange(6.0, 11.0 + 1e-3, 0.5) * u.kpc

    with h5py.File(pdata_file, "w") as f, h5py.File(bdata_file, "w") as bf:
        for i, Rg_c in enumerate(Rg_bins):
            dR = 1.0 * u.kpc
            mask = (np.abs(Rg - Rg_c) < dR) & (
                np.abs(w.cylindrical.v_rho) < 15 * u.km / u.s
            )
            sub_w = w[mask]

            group = f.create_group(str(i))
            group.attrs["Rg"] = Rg_c.value

            tbl = at.QTable()
            tbl["R"] = sub_w.cylindrical.rho
            tbl["v_R"] = sub_w.cylindrical.v_rho
            tbl["z"] = sub_w.z.to(u.kpc)
            tbl["v_z"] = sub_w.v_z.to(u.km / u.s)
            tbl.write(group, serialize_meta=True)

            # Binned data:
            bdata = oti.get_binned_counts(
                pos=sub_w.z, vel=sub_w.v_z, bins=zvz_bins, units=galactic
            )
            bdata["label_err"] = np.sqrt(bdata["counts"]) / bdata["counts"]
            bgroup = bf.create_group(str(i))
            bgroup.attrs["Rg"] = Rg_c.value
            for k in bdata.keys():
                bgroup.create_dataset(k, data=bdata[k])

    return Rg_bins


def fit_model(i, bdata, cache_path, overwrite=False):
    opt_file = cache_path / f"{short_name}-{i}-params-opt.pkl"
    if opt_file.exists() and not overwrite:
        print("Optimized parameters already exist - skipping")
        with open(cache_path / f"{short_name}-{i}-model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(opt_file, "rb") as f:
            params = pickle.load(f)
        return model, params

    model, bounds, init_params = oti.TorusImaging1DSpline.auto_init(
        bdata,
        label_knots=12,
        e_knots={2: 16, 4: 6},
        label_l2_sigma=1.0,
        label_smooth_sigma=0.5,
        e_l2_sigmas={2: 1.0, 4: 1.0},
        e_smooth_sigmas={2: 0.2, 4: 0.2},
        dacc_strength=1e3,
        label_knots_spacing_power=0.75,
        e_knots_spacing_power=0.5,
    )

    init_params["e_params"][2]["vals"] = np.full_like(
        init_params["e_params"][2]["vals"], -4
    )
    init_params["e_params"][4]["vals"] = np.full_like(
        init_params["e_params"][4]["vals"],
        np.log(0.05 / model._label_knots.max()),
    )

    with open(cache_path / f"{short_name}-{i}-model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(cache_path / f"{short_name}-{i}-params-init.pkl", "wb") as f:
        pickle.dump(init_params, f)

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

    with open(opt_file, "wb") as f:
        pickle.dump(res.params, f)

    return model, res.params


def main(overwrite_data=False, overwrite_fit=False):
    pdata_file = cache_path / f"{short_name}-pdata.hdf5"
    bdata_file = cache_path / f"{short_name}-bdata.hdf5"

    if not pdata_file.exists() or overwrite_data:
        print("Generating particle data...")
        Rgs = make_gaia_data(pdata_file, bdata_file)
        print(f"Particle data generated and cached to file {pdata_file!s}")

    else:
        with h5py.File(pdata_file, "r") as f:
            Rgs = np.array([float(f[k].attrs["Rg"]) for k in f.keys()]) * u.kpc

    Rgs = np.sort(Rgs)

    all_models = []
    all_params = []
    with h5py.File(bdata_file, "r") as bf:
        for i, Rg_c in enumerate(Rgs):
            print("-" * 79)
            print(f"Rg bin {i}, centered at Rg = {Rg_c:.1f}")
            group = bf[str(i)]
            bdata = {k: group[k][:] for k in group.keys()}

            model, params = fit_model(i, bdata, cache_path, overwrite=overwrite_fit)
            all_models.append(model)
            all_params.append(params)

    with h5py.File(pdata_file, "r") as pf:
        for i, Rg_c in enumerate(Rgs):
            pdata = at.QTable.read(pf[str(i)])
            model = all_models[i]
            params = all_params[i]

            # Split into batches because we get memory issues otherwise:
            idx = np.arange(0, len(pdata) - 1, 100_000)
            if idx[-1] != len(pdata) - 1:
                idx = np.concatenate([idx, [len(pdata) - 1]])

            aaf_batches = []
            for i1, i2 in zip(idx[:-1], idx[1:]):
                aaf_batches.append(
                    model.compute_action_angle(
                        pdata["z"][i1:i2],
                        pdata["v_z"][i1:i2],
                        params,
                        N_grid=15,
                    )
                )
            aaf = at.vstack(aaf_batches)
            aaf.write(cache_path / f"{short_name}-{i}-aaf.hdf5", overwrite=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite-data", action="store_true", default=False)
    parser.add_argument("--overwrite-fit", action="store_true", default=False)
    args = parser.parse_args()

    main(overwrite_data=args.overwrite_data, overwrite_fit=args.overwrite_fit)
