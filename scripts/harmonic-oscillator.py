import astropy.units as u
import gala.potential as gp
import jax
import numpy as np
import torusimaging as oti
from gala.units import galactic

jax.config.update("jax_enable_x64", True)

Omega = 0.08 * u.rad / u.Myr

scale_vz = 50 * u.km / u.s
sz = (scale_vz / np.sqrt(Omega)).decompose(galactic)  # .value

gala_pot = gp.HarmonicOscillatorPotential(Omega / u.rad, units=galactic)

N = 2**20

rng = np.random.default_rng(42)
with u.set_enabled_equivalencies(u.dimensionless_angles()):
    Jzs = (rng.exponential(scale=sz.value**2, size=N) * sz.unit**2).to(
        galactic["length"] ** 2 / galactic["time"]
    )
    thzs = rng.uniform(0, 2 * np.pi, size=N)

sim_mgfe_std = 0.05

with u.set_enabled_equivalencies(u.dimensionless_angles()):
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

    pdata["mgfe_err"] = np.exp(rng.uniform(np.log(0.005), np.log(0.1), size=len(Jzs)))
    pdata["mgfe"] = rng.normal(pdata["mgfe"], pdata["mgfe_err"])


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

model, bounds, init_params = oti.TorusImaging1DSpline.auto_init(
    bdata,
    label_knots=8,
    e_knots={2: 8},
    label_l2_sigma=1.0,
    label_smooth_sigma=0.5,
    e_l2_sigmas={2: 0.1},
    e_smooth_sigmas={2: 0.2},
)

print(init_params)
print(bounds)

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
print(model.objective_gaussian(init_params, **data_kw))

res = model.optimize(init_params, objective="gaussian", bounds=bounds, **data_kw)
print(res)

states, mcmc_samples = model.mcmc_run_label(
    bdata, p0=res.params, bounds=bounds, num_warmup=500, num_steps=200
)
