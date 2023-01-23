# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: apw-py3-2022
#     language: python
#     name: apw-py3-2022
# ---

import pathlib

import agama
import astropy.table as at
import astropy.units as u
import gala.integrate as gi
import matplotlib as mpl
import matplotlib.pyplot as plt

# %matplotlib inline
import numpy as np
from make_toy_df import get_potentials
from scipy.ndimage import gaussian_filter

agama.setUnits(mass=u.Msun, length=u.kpc, time=u.Myr)

# +
try:
    this_path = pathlib.Path(__file__).absolute().parent
except NameError:
    this_path = pathlib.Path(".").absolute()

data_path = this_path.parent / "data"
figure_path = this_path.parent / "figures"
# -

tbl = at.QTable.read(data_path / "toy-df.fits")
len(tbl)

gala_pot, agama_pot = get_potentials()
R0 = 8.275 * u.kpc
vcirc = gala_pot.circular_velocity(R0 * [1.0, 0, 0])[0]

mask = np.ones(len(tbl), dtype=bool)
# mask = (
#     (tbl['J'][:, 0].value < 0.05)
#     & (np.abs(tbl['J'][:, 2] - np.median(tbl['J'][:, 2])).value < 0.05)
# #     & (tbl['R'] >= 8.27*u.kpc)& (tbl['R'] <= 8.9*u.kpc)
# )
mask.sum()

plt.hist2d(
    tbl["R"].value[mask],
    tbl["z"].value[mask],
    bins=(np.linspace(8, 10, 128), np.linspace(-10, 10, 128)),
    norm=mpl.colors.LogNorm(),
)

# Jr, Jz, Jphi
JR = 0.0 * u.km / u.s * u.kpc
# JR = np.median(tbl['J'][mask, 0])
# JR = 1e-4 * u.km / u.s * u.kpc
Jphi = vcirc * R0

# +
Norbits = 12
orbits = []

Nt = 1024
zeros = np.zeros(Nt)

# Jzs = np.linspace(1.1e-2, np.sqrt(0.1), Norbits) ** 2 * u.kpc**2 / u.Myr
Jzs = np.linspace(1e-2, np.sqrt(0.1), Norbits) ** 2 * u.kpc**2 / u.Myr
Omzs = []
for Jz in Jzs:
    act = u.Quantity([JR, Jz, Jphi]).to_value(u.kpc**2 / u.Myr)
    torus_mapper = agama.ActionMapper(agama_pot, act, tol=1e-7)
    Omzs.append(torus_mapper.Omegaz)

    T = 10 * 2 * np.pi / torus_mapper.Omegaz
    t_grid = np.linspace(0, T, Nt)

    xv0 = torus_mapper([0.0, 0, 0]).T
    #     print(xv0)
    orbit = gala_pot.integrate_orbit(xv0, t=t_grid, Integrator=gi.DOPRI853Integrator)
    orbits.append((orbit.z.to_value(u.kpc), orbit.v_z.to_value(u.km / u.s)))

#     thz = torus_mapper.Omegaz * t_grid
#     angles = np.stack((zeros, thz, zeros)).T
#     z, vz = torus_mapper(angles)[:, [2, 5]].T
#     vz = (vz * u.kpc / u.Myr).to_value(u.km / u.s)
#     orbits.append((z, vz))
# -

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
for (z, vz), Jz in zip(orbits, Jzs):
    ax.plot(vz, z, marker="", ls="-", color="k", lw=2)
ax.set_xlim(-100, 100)
ax.set_ylim(-2.5, 2.5)

# +
Nbins = 150
bins = (np.linspace(-100, 100, Nbins), np.linspace(-2.5, 2.5, Nbins))
H, xe, ye = np.histogram2d(
    tbl["vz"].to_value(u.km / u.s)[mask], tbl["z"].to_value(u.kpc)[mask], bins=bins
)
H /= H.max()
H = gaussian_filter(H, 2)
xc = 0.5 * (xe[:-1] + xe[1:])
yc = 0.5 * (ye[:-1] + ye[1:])

ii = len(xc) // 2
dens_grid = H[ii:, np.abs(yc).argmin()]
levels = [1.0]
for (_, vz) in orbits:
    vz_grid = xc[ii:]
    jj = np.abs(vz_grid - vz[0]).argmin()
    levels.append(dens_grid[jj])

# +
cm = plt.get_cmap("magma")
norm = mpl.colors.Normalize(vmin=0, vmax=120)

fig, axes = plt.subplots(
    1, 2, figsize=(10.5, 6.3), constrained_layout=True, sharex=True, sharey=True
)

# Orbits:
ax = axes[0]
for (z, vz), Jz in zip(orbits, Jzs):
    ax.plot(
        vz, z, marker="", ls="-", lw=2, color=cm(norm(Jz.to_value(u.km / u.s * u.kpc)))
    )

ax.set_xlim(-100, 100)
ax.set_ylim(-2.5, 2.5)

smap = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
cb = fig.colorbar(smap, ax=axes[0], orientation="horizontal")
cb.set_label(f"$J_z$ [{u.kpc*u.km/u.s:latex_inline}]")

# DF:
ax = axes[1]
# levels = np.linspace(0, np.max(H), len(Jzs) + 2)
cs = ax.contourf(xc, yc, H.T, levels=sorted(levels), cmap="Blues")

cb = fig.colorbar(cs, ax=ax, orientation="horizontal")
cb.set_label("scaled density")
cb.set_ticklabels([])

for ax in axes:
    ax.set_xlabel(f"$v_z$ [{u.km/u.s:latex_inline}]")
axes[0].set_ylabel(f"$z$ [{u.kpc:latex_inline}]")

axes[0].set_title("Orbits")
axes[1].set_title("Phase-space Density")

fig.savefig(figure_path / "illustrate-zvz.pdf")
# -
