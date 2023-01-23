# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python [conda env:root] *
#     language: python
#     name: conda-root-py
# ---

import pathlib

import agama
import astropy.table as at
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt

# %matplotlib inline
import numpy as np
from make_toy_df import get_potentials

agama.setUnits(mass=u.Msun, length=u.kpc, time=u.Myr)
# -

try:
    this_path = pathlib.Path(__file__).absolute().parent
except NameError:
    this_path = pathlib.Path(".").absolute().parent
data_path = this_path.parent / "data"
tbl = at.QTable.read(data_path / "toy-df.fits")

gala_pot, agama_pot = get_potentials()
R0 = 8.275 * u.kpc
vcirc = gala_pot.circular_velocity(R0 * [1.0, 0, 0])[0]

# Jr, Jz, Jphi
JR = 0 * u.km / u.s * u.kpc
Jphi = vcirc * R0

# +
Norbits = 12
orbits = []

Nt = 1024
zeros = np.zeros(Nt)

Jzs = np.linspace(1.5e-2, np.sqrt(0.12), Norbits) ** 2 * u.kpc**2 / u.Myr
Omzs = []
for Jz in Jzs:
    act = u.Quantity([JR, Jz, Jphi]).to_value(u.kpc**2 / u.Myr)
    torus_mapper = agama.ActionMapper(agama_pot, act)

    t_grid = np.linspace(0, 2 * np.pi / torus_mapper.Omegaz, Nt)
    thz = torus_mapper.Omegaz * t_grid
    Omzs.append(torus_mapper.Omegaz)
    angles = np.stack((zeros, thz, zeros)).T
    z, vz = torus_mapper(angles)[:, [2, 5]].T

    vz = (vz * u.kpc / u.Myr).to_value(u.km / u.s)

    orbits.append((z, vz))

# +
cm = plt.get_cmap("viridis")
norm = mpl.colors.Normalize(vmin=0, vmax=120)

fig, axes = plt.subplots(
    1, 3, figsize=(15, 6.0), constrained_layout=True, sharex=True, sharey=True
)

# Orbits:
ax = axes[0]
for (z, vz), Jz in zip(orbits, Jzs):
    ax.plot(
        vz, z, marker="", ls="-", lw=2, color=cm(norm(Jz.to_value(u.km / u.s * u.kpc)))
    )

ax.set_xlim(-100, 100)
ax.set_ylim(-3, 3)

smap = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
cb = fig.colorbar(smap, ax=ax, orientation="horizontal")
cb.set_label(f"$J_z$ [{u.kpc*u.km/u.s:latex_inline}]")

# DF:
ax = axes[1]

bins = (np.linspace(-100, 100, 151), np.linspace(-3, 3, 151))
H, xe, ye = np.histogram2d(
    tbl["vz"].to_value(u.km / u.s), tbl["z"].to_value(u.kpc), bins=bins
)
H /= H.max()
xc = 0.5 * (xe[:-1] + xe[1:])
yc = 0.5 * (ye[:-1] + ye[1:])
cs = ax.contourf(xc, yc, H.T, levels=np.linspace(0, np.max(H), len(Jzs) + 2))

cb = fig.colorbar(cs, ax=ax, orientation="horizontal")
cb.set_label("scaled density")

for ax in axes:
    ax.set_xlabel(f"$v_z$ [{u.km/u.s:latex_inline}]")
axes[0].set_ylabel(f"$z$ [{u.kpc:latex_inline}]")

# fig.savefig("../tex/figures/illustrate-zvz.pdf")
# -
