import pathlib

import agama
import astropy.coordinates as coord
import astropy.units as u
import gala.potential as gp
from gala.units import galactic

this_path = pathlib.Path(__file__).absolute().parent
tex_path = this_path.parent
data_path = tex_path / "data"

agama.setUnits(mass=u.Msun, length=u.kpc, time=u.Myr)


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

R0 = 8.275 * u.kpc
vc0 = 229 * u.km / u.s

galcen_frame = coord.Galactocentric(
    galcen_distance=R0, galcen_v_sun=[8.4, 251.8, 8.4] * u.km / u.s
)


def plot_data_model_residual(
    model, bdata, params, zlim, vzlim=None, aspect=True, residual_lim=0.05
):
    import matplotlib.pyplot as plt
    import numpy as np

    title_fontsize = 20
    title_pad = 10

    cb_labelsize = 16
    mgfe_cbar_xlim = (0, 0.15)
    mgfe_cbar_vlim = (-0.05, 0.18)

    tmp_aaf = model.label_model.compute_action_angle(
        np.atleast_1d(zlim) * 0.75, [0.0] * u.km / u.s, params
    )
    Omega = tmp_aaf["Omega"][0]
    if vzlim is None:
        vzlim = zlim * Omega
    vzlim = vzlim.to_value(u.km / u.s, u.dimensionless_angles())

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5.1), sharex=True, sharey=True, layout="constrained"
    )

    cs = axes[0].pcolormesh(
        bdata["vel"].to_value(u.km / u.s),
        bdata["pos"].to_value(u.kpc),
        bdata["mgfe"],
        cmap="magma",
        rasterized=True,
        vmin=mgfe_cbar_vlim[0],
        vmax=mgfe_cbar_vlim[1],
    )
    cb = fig.colorbar(cs, ax=axes[0:2])
    cb.set_label("mean [Mg/Fe]", fontsize=cb_labelsize)
    cb.ax.set_ylim(mgfe_cbar_xlim)
    cb.ax.set_yticks(np.arange(mgfe_cbar_xlim[0], mgfe_cbar_xlim[1] + 1e-3, 0.05))
    cb.ax.yaxis.set_tick_params(labelsize=14)

    model_mgfe = np.array(model.label_model.label(bdata["pos"], bdata["vel"], params))
    cs = axes[1].pcolormesh(
        bdata["vel"].to_value(u.km / u.s),
        bdata["pos"].to_value(u.kpc),
        model_mgfe,
        cmap="magma",
        rasterized=True,
        vmin=mgfe_cbar_vlim[0],
        vmax=mgfe_cbar_vlim[1],
    )

    cs = axes[2].pcolormesh(
        bdata["vel"].to_value(u.km / u.s),
        bdata["pos"].to_value(u.kpc),
        bdata["mgfe"] - model_mgfe,
        cmap="RdBu_r",
        vmin=-residual_lim,
        vmax=residual_lim,
        rasterized=True,
    )
    cb = fig.colorbar(cs, ax=axes[2])  # , orientation="horizontal")
    cb.set_label("data $-$ model", fontsize=cb_labelsize)
    cb.ax.yaxis.set_tick_params(labelsize=14)

    # Titles
    axes[0].set_title("Simulated Data", fontsize=title_fontsize, pad=title_pad)
    axes[1].set_title("Optimized OTI Model", fontsize=title_fontsize, pad=title_pad)
    axes[2].set_title("Residuals", fontsize=title_fontsize, pad=title_pad)
    fig.suptitle("Demonstration with Simulated Data: Harmonic Oscillator", fontsize=24)

    # Labels
    axes[0].set_ylabel(f"$z$ [{u.kpc:latex_inline}]")
    for ax in axes:
        ax.set_xlabel(f"$v_z$ [{u.km/u.s:latex_inline}]")

    # Ticks
    axes[0].set_xticks(np.arange(-300, 300 + 1, 100))
    axes[0].set_xticks(np.arange(-300, 300 + 1, 50), minor=True)
    axes[1].set_yticks(np.arange(-3, 3 + 1e-3, 1))
    axes[1].set_yticks(np.arange(-3, 3 + 1e-3, 0.5), minor=True)

    for ax in axes:
        if aspect:
            # ax.set_aspect(2 * np.pi / Omega.value)
            ax.set_aspect(1 / Omega.value)
        ax.set_xlim(-vzlim, vzlim)
        ax.set_ylim(-zlim.to_value(u.kpc), zlim.to_value(u.kpc))

    return fig, axes
