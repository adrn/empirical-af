import astropy.units as u
import jax
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def plot_bdata(bdata, label_vlim=None):
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 5), constrained_layout=True, sharex=True, sharey=True
    )
    cs = axes[0].pcolormesh(
        bdata["vel"].to_value(u.km / u.s),
        bdata["pos"].to_value(u.kpc),
        bdata["counts"],
        norm=mpl.colors.LogNorm(),
        cmap="Greys",
    )
    fig.colorbar(cs, ax=axes[0])

    if label_vlim is None:
        label_vlim = (None, None)
    cs = axes[1].pcolormesh(
        bdata["vel"].to_value(u.km / u.s),
        bdata["pos"].to_value(u.kpc),
        bdata["label"],
        vmin=label_vlim[0],
        vmax=label_vlim[1],
    )
    fig.colorbar(cs, ax=axes[1])

    cs = axes[2].pcolormesh(
        bdata["vel"].to_value(u.km / u.s),
        bdata["pos"].to_value(u.kpc),
        bdata["label_err"],
        norm=mpl.colors.LogNorm(),
        cmap="Blues",
    )
    fig.colorbar(cs, ax=axes[2])

    axes[0].set_title("Log counts")
    axes[1].set_title("Mean label")
    axes[2].set_title("Label error")

    for ax in axes:
        ax.set_xlabel("$v_z$ [km/s]")
    axes[0].set_ylabel("$z$ [kpc]")

    return fig, axes


def plot_spline_functions(model, params):
    r_e_grid = np.linspace(0, model._label_knots.max(), 128)
    e_vals = model._get_es(r_e_grid, params["e_params"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")

    ax = axes[0]
    sum_ = None
    for m, vals in e_vals.items():
        (l,) = ax.plot(r_e_grid, vals, marker="", label=f"$e_{m}$")  # noqa
        ax.scatter(
            model._e_knots[m],
            model.e_funcs[m](model._e_knots[m], params["e_params"][m]["vals"]),
            color=l.get_color(),
        )

        if sum_ is None:
            sum_ = vals
        else:
            sum_ += vals

    ax.plot(r_e_grid, sum_, ls="--", marker="")
    ax.set_title("$e_m$ functions")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlabel("$r_e$")

    ax = axes[1]
    l_vals = model.label_func(r_e_grid, **params["label_params"])
    (l,) = ax.plot(r_e_grid, l_vals, marker="")  # noqa

    l_vals = model.label_func(model._label_knots, **params["label_params"])
    ax.scatter(model._label_knots, l_vals, color=l.get_color())
    ax.set_xlabel("$r$")
    ax.set_title("Label function")

    return fig, axes


def plot_data_model_residual(
    model,
    bdata,
    params,
    zlim,
    vzlim=None,
    aspect=True,
    residual_lim=0.05,
    subplots_kwargs=None,
    suptitle1="Demonstration with Simulated Data:",
    suptitle2="",
):
    title_fontsize = 20
    title_pad = 10

    cb_labelsize = 16
    mgfe_cbar_xlim = (0, 0.15)
    mgfe_cbar_vlim = (-0.05, 0.18)

    tmp_aaf = model.compute_action_angle(
        np.atleast_1d(zlim) * 0.75, [0.0] * u.km / u.s, params
    )
    Omega = tmp_aaf["Omega"][0]
    if vzlim is None:
        vzlim = zlim * Omega
    vzlim = vzlim.to_value(u.km / u.s, u.dimensionless_angles())

    if subplots_kwargs is None:
        subplots_kwargs = dict()
    subplots_kwargs.setdefault("figsize", (16, 5.3))
    subplots_kwargs.setdefault("sharex", True)
    subplots_kwargs.setdefault("sharey", True)
    subplots_kwargs.setdefault("layout", "constrained")
    fig, axes = plt.subplots(1, 3, **subplots_kwargs)

    cs = axes[0].pcolormesh(
        bdata["vel"].to_value(u.km / u.s),
        bdata["pos"].to_value(u.kpc),
        bdata["label"],
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

    model_mgfe = np.array(model._get_label(bdata["pos"], bdata["vel"], params))
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
        bdata["label"] - model_mgfe,
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
    fig.suptitle(f"{suptitle1} {suptitle2}", fontsize=24)

    # Labels
    axes[0].set_ylabel(f"$z$ [{u.kpc:latex_inline}]")
    for ax in axes:
        ax.set_xlabel(f"$v_z$ [{u.km/u.s:latex_inline}]")

    # Ticks
    axes[0].set_xticks(np.arange(-300, 300 + 1, 100))
    axes[0].set_xticks(np.arange(-300, 300 + 1, 50), minor=True)
    axes[1].set_yticks(np.arange(-3, 3 + 1e-3, 1))
    axes[1].set_yticks(np.arange(-3, 3 + 1e-3, 0.5), minor=True)

    aspect = Omega.to_value(u.km / u.s / u.kpc, u.dimensionless_angles())
    for ax in axes:
        if aspect:
            ax.set_aspect(aspect)
        ax.set_xlim(-vzlim, vzlim)
        ax.set_ylim(-zlim.to_value(u.kpc), zlim.to_value(u.kpc))

    return fig, axes


@u.quantity_input
def plot_az_Jz(
    acc_MAP, acc_samples, z_grid: u.Quantity[u.kpc], aaf, true_acc, true_Jz, axes=None
):
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6.0), layout="constrained")
    else:
        fig = axes[0].figure

    J_unit = u.km / u.s * u.kpc
    a_unit = u.km / u.s / u.Myr

    z_grid = z_grid.to_value(u.kpc)
    acc_samples = acc_samples.to_value(a_unit)

    ax = axes[0]
    ax.plot(
        z_grid,
        acc_MAP.to_value(a_unit),
        color="k",
        marker="",
        label="OTI inferred $a_z$",
        zorder=10,
    )
    ax.fill_between(
        z_grid,
        np.nanpercentile(acc_samples, 16, axis=0),
        np.nanpercentile(acc_samples, 84, axis=0),
        color="#cccccc",
        lw=0,
        label=r"uncertainty",
        zorder=2,
    )
    ax.plot(
        z_grid,
        true_acc.to_value(a_unit),
        ls="--",
        color="tab:green",
        marker="",
        label=r"true $a_z(z)$",
        zorder=15,
    )
    ax.set_xlim(z_grid.min(), z_grid.max())
    ax.axhline(0, zorder=-10, color="tab:blue", alpha=0.4, ls=":")
    ax.legend(loc="lower left", fontsize=16)

    ax.set_xlabel(f"vertical position, $z$ [{u.kpc:latex}]")
    ax.set_ylabel(
        f"vertical acceleration, $a_z$\n"
        f"[{u.km/u.s:latex_inline} {u.Myr**-1:latex_inline}]"
    )
    ax.set_title("acceleration profile")

    # -----------------------------------------------------------------------
    ax = axes[1]
    Jz_max = true_Jz.max().to_value(J_unit)
    ax.plot(
        aaf["J"].to_value(J_unit),
        true_Jz.to_value(J_unit),
        marker="o",
        ls="none",
        alpha=0.25,
        mew=0,
        ms=2.0,
        label="particle $J_z$ values",
        rasterized=True,
    )
    ax.axline(
        [0, 0],
        [Jz_max] * 2,
        ls="--",
        color="tab:green",
        marker="",
        label="one-to-one line",
    )
    ax.set_xlabel(f"inferred $J_z$ [{J_unit:latex_inline}]")
    ax.set_ylabel(f"true $J_z$ [{J_unit:latex_inline}]")
    ax.set_xticks(np.arange(0, 2 * Jz_max, 50))
    ax.set_yticks(np.arange(0, 2 * Jz_max, 50))

    ax.set_xlim(0, Jz_max)
    ax.set_ylim(0, Jz_max)
    ax.set_title("vertical action $J_z$")

    ax.legend(loc="lower right", fontsize=16)

    return fig, axes


def plot_orbit_shapes(model, params, true_orbit_zvzs, zlim, vzlim, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    th_e_grid = np.linspace(0, 2 * np.pi, 1024)

    get_pos_vec = jax.vmap(model._get_pos, in_axes=[None, 0, None, None])
    get_vel_vec = jax.vmap(model._get_vel, in_axes=[None, 0, None, None])

    for i, (true_z, true_vz) in enumerate(true_orbit_zvzs):
        rr = model._get_r(
            *model._get_elliptical_coords(0.0, true_vz[0], params),
            params["e_params"],
        )

        oti_pp = get_pos_vec(rr, th_e_grid, params, None)
        oti_vv = (get_vel_vec(rr, th_e_grid, params, None) * u.kpc / u.Myr).to_value(
            u.km / u.s
        )
        if i == 0:
            kw1 = dict(label="OTI inferred")
            kw2 = dict(label="true orbit")
        else:
            kw1 = {}
            kw2 = {}
        ax.plot(oti_vv, oti_pp, marker="", ls="-", color="k", zorder=4, **kw1)

        ax.plot(
            (true_vz * u.kpc / u.Myr).to_value(u.km / u.s),
            true_z,
            marker="",
            color="tab:green",
            ls="--",
            zorder=10,
            **kw2,
        )

    # Labels
    ax.set_ylabel(f"vertical position, $z$ [{u.kpc:latex_inline}]")
    ax.set_xlabel(f"vertical velocity, $v_z$ [{u.km/u.s:latex_inline}]")

    # Ticks
    ax.set_xticks(np.arange(-200, 200 + 1, 100))
    ax.set_xticks(np.arange(-200, 200 + 1, 50), minor=True)
    ax.set_yticks(np.arange(-3, 3 + 1e-3, 1))
    ax.set_yticks(np.arange(-3, 3 + 1e-3, 0.5), minor=True)
    ax.set_xlim(vzlim)
    ax.set_ylim(zlim)
    ax.set_title("orbit shapes")
    ax.legend(loc="lower left", fontsize=16).set_zorder(10)

    return fig, ax
