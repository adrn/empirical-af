import astropy.units as u
import gala.dynamics as gd
import gala.potential as gp
import jax
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from gala.dynamics.actionangle.tests.staeckel_helpers import galpy_find_actions_staeckel
from gala.units import galactic
from pysr import PySRRegressor
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)
# import agama
# agama.setUnits(mass=u.Msun, length=u.kpc, time=u.Myr)


# m = 5.4e11
def eval_efunc(z_grid, m, b, R0, vc0=229 * u.km / u.s):
    m = m * 1e10
    h_R = 2.6

    def obj(logmh):
        gala_pot = gp.CCompositePotential(
            disk=gp.MN3ExponentialDiskPotential(m=m, h_R=h_R, h_z=b, units=galactic),
            halo=gp.NFWPotential(m=np.exp(logmh), r_s=15.63, units=galactic),
        )
        return (
            np.squeeze((gala_pot.circular_velocity(R0 * [1.0, 0, 0]) - vc0).value) ** 2
        )

    res = minimize(obj, x0=np.log(5.4e11), method="BFGS")

    gala_pot = gp.CCompositePotential(
        disk=gp.MN3ExponentialDiskPotential(m=m, h_R=h_R, h_z=b, units=galactic),
        halo=gp.NFWPotential(m=np.exp(res.x[0]), r_s=15.63, units=galactic),
    )

    # agama_pot = agama.Potential(
    #     dict(
    #         type="miyamotonagai",
    #         mass=gala_pot["disk"].parameters["m"].value,
    #         scaleradius=gala_pot["disk"].parameters["a"].value,
    #         scaleheight=gala_pot["disk"].parameters["b"].value,
    #     ),
    #     dict(
    #         type="nfw",
    #         mass=gala_pot["halo"].parameters["m"].value,
    #         scaleradius=gala_pot["halo"].parameters["r_s"].value,
    #     ),
    # )

    xyz0 = [1.0, 0, 0] * R0
    vc = gala_pot.circular_velocity(xyz0)
    assert u.isclose(vc, vc0)

    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        # Omega0 = np.sqrt(gala_pot.density(xyz0) * (4 * np.pi * G)).to(u.rad / u.Myr)
        Omega0 = np.sqrt(gala_pot.hessian(xyz0)[2, 2, 0]).to(u.rad / u.Myr)

    rzp = z_grid * np.sqrt(Omega0)

    xv = np.zeros((z_grid.size, 6))
    xv[:, 0] = R0.decompose(galactic).value
    xv[:, 2] = z_grid.decompose(galactic).value
    xv[:, 4] = vc0.decompose(galactic).value

    # af = agama.ActionFinder(agama_pot)
    # J, ang, freq = af(xv, angles=True)
    # Omz = freq[:, 1] * u.rad/u.Myr

    w = gd.PhaseSpacePosition.from_w(xv.T, units=galactic)
    aaf = galpy_find_actions_staeckel(gala_pot, w)
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        Omz = aaf["freqs"][:, 2].to(u.rad / u.Myr)

    rz = z_grid * np.sqrt(Omz)
    e_sum = -(rz / rzp - 1)

    return rzp, rz, e_sum


def main(filename):
    N_p = 2
    N_k = 12
    N_z = 32

    par_names = ["m", "b"]
    m_1d = np.linspace(1, 10, N_k)
    b_1d = np.geomspace(0.075, 1.0, N_k)

    # z_1d = np.linspace(1e-3**0.5, 3.0**0.5, N_z) ** 2
    z_1d = np.geomspace(2e-3, 3.0, N_z)

    pars_1d = np.stack(np.meshgrid(m_1d, b_1d))
    pars = pars_1d.reshape(N_p, 1, -1)
    pars = np.broadcast_to(pars, (N_p, N_z, pars.shape[-1]))

    X = np.full((N_k**N_p, N_z, 1 + N_p), np.nan)
    y = np.full((N_k**N_p, N_z), np.nan)
    print(f"X shape = {X.shape}, y shape = {y.shape}")

    # Populate X and y arrays:
    for i, (m, b) in enumerate(pars_1d.reshape(N_p, -1).T):
        try:
            rzp, _, e_sum = eval_efunc(z_grid=z_1d * u.kpc, m=m, b=b, R0=8 * u.kpc)
        except AssertionError:
            print(m, b)
            rzp = np.full_like(rzp, np.nan)
            e_sum = np.full_like(e_sum, np.nan)

        X[i, :, 0] = np.log(rzp.value)  # NOTE: updated from linear "rzp.value"!
        # X[i, :, 0] = rzp.value
        X[i, :, 1] = m
        X[i, :, 2] = b
        y[i] = e_sum

    flat_X = X.reshape(-1, 1 + N_p)
    flat_y = y.ravel()

    mask = flat_y > 1e-13
    flat_X = flat_X[mask]
    flat_y = flat_y[mask]
    flat_log_y = np.log(flat_y)
    print(f"flat_X shape = {flat_X.shape}, flat_y shape = {flat_y.shape}")

    np.savez(
        "Xy.npz",
        X=X,
        y=y,
        flat_X=flat_X,
        flat_y=flat_y,
        flat_log_y=flat_log_y,
        mask=mask,
    )

    # Plot all r_e vs. e_sum curves
    fig, axes = plt.subplots(
        1, N_p, figsize=(4 * N_p, 4), sharex=True, sharey=True, layout="constrained"
    )
    for j, i in enumerate(range(1, N_p + 1)):
        norm = mpl.colors.LogNorm(flat_X[:, i].min(), flat_X[:, i].max())
        cmap = plt.get_cmap("viridis")
        axes.flat[j].scatter(flat_X[:, 0], flat_y, color=cmap(norm(flat_X[:, i])), s=2)
        axes.flat[j].set_title(par_names[j])
    fig.savefig("flat_X-r_e-curves.png", dpi=250)
    plt.close(fig)

    # ----------------------------------------------------------------------------------
    # Run PySR
    model = PySRRegressor(
        # binary_operators=["+", "-", "/", "*", "^"],
        unary_operators=["log", "exp", "square", "cube", "sqrt"],
        # unary_operators=["log", "exp", "sqrt"],
        maxsize=30,
        # parsimony=1e-5,
        # batching=True,
        # niterations=128,
        niterations=1024,
        populations=64,
        early_stop_condition=(
            "stop_if(loss, complexity) = loss < 1e-5 && complexity < 20"
            # Stop early if we find a good and simple equation
        ),
        precision=64,
        # ^ Higher precision calculations.
        # constraints={"^": (-1, 1)},
    )
    model.fit(flat_X, flat_log_y, variable_names=["r_e", "m", "b"])

    pred_flat_y = np.exp(model.predict(flat_X))
    pred_y = np.full_like(y, np.nan)
    pred_y.flat[mask] = pred_flat_y
    print(f"All positive? {np.all(pred_flat_y > 0)}")

    fig = plt.figure(figsize=(12, 3))
    plt.imshow((y - pred_y).T, origin="lower", cmap="RdBu", vmin=-0.04, vmax=0.04)
    plt.colorbar()
    plt.savefig("y-residuals.png", dpi=250)
    plt.close(fig)

    fig, axes = plt.subplots(
        6, 6, figsize=(12, 12), sharex=True, sharey=True, layout="constrained"
    )
    for j, i in enumerate(np.linspace(0, X.shape[0] - 1, axes.size).astype(int)):
        ax = axes.flat[j]
        ax.plot(X[i, :, 0], y[i])
        ax.plot(X[i, :, 0], pred_y[i])
    fig.savefig("X-y-compare.png", dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = None
    main(filename)
