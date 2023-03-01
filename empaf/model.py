import abc
from functools import partial
from warnings import warn

import astropy.table as at
import astropy.units as u
import jax
import jax.numpy as jnp
import jaxopt
import scipy.interpolate as sci
from astropy.stats import median_absolute_deviation as MAD
from gala.units import UnitSystem, galactic
from jax.scipy.special import gammaln
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
from jaxopt import Bisection
from scipy.stats import binned_statistic, binned_statistic_2d

from empaf.jax_helpers import simpson
from empaf.model_helpers import custom_tanh_func_alt

__all__ = ["DensityOrbitModel", "LabelOrbitModel"]


class OrbitModelBase:
    _param_names = ["ln_Omega", "e_params", "z0", "vz0"]

    def __init__(self, e_funcs=None, unit_sys=galactic):
        r"""
        Notation:
        - :math:`\Omega_0` or ``Omega_0``: A scale frequency used to compute the
          elliptical radius ``rz_prime``. This can be interpreted as the asymptotic
          midplane orbital frequency.
        - :math:`r_z'` or ``rz_prime``: The "raw" elliptical radius :math:`\sqrt{z^2\,
          \Omega_0 + v_z^2 \, \Omega_0^{-1}}`.
        - :math:`\theta'` or ``theta_prime``: The "raw" z angle defined as :math:`\tan
          {\theta'} = \frac{z}{v_z}\,\Omega_0`.
        - :math:`r_z` or ``rz``: The distorted elliptical radius :math:`r_z = r_z' \,
          f(r_z', \theta_z')` where :math:`f(\cdot)` is the distortion function.
        - :math:`\theta` or ``theta``: The true vertical angle.
        - :math:`f(r_z', \theta_z')`: The distortion function is a Fourier expansion,
          defined as: :math:`f(r_z', \theta_z') = 1+\sum_m e_m(r_z')\,\cos(m\,\theta')`

        Parameters
        ----------
        e_funcs : dict (optional)
            A dictionary that provides functions that specify the dependence of the
            Fourier distortion coefficients :math:`e_m(r_z')`. Keys should be the
            (integer) "m" order of the distortion term (for the distortion function),
            and values should be Python callable objects that can be passed to
            `jax.jit()`.
        unit_sys : `gala.units.UnitSystem` (optional)
            The unit system to work in. Default is to use the "galactic" unit system
            from Gala: (kpc, Myr, Msun).

        """
        if e_funcs is None:
            # Default functions:
            self.e_funcs = {
                2: lambda *a, **k: custom_tanh_func_alt(*a, f0=0.0, **k),
                4: lambda *a, **k: custom_tanh_func_alt(*a, f0=0.0, **k),
            }
            self._default_e_funcs = True
        else:
            self.e_funcs = {int(m): jax.jit(e_func) for m, e_func in e_funcs.items()}
            self._default_e_funcs = False

        # Unit system:
        self.unit_sys = UnitSystem(unit_sys)

    @partial(jax.jit, static_argnames=["self"])
    def get_es(self, rz_prime, e_params):
        """
        Compute the Fourier m-order coefficients
        """
        es = {}
        for m, pars in e_params.items():
            es[m] = self.e_funcs[m](rz_prime, **pars)
        return es

    @partial(jax.jit, static_argnames=["self"])
    def z_vz_to_rz_theta_prime(self, z, vz, params):
        r"""
        Compute :math:`r_z'` (``rz_prime``) and :math:`\theta_z'` (``theta_prime``)
        """
        x = (vz - params["vz0"]) / jnp.sqrt(jnp.exp(params["ln_Omega"]))
        y = (z - params["z0"]) * jnp.sqrt(jnp.exp(params["ln_Omega"]))

        rz_prime = jnp.sqrt(x**2 + y**2)
        th_prime = jnp.arctan2(y, x)

        return rz_prime, th_prime

    z_vz_to_rz_theta_prime_arr = jax.vmap(
        z_vz_to_rz_theta_prime, in_axes=[None, 0, 0, None]
    )

    @partial(jax.jit, static_argnames=["self"])
    def get_rz(self, rz_prime, theta_prime, e_params):
        """
        Compute the distorted radius :math:`r_z`
        """
        es = self.get_es(rz_prime, e_params)
        return rz_prime * (
            1
            + jnp.sum(
                jnp.array([e * jnp.cos(n * theta_prime) for n, e in es.items()]), axis=0
            )
        )

    @partial(jax.jit, static_argnames=["self"])
    def get_rz_prime(self, rz, theta_prime, e_params):
        """
        Compute the raw radius :math:`r_z'` by inverting the distortion transformation
        """
        bisec = Bisection(
            lambda x, rrz, tt_prime, ee_params: self.get_rz(x, tt_prime, ee_params)
            - rrz,
            lower=0.0,
            upper=1.0,
            maxiter=30,
            jit=True,
            unroll=True,
            check_bracket=False,
            tol=1e-4,
        )
        return bisec.run(rz, rrz=rz, tt_prime=theta_prime, ee_params=e_params).params

    @partial(jax.jit, static_argnames=["self"])
    def get_z(self, rz, theta_prime, params):
        rzp = self.get_rz_prime(rz, theta_prime, params["e_params"])
        return rzp * jnp.sin(theta_prime) / jnp.sqrt(jnp.exp(params["ln_Omega"]))

    @partial(jax.jit, static_argnames=["self"])
    def get_vz(self, rz, theta_prime, params):
        rzp = self.get_rz_prime(rz, theta_prime, params["e_params"])
        return rzp * jnp.cos(theta_prime) * jnp.sqrt(jnp.exp(params["ln_Omega"]))

    @partial(jax.jit, static_argnames=["self", "N_grid"])
    def _get_Tz_Jz_thetaz(self, z, vz, params, N_grid):
        rzp_, thp_ = self.z_vz_to_rz_theta_prime(z, vz, params)
        rz = self.get_rz(rzp_, thp_, params["e_params"])

        dz_dthp_func = jax.vmap(
            jax.grad(self.get_z, argnums=1), in_axes=[None, 0, None]
        )

        get_vz = jax.vmap(self.get_vz, in_axes=[None, 0, None])

        # Grid of theta_prime to do the integral over:
        thp_grid = jnp.linspace(0, jnp.pi / 2, N_grid)
        vz_th = get_vz(rz, thp_grid, params)
        dz_dthp = dz_dthp_func(rz, thp_grid, params)

        Tz = 4 * simpson(dz_dthp / vz_th, thp_grid)
        Jz = 4 / (2 * jnp.pi) * simpson(dz_dthp * vz_th, thp_grid)

        thp_partial = jnp.linspace(0, thp_, N_grid)
        vz_th_partial = get_vz(rz, thp_partial, params)
        dz_dthp_partial = dz_dthp_func(rz, thp_partial, params)
        dt = simpson(dz_dthp_partial / vz_th_partial, thp_partial)
        thz = 2 * jnp.pi * dt / Tz

        return Tz, Jz, thz

    _get_Tz_Jz_thetaz = jax.vmap(_get_Tz_Jz_thetaz, in_axes=[None, 0, 0, None, None])

    @u.quantity_input
    def compute_action_angle(self, z: u.kpc, vz: u.km / u.s, params, N_grid=101):
        """
        Compute the vertical period, action, and angle given input phase-space
        coordinates.

        Parameters
        ----------
        TODO
        """
        z = z.decompose(self.unit_sys).value
        vz = vz.decompose(self.unit_sys).value
        Tz, Jz, thz = self._get_Tz_Jz_thetaz(z, vz, params, N_grid)

        tbl = at.QTable()
        tbl["T_z"] = Tz * self.unit_sys["time"]
        tbl["Omega_z"] = 2 * jnp.pi * u.rad / tbl["T_z"]
        tbl["J_z"] = Jz * self.unit_sys["length"] ** 2 / self.unit_sys["time"]
        tbl["theta_z"] = thz * self.unit_sys["angle"]

        return tbl

    @abc.abstractmethod
    def objective(self):
        pass

    def optimize(self, params0, bounds=None, jaxopt_kwargs=None, **data):
        """
        Parameters
        ----------
        params0 : dict (optional)
        bounds : tuple of dict (optional)
        jaxopt_kwargs : dict (optional)
        **data
            Passed through to the objective function

        Returns
        -------
        jaxopt_result : TODO
            TODO
        """
        if jaxopt_kwargs is None:
            jaxopt_kwargs = dict()
        jaxopt_kwargs.setdefault("maxiter", 16384)

        if bounds is not None:
            # Detect packed bounds (a single dict):
            if isinstance(bounds, dict):
                bounds = self.unpack_bounds(bounds)

            jaxopt_kwargs.setdefault("method", "L-BFGS-B")
            optimizer = jaxopt.ScipyBoundedMinimize(
                fun=self.objective,
                **jaxopt_kwargs,
            )
            res = optimizer.run(init_params=params0, bounds=bounds, **data)

        else:
            jaxopt_kwargs.setdefault("method", "BFGS")
            raise NotImplementedError("TODO")

        # warn if optimization was not successful, set state if successful
        if not res.state.success:
            warn(
                "Optimization failed! See the returned result object for more "
                "information, but the model state was not updated"
            )

        return res

    @classmethod
    def unpack_bounds(cls, bounds):
        """
        Split a bounds dictionary that is specified like: {"key": (lower, upper)} into
        two bounds dictionaries for the lower and upper bounds separately, e.g., for the
        example above: {"key": lower} and {"key": upper}.
        """
        import numpy as np

        def clean_dict(d):
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items()}
            else:
                d = np.array(d)
                assert d.shape[0] == 2
                return d

        # Make sure all tuples / lists become arrays:
        clean_bounds = clean_dict(bounds)

        vals, treedef = jax.tree_util.tree_flatten(clean_bounds)

        bounds_l = treedef.unflatten([x[0] for x in vals])
        bounds_r = treedef.unflatten([x[1] for x in vals])

        return bounds_l, bounds_r


class DensityOrbitModel(OrbitModelBase):
    def __init__(self, ln_dens_func, e_funcs=None, unit_sys=galactic):
        """
        {intro}

        Parameters
        ----------
        ln_dens_func : callable (optional)
            TODO
        {params}
        """
        super().__init__(e_funcs=e_funcs, unit_sys=unit_sys)
        self.ln_dens_func = jax.jit(ln_dens_func)

    __init__.__doc__ = __init__.__doc__.format(
        intro=OrbitModelBase.__init__.__doc__.split("Parameters")[0].rstrip(),
        params=OrbitModelBase.__init__.__doc__.split("----------")[1].lstrip(),
    )

    @u.quantity_input
    def get_params_init(self, z: u.kpc, vz: u.km / u.s, ln_dens_params0=None):
        """
        Estimate initial model parameters from the data

        Parameters
        ----------
        z : quantity-like or array-like
        vz : quantity-like or array-like

        Returns
        -------
        model : `VerticalOrbitModel`
            A copy of the initial model with state set to initial parameter estimates.
        """
        import numpy as np

        z = z.decompose(self.unit_sys).value
        vz = vz.decompose(self.unit_sys).value

        std_z = 1.5 * MAD(z, ignore_nan=True)
        std_vz = 1.5 * MAD(vz, ignore_nan=True)
        nu = std_vz / std_z

        pars0 = {"z0": np.nanmedian(z), "vz0": np.nanmedian(vz), "ln_Omega": np.log(nu)}
        rzp, _ = self.z_vz_to_rz_theta_prime_arr(z, vz, pars0)

        max_rz = np.nanpercentile(rzp, 99.5)
        rz_bins = np.linspace(0, max_rz, 25)  # TODO: fixed number
        dens_H, xe = np.histogram(rzp, bins=rz_bins)
        xc = 0.5 * (xe[:-1] + xe[1:])
        ln_dens = np.log(dens_H) - np.log(2 * np.pi * xc * (xe[1:] - xe[:-1]))

        # TODO: WTF - this is a total hack -- why is this needed???
        ln_dens = ln_dens - 8.6

        if ln_dens_params0 is not None:
            # Fit the specified ln_dens_func to the measured densities
            # This is a BAG O' HACKS!
            spl = sci.InterpolatedUnivariateSpline(xc, ln_dens, k=3)
            xeval = np.geomspace(1e-3, np.nanmax(xc), 32)  # MAGIC NUMBER:

            def obj(params, x, data_y):
                model_y = self.ln_dens_func(x, **params)
                return jnp.sum((model_y - data_y) ** 2)

            optim = jaxopt.ScipyMinimize(fun=obj, method="BFGS")
            res = optim.run(init_params=ln_dens_params0, x=xeval, data_y=spl(xeval))
            if res.state.success:
                pars0["ln_dens_params"] = res.params
            else:
                # TODO: warn!
                pass

        # If default e_funcs, we can specify some defaults:
        if self._default_e_funcs:
            pars0["e_params"] = {2: {}, 4: {}}
            pars0["e_params"][2]["f1"] = 0.1
            pars0["e_params"][2]["alpha"] = 0.33
            pars0["e_params"][2]["x0"] = 3.0

            pars0["e_params"][4]["f1"] = -0.02
            pars0["e_params"][4]["alpha"] = 0.45
            pars0["e_params"][4]["x0"] = 3.0
        else:
            # TODO: warn!
            pass

        return pars0

    @u.quantity_input
    def get_data_im(self, z: u.kpc, vz: u.km / u.s, bins):
        """
        Convert the raw data (stellar positions and velocities z, vz) into a binned 2D
        histogram / image of number counts.

        Parameters
        ----------
        z : quantity-like
        vz : quantity-like
        bins : dict
        """
        data_H, xe, ye = jnp.histogram2d(
            vz.decompose(self.unit_sys).value,
            z.decompose(self.unit_sys).value,
            bins=(bins["vz"], bins["z"]),
        )
        xc = 0.5 * (xe[:-1] + xe[1:])
        yc = 0.5 * (ye[:-1] + ye[1:])
        xc, yc = jnp.meshgrid(xc, yc)

        return {"z": jnp.array(yc), "vz": jnp.array(xc), "H": jnp.array(data_H.T)}

    @partial(jax.jit, static_argnames=["self"])
    def get_ln_dens(self, rz, params):
        return self.ln_dens_func(rz, **params["ln_dens_params"])

    @partial(jax.jit, static_argnames=["self"])
    def ln_density(self, z, vz, params):
        rzp, thp = self.z_vz_to_rz_theta_prime(z, vz, params)
        rz = self.get_rz(rzp, thp, params["e_params"])
        return self.get_ln_dens(rz, params)

    @partial(jax.jit, static_argnames=["self"])
    def ln_poisson_likelihood(self, params, z, vz, H):
        # Expected number:
        ln_Lambda = self.ln_density(z, vz, params)

        # gammaln(x+1) = log(factorial(x))
        return (H * ln_Lambda - jnp.exp(ln_Lambda) - gammaln(H + 1)).sum()

    @partial(jax.jit, static_argnames=["self"])
    def objective(self, params, z, vz, H):
        return -(self.ln_poisson_likelihood(params, z, vz, H)) / H.size


class LabelOrbitModel(OrbitModelBase):
    fit_name = "label"
    fit_param_names = ["A", "alpha", "x0"]
    # TODO: might need others to control offset, etc.

    def __init__(
        self, label_knots, e_knots, e_signs, e_k=1, label_k=3, unit_sys=galactic
    ):
        f"""
        {OrbitModelBase.__init__.__doc__.split('Parameters')[0]}

        Parameters
        ----------
        label_knots : array_like
            The knot locations for the spline that controls the label function. These
            are locations in :math:`r_z`.
        {OrbitModelBase.__init__.__doc__.split('----------')[1]}

        """
        self.label_knots = jnp.array(label_knots)
        self.label_k = int(label_k)
        super().__init__(e_knots=e_knots, e_signs=e_signs, e_k=e_k, unit_sys=unit_sys)

    @u.quantity_input
    def get_params_init(self, z: u.kpc, vz: u.km / u.s, label_stat):
        import numpy as np
        import scipy.interpolate as sci

        z = z.decompose(self.unit_sys).value
        vz = vz.decompose(self.unit_sys).value

        model = self.copy()

        # First, estimate nu0 with some crazy bullshit:
        med_stat = np.nanpercentile(label_stat, 15)

        fac = 0.02  # MAGIC NUMBER
        for _ in range(16):  # max number of iterations
            annulus_idx = np.abs(label_stat.ravel() - med_stat) < fac * np.abs(med_stat)
            if annulus_idx.sum() < 0.05 * len(annulus_idx):  # MAGIC NUMBER
                fac *= 2
            else:
                break

        else:
            raise ValueError("Shit!")

        nu = np.nanpercentile(np.abs(vz.ravel()[annulus_idx]), 25) / np.nanpercentile(
            np.abs(z.ravel()[annulus_idx]), 25
        )

        model.set_state({"z0": 0.0, "vz0": 0.0, "nu": nu})
        rzp, _ = model.z_vz_to_rz_theta_prime_arr(z, vz)

        # Now estimate the label function spline values, again, with some craycray:
        bins = np.linspace(0, 1.0, 9) ** 2  # TODO: arbitrary bin max = 1
        stat = binned_statistic(
            rzp.ravel(), label_stat.ravel(), bins=bins, statistic=np.nanmedian
        )
        xc = 0.5 * (stat.bin_edges[:-1] + stat.bin_edges[1:])

        simple_spl = sci.InterpolatedUnivariateSpline(
            xc[np.isfinite(stat.statistic)],
            stat.statistic[np.isfinite(stat.statistic)],
            k=1,
        )
        model.set_state({"label_vals": simple_spl(model.label_knots)})

        # Lastly, set all e vals to 0
        e_vals = {}
        e_vals[2] = jnp.full(len(self.e_knots[2]) - 1, 0.1 / len(self.e_knots[2]))
        e_vals[4] = jnp.full(len(self.e_knots[4]) - 1, 0.05 / len(self.e_knots[4]))
        for m in self.e_knots.keys():
            if m in e_vals:
                continue
            e_vals[m] = jnp.zeros(len(self.e_knots[m]) - 1)
        model.set_state({"e_vals": e_vals})

        model._validate_state()

        return model

    @u.quantity_input
    def get_data_im(
        self, z: u.kpc, vz: u.km / u.s, label, bins, **binned_statistic_kwargs
    ):
        import numpy as np

        stat = binned_statistic_2d(
            vz.decompose(self.unit_sys).value,
            z.decompose(self.unit_sys).value,
            label,
            bins=(bins["vz"], bins["z"]),
            **binned_statistic_kwargs,
        )
        xc = 0.5 * (stat.x_edge[:-1] + stat.x_edge[1:])
        yc = 0.5 * (stat.y_edge[:-1] + stat.y_edge[1:])
        xc, yc = jnp.meshgrid(xc, yc)

        # Compute label statistic error
        err_floor = 0.1
        stat_err = binned_statistic_2d(
            vz.decompose(self.unit_sys).value,
            z.decompose(self.unit_sys).value,
            label,
            bins=(bins["vz"], bins["z"]),
            statistic=lambda x: np.sqrt((1.5 * MAD(x)) ** 2 + err_floor**2)
            / np.sqrt(len(x)),
        )

        return {
            "z": jnp.array(yc),
            "vz": jnp.array(xc),
            "label_stat": jnp.array(stat.statistic.T),
            "label_stat_err": jnp.array(stat_err.statistic.T),
        }

    @partial(jax.jit, static_argnames=["self"])
    def get_label(self, rz):
        self._validate_state()
        label_vals = self.state["label_vals"]
        spl = InterpolatedUnivariateSpline(self.label_knots, label_vals, k=self.label_k)
        return spl(rz)

    @partial(jax.jit, static_argnames=["self"])
    def label(self, z, vz):
        self._validate_state()
        rzp, thp = self.z_vz_to_rz_theta_prime(z, vz)
        rz = self.get_rz(rzp, thp)
        return self.get_label(rz)

    @partial(jax.jit, static_argnames=["self"])
    def ln_label_likelihood(self, params, z, vz, label_stat, label_stat_err):
        self.set_state(params, overwrite=True)
        self._validate_state()

        rzp, thp = self.z_vz_to_rz_theta_prime(z, vz)
        rz = self.get_rz(rzp, thp)
        model_label = self.get_label(rz)

        # log-normal
        return -0.5 * jnp.nansum((label_stat - model_label) ** 2 / label_stat_err**2)

    @partial(jax.jit, static_argnames=["self"])
    def objective(self, params, z, vz, label_stat, label_stat_err):
        return (
            -(self.ln_label_likelihood(params, z, vz, label_stat, label_stat_err))
            / label_stat.size
        )
