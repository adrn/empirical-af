from functools import partial

import astropy.units as u
import jax.numpy as jnp
import numpy as np
import torusimaging as oti
from astropy.constants import G
from gala.units import galactic


def label_func_base(r, label_vals, knots):
    return oti.model_helpers.monotonic_quadratic_spline(knots, label_vals, r)


def e_func_base(r_e, vals, sign, knots):
    return sign * oti.model_helpers.monotonic_quadratic_spline(
        knots, jnp.concatenate((jnp.array([0.0]), vals)), r_e
    )


def regularization_func_base(params, e_regularize: bool, e_regularize_sigmas: dict):
    p = 0.0

    if e_regularize:
        # L2
        for m in params["e_params"]:
            p += jnp.sum((params["e_params"][m]["vals"] / e_regularize_sigmas[m]) ** 2)

        # # L1
        # for m in params["e_params"]:
        #     p += jnp.sum(
        #         jnp.abs(params["e_params"][m]["vals"] / e_regularize_sigmas[m])
        #     )

    return p


class SplineLabelModelWrapper:
    def __init__(
        self,
        r_e_max: float,
        label_n_knots: int,
        label0_bounds: tuple,
        label_grad_sign: float,
        e_n_knots: dict,
        e_knots_scale=None,
        e_signs=None,
        e_regularize=True,
        e_regularize_sigmas=None,
        unit_sys=galactic,
    ):
        self.unit_sys = unit_sys

        # ------------------------------------------------------------------------------
        # Set up the label function bits:

        # Knot locations, spaced equally in r_z
        self.label_knots = jnp.linspace(0, r_e_max, label_n_knots)
        label_func = partial(label_func_base, knots=self.label_knots)

        if label_grad_sign > 0:
            label_func_bounds = {
                "label_vals": (
                    np.concatenate(
                        ([label0_bounds[0]], jnp.full(label_n_knots - 1, 0.0))
                    ),
                    np.concatenate(
                        ([label0_bounds[1]], jnp.full(label_n_knots - 1, 10.0))
                    ),
                )
            }
        else:
            label_func_bounds = {
                "label_vals": (
                    np.concatenate(
                        ([label0_bounds[0]], jnp.full(label_n_knots - 1, -10.0))
                    ),
                    np.concatenate(
                        ([label0_bounds[1]], jnp.full(label_n_knots - 1, 0.0))
                    ),
                )
            }

        # ------------------------------------------------------------------------------
        # Set up the e function components:
        if e_knots_scale is None:
            e_knots_scale = {}
        for m in e_n_knots:
            e_knots_scale.setdefault(m, (lambda x: x, lambda x: x))

        self.e_n_knots = e_n_knots
        self.e_knots = {
            m: e_knots_scale[m][0](jnp.linspace(0, e_knots_scale[m][1](r_e_max), n))
            for m, n in e_n_knots.items()
        }
        if e_signs is None:
            e_signs = {m: (-1.0 if (m / 2) % 2 == 0 else 1.0) for m in self.e_knots}

        e_funcs = {
            m: partial(e_func_base, sign=e_signs[m], knots=self.e_knots[m])
            for m in self.e_knots
        }

        e_bounds = {
            m: {"vals": (jnp.full(n - 1, 0), jnp.full(n - 1, 10.0))}  # TODO: magic
            for m, n in e_n_knots.items()
        }

        if e_regularize_sigmas is None:
            # Default value of L2 regularization stddev:
            e_regularize_sigmas = {m: 0.1 for m in self.e_knots}

        # ------------------------------------------------------------------------------
        # Setup the regularization function:
        reg_func = partial(
            regularization_func_base,
            e_regularize=e_regularize,
            e_regularize_sigmas=e_regularize_sigmas,
        )

        self.label_model = oti.LabelOrbitModel(
            label_func=label_func,
            e_funcs=e_funcs,
            regularization_func=reg_func,
            unit_sys=self.unit_sys,
        )

        self._bounds = {}

        # Reasonable bounds for the midplane density
        dens0_bounds = [0.01, 10] * u.Msun / u.pc**3
        self._bounds["ln_Omega"] = 0.5 * np.log(
            (4 * np.pi * G * dens0_bounds).decompose(self.unit_sys).value
        )
        self._bounds["z0"] = (-0.5, 0.5)
        self._bounds["vz0"] = (-0.05, 0.05)  # ~50 km/s
        self._bounds["e_params"] = e_bounds
        self._bounds["label_params"] = label_func_bounds

    def get_init_params(self, oti_data, label_name=None):
        if label_name is None:
            if len(oti_data.labels) == 1:
                label_name = list(oti_data.labels.keys())[0]
            else:
                raise ValueError("must specify label_name")

        label = oti_data.labels[label_name]

        params0 = self.label_model.get_params_init(oti_data.pos, oti_data.vel, label)
        r_e, _ = self.label_model.get_elliptical_coords(
            oti_data._pos, oti_data._vel, params0
        )

        params0["e_params"] = {
            m: {"vals": jnp.zeros(self.e_n_knots[m] - 1)} for m in self.e_knots
        }

        # Estimate the label value near r_e = 0 and slopes for knot values:
        r1, r2 = np.nanpercentile(r_e, [10, 90])
        label0 = np.nanmean(label[r_e <= r1])
        label_slope = (np.nanmedian(label[r_e >= r2]) - label0) / (r2 - r1)

        params0["label_params"] = {
            "label_vals": np.concatenate(
                (
                    [label0],
                    np.full(len(self.label_knots) - 1, label_slope),
                )
            )
        }

        return params0

    def run(
        self,
        oti_data,
        bins,
        label_name=None,
        label_err_floor=0.01,
        data_kw=None,
        jaxopt_kw=None,
    ):
        if data_kw is None:
            data_kw = {}
        if jaxopt_kw is None:
            jaxopt_kw = {}
        jaxopt_kw.setdefault("tol", 1e-12)

        bdata, label_name = oti_data.get_binned_label(
            bins, label_name=label_name, **data_kw
        )

        tmp, _ = oti_data.get_binned_label(
            bins,
            label_name=label_name,
            statistic=lambda x: np.sqrt(
                label_err_floor**2 + np.nanvar(x) / (len(x) + 1)
            ),
            **data_kw,
        )
        bdata[f"{label_name}_err"] = tmp[label_name]
        p0 = self.get_init_params(oti_data, label_name=label_name)

        # First check that objective evaluates to a finite value:
        mask = np.isfinite(bdata[label_name])
        data = dict(
            pos=bdata["pos"].decompose(self.unit_sys).value[mask],
            vel=bdata["vel"].decompose(self.unit_sys).value[mask],
            label=bdata[label_name][mask],
            label_err=bdata[f"{label_name}_err"][mask],
        )
        test_val = self.label_model.objective(p0, **data)
        if not np.isfinite(test_val):
            raise RuntimeError("Objective function evaluated to non-finite value")

        res = self.label_model.optimize(
            params0=p0, bounds=self._bounds, jaxopt_kwargs=jaxopt_kw, **data
        )

        return bdata, res
