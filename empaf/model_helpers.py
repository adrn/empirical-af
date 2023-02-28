import jax

__all__ = ["monotonic_designer_func", "monotonic_designer_func_alt"]


@jax.jit
def monotonic_designer_func(x, A, alpha, x0, c=1.0):
    """
    This is a custom family of functions that can be controlled such that they are
    monotonic and have constant sign of the curvature (second derivative). This is
    mostly used internally to set the dependence of the :math:`e_m(r_z)` functions. This
    was inspired by the discussion in this StackExchange post:
    https://math.stackexchange.com/questions/65641/i-need-to-define-a-family-one-parameter-of-monotonic-curves
    """
    beta = (1 - alpha) / alpha
    return A * (c - (1 - (x / x0) ** (1 / beta)) ** beta)


@jax.jit
def monotonic_designer_func_alt(x, f0, f1, alpha, x0):
    """
    An alternate parametrization of the designer function ``monotonic_designer_func()``

    This is a custom family of functions that can be controlled such that they are
    monotonic and have constant sign of the curvature (second derivative). This is
    mostly used internally to set the dependence of the :math:`e_m(r_z)` functions. This
    was inspired by the discussion in this StackExchange post:
    https://math.stackexchange.com/questions/65641/i-need-to-define-a-family-one-parameter-of-monotonic-curves
    """
    A = (f1 - f0) / (1 + monotonic_designer_func(1.0, 1.0, alpha, x0, c=0.0))
    offset = f0 + A
    return monotonic_designer_func(x, c=0.0, A=A, alpha=alpha, x0=x0) + offset
