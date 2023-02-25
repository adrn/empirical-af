import jax
import jax.numpy as jnp

__all__ = ["simpson"]


@jax.jit
def simpson(y, x):
    """
    Evaluate the definite integral of a function using Simpson's rule

    Note: x must be a regularly-spaced grid of points!
    """

    dx = jnp.diff(x)[0]
    num_points = len(x)
    if num_points % 2 == 0:
        raise ValueError("Because of laziness, the input size must be odd")

    weights_first = jnp.asarray([1.0])
    weights_mid = jnp.tile(jnp.asarray([4.0, 2.0]), [(num_points - 3) // 2])
    weights_last = jnp.asarray([4.0, 1.0])
    weights = jnp.concatenate([weights_first, weights_mid, weights_last], axis=0)

    return dx / 3 * jnp.sum(y * weights, axis=-1)


@jax.jit
def designer_func(x, A, alpha, x0, c=1.0):
    """
    This is a custom family of functions that can be controlled such that they are
    monotonic and have constant sign of the curvature (second derivative). This is
    mostly used internally to set the dependence of the :math:`e_m(r_z)` functions. This
    was inspired by the discussion in this StackExchange post:
    https://math.stackexchange.com/questions/65641/i-need-to-define-a-family-one-parameter-of-monotonic-curves
    """
    beta = (1 - alpha) / alpha
    return A * (c - (1 - (x / x0) ** (1 / beta)) ** beta)
