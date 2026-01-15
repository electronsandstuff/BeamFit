from gaussufunc import supergaussian_internal, supergaussian_grad_internal
import numpy as np


def supergaussian(x, y, mu_x, mu_y, sigma_xx, sigma_xy, sigma_yy, n, a, o):
    """
    Compiled numpy compatible ufunc which computes the bivariate super-Gaussian function.

    Computes the function:

        f(r) = a*exp(-(1/2(r - mu)^T Sigma^{-1} (r - mu))^n) + o

    where r is the vector {x, y}, mu is the centroid vector {mu_x, mu_y}, and
    Sigma is the covariance matrix {{sigma_xx, sigma_xy}, {sigma_xy, sigma_yy}}.

    As a ufunc, sending a numpy array into the values will broadcast them against
    each other as is possible.

    Parameters
    ----------
    x : float or np.ndarray
        x value(s) at which the super-Gaussian is evaluated.
    y : float or np.ndarray
        y value(s) at which the super-Gaussian is evaluated.
    mu_x : float
        x component of centroid.
    mu_y : float
        y component of centroid.
    sigma_xx : float
        x variance.
    sigma_xy : float
        xy correlation.
    sigma_yy : float
        y variance.
    n : float
        Super-Gaussian parameter.
    a : float
        Amplitude.
    o : float
        Offset.

    Returns
    -------
    np.ndarray
        The values of the super-Gaussian.
    """
    return supergaussian_internal(
        x, y, mu_x, mu_y, sigma_xx, sigma_xy, sigma_yy, n, a, o
    )


def supergaussian_grad(x, y, mu_x, mu_y, sigma_xx, sigma_xy, sigma_yy, n, a, o):
    """
    Calculate the Jacobian of the super-Gaussian with respect to the parameters.

    Computes the Jacobian matrix of the super-Gaussian function with respect to
    the parameters (mu_x, mu_y, sigma_xx, sigma_xy, sigma_yy, n, a, o).

    Parameters
    ----------
    x : np.ndarray, shape (m,)
        Array of x values to evaluate at.
    y : np.ndarray, shape (m,)
        Array of y values to evaluate at.
    mu_x : float
        x component of centroid.
    mu_y : float
        y component of centroid.
    sigma_xx : float
        x variance.
    sigma_xy : float
        xy correlation.
    sigma_yy : float
        y variance.
    n : float
        Super-Gaussian parameter.
    a : float
        Amplitude.
    o : float
        Offset.

    Returns
    -------
    np.ndarray, shape (m, 8)
        The Jacobian of the super-Gaussian with respect to all parameters.
    """
    return np.array(
        supergaussian_grad_internal(
            x, y, mu_x, mu_y, sigma_xx, sigma_xy, sigma_yy, n, a, o
        )
    ).T
