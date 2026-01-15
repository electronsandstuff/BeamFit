from gaussufunc import supergaussian_internal, supergaussian_grad_internal
import numpy as np


def supergaussian(x, y, mu_x, mu_y, sigma_xx, sigma_xy, sigma_yy, n, a, o):
    """
    Compiled (fast) numpy compatible ufunc which computes the bivariate super-Gaussian function
      f(r) = a*exp(-(1/2(r - mu)^T Sigma^{-1} (r - mu))^n) + o
    where r is the vector {x, y}, mu is the centroid vector {mu_x, mu_y}, and Sigma is the covariance matrix
    {{sigma_xx, sigma_xy}, {sigma_xy, sigma_yy}}.

    As a ufunc, sending a numpy array into the values will broadcast them against each other as is possible.

    :param x: x value at which the super-Gaussian is evaluated at
    :param y: y value at which the super-Gaussian is evaluated at
    :param mu_x: x component of centroid
    :param mu_y: y component of centroid
    :param sigma_xx: x variance
    :param sigma_xy: xy correlation
    :param sigma_yy: y variance
    :param n: super-Gaussian parameter
    :param a: amplitude
    :param o: offset
    :return: np.ndarray, the values of the supergaussian
    """
    return supergaussian_internal(
        x, y, mu_x, mu_y, sigma_xx, sigma_xy, sigma_yy, n, a, o
    )


def supergaussian_grad(x, y, mu_x, mu_y, sigma_xx, sigma_xy, sigma_yy, n, a, o):
    """
    Calculates the Jacobian of the supergaussian WRT to the values (mu_x, mu_y, sigma_xx, sigma_xy, sigma_yy, n, a, o)

    :param x: (m,) np.ndarray, array of x values to evaluate at
    :param y: (m,) np.ndarray, array of y values to evaluate at
    :param mu_x: x component of centroid
    :param mu_y: y component of centroid
    :param sigma_xx: x variance
    :param sigma_xy: xy correlation
    :param sigma_yy: y variance
    :param n: super-Gaussian parameter
    :param a: amplitude
    :param o: offset
    :return: (m, 8) np.ndarray, the Jacobian of the supergaussian WRT to all parameters
    """
    return np.array(
        supergaussian_grad_internal(
            x, y, mu_x, mu_y, sigma_xx, sigma_xy, sigma_yy, n, a, o
        )
    ).T
