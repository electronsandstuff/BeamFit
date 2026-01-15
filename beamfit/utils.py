from __future__ import annotations
import numpy as np
import scipy.special as special

from . import factory
from .base import AnalysisMethod, AnalysisResult


def get_image_and_weight(
    raw_images: list[np.ndarray], dark_fields: list[np.ndarray], mask: np.ndarray
):
    """
    Convenience function to get an averaged and background subtracked image combined with a weighting value for each pixel
    based on the amount of noise in it.

    Parameters
    ----------
    raw_images : list[np.ndarray]
        The images of the beam to process
    dark_fields : list[np.ndarray]
        Images without the beam for background subtraction
    mask : np.ndarray
        A boolean image where a true pixel excludes it from fitting

    Returns
    -------
    np.ndarray, np.ndarray
        The processed image and its weights
    """
    image = np.ma.masked_array(
        data=np.mean(raw_images, axis=0) - np.mean(dark_fields, axis=0), mask=mask
    )
    std_image = np.ma.masked_array(
        data=np.sqrt(
            np.std(raw_images, axis=0) ** 2 + np.std(dark_fields, axis=0) ** 2
        ),
        mask=mask,
    )
    image_weight = len(raw_images) / std_image**2
    return image, image_weight


def get_config_dict_analysis_method(m: AnalysisMethod):
    return {"type": type(m).__name__, "config": m.get_config_dict()}


def create_analysis_method_from_dict(d):
    return factory.create("analysis", d["type"], **d["config"])


def super_gaussian_scaling_factor(n: float) -> float:
    """
    Factor applied to the internal covariance-matrix-like parameters of the supergaussian function to convert it
    to the actual covariance matrix (ie Sigma = f(n) * Sigma_{sg}).

    Parameters
    ----------
    n : float
        Supergaussian parameter n

    Returns
    -------
    float
        The conversion factor
    """
    return special.gamma((2 + n) / n) / 2 / special.gamma(1 + 1 / n)


def super_gaussian_scaling_factor_grad(n):
    """
    The derivative of `super_gaussian_scaling_factor` for uncertainty propogation.

    Parameters
    ----------
    n : float
        Supergaussian parameter n

    Returns
    -------
    float
        Derivative of the conversion factor
    """
    n = n
    scaling_factor_deriv = (
        special.gamma((2 + n) / n) / 2 / n**2 * special.polygamma(0, 1 + 1 / n)
    )
    scaling_factor_deriv += (
        (1 / n - (2 + n) / n**2)
        * special.gamma((2 + n) / n)
        * special.polygamma(0, (2 + n) / n)
        / 2
    )
    scaling_factor_deriv /= special.gamma(1 + 1 / n)
    return scaling_factor_deriv


class SuperGaussianResult(AnalysisResult):
    """
    Represents the results of a fitting process where the model takes the form of a supergaussian (or gaussian where
    `n` is set to 1)
    """

    def __init__(
        self, mu=np.zeros(2), sigma=np.identity(2), a=1.0, o=0.0, n=1.0, c=None, h=None
    ):
        if h is None:
            self.mu = mu  # Centroid
            self.sigma = sigma  # Variance-covariance matrix
            self.a = a  # Amplitude
            self.o = o  # Background offset
            self.n = n  # Supergaussian parameter
        else:
            self.h = h
        self.c = c  # covariance matrix of h

    @property
    def h(self):
        return np.array(
            [
                self.mu[0],
                self.mu[1],
                self.sigma[0, 0],
                self.sigma[0, 1],
                self.sigma[1, 1],
                self.n,
                self.a,
                self.o,
            ]
        )

    @h.setter
    def h(self, h):
        self.mu = np.array([h[0], h[1]])
        self.sigma = np.array([[h[2], h[3]], [h[3], h[4]]])
        self.n = h[5]
        self.a = h[6]
        self.o = h[7]

    def get_mean(self):
        return self.mu

    def get_covariance_matrix(self):
        return self.sigma * super_gaussian_scaling_factor(self.n)

    def get_mean_std(self):
        return np.sqrt(np.array([self.c[0, 0], self.c[1, 1]]))

    def get_covariance_matrix_std(self):
        # Find the Jacobian of the scaling transformation
        scaling_j = np.identity(4)
        scaling_j[:3, :3] *= super_gaussian_scaling_factor(self.n)
        scaling_j[:3, 3] = self.h[2:5] * super_gaussian_scaling_factor_grad(self.n)

        # Get the covariance matrix of our variables
        sigma_n_cov = self.c[2:6, 2:6]

        # Transform it
        sigma_n_cov_scaled = scaling_j @ sigma_n_cov @ scaling_j.T
        return np.sqrt(
            np.array(
                [
                    [sigma_n_cov_scaled[0, 0], sigma_n_cov_scaled[1, 1]],
                    [sigma_n_cov_scaled[1, 1], sigma_n_cov_scaled[2, 2]],
                ]
            )
        )
