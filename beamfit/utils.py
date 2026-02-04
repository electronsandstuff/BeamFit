from __future__ import annotations
import numpy as np
import scipy.special as special
from typing import Any
from pydantic import field_validator, model_validator, Field

from . import factory
from .base import AnalysisMethod, AnalysisResult
from .pydantic import NumpyArray


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
        The get_processed() image and its weights
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

    # Parameters of best fit
    mu: NumpyArray = Field(description="Centroid of distribution (x, y)")
    sigma: NumpyArray = Field(
        description="Covariance matrix of underlying Gaussian ((xx, xy), (yx, yy))"
    )
    a: float = Field(1.0, description="Amplitude")
    o: float = Field(0.0, description="Offset")
    n: float = Field(1.0, description="Supergaussian parameter")

    # Parameter covariances
    c: NumpyArray | None = Field(
        None, description="Covariance between all parameters of best fit `h`"
    )

    @model_validator(mode="before")
    @classmethod
    def handle_h_init(cls, data: Any) -> Any:
        """Handle legacy initialization with h parameter"""
        if not isinstance(data, dict):
            return data

        if "h" in data:
            h = np.asarray(data.pop("h"))
            # h format: [mu_x, mu_y, sigma_xx, sigma_xy, sigma_yy, n, a, o]
            data["mu"] = np.array([h[0], h[1]])
            data["sigma"] = np.array([[h[2], h[3]], [h[3], h[4]]])
            data["n"] = h[5]
            data["a"] = h[6]
            data["o"] = h[7]

        return data

    @field_validator("mu")
    @classmethod
    def validate_mu_shape(cls, v):
        """Validate that mu is a 2D vector"""
        if v.shape != (2,):
            raise ValueError(
                f"mu must be a 2D vector with shape (2,), got shape {v.shape}"
            )
        return v

    @field_validator("sigma")
    @classmethod
    def validate_sigma_shape(cls, v):
        """Validate that sigma is a 2x2 matrix"""
        if v.shape != (2, 2):
            raise ValueError(
                f"sigma must be a 2x2 matrix with shape (2, 2), got shape {v.shape}"
            )
        return v

    @field_validator("c")
    @classmethod
    def validate_c_shape(cls, v):
        """Validate that c is an 8x8 covariance matrix"""
        if v.shape != (8, 8):
            raise ValueError(
                f"c must be an 8x8 matrix with shape (8, 8), got shape {v.shape}"
            )
        return v

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
