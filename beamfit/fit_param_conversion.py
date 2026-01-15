import numpy as np
import warnings

from .utils import SuperGaussianResult


def get_mu_sigma(h, pixel_size):
    """
    Legacy. Get the centroid (mu) and second moments (sigma) from the supergaussian
    best fit parameters (h).

    Parameters
    ----------
    h : np.ndarray
        Supergaussian parameters of best fit
    pixel_size : float
        Conversion from pixel size to the size of the image

    Returns
    -------
    np.ndarray, np.ndarry
        The centroids and second moments
    """
    # Issue deprecation warning
    warnings.warn(
        "get_mu_sigma is deprecated, use `SuperGaussianResult` instead",
        DeprecationWarning,
    )

    r = SuperGaussianResult(h=h)
    return r.get_mean() * pixel_size, r.get_covariance_matrix() * pixel_size**2


def get_mu_sigma_std(h, c, pixel_size, pixel_size_std):
    """
    Get the fit covariances of the first and second moments from model fit output.

    Parameters
    ----------
    h : np.ndarray
        Parameters of best fit for supergaussian model
    c : np.ndarray
        Covariance matrix of the parameters of best fit
    pixel_size : float
        Conversion from pixel size to the size of the image
    pixel_size_std : float
        Standard deviation in the pixel size

    Returns
    -------
    np.ndarray, np.ndarray
        Estimated standard deviations of the first and second moments.
    """
    # Issue deprecation warning
    warnings.warn(
        "get_mu_sigma_std is deprecated, use `SuperGaussianResult` instead",
        DeprecationWarning,
    )

    r = SuperGaussianResult(h=h, c=c)
    mu = r.get_mean()
    sigma = r.get_covariance_matrix()
    mu_var = r.get_mean_std() ** 2
    sigma_var = r.get_covariance_matrix_std() ** 2

    # Scale by the pixel size and calculate variances
    pixel_size_var = pixel_size_std**2
    mu_scaled_var = (
        mu_var * pixel_size_var + pixel_size_var * mu**2 + mu_var * pixel_size**2
    )
    pixel_size_squared_var = 4 * pixel_size**2 * pixel_size_var
    sigma_scaled_var = (
        sigma_var * pixel_size_squared_var
        + pixel_size_squared_var * sigma**2
        + sigma_var * pixel_size**4
    )

    # Return them
    return np.sqrt(mu_scaled_var), np.sqrt(sigma_scaled_var)
