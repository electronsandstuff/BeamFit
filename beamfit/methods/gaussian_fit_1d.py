import numpy as np
from scipy import special, optimize as opt
from typing import Union, Any

from ..base import AnalysisMethod, Setting
from ..utils import SuperGaussianResult


class GaussianProfile1D(AnalysisMethod):
    """
    Calculate the centroid and beam moments from the marginal distributions on the X and Y axis which are fit to Gaussians.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __fit__(self, image, image_sigmas=None):
        """
        Integrates the image across each axis and fits a Gaussian function with offset to each axis.  Predicts the 2D
        Gaussian of best fit from the resulting data.

        Note:  This function only works for Gaussians with positive amplitude at the moment.  Invert any images where the
        Gaussian points downward.
        """
        if image_sigmas is None:
            image_sigmas = np.ones_like(image.data)

        def fitfun(x, mu, sigma, a, c):
            return a * np.exp(-((x - mu) ** 2) / 2 / sigma**2) + c

        # Fit the projection on each axis
        fits = []
        fit_uncertainties = []
        for axis in range(2):
            # Integrate the profile onto one axis and make the x values
            y = np.sum(image, axis=axis)
            y_sigma = np.sqrt(np.sum(image_sigmas**2, axis=axis))

            # Estimate the fit parameters
            lo, hi = y.min(), y.max()
            y_norm = (y - lo) / (hi - lo)
            y_sigma_norm = y_sigma / (hi - lo)
            p0 = np.array([np.argmax(y_norm), np.sum(y_norm > np.exp(-1 / 8)), 1, 0.0])

            # Run the fit
            hopt, c = opt.curve_fit(
                fitfun, np.arange(y_norm.size), y_norm, p0, sigma=y_sigma_norm
            )

            # Add in the optimal values
            hopt[2] *= hi - lo
            hopt[3] = hopt[3] * (hi - lo) + lo
            fits.append(hopt)

            # Transform and add the uncertainties
            j_norm = np.identity(4)
            j_norm[2, 2] = hi - lo
            j_norm[3, 3] = hi - lo
            c = j_norm @ c @ j_norm.T
            fit_uncertainties.append(c)

        # Merge fits and uncertainties into single array
        all_params = np.concatenate((fits[1], fits[0]))
        all_uncertainties = np.identity(8)
        all_uncertainties[:4, :4] = fit_uncertainties[1]
        all_uncertainties[4:, 4:] = fit_uncertainties[0]

        # Calculate scaling values for x/y for amplitude. This is required since gaussian can clip on the edge of the
        # image meaning the integral involves the error function
        sy, sx = [
            2
            / (
                special.erf((s - f[0]) / f[1] / np.sqrt(2))
                - special.erf(-f[0] / f[1] / np.sqrt(2))
            )
            / (f[1] * np.sqrt(2 * np.pi))
            for s, f in zip(image.shape, reversed(fits))
        ]

        # Convert the parameters to h for the 2D Super-Gaussian
        h = np.array(
            [
                all_params[0],
                all_params[4],
                all_params[1] ** 2,
                0.0,
                all_params[5] ** 2,
                1.0,
                (all_params[2] * sx + all_params[6] * sy) / 2,
                (all_params[3] / image.shape[1] + all_params[7] / image.shape[0]) / 2,
            ]
        )

        # Get the Jacobian of the transformation
        j = np.zeros((8, 8))
        j[0, 0] = 1
        j[1, 4] = 1
        j[2, 1] = 2 * all_params[1]
        j[4, 5] = 2 * all_params[5]
        j[6, 2] = sx / 2
        j[6, 6] = sy / 2
        j[7, 3] = 1 / image.shape[1] / 2
        j[7, 7] = 1 / image.shape[0] / 2

        # Transform the uncertainties
        c_all = j @ all_uncertainties @ j.T

        # Return it
        ret = SuperGaussianResult(h=h, c=c_all)
        return ret

    def __get_settings__(self) -> list[Setting]:
        return []

    def __set_from_settings__(self, settings: dict[str, Union[str, dict[str, Any]]]):
        pass


def fit_gaussian_1d(image):  # Backwards compatibility
    return GaussianProfile1D().fit(image).h
