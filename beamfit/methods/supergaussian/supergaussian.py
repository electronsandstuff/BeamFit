import numpy as np
import scipy.optimize as opt
from typing import Union, Any

from ... import factory
from ...base import AnalysisMethod, Setting
from ...utils import SuperGaussianResult


def supergaussian_python(x, y, mu_x, mu_y, sigma_xx, sigma_xy, sigma_yy, n, a, o):
    """
    Pure Python implementation of the bivariate super-Gaussian function.

    Computes the function:
        f(r) = a*exp(-(1/2(r - mu)^T Sigma^{-1} (r - mu))^n) + o

    where r is the vector {x, y}, mu is the centroid vector {mu_x, mu_y}, and
    Sigma is the covariance matrix {{sigma_xx, sigma_xy}, {sigma_xy, sigma_yy}}.

    This is a direct translation of the C implementation in src/gaussian.c.

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
    float or np.ndarray
        The values of the super-Gaussian.
    """
    # Compute inverse of determinant: 1 / (sigma_xx * sigma_yy - sigma_xy^2)
    inv_det = 1.0 / (sigma_xx * sigma_yy - sigma_xy * sigma_xy)

    # Compute displacements from centroid
    dx = x - mu_x
    dy = y - mu_y

    # Compute the quadratic form: (r - mu)^T Sigma^{-1} (r - mu)
    # Using matrix inverse formula for 2x2 matrix:
    # Sigma^{-1} = (1/det) * [[sigma_yy, -sigma_xy], [-sigma_xy, sigma_xx]]
    quad_x = sigma_yy * inv_det * dx - sigma_xy * inv_det * dy
    quad_y = -sigma_xy * inv_det * dx + sigma_xx * inv_det * dy

    # Complete the quadratic form
    quad_form = dx * quad_x + dy * quad_y

    # Compute the super-Gaussian: a * exp(-(quad_form^n) / 2^n) + o
    # Equivalent to: a / exp((quad_form^n) / 2^n) + o
    result = a / np.exp(np.abs(quad_form) ** n / (2.0**n)) + o

    return result


def supergaussian_grad_python(x, y, mu_x, mu_y, sigma_xx, sigma_xy, sigma_yy, n, a, o):
    """
    Pure Python implementation of the super-Gaussian gradient (Jacobian).

    Calculate the Jacobian of the super-Gaussian with respect to the parameters.

    Computes the Jacobian matrix of the super-Gaussian function with respect to
    the parameters (mu_x, mu_y, sigma_xx, sigma_xy, sigma_yy, n, a, o).

    This is a direct translation of the C implementation in src/gaussian.c.

    Parameters
    ----------
    x : np.ndarray
        Array of x values to evaluate at.
    y : np.ndarray
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
        Columns are: [d/dmu_x, d/dmu_y, d/dsigma_xx, d/dsigma_xy, d/dsigma_yy, d/dn, d/da, d/do]
    """
    # Pre-compute common terms
    TMP_180 = mu_x - x  # Note: reversed from C code (args[2] - args[0])
    TMP_172 = sigma_xy * sigma_xy
    TMP_178 = sigma_xx * mu_y * mu_y
    TMP_185 = sigma_xx * y * y
    TMP_189 = mu_x * sigma_xy - sigma_xy * x + sigma_xx * y
    TMP_191 = (
        TMP_178
        + sigma_yy * TMP_180 * TMP_180
        + 2.0 * sigma_xy * y * TMP_180
        + TMP_185
        - 2.0 * mu_y * TMP_189
    )
    TMP_170 = np.power(2.0, 1.0 - n)
    TMP_171 = np.power(2.0, -n)
    TMP_175 = sigma_xx * sigma_yy - TMP_172
    TMP_192 = TMP_191 / TMP_175
    TMP_193 = np.power(np.abs(TMP_192), n)
    TMP_195 = np.exp(-TMP_171 * TMP_193)
    TMP_199 = np.power(np.abs(sigma_xx * sigma_yy - TMP_172), -n)
    TMP_205 = -1.0 + n
    TMP_206 = np.power(np.abs(TMP_191) + 1.0e-100, TMP_205)
    TMP_200 = mu_y * sigma_xy
    TMP_202 = sigma_yy * x
    TMP_204 = TMP_200 - mu_x * sigma_yy + TMP_202 - sigma_xy * y
    TMP_214 = 1.0 / TMP_175 / TMP_175
    TMP_216 = np.power(np.abs(TMP_192) + 1.0e-100, TMP_205)
    TMP_212 = mu_y * sigma_xx - mu_x * sigma_xy + sigma_xy * x - sigma_xx * y

    # Compute gradients (matching C code lines 97-109)
    grad_mu_x = a * n * TMP_170 * TMP_195 * TMP_199 * TMP_204 * TMP_206
    grad_mu_y = -(a * n * TMP_170 * TMP_195 * TMP_199 * TMP_206 * TMP_212)
    grad_sigma_xx = a * n * TMP_171 * TMP_195 * TMP_204 * TMP_204 * TMP_214 * TMP_216
    grad_sigma_xy = (
        a
        * n
        * TMP_170
        * (-(mu_y * sigma_xx) + TMP_189)
        * TMP_195
        * TMP_204
        * TMP_214
        * TMP_216
    )
    grad_sigma_yy = a * n * TMP_171 * TMP_195 * TMP_212 * TMP_212 * TMP_214 * TMP_216

    # Gradient w.r.t. n - handle small values to avoid log(0)
    grad_n = np.where(
        TMP_191 < 1e-50,
        0.0,
        a
        * TMP_171
        * TMP_193
        * TMP_195
        * (
            np.log(np.abs(2.0 * sigma_xx * sigma_yy - 2.0 * TMP_172))
            - np.log(np.abs(TMP_191))
        ),
    )

    grad_a = TMP_195
    grad_o = np.ones_like(x)

    # Stack into matrix form (m, 8) - transposed from C implementation
    result = np.array(
        [
            grad_mu_x,
            grad_mu_y,
            grad_sigma_xx,
            grad_sigma_xy,
            grad_sigma_yy,
            grad_n,
            grad_a,
            grad_o,
        ]
    ).T

    return result


class SuperGaussian(AnalysisMethod):
    """
    Fit a superguassian model to the image and extract beam centroids, moments.
      f(r) = a*exp(-(1/2(r - mu)^T Sigma^{-1} (r - mu))^n) + o
    """

    def __init__(
        self,
        predfun="GaussianProfile1D",
        predfun_args=None,
        sig_param="LogCholesky",
        sig_param_args=None,
        maxfev=100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if sig_param_args is None:
            sig_param_args = {}
        if predfun_args is None:
            predfun_args = {}
        self.predfun = factory.create("analysis", predfun, **predfun_args)
        self.predfun_args = predfun_args
        self.maxfev = maxfev
        self.sig_param = factory.create("sig_param", sig_param, **sig_param_args)
        self.sig_param_args = sig_param_args

    def __fit__(self, image, image_sigmas=None):
        lo, hi = image.min(), image.max()  # Normalize image
        image = (image - lo) / (hi - lo)

        # Get the x and y data for the fit
        m, n = np.mgrid[: image.shape[0], : image.shape[1]]
        x = np.vstack((m[~image.mask], n[~image.mask]))
        y = np.array(image[~image.mask])

        # Setup the fitting functions
        def h_to_theta(h):
            # Break out the variables
            mu = h[:2]
            sigma = np.array([[h[2], h[3]], [h[3], h[4]]])
            n = h[5]
            a = h[6]
            o = h[7]

            # Transform parameters
            st = self.sig_param.forward(sigma)  # The sigma parameterization
            nt = np.log(n)  # n is positive
            return np.array([mu[0], mu[1], st[0], st[1], st[2], nt, a, o])

        def theta_to_h(theta):
            # Break out the variables
            mu = theta[:2]
            st = theta[2:5]
            nt = theta[5]
            a = theta[6]
            o = theta[7]

            # Transform sigma and n back
            sigma = self.sig_param.reverse(st)
            n = np.exp(nt)
            return np.array(
                [mu[0], mu[1], sigma[0, 0], sigma[0, 1], sigma[1, 1], n, a, o]
            )

        def theta_to_h_grad(theta):
            # Break out the parameters
            st = theta[2:5]
            nt = theta[5]

            # Construct the jacobian
            j = np.identity(8)
            j[2:5, 2:5] = self.sig_param.reverse_grad(
                st
            )  # Add the sigma parameterization gradient
            j[5, 5] = np.exp(nt)
            return j

        def fitfun(xdata, *theta):
            return supergaussian_python(xdata[0], xdata[1], *theta_to_h(theta))

        def fitfun_grad(xdata, *theta):
            jacf = theta_to_h_grad(theta)
            jacg = supergaussian_grad_python(xdata[0], xdata[1], *theta_to_h(theta))
            return jacg @ jacf  # Chain rule

        if image_sigmas is None:
            theta_opt, theta_c = opt.curve_fit(
                fitfun,
                x,
                y,
                h_to_theta(self.predfun.fit(image).h),
                jac=fitfun_grad,
                maxfev=self.maxfev,
            )
        else:
            sigma = image_sigmas[~image.mask] / (hi - lo)
            theta_opt, theta_c = opt.curve_fit(
                fitfun,
                x,
                y,
                h_to_theta(self.predfun.fit(image).h),
                sigma=sigma,
                jac=fitfun_grad,
                absolute_sigma=True,
                maxfev=self.maxfev,
            )
        h_opt = theta_to_h(theta_opt)
        j = theta_to_h_grad(theta_opt)
        h_c = j @ theta_c @ j.T

        # Transform c according to normalization
        j_norm = np.identity(8)
        j_norm[6, 6] = hi - lo
        j_norm[7, 7] = hi - lo
        h_c = j_norm @ h_c @ j_norm.T

        # Return the fit and the covariance variance matrix
        return SuperGaussianResult(
            mu=np.array([h_opt[0], h_opt[1]]),
            sigma=np.array([[h_opt[2], h_opt[3]], [h_opt[3], h_opt[4]]]),
            n=h_opt[5],
            a=h_opt[6] * (hi - lo),
            o=h_opt[7] * (hi - lo) + lo,
            c=h_c,
        )

    def __get_config_dict__(self):
        return {
            "predfun": type(self.predfun).__name__,
            "predfun_args": self.predfun_args,
            "sig_param": type(self.sig_param).__name__,
            "sig_param_args": self.sig_param_args,
            "maxfev": self.maxfev,
        }

    def __get_settings__(self) -> list[Setting]:
        pred_funs = [x for x in factory.get_names("analysis") if x != "SuperGaussian"]
        pred_fun_settings = {
            x: factory.create("analysis", x).get_settings() for x in pred_funs
        }
        return [
            Setting(
                "Intial Prediction Method",
                "GaussianProfile1D",
                stype="settings_list",
                list_values=pred_funs,
                list_settings=pred_fun_settings,
            ),
            Setting(
                "Covariance Matrix Parameterization",
                "LogCholesky",
                stype="list",
                list_values=factory.get_names("sig_param"),
            ),
            Setting("Max Function Evaluation", "100"),
        ]

    def __set_from_settings__(self, settings: dict[str, Union[str, dict[str, Any]]]):
        self.predfun = factory.create(
            "analysis", settings["Intial Prediction Method"]["name"]
        )
        self.predfun.set_from_settings(settings["Intial Prediction Method"]["settings"])
        self.sig_param = factory.create(
            "sig_param", settings["Covariance Matrix Parameterization"]
        )
        maxfev = int(settings["Max Function Evaluation"])
        if maxfev < 1:
            raise ValueError(f"maxfev must be greater than zero, got {maxfev}")
        self.maxfev = maxfev


def fit_supergaussian(
    image,
    image_weights=None,
    prediction_func="2D_linear_Gaussian",
    sigma_threshold=3,
    sigma_threshold_guess=1,
    smoothing=5,
    maxfev=100,
):  # Backwards compatibility
    predfun = {
        "2D_linear_Gaussian": "GaussianLinearLeastSquares",
        "1D_Gaussian": "GaussianProfile1D",
    }[prediction_func]
    ret = SuperGaussian(
        predfun=predfun,
        predfun_args={
            "sigma_threshold": sigma_threshold_guess,
            "median_filter_size": smoothing,
        },
        sigma_threshold=sigma_threshold,
        maxfev=maxfev,
    ).fit(image, image_weights)
    return ret.h, ret.c
