import numpy as np
import scipy.optimize as opt
from typing import Union, Any

from .. import factory
from ..base import AnalysisMethod, Setting
from ..utils import SuperGaussianResult
from ..supergaussian_c_drivers import supergaussian, supergaussian_grad


class SigmaTrans:
    def __init__(self):
        pass

    def forward(self, h):
        raise NotImplementedError

    def reverse(self, h):
        raise NotImplementedError

    def forward_grad(self, h, grad):
        raise NotImplementedError


class SuperGaussian(AnalysisMethod):
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
            return supergaussian(xdata[0], xdata[1], *theta_to_h(theta))

        def fitfun_grad(xdata, *theta):
            jacf = theta_to_h_grad(theta)
            jacg = supergaussian_grad(xdata[0], xdata[1], *theta_to_h(theta))
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
