import numpy as np
from typing import Union, Any

from .base import AnalysisMethod, Setting
from .utils import SuperGaussianResult


def x_to_h(x):
    return np.array(
        [
            (-1 * x[3] * x[4] + 2 * x[1] * x[5]) / (x[4] ** 2 - 4 * x[2] * x[5]),
            (-2 * x[2] * x[3] + x[1] * x[4]) / (-1 * x[4] ** 2 + 4 * x[2] * x[5]),
            -x[5] / (x[2] * x[5] - x[4] ** 2 / 4) / 2,
            x[4] / (x[2] * x[5] - x[4] ** 2 / 4) / 4,
            -x[2] / (x[2] * x[5] - x[4] ** 2 / 4) / 2,
            1.0,
            x[6] - x[7],
            x[7],
        ]
    )


def x_to_h_grad(x):
    j = np.zeros((8, 8))
    j[0, 1] = (2 * x[5]) / (x[4] ** 2 - 4 * x[2] * x[5])
    j[0, 2] = (
        -(-1 * x[3] * x[4] + 2 * x[1] * x[5])
        / (x[4] ** 2 - 4 * x[2] * x[5]) ** 2
        * (-4 * x[5])
    )
    j[0, 3] = (-1 * x[4]) / (x[4] ** 2 - 4 * x[2] * x[5])
    j[0, 4] = (-1 * x[3]) / (x[4] ** 2 - 4 * x[2] * x[5]) - (
        -1 * x[3] * x[4] + 2 * x[1] * x[5]
    ) / (x[4] ** 2 - 4 * x[2] * x[5]) ** 2 * x[4] * 2
    j[0, 5] = (2 * x[1]) / (x[4] ** 2 - 4 * x[2] * x[5]) - (
        -1 * x[3] * x[4] + 2 * x[1] * x[5]
    ) / (x[4] ** 2 - 4 * x[2] * x[5]) ** 2 * (-4 * x[2])
    j[1, 1] = (x[4]) / (-1 * x[4] ** 2 + 4 * x[2] * x[5])
    j[1, 2] = (-2 * x[3]) / (-1 * x[4] ** 2 + 4 * x[2] * x[5]) - (
        -2 * x[2] * x[3] + x[1] * x[4]
    ) / (-1 * x[4] ** 2 + 4 * x[2] * x[5]) ** 2 * (4 * x[5])
    j[1, 3] = (-2 * x[2]) / (-1 * x[4] ** 2 + 4 * x[2] * x[5])
    j[1, 4] = (x[1]) / (-1 * x[4] ** 2 + 4 * x[2] * x[5]) - (
        -2 * x[2] * x[3] + x[1] * x[4]
    ) / (-1 * x[4] ** 2 + 4 * x[2] * x[5]) ** 2 * (-2 * x[4])
    j[1, 5] = (
        -(-2 * x[2] * x[3] + x[1] * x[4])
        / (-1 * x[4] ** 2 + 4 * x[2] * x[5]) ** 2
        * (4 * x[2])
    )
    j[2, 2] = 2 * x[5] / (x[2] * x[5] - x[4] ** 2 / 4) ** 2 * x[5] / 4
    j[2, 4] = 2 * x[5] / (x[2] * x[5] - x[4] ** 2 / 4) ** 2 * (-2 * x[4]) / 4 / 4
    j[2, 5] = (
        -2 / (x[2] * x[5] - x[4] ** 2 / 4) / 4
        + 2 * x[5] / (x[2] * x[5] - x[4] ** 2 / 4) ** 2 * x[2] / 4
    )
    j[3, 2] = -x[4] / (x[2] * x[5] - x[4] ** 2 / 4) ** 2 * x[5] / 4
    j[3, 4] = (
        1 / (x[2] * x[5] - x[4] ** 2 / 4) / 4
        - x[4] / (x[2] * x[5] - x[4] ** 2 / 4) ** 2 * (-2) * x[4] / 4 / 4
    )
    j[3, 5] = -x[4] / (x[2] * x[5] - x[4] ** 2 / 4) ** 2 * x[2] / 4
    j[4, 2] = (
        -2 / (x[2] * x[5] - x[4] ** 2 / 4) / 4
        + 2 * x[2] / (x[2] * x[5] - x[4] ** 2 / 4) ** 2 * x[5] / 4
    )
    j[4, 4] = 2 * x[2] / (x[2] * x[5] - x[4] ** 2 / 4) ** 2 * (-2) * x[4] / 4 / 4
    j[4, 5] = 2 * x[2] / (x[2] * x[5] - x[4] ** 2 / 4) ** 2 * x[2] / 4
    j[6, 6] = 1
    j[6, 7] = -1
    j[7, 7] = 1
    return j


class GaussianLinearLeastSquares(AnalysisMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __fit__(self, image, image_sigmas=None):
        if image_sigmas is None:
            image_sigmas = np.ones_like(image)

        # Normalize image z axis
        lo, hi = image.min(), image.max()
        image_norm = ((image - lo) / (hi - lo) + np.exp(-10)) / (1 + np.exp(-10))

        # Create coefficient matrices for fit
        m, n = np.mgrid[: image_norm.shape[0], : image_norm.shape[1]]
        mm = m[~image_norm.mask]
        nn = n[~image_norm.mask]
        x = np.array([np.ones_like(mm), mm, mm**2, nn, nn * mm, nn**2]).T
        expy = image_norm[~image_norm.mask].data
        y = np.log(expy)

        # Weight the values and pass the uncertainties through the nonlinear transformation of y
        w = np.abs(
            (hi - lo) * expy / np.array(image_sigmas[~image_norm.mask])
        )  # 1/sigma
        wy = y * w
        wx = x * w[:, None]

        # Solve the linear least squares problem with QR factorization
        q, r = np.linalg.qr((wx.T * wy).T)
        x = np.linalg.solve(r, q.T @ wy**2)
        x = np.concatenate((x, np.array([hi, lo])))
        c = np.zeros((8, 8))
        c[:6, :6] = np.linalg.inv(wx.T @ wx)

        # Transform the best fit parameters into h for super-Gaussian
        h = x_to_h(x)
        j = x_to_h_grad(x)
        c = j @ c @ j.T

        # Return the fit
        return SuperGaussianResult(h=h, c=c)

    def __get_settings__(self) -> list[Setting]:
        return []

    def __set_from_settings__(self, settings: dict[str, Union[str, dict[str, Any]]]):
        pass


def fit_gaussian_linear_least_squares(
    image, sigma_threshold=2, plot=False
):  # Backwards compatibility
    return GaussianLinearLeastSquares(sigma_threshold=sigma_threshold).fit(image).h
