import numpy as np
from typing import Union, Any

from ..base import AnalysisMethod, Setting
from ..utils import SuperGaussianResult


class RMSIntegration(AnalysisMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __fit__(self, image, image_sigmas=None):
        lo, hi = image.min(), image.max()  # Normalize image
        image = (image - lo) / (hi - lo)

        m, n = np.mgrid[: image.shape[0], : image.shape[1]]  # Compute image moments
        mmnts = np.array(
            [
                [
                    float("nan") if i + j > 2 else np.sum(m**i * n**j * image)
                    for j in range(3)
                ]
                for i in range(3)
            ]
        )

        mu = np.array([mmnts[1, 0], mmnts[0, 1]]) / mmnts[0, 0]
        nu = (
            np.array([[mmnts[2, 0], mmnts[1, 1]], [mmnts[1, 1], mmnts[0, 2]]])
            / mmnts[0, 0]
        )
        return SuperGaussianResult(
            mu=mu, sigma=nu - mu[:, None] * mu[None, :], a=(hi - lo), o=lo
        )

    def __get_settings__(self) -> list[Setting]:
        return []

    def __set_from_settings__(self, settings: dict[str, Union[str, dict[str, Any]]]):
        pass
