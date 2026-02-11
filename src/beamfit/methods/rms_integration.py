import numpy as np
from typing import Union, Any, Literal

from ..base import AnalysisMethod, Setting
from ..utils import SuperGaussianResult


class RMSIntegration(AnalysisMethod):
    """
    Estimate beam centroids and beam moments by numerically integrating the distribution. Ie
    $\sigma_{x,y} = \sqrt{E[{x,y}^2] - E[{x,y}]^2}$. Where the expectation value is calculated as
    $E[g(x,y)] = \int\int f(x,y)*g(x,y)*dx*dy$ for the normalized distribution $f(x,y)$.
    """

    type: Literal["RMSIntegration"] = "RMSIntegration"

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
