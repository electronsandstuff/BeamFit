from __future__ import annotations
import numpy as np
import scipy.signal as signal
from typing import Literal, Union
from typing_extensions import Annotated
from abc import ABC
from pydantic import BaseModel, Field, model_validator, Discriminator

from .exceptions import FilterError


class ImageFilter(BaseModel, ABC):
    """
    Base class for image preprocessing filters.
    """

    type: str = Field(..., description="Filter type discriminator")

    def apply(self, image: np.ma.MaskedArray) -> np.ma.MaskedArray:
        """
        Apply the filter to the image.

        Parameters
        ----------
        image : np.ma.MaskedArray
            The masked array image to filter.

        Returns
        -------
        np.ma.MaskedArray
            The filtered image.
        """
        raise NotImplementedError


class SigmaThresholdFilter(ImageFilter):
    """
    Filter that masks pixels below a threshold based on sigma value.
    Masks pixels where intensity < max_intensity * exp(-sigma^2).
    """

    type: Literal["SigmaThresholdFilter"] = "SigmaThresholdFilter"
    sigma: float = Field(..., description="Sigma threshold value", gt=0)

    def apply(self, image: np.ma.MaskedArray) -> np.ma.MaskedArray:
        """Apply sigma threshold masking to the image."""
        threshold = image.max() * np.exp(-(self.sigma**2))
        image.mask = np.bitwise_or(image.mask, image < threshold)
        return image


class MedianFilter(ImageFilter):
    """
    Filter that applies median filtering to the image.
    """

    type: Literal["MedianFilter"] = "MedianFilter"
    kernel_size: int = Field(..., description="Median filter kernel size", ge=3)

    @model_validator(mode="after")
    def validate_kernel_size(self):
        """Ensure kernel size is odd."""
        if self.kernel_size % 2 == 0:
            raise FilterError(
                f"Median filter kernel size must be odd, got {self.kernel_size}"
            )
        return self

    def apply(self, image: np.ma.MaskedArray) -> np.ma.MaskedArray:
        """Apply median filtering to the image."""
        filtered = signal.medfilt2d(image.data, kernel_size=self.kernel_size)
        return np.ma.array(filtered, mask=image.mask)


# Discriminated union of all filter types
FilterUnion = Annotated[
    Union[SigmaThresholdFilter, MedianFilter], Discriminator("type")
]
