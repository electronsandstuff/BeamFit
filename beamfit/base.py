from __future__ import annotations
import numpy as np
import scipy.signal as signal
from dataclasses import dataclass
from typing import List, Dict, Union, Any
from abc import ABC


@dataclass
class Setting:
    name: str
    default: str
    stype: str = "str"  # can be 'str', 'list', 'settings_list'
    list_values: List[str] = None  # Options used with 'list' type
    list_settings: Dict[str, List[Setting]] = (
        None  # Map from names in list_values to lists of settings for that option
    )


class AnalysisMethod(ABC):
    def __init__(self, sigma_threshold=None, median_filter_size=None):
        self.sigma_threshold = sigma_threshold
        self.median_filter_size = median_filter_size

    def fit(self, image, image_sigmas=None):
        """
        Measure the RMS size and centroid of the supplied image. If the image is passed as a masked array then
        (depending on support by the fitting method) only unmasked pixels are considered. This can be useful for
        selecting oddly shaped regions of interest.

        :param image: 2D np.ndarray or np.ma.array, the image as a grayscale array or masked array of pixels
        :param image_sigmas: (optional) the uncertainty in each pixel intensity
        :return: AnalysisResult object (depends on analysis method)
        """
        if not np.ma.isMaskedArray(image):  # Make a mask if there isn't one
            image = np.ma.array(image)
        if self.median_filter_size is not None:  # Median filter the image if required
            image = np.ma.array(
                signal.medfilt2d(image, kernel_size=self.median_filter_size),
                mask=image.mask,
            )
        if self.sigma_threshold is not None:  # Apply threshold if provided
            image.mask = np.bitwise_or(
                image.mask, image < (image.max() * np.exp(-(self.sigma_threshold**2)))
            )
        return self.__fit__(image, image_sigmas)

    def __fit__(self, image, image_sigmas=None):
        raise NotImplementedError

    def get_config_dict(self):
        ret = {
            "sigma_threshold": self.sigma_threshold,
            "median_filter_size": self.median_filter_size,
        }
        ret.update(self.__get_config_dict__())
        return ret

    def __get_config_dict__(self):
        return {}

    def get_settings(self) -> List[Setting]:
        arr = [
            Setting("Sigma Threshold", "Off", stype="list", list_values=["Off", "On"]),
            Setting("Sigma Threshold Size", "3.0"),
            Setting("Median Filter", "Off", stype="list", list_values=["Off", "On"]),
            Setting("Median Filter Size", "3"),
        ]
        return arr + self.__get_settings__()

    def __get_settings__(self) -> List[Setting]:
        raise NotImplementedError()

    def set_from_settings(self, settings: Dict[str, str]):
        if settings["Sigma Threshold"] == "On":
            sigma_threshold = float(settings["Sigma Threshold Size"])
            if sigma_threshold <= 0.0:
                raise ValueError(
                    f"Sigma threshold must be greater than zero, got {sigma_threshold}"
                )
            self.sigma_threshold = sigma_threshold
        elif settings["Sigma Threshold"] == "Off":
            self.sigma_threshold = None
        else:
            raise ValueError(
                f'Unrecognized value for "Sigma Threshold": "{settings["Sigma Threshold"]}"'
            )
        if settings["Median Filter"] == "On":
            median_filter_size = int(settings["Median Filter Size"])
            if median_filter_size < 3:
                raise ValueError(
                    f"Median filter size must be at least 3, got {median_filter_size}"
                )
            if median_filter_size % 2 == 0:
                raise ValueError(
                    f"Median filter size must be odd integer, not {median_filter_size}"
                )
            self.median_filter_size = median_filter_size
        elif settings["Median Filter"] == "Off":
            self.median_filter_size = None
        else:
            raise ValueError(
                f'Unrecognized value for "Median Filter": "{settings["Median Filter"]}"'
            )
        self.__set_from_settings__(settings)

    def __set_from_settings__(self, settings: Dict[str, Union[str, Dict[str, Any]]]):
        raise NotImplementedError()


class AnalysisResult(ABC):
    def get_mean(self):
        raise NotImplementedError

    def get_covariance_matrix(self):
        raise NotImplementedError

    def get_mean_std(self):
        return None

    def get_covariance_matrix_std(self):
        return None
