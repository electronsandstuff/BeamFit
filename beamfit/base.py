from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Union, Any
from abc import ABC
from pydantic import BaseModel, model_validator

from .filters import FilterUnion, SigmaThresholdFilter, MedianFilter
from .image import BeamImage


@dataclass
class Setting:
    """
    Used to programatically pass information about what settings are available for user facing code to modify.
    """

    name: str
    default: str
    stype: str = "str"  # can be 'str', 'list', 'settings_list'
    list_values: list[str] = None  # Options used with 'list' type
    list_settings: dict[str, list[Setting]] = (
        None  # Map from names in list_values to lists of settings for that option
    )


class AnalysisMethod(BaseModel, ABC):
    """
    Parent class for all methods of getting first and second moments from a beam image.
    """

    filters: list[FilterUnion] = []

    @model_validator(mode="before")
    @classmethod
    def convert_legacy_fields(cls, data: Any) -> Any:
        """
        Convert legacy sigma_threshold and median_filter_size fields to the new filters list.
        Raises an error if both legacy fields and filters list are specified.
        """
        if isinstance(data, dict):
            has_legacy = "sigma_threshold" in data or "median_filter_size" in data
            has_filters = "filters" in data

            if has_legacy and has_filters:
                raise ValueError(
                    "Cannot specify both legacy fields (sigma_threshold, median_filter_size) "
                    "and new filters list. Please use only the filters list."
                )

            if has_legacy:
                filters = []

                # Convert sigma_threshold to SigmaThresholdFilter
                if data.get("sigma_threshold") is not None:
                    filters.append(SigmaThresholdFilter(sigma=data["sigma_threshold"]))

                # Convert median_filter_size to MedianFilter
                if data.get("median_filter_size") is not None:
                    filters.append(MedianFilter(kernel_size=data["median_filter_size"]))

                data["filters"] = filters
                # Remove legacy fields from data
                data.pop("sigma_threshold", None)
                data.pop("median_filter_size", None)

        return data

    def fit(
        self, image: Union[BeamImage, np.ndarray, np.ma.MaskedArray], image_sigmas=None
    ):
        """
        Measure the RMS size and centroid of the supplied image.

        If the image is passed as a masked array then (depending on support by the
        fitting method) only unmasked pixels are considered. This can be useful for
        selecting oddly shaped regions of interest.

        Parameters
        ----------
        image : BeamImage or 2D array
            The image as image object or a grayscale array of (possibly) masked of pixels.
        image_sigmas : array-like, optional
            The uncertainty in each pixel intensity. Not used if `BeamImage` object is passed.

        Returns
        -------
        AnalysisResult
            Analysis result object (depends on analysis method).
        """
        if isinstance(image, BeamImage):
            _img = image.processed
            _sigmas = image.pixel_std_devs
            if image_sigmas is not None:
                raise ValueError("When image is a `BeamImage`, cannot use image_sigmas")
        else:
            if not np.ma.isMaskedArray(image):  # Make a mask if there isn't one
                image = np.ma.array(image)
            _img = image
            _sigmas = image_sigmas

        # Apply all filters in order
        for filter in self.filters:
            _img = filter.apply(_img)

        return self.__fit__(image, _sigmas)

    def __fit__(self, _img, image_sigmas=None):
        """
        Implement the actual fitting method in child classes using this method.
        """
        raise NotImplementedError

    def get_config_dict(self):
        """
        Returns all information to configure the class as a dict.

        Returns
        -------
        dict
            Class config information
        """
        ret = {
            "filters": [filter.model_dump() for filter in self.filters],
        }
        ret.update(self.__get_config_dict__())
        return ret

    def __get_config_dict__(self):
        """
        Internal method implemented by children to add their own configuration.

        Returns
        -------
        dict
            Class config information
        """
        return {}

    def get_settings(self) -> list[Setting]:
        """
        Returns a list of settings for user code to programatically modify the method.

        Returns
        -------
        list[Setting]
            List of the user changeable settings
        """
        arr = [
            Setting("Sigma Threshold", "Off", stype="list", list_values=["Off", "On"]),
            Setting("Sigma Threshold Size", "3.0"),
            Setting("Median Filter", "Off", stype="list", list_values=["Off", "On"]),
            Setting("Median Filter Size", "3"),
        ]
        return arr + self.__get_settings__()

    def __get_settings__(self) -> list[Setting]:
        """
        Internal method used by children to return their additional settings.

        Returns
        -------
        list[Setting]
            The settings usable by children
        """
        raise NotImplementedError()

    def set_from_settings(self, settings: dict[str, str]):
        """
        Update the class based on settings returned from user code.

        Parameters
        ----------
        settings : dict[str, str]
            Mapping from setting name to setting value
        """
        filters = []

        if settings["Sigma Threshold"] == "On":
            sigma_threshold = float(settings["Sigma Threshold Size"])
            if sigma_threshold <= 0.0:
                raise ValueError(
                    f"Sigma threshold must be greater than zero, got {sigma_threshold}"
                )
            filters.append(SigmaThresholdFilter(sigma=sigma_threshold))
        elif settings["Sigma Threshold"] != "Off":
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
            filters.append(MedianFilter(kernel_size=median_filter_size))
        elif settings["Median Filter"] != "Off":
            raise ValueError(
                f'Unrecognized value for "Median Filter": "{settings["Median Filter"]}"'
            )

        self.filters = filters
        self.__set_from_settings__(settings)

    def __set_from_settings__(self, settings: dict[str, Union[str, dict[str, Any]]]):
        """
        Implemented by children to apply settings to their internal parameters.
        """
        raise NotImplementedError()


class AnalysisResult(BaseModel, ABC):
    """
    Parent class for the output of one of the analysis methods.
    """

    def get_mean(self) -> np.ndarray:
        """
        Return the centroid vector in pixel coordinates.

        Returns
        -------
        np.ndarray
            The centroid vector [mu_x, mu_y]
        """
        raise NotImplementedError

    def get_covariance_matrix(self) -> np.ndarray:
        """
        Return the covariance matrix in pixel coordinates.

        Returns
        -------
        np.ndarray
            The covariance matrix [[sigma_xx, sigma_xy], [sigma_yx, sigma_yy]] in units of pixels
        """
        raise NotImplementedError

    def get_mean_std(self) -> np.ndarray:
        """
        Get an estimate of uncertainty for each component of the centroid vector.

        Returns
        -------
        np.ndarray
            The vector [std(mu_x), std(mu_y)]
        """
        return None

    def get_covariance_matrix_std(self) -> np.ndarray:
        """
        Return an estimate of uncertainty in the covariance matrix.

        Returns
        -------
        np.ndarray
            The matrix [[std(sigma_xx), std(sigma_xy)], [std(sigma_yx), std(sigma_yy)]]
        """
        return None
