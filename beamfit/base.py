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
        # Handle types for image_sigmas
        if not isinstance(image_sigmas, (type(None), np.ndarray)):
            raise ValueError(f"Invalid type for `image_sigmas`: {type(image_sigmas)}")

        # Handle different image types
        if isinstance(image, BeamImage):
            _img = image.get_avg_and_subtracted()
            if image.can_estimate_variance:
                _sigmas = image.get_std_error()
            else:
                _sigmas = None
            if image_sigmas is not None:
                raise ValueError("When image is a `BeamImage`, cannot use image_sigmas")
        elif isinstance(image, np.ndarray):
            _img = np.ma.array(image)
            _sigmas = image_sigmas
        elif isinstance(image, np.ma.MaskedArray):
            _img = image
            _sigmas = image_sigmas
        else:
            raise ValueError(f"Invalid type for `image`: {type(image)}")

        # Sanity checks before going on
        if (len(_img.shape) != 2) or not (_img.shape[0] > 8 and _img.shape[1] > 8):
            raise ValueError(f"Invalid shape for image array: {_img.shape}")
        if (_sigmas is not None) and (
            (len(_img.shape) != len(_sigmas.shape))
            or (_img.shape[0] != _sigmas.shape[0])
            or (_img.shape[1] != _sigmas.shape[1])
        ):
            raise ValueError(
                f"Image and sigmas array must match in shape (_img.shape={_img.shape}, _sigmas.shape={_sigmas.shape})"
            )

        # Apply all filters in order
        for filter in self.filters:
            _img = filter.apply(_img)

        return self.__fit__(_img, _sigmas)

    def __fit__(self, image, image_sigmas=None):
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
        cov = self.get_uncertainty_matrix()
        if cov is not None:
            cov = np.sqrt(np.diag(cov)[:2])
        return cov

    def get_covariance_matrix_std(self) -> np.ndarray:
        """
        Return an estimate of uncertainty in the covariance matrix.

        Returns
        -------
        np.ndarray
            The matrix [[std(sigma_xx), std(sigma_xy)], [std(sigma_yx), std(sigma_yy)]]
        """
        cov = self.get_uncertainty_matrix()
        if cov is not None:
            cov = np.sqrt(np.diag(cov)[2:])
            cov = np.array([[cov[0], cov[1]], [cov[1], cov[2]]])
        return cov

    def get_uncertainty_matrix(self) -> np.ndarray | None:
        """
        Get the estimated covariances between all fit values parameters (or None if not estimated). The result is returned
        as a 2D covariance matrix with the parameters ordered as [mu_x, mu_y, sig_xx, sig_xy, sig_yy].

        Returns
        -------
        np.ndarray | None
            The covariances of the best fit parameters (or None if no estimate)
        """
        raise NotImplementedError
