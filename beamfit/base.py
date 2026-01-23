from __future__ import annotations
import numpy as np
import scipy.signal as signal
from dataclasses import dataclass
from typing import Union, Any
from abc import ABC
from pydantic import BaseModel


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

    sigma_threshold: float | None = None
    median_filter_size: int | None = None

    def fit(self, image, image_sigmas=None):
        """
        Measure the RMS size and centroid of the supplied image.

        If the image is passed as a masked array then (depending on support by the
        fitting method) only unmasked pixels are considered. This can be useful for
        selecting oddly shaped regions of interest.

        Parameters
        ----------
        image : np.ndarray or np.ma.array, 2D
            The image as a grayscale array or masked array of pixels.
        image_sigmas : array-like, optional
            The uncertainty in each pixel intensity.

        Returns
        -------
        AnalysisResult
            Analysis result object (depends on analysis method).
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
            "sigma_threshold": self.sigma_threshold,
            "median_filter_size": self.median_filter_size,
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
