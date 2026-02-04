import numpy as np
from typing import Union


class BeamImage:
    def __init__(
        self,
        *,
        data_images: list[Union[np.ndarray, np.ma.MaskedArray]],
        darkfield_images: list[Union[np.ndarray, np.ma.MaskedArray]] | None = None,
        mask: np.ndarray | None = None,
    ):
        """
        All images acquired for a single beam. This includes multiple images of the beam for averaging purposes, background
        subtraction images and a mask to indicate which regions of the image are meant to be included in the analysis.

        Parameters
        ----------
        data_images : list[Union[np.ndarray, np.ma.MaskedArray]]
            Images including the beam data, to be averaged
        darkfield_images : list[Union[np.ndarray, np.ma.MaskedArray]] | None, optional
            Images for background subtraction (ie turn off beam source and retake images), to be averaged, by default None
        mask : np.ndarray | None, optional
            Boolean image, true (greater pixel value than zero) means pixel is "masked" (not used in analysis), by default None
        """
        if darkfield_images is None:
            darkfield_images = []
        if len(data_images) < 1:
            raise ValueError("Must supply at least one data image")

        # Type checks
        for idx, img in enumerate(data_images):
            if not isinstance(img, (np.ndarray, np.ma.MaskedArray)):
                raise ValueError(
                    f"Data images must np.array or np.ma.MaskedArray; type(data_images[{idx}])={type(img)}"
                )
        for idx, img in enumerate(darkfield_images):
            if not isinstance(img, (np.ndarray, np.ma.MaskedArray)):
                raise ValueError(
                    f"Darkfield images must np.array or np.ma.MaskedArray; type(darkfield_images[{idx}])={type(img)}"
                )
        if not isinstance(mask, (np.ndarray, type(None))):
            raise ValueError(f"Mask must be None or np.array; type(mask)={type(mask)}")

        # Check all image shapes
        img_shapes_list = [img.shape for img in data_images] + [
            img.shape for img in darkfield_images
        ]
        if mask is not None:
            img_shapes_list.append(mask.shape)
        img_shapes = set(img_shapes_list)
        if len(img_shapes) > 1:
            raise ValueError(
                f"All image arrays must have the same shape, detected image shapes: {img_shapes}"
            )
        for shape in img_shapes:
            if len(shape) != 2:
                raise ValueError(f"Image arrays must be 2D; image shape: {shape}")
            if (shape[0] < 8) or (shape[1] < 8):
                raise ValueError(
                    f"Images are too small for useful calculations! Minimum set to (8, 8). Image shape: {shape}"
                )

        # Copy (to avoid mutating user data) and cast to float64 np.ndarray (dropping np.ma.array mask if provided)
        self._data_images = [np.array(img, dtype=np.float64) for img in data_images]
        self._darkfield_images = [
            np.array(img, dtype=np.float64) for img in darkfield_images
        ]

        # Add to class parameters
        if mask is not None:
            self._mask = np.array(mask, dtype=bool)
        else:
            self._mask = np.ma.nomask

    @property
    def processed(self) -> np.ma.MaskedArray:
        """
        Return the processed image with averaging, background subtraction, and masking applied.

        Returns
        -------
        np.ma.MaskedArray
            Processed beam image as a masked array
        """
        # Average data images
        result = np.mean(self._data_images, axis=0)

        # Subtract averaged darkfield if available
        if len(self._darkfield_images) > 0:
            darkfield_avg = np.mean(self._darkfield_images, axis=0)
            result = result - darkfield_avg

        # Create masked array
        return np.ma.masked_array(result, mask=self._mask)

    @property
    def pixel_std_error(self) -> np.ma.MaskedArray:
        """
        Std. deviation of the estimated value of each pixel after averaging and background subtraction.
        """
        if not self.can_estimate_variance:
            raise ValueError("There is not enough data to estimate pixel variances")

        # Calculate variance
        var = np.var(self._data_images, axis=0) / len(self._data_images)
        if self._darkfield_images:
            var = var + np.var(self._darkfield_images, axis=0) / len(
                self._darkfield_images
            )

        # Pack into masked array
        return np.ma.masked_array(data=np.sqrt(var), mask=self._mask)

    @property
    def can_estimate_variance(self) -> bool:
        """
        Is there enough data for a valid prediction of the pixel sigmas?
        """
        return (len(self._data_images) > 1) or (len(self._darkfield_images) > 1)
