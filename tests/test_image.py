import numpy as np
import pytest
from beamfit.image import BeamImage
from beamfit.exceptions import ImageError, UncertaintyEstimateError


def test_no_data_images():
    """Test that error is raised when no data images are provided."""
    with pytest.raises(ImageError):
        BeamImage(signal_images=[])


@pytest.mark.parametrize(
    "signal_images,background_images,mask",
    [
        # Wrong type in signal_images
        ([[1, 2, 3]], None, None),
        # Wrong type in background_images
        ([np.ones((10, 10))], [[1, 2, 3]], None),
        # Wrong type in mask
        ([np.ones((10, 10))], None, [[True, False]]),
    ],
)
def test_wrong_types(signal_images, background_images, mask):
    """Test that error is raised for wrong types."""
    with pytest.raises(ImageError):
        BeamImage(
            signal_images=signal_images,
            background_images=background_images,
            mask=mask,
        )


@pytest.mark.parametrize(
    "signal_images,background_images,mask",
    [
        ([np.ones((10, 10)), np.ones((10, 12))], None, None),
        ([np.ones((10, 10))], [np.ones((10, 12))], None),
        (
            [np.ones((10, 10)), np.ones((10, 12))],
            [np.ones((10, 10)), np.ones((10, 10))],
            None,
        ),
        (
            [np.ones((10, 10)), np.ones((10, 10))],
            [np.ones((10, 10)), np.ones((10, 12))],
            None,
        ),
        (
            [np.ones((10, 10)), np.ones((10, 10))],
            [np.ones((10, 10)), np.ones((10, 10))],
            np.ones((10, 12)),
        ),
        ([np.ones((10, 10))], None, np.zeros((10, 12))),
    ],
)
def test_inconsistent_shapes(signal_images, background_images, mask):
    """Test that error is raised for inconsistent shapes."""
    with pytest.raises(ImageError):
        BeamImage(
            signal_images=signal_images,
            background_images=background_images,
            mask=mask,
        )


def test_non_2d_array():
    """Test that error is raised for non-2D arrays."""
    with pytest.raises(ImageError):
        BeamImage(signal_images=[np.ones((10, 10, 3))])


def test_too_small_images():
    """Test that error is raised for images smaller than 8x8."""
    with pytest.raises(ImageError):
        BeamImage(signal_images=[np.ones((5, 5))])


def test_data_cast_to_float64():
    """Test that data is cast to float64."""
    # Test with int array (must be at least 8x8)
    data_int = np.ones((10, 10), dtype=np.int32)
    darkfield_int = np.zeros((10, 10), dtype=np.int32)

    beam = BeamImage(signal_images=[data_int], background_images=[darkfield_int])

    assert beam._signal_images[0].dtype == np.float64
    assert beam._background_images[0].dtype == np.float64


def test_background_subtraction():
    """Test that background subtraction works correctly."""
    # Create data images with values 4, 5, 6
    data1 = np.full((10, 10), 4.0)
    data2 = np.full((10, 10), 5.0)
    data3 = np.full((10, 10), 6.0)

    # Create darkfield images with values 1, 2, 3
    dark1 = np.full((10, 10), 1.0)
    dark2 = np.full((10, 10), 2.0)
    dark3 = np.full((10, 10), 3.0)

    beam = BeamImage(
        signal_images=[data1, data2, data3], background_images=[dark1, dark2, dark3]
    )

    result = beam.get_avg_and_subtracted()

    # Expected: mean([4, 5, 6]) - mean([1, 2, 3]) = 5.0 - 2.0 = 3.0
    expected = 3.0

    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "mask_dtype",
    [bool, int, float],
)
def test_mask_works(mask_dtype):
    """Test that mask works correctly with different dtypes."""
    data = np.ones((10, 10))
    mask = np.zeros((10, 10), dtype=mask_dtype)
    mask[0:5, 0:5] = 1  # Mask upper-left quadrant

    beam = BeamImage(signal_images=[data], mask=mask)
    result = beam.get_avg_and_subtracted()

    # Check that result is masked array
    assert isinstance(result, np.ma.MaskedArray)

    # Check upper-left quadrant is masked
    assert result.mask[0, 0]
    assert result.mask[4, 4]
    assert np.ma.is_masked(result[0, 0])
    assert np.ma.is_masked(result[4, 4])

    # Check upper-right quadrant is not masked
    assert not result.mask[0, 5]
    assert not result.mask[4, 9]
    assert not np.ma.is_masked(result[0, 5])
    assert not np.ma.is_masked(result[4, 9])
    assert result[0, 5] == 1.0
    assert result[4, 9] == 1.0

    # Check lower-left quadrant is not masked
    assert not result.mask[5, 0]
    assert not result.mask[9, 4]
    assert not np.ma.is_masked(result[5, 0])
    assert not np.ma.is_masked(result[9, 4])
    assert result[5, 0] == 1.0
    assert result[9, 4] == 1.0

    # Check lower-right quadrant is not masked
    assert not result.mask[5, 5]
    assert not result.mask[9, 9]
    assert not np.ma.is_masked(result[5, 5])
    assert not np.ma.is_masked(result[9, 9])
    assert result[5, 5] == 1.0
    assert result[9, 9] == 1.0


def test_no_darkfield():
    """Test that processing works without darkfield images."""
    data1 = np.full((10, 10), 2.0)
    data2 = np.full((10, 10), 4.0)

    beam = BeamImage(signal_images=[data1, data2])
    result = beam.get_avg_and_subtracted()

    # Expected: mean([2, 4]) = 3.0
    assert np.allclose(result, 3.0)


def test_no_mask():
    """Test that processing works without a mask."""
    data = np.ones((10, 10))

    beam = BeamImage(signal_images=[data])
    result = beam.get_avg_and_subtracted()

    # Check that result is masked array
    assert isinstance(result, np.ma.MaskedArray)

    # Check that no pixels are masked
    assert result.mask is np.ma.nomask or not result.mask.any()


def test_pixel_std_error_one_data_no_darkfield():
    """Test get_std_error() with one data image and no darkfield."""
    data1 = np.full((10, 10), 5.0)

    beam = BeamImage(signal_images=[data1])
    with pytest.raises(UncertaintyEstimateError):
        beam.get_std_error()


def test_pixel_std_error_one_data_one_darkfield():
    """Test get_std_error() with one data image and one darkfield."""
    data1 = np.full((10, 10), 5.0)
    dark1 = np.full((10, 10), 1.0)

    beam = BeamImage(signal_images=[data1], background_images=[dark1])
    with pytest.raises(UncertaintyEstimateError):
        beam.get_std_error()


def test_pixel_std_error_one_data_multiple_darkfield():
    """Test get_std_error() with one data image and multiple darkfield images."""
    data1 = np.full((10, 10), 5.0)
    dark1 = np.full((10, 10), 1.0)
    dark2 = np.full((10, 10), 2.0)
    dark3 = np.full((10, 10), 3.0)

    beam = BeamImage(signal_images=[data1], background_images=[dark1, dark2, dark3])
    std_devs = beam.get_std_error()

    # Check no errors and no NaN
    assert isinstance(std_devs, np.ma.MaskedArray)
    assert not np.any(np.isnan(std_devs))

    expected_std = np.std([1.0, 2.0, 3.0]) / np.sqrt(3)
    assert np.allclose(std_devs, expected_std)


def test_pixel_std_error_multiple_data_no_darkfield():
    """Test get_std_error() with multiple data images and no darkfield."""
    data1 = np.full((10, 10), 4.0)
    data2 = np.full((10, 10), 5.0)
    data3 = np.full((10, 10), 6.0)

    beam = BeamImage(signal_images=[data1, data2, data3])
    std_devs = beam.get_std_error()

    # Check no errors and no NaN
    assert isinstance(std_devs, np.ma.MaskedArray)
    assert not np.any(np.isnan(std_devs))

    expected_std = np.std([4.0, 5.0, 6.0]) / np.sqrt(3)
    assert np.allclose(std_devs, expected_std)


def test_pixel_std_error_multiple_data_one_darkfield():
    """Test get_std_error() with multiple data images and one darkfield."""
    data1 = np.full((10, 10), 4.0)
    data2 = np.full((10, 10), 5.0)
    data3 = np.full((10, 10), 6.0)
    dark1 = np.full((10, 10), 1.0)

    beam = BeamImage(signal_images=[data1, data2, data3], background_images=[dark1])
    std_devs = beam.get_std_error()

    # Check no errors and no NaN
    assert isinstance(std_devs, np.ma.MaskedArray)
    assert not np.any(np.isnan(std_devs))

    expected_std = np.std([4.0, 5.0, 6.0]) / np.sqrt(3)
    assert np.allclose(std_devs, expected_std)


def test_pixel_std_error_multiple_data_multiple_darkfield():
    """Test get_std_error() with multiple data and darkfield images."""
    data1 = np.full((10, 10), 4.0)
    data2 = np.full((10, 10), 5.0)
    data3 = np.full((10, 10), 6.0)
    dark1 = np.full((10, 10), 1.0)
    dark2 = np.full((10, 10), 2.0)
    dark3 = np.full((10, 10), 3.0)

    beam = BeamImage(
        signal_images=[data1, data2, data3], background_images=[dark1, dark2, dark3]
    )
    std_devs = beam.get_std_error()

    # Check no errors and no NaN
    assert isinstance(std_devs, np.ma.MaskedArray)
    assert not np.any(np.isnan(std_devs))

    data_std = np.std([4.0, 5.0, 6.0]) / np.sqrt(3)
    dark_std = np.std([1.0, 2.0, 3.0]) / np.sqrt(3)
    expected_std = np.sqrt(data_std**2 + dark_std**2)
    assert np.allclose(std_devs, expected_std)
