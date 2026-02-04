import numpy as np
import pytest
from beamfit.image import BeamImage


def test_no_data_images():
    """Test that error is raised when no data images are provided."""
    with pytest.raises(ValueError):
        BeamImage(data_images=[])


@pytest.mark.parametrize(
    "data_images,darkfield_images,mask",
    [
        # Wrong type in data_images
        ([[1, 2, 3]], None, None),
        # Wrong type in darkfield_images
        ([np.ones((10, 10))], [[1, 2, 3]], None),
        # Wrong type in mask
        ([np.ones((10, 10))], None, [[True, False]]),
    ],
)
def test_wrong_types(data_images, darkfield_images, mask):
    """Test that error is raised for wrong types."""
    with pytest.raises(ValueError):
        BeamImage(
            data_images=data_images,
            darkfield_images=darkfield_images,
            mask=mask,
        )


@pytest.mark.parametrize(
    "data_images,darkfield_images,mask",
    [
        # Inconsistent shapes in data_images
        ([np.ones((10, 10)), np.ones((10, 12))], None, None),
        # Inconsistent shape between data and darkfield
        ([np.ones((10, 10))], [np.ones((10, 12))], None),
        # Inconsistent shape between data and mask
        ([np.ones((10, 10))], None, np.zeros((10, 12))),
    ],
)
def test_inconsistent_shapes(data_images, darkfield_images, mask):
    """Test that error is raised for inconsistent shapes."""
    with pytest.raises(ValueError):
        BeamImage(
            data_images=data_images,
            darkfield_images=darkfield_images,
            mask=mask,
        )


def test_non_2d_array():
    """Test that error is raised for non-2D arrays."""
    data = np.ones((10, 10, 3))
    with pytest.raises(ValueError):
        BeamImage(data_images=[data])


def test_too_small_images():
    """Test that error is raised for images smaller than 8x8."""
    data = np.ones((5, 5))
    with pytest.raises(ValueError):
        BeamImage(data_images=[data])


def test_data_cast_to_float64():
    """Test that data is cast to float64."""
    # Test with int array (must be at least 8x8)
    data_int = np.ones((10, 10), dtype=np.int32)
    darkfield_int = np.zeros((10, 10), dtype=np.int32)

    beam = BeamImage(data_images=[data_int], darkfield_images=[darkfield_int])

    assert beam._data_images[0].dtype == np.float64
    assert beam._darkfield_images[0].dtype == np.float64


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
        data_images=[data1, data2, data3], darkfield_images=[dark1, dark2, dark3]
    )

    result = beam.processed

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

    beam = BeamImage(data_images=[data], mask=mask)
    result = beam.processed

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

    beam = BeamImage(data_images=[data1, data2])
    result = beam.processed

    # Expected: mean([2, 4]) = 3.0
    assert np.allclose(result, 3.0)


def test_no_mask():
    """Test that processing works without a mask."""
    data = np.ones((10, 10))

    beam = BeamImage(data_images=[data])
    result = beam.processed

    # Check that result is masked array
    assert isinstance(result, np.ma.MaskedArray)

    # Check that no pixels are masked
    assert result.mask is np.ma.nomask or not result.mask.any()


def test_pixel_std_error_one_data_no_darkfield():
    """Test pixel_std_error with one data image and no darkfield."""
    data1 = np.full((10, 10), 5.0)

    beam = BeamImage(data_images=[data1])
    with pytest.raises(ValueError):
        beam.pixel_std_error


def test_pixel_std_error_one_data_one_darkfield():
    """Test pixel_std_error with one data image and one darkfield."""
    data1 = np.full((10, 10), 5.0)
    dark1 = np.full((10, 10), 1.0)

    beam = BeamImage(data_images=[data1], darkfield_images=[dark1])
    with pytest.raises(ValueError):
        beam.pixel_std_error


def test_pixel_std_error_one_data_multiple_darkfield():
    """Test pixel_std_error with one data image and multiple darkfield images."""
    data1 = np.full((10, 10), 5.0)
    dark1 = np.full((10, 10), 1.0)
    dark2 = np.full((10, 10), 2.0)
    dark3 = np.full((10, 10), 3.0)

    beam = BeamImage(data_images=[data1], darkfield_images=[dark1, dark2, dark3])
    std_devs = beam.pixel_std_error

    # Check no errors and no NaN
    assert isinstance(std_devs, np.ma.MaskedArray)
    assert not np.any(np.isnan(std_devs))

    # Expected: sqrt(std([5])^2 + std([1, 2, 3])^2) = sqrt(0 + std([1, 2, 3])^2)
    expected_std = np.std([1.0, 2.0, 3.0]) / np.sqrt(3)
    assert np.allclose(std_devs, expected_std)


def test_pixel_std_error_multiple_data_no_darkfield():
    """Test pixel_std_error with multiple data images and no darkfield."""
    data1 = np.full((10, 10), 4.0)
    data2 = np.full((10, 10), 5.0)
    data3 = np.full((10, 10), 6.0)

    beam = BeamImage(data_images=[data1, data2, data3])
    std_devs = beam.pixel_std_error

    # Check no errors and no NaN
    assert isinstance(std_devs, np.ma.MaskedArray)
    assert not np.any(np.isnan(std_devs))

    # Expected: sqrt(std([4, 5, 6])^2 + 0^2) = std([4, 5, 6])
    expected_std = np.std([4.0, 5.0, 6.0]) / np.sqrt(3)
    assert np.allclose(std_devs, expected_std)


def test_pixel_std_error_multiple_data_one_darkfield():
    """Test pixel_std_error with multiple data images and one darkfield."""
    data1 = np.full((10, 10), 4.0)
    data2 = np.full((10, 10), 5.0)
    data3 = np.full((10, 10), 6.0)
    dark1 = np.full((10, 10), 1.0)

    beam = BeamImage(data_images=[data1, data2, data3], darkfield_images=[dark1])
    std_devs = beam.pixel_std_error

    # Check no errors and no NaN
    assert isinstance(std_devs, np.ma.MaskedArray)
    assert not np.any(np.isnan(std_devs))

    # Expected: sqrt(std([4, 5, 6])^2 + std([1])^2) = sqrt(std([4, 5, 6])^2 + 0)
    expected_std = np.std([4.0, 5.0, 6.0]) / np.sqrt(3)
    assert np.allclose(std_devs, expected_std)


def test_pixel_std_error_multiple_data_multiple_darkfield():
    """Test pixel_std_error with multiple data and darkfield images."""
    data1 = np.full((10, 10), 4.0)
    data2 = np.full((10, 10), 5.0)
    data3 = np.full((10, 10), 6.0)
    dark1 = np.full((10, 10), 1.0)
    dark2 = np.full((10, 10), 2.0)
    dark3 = np.full((10, 10), 3.0)

    beam = BeamImage(
        data_images=[data1, data2, data3], darkfield_images=[dark1, dark2, dark3]
    )
    std_devs = beam.pixel_std_error

    # Check no errors and no NaN
    assert isinstance(std_devs, np.ma.MaskedArray)
    assert not np.any(np.isnan(std_devs))

    # Expected: sqrt(std([4, 5, 6])^2 + std([1, 2, 3])^2)
    data_std = np.std([4.0, 5.0, 6.0]) / np.sqrt(3)
    dark_std = np.std([1.0, 2.0, 3.0]) / np.sqrt(3)
    expected_std = np.sqrt(data_std**2 + dark_std**2)
    assert np.allclose(std_devs, expected_std)
