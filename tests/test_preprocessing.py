import numpy as np
import beamfit


def test_deserialize_with_filters():
    """Test deserialization of AnalysisMethod with new filters format."""
    # Test with filters list format
    config = {
        "filters": [
            {"type": "SigmaThresholdFilter", "sigma": 2.0},
            {"type": "MedianFilter", "kernel_size": 3},
        ]
    }
    method = beamfit.AnalysisMethodDebugger(**config)
    assert len(method.filters) == 2
    assert method.filters[0].type == "SigmaThresholdFilter"
    assert method.filters[0].sigma == 2.0
    assert method.filters[1].type == "MedianFilter"
    assert method.filters[1].kernel_size == 3


def test_deserialize_with_legacy_format():
    """Test deserialization of AnalysisMethod with legacy format."""
    # Test with legacy format
    config = {"sigma_threshold": 2.0, "median_filter_size": 3}
    method = beamfit.AnalysisMethodDebugger(**config)
    assert len(method.filters) == 2
    assert method.filters[0].type == "SigmaThresholdFilter"
    assert method.filters[0].sigma == 2.0
    assert method.filters[1].type == "MedianFilter"
    assert method.filters[1].kernel_size == 3


def test_threshold():
    m, n = np.mgrid[:64, :64]
    image = beamfit.supergaussian(n, m, 32, 32, 8**2, 0, 8**2, 1, 1, 0)
    o = beamfit.AnalysisMethodDebugger(sigma_threshold=2).fit(image)
    assert np.sum(o.mask) == np.sum(image < np.exp(-(2**2)))


def test_median_filter():
    m, n = np.mgrid[:64, :64]
    image = beamfit.supergaussian(n, m, 32, 32, 8**2, 0, 8**2, 1, 1, 0)
    image[4, 24] = 100  # Set some random hot pixels
    image[32, 15] = 100
    image[54, 22] = 100
    o = beamfit.AnalysisMethodDebugger(median_filter_size=3).fit(image)
    assert abs(o.max() - 1) < 0.1  # equivalent to assertAlmostEqual with places=1


def test_masked():
    m, n = np.mgrid[:64, :64]
    image = beamfit.supergaussian(n, m, 32, 32, 8**2, 0, 8**2, 1, 1, 0)
    o = beamfit.AnalysisMethodDebugger(sigma_threshold=2, median_filter_size=3).fit(
        image
    )
    assert np.ma.isMaskedArray(o)
