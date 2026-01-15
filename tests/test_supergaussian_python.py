import numpy as np
from beamfit.methods.supergaussian.c_drivers import supergaussian, supergaussian_grad
from beamfit.methods.supergaussian.supergaussian import (
    supergaussian_python,
    supergaussian_grad_python,
)


def test_supergaussian_python_matches_c():
    """Test that supergaussian_python matches C implementation with random inputs"""
    np.random.seed(42)

    # Random input coordinates (256x256 image with ~80% valid pixels)
    m, n = np.mgrid[:256, :256]
    mask = np.random.rand(256, 256) < 0.8
    x = m[mask].astype(float)
    y = n[mask].astype(float)

    # Random but reasonable parameters
    mu_x = np.random.uniform(64, 192)
    mu_y = np.random.uniform(64, 192)
    sigma_xx = np.random.uniform(400, 2500)  # 20^2 to 50^2
    sigma_xy = np.random.uniform(-500, 500)
    sigma_yy = np.random.uniform(400, 2500)
    n_param = np.random.uniform(0.5, 2.5)
    a = np.random.uniform(0.5, 5.0)
    o = np.random.uniform(-0.5, 0.5)

    # Compare C and Python implementations
    c_result = supergaussian(
        x, y, mu_x, mu_y, sigma_xx, sigma_xy, sigma_yy, n_param, a, o
    )
    py_result = supergaussian_python(
        x, y, mu_x, mu_y, sigma_xx, sigma_xy, sigma_yy, n_param, a, o
    )

    np.testing.assert_allclose(py_result, c_result, rtol=1e-13, atol=1e-14)


def test_supergaussian_grad_python_matches_c():
    """Test that supergaussian_grad_python matches C implementation with random inputs"""
    np.random.seed(123)

    # Random input coordinates (256x256 image with ~80% valid pixels)
    m, n = np.mgrid[:256, :256]
    mask = np.random.rand(256, 256) < 0.8
    x = m[mask].astype(float)
    y = n[mask].astype(float)

    # Random but reasonable parameters
    mu_x = np.random.uniform(64, 192)
    mu_y = np.random.uniform(64, 192)
    sigma_xx = np.random.uniform(400, 2500)  # 20^2 to 50^2
    sigma_xy = np.random.uniform(-500, 500)
    sigma_yy = np.random.uniform(400, 2500)
    n_param = np.random.uniform(0.5, 2.5)
    a = np.random.uniform(0.5, 5.0)
    o = np.random.uniform(-0.5, 0.5)

    # Compare C and Python implementations
    # Note: C implementation returns shape (8, m), Python returns (m, 8)
    c_result = supergaussian_grad(
        x, y, mu_x, mu_y, sigma_xx, sigma_xy, sigma_yy, n_param, a, o
    )
    py_result = supergaussian_grad_python(
        x, y, mu_x, mu_y, sigma_xx, sigma_xy, sigma_yy, n_param, a, o
    ).T

    np.testing.assert_allclose(py_result, c_result.T, rtol=1e-12, atol=1e-14)
