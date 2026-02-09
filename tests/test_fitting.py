import os
import pickle
import pytest
import numpy as np
import beamfit
from beamfit.filters import SigmaThresholdFilter, MedianFilter


def get_mu_sigma_numerical(h):
    """
    Evaluates the mean and variance/co-variance of the super-Gaussian numerically.  Generates an image with 10 sigma
    worth of tails.  Sums up the values X*rho, Y*rho, X^2*rho, X*Y*rho, and Y^2*rho.
    """
    # Estimate what the mean and standard deviation of the thing should be and make a test image to numerically
    # integrate at 10 sigma on each side
    mu, std = beamfit.get_mu_sigma(h, 1.0)
    sigma = 10
    xmin = int(mu[0] - sigma * np.sqrt(std[0, 0]))
    xmax = int(mu[0] + sigma * np.sqrt(std[0, 0]))
    ymin = int(mu[1] - sigma * np.sqrt(std[1, 1]))
    ymax = int(mu[1] + sigma * np.sqrt(std[1, 1]))
    Xn, Yn = np.mgrid[xmin:xmax, ymin:ymax]
    sgn = beamfit.supergaussian(Xn, Yn, *h) - h[-1]

    # Numerically integrate to find mean and variance
    norm = np.sum(sgn)
    mux = np.sum(Xn * sgn) / norm
    muy = np.sum(Xn * sgn) / norm
    Vxx = np.sum(Xn**2 * sgn) / norm
    Vxy = np.sum(Xn * Yn * sgn) / norm
    Vyy = np.sum(Yn**2 * sgn) / norm
    mu = np.array([mux, muy])
    sigma = np.array([[Vxx, Vxy], [Vxy, Vyy]]) - mu[:, None] * mu[None, :]

    return mu, sigma


def calc_gradient_central_difference(
    fn, x0=np.array([0, 0, 0]), h=1e-5, atol=1e-9, fn_type="sigma_mat"
):
    # Coefficients for the finite differences schemes
    coef = [
        np.array([0.0, -1 / 2, 0.0, 1 / 2, 0.0]),
        np.array([1 / 12, -2 / 3, 0.0, 2 / 3, -1 / 12]),
    ]

    # Find the x values to evaluate at
    n = coef[-2].size
    x = (
        np.arange(-(n // 2), n // 2 + 1)[None, None, :]
        * np.identity(len(x0))[:, :, None]
        * h
    )
    x = np.reshape(x, (len(x0), len(x0) * x.shape[2])).T
    x = x + x0[None, :]

    # Evaluate the Jacobian
    if fn_type == "sigma_mat":

        def fn_internal(x):
            return fn(x).ravel()[[True, True, False, True]]
    elif fn_type == "vector":
        fn_internal = fn
    elif fn_type == "scalar":

        def fn_internal(x):
            return np.array([fn(x)])
    else:
        raise ValueError(f'Unrecognized value for "fn_type": "{fn_type}"')

    y = np.array([fn_internal(xx) for xx in x])
    y = np.reshape(y.T, (y.shape[1], len(x0), y.shape[0] // len(x0)))
    j = np.array([np.sum(y * c[None, None, :] / h, axis=2) for c in coef])

    # Get the error estimate and set places where Jacobian is zero to nan manually
    err = np.abs((j[1] - j[0]) / j[1])
    err[np.abs(j[1]) < atol] = float("nan")

    if fn_type == "scalar":
        return j[1][0, 0], err[0, 0]
    return j[1], err  # Return the Jacobian and error estimates


@pytest.fixture
def test_data():
    """Fixture that loads the test data for all tests"""
    # Get our path and the path of the data file
    this_path = os.path.dirname(__file__)
    filename = os.path.join(this_path, "test_data.pickle")

    # Load the test data
    with open(filename, "rb") as f:
        return pickle.load(f)


def test_fit_supergaussian(test_data):
    # Pull out the test image and validation data
    test_image = test_data["supergaussian_fit_data"]["images"][0]
    valid_h = test_data["supergaussian_fit_data"]["labels"][0][0]
    # valid_C = test_data["supergaussian_fit_data"]["labels"][0][1]

    # Fit it and Compare
    res = beamfit.SuperGaussian().fit(test_image)
    # test_h, test_C = beamfit.fit_supergaussian(test_image)
    np.testing.assert_allclose(res.h, valid_h, rtol=0.1)
    res = beamfit.SuperGaussian().fit(test_image, np.ones_like(test_image))
    np.testing.assert_allclose(res.h, valid_h, rtol=0.1)


def test_fit_supergaussian_beam_image(test_data):
    # Pull out the test image and validation data
    test_image = beamfit.BeamImage(
        signal_images=[test_data["supergaussian_fit_data"]["images"][0]]
    )

    # Fit it and Compare
    res = beamfit.SuperGaussian().fit(test_image)
    valid_h = test_data["supergaussian_fit_data"]["labels"][0][0]
    np.testing.assert_allclose(res.h, valid_h, rtol=0.1)


def test_supergaussian(test_data):
    # Pull out the test data
    X = test_data["gaussufunc"]["X"]
    Y = test_data["gaussufunc"]["Y"]
    h = test_data["gaussufunc"]["h"]
    valid = test_data["gaussufunc"]["supergaussian"]

    # Compute the test function
    test = beamfit.supergaussian(X, Y, *h)

    # Test it
    np.testing.assert_allclose(test, valid)


def test_supergaussian_grad(test_data):
    # Pull out the test data
    X = test_data["gaussufunc"]["X"]
    Y = test_data["gaussufunc"]["Y"]
    h = test_data["gaussufunc"]["h"]
    valid = test_data["gaussufunc"]["supergaussian_grad"]

    # Compute the test function
    test = beamfit.supergaussian_grad(X, Y, *h).T

    # Test it
    for t, v in zip(valid, test):
        np.testing.assert_allclose(t, v, atol=1e-9)


def test_get_mu_sigma():
    # Numerically find the reference values for the image
    h_test = np.array([128, 128, 50**2, 0.0, 40**2, 0.8, 1.0, 0.05])
    mu_ref, sigma_ref = get_mu_sigma_numerical(h_test)

    # Find the test values from the library
    mu_test, sigma_test = beamfit.get_mu_sigma(h_test, 1.0)

    # Compare them
    np.testing.assert_allclose(mu_test, mu_ref, atol=1e-9)
    np.testing.assert_allclose(sigma_test, sigma_ref, atol=1e-9)


@pytest.mark.parametrize(
    "profile_class",
    [
        beamfit.GaussianProfile1D,
        beamfit.GaussianLinearLeastSquares,
        beamfit.RMSIntegration,
    ],
)
@pytest.mark.parametrize(
    "h_ref",
    [
        np.array([128, 256, 32**2, 0.0, 8**2, 1, 1.0, 0.06]),
        np.array([115, 345, 32**2, 0.0, 16**2, 1, 10.0, 0.15]),
        np.array([132, 375, 16**2, 0.0, 16**2, 1, 432.0, -23]),
        np.array([142, 128, 25**2, 0.0, 19**2, 1, 253.0, 432]),
    ],
)
def test_gaussian_profiles(profile_class, h_ref):
    """Test Gaussian profile fitting with various parameter sets"""
    # Generate the image
    X, Y = np.mgrid[:256, :512]
    sg = beamfit.supergaussian(X, Y, *h_ref)

    # Create profile instance and test fitting
    profile = profile_class()
    np.testing.assert_allclose(profile.fit(sg).h, h_ref, rtol=0.005, atol=1e-6)


@pytest.mark.parametrize("x0", [0.5, 0.75, 1.0, 1.25, 1.5])
def test_supergaussian_scaling_grad(x0):
    """Test supergaussian scaling gradient for various x0 values"""
    j, _ = calc_gradient_central_difference(
        beamfit.super_gaussian_scaling_factor,
        x0=np.array([x0]),
        fn_type="scalar",
    )
    np.testing.assert_allclose(
        beamfit.super_gaussian_scaling_factor_grad(x0), j, atol=1e-9
    )


@pytest.mark.parametrize(
    "x0",
    [
        np.array([1, 1, 1, 1, 1, 1, 1, 0]),
        np.array([1, 2, 3, 4, 5, 6, 1, 0]),
        np.array([6, 5, 4, 3, 2, 1, 1, 0]),
        np.array([6, 5, 4, 4, 5, 6, 1, 0]),
    ],
)
def test_gaussian_linear_least_squares_trans_grad(x0):
    """Test Gaussian linear least squares transformation gradient for various x0 values"""
    j, _ = calc_gradient_central_difference(
        beamfit.gaussian_lls_trans, x0=x0, fn_type="vector"
    )
    np.testing.assert_allclose(beamfit.gaussian_lls_trans_grad(x0), j, atol=1e-10)


@pytest.mark.parametrize(
    "predfun,expected_class",
    [
        ("GaussianProfile1D", beamfit.GaussianProfile1D),
        ("GaussianLinearLeastSquares", beamfit.GaussianLinearLeastSquares),
        ("RMSIntegration", beamfit.RMSIntegration),
    ],
)
def test_supergaussian_init_predfun_string(predfun, expected_class):
    """Test SuperGaussian initialization with string predfun names"""
    sg = beamfit.SuperGaussian(predfun=predfun)
    assert isinstance(sg.predfun, expected_class)
    assert sg.predfun.type == predfun


@pytest.mark.parametrize(
    "sig_param,expected_class",
    [
        ("Cholesky", beamfit.Cholesky),
        ("LogCholesky", beamfit.LogCholesky),
        ("Spherical", beamfit.Spherical),
        ("MatrixLogarithm", beamfit.MatrixLogarithm),
        ("Givens", beamfit.Givens),
    ],
)
def test_supergaussian_init_sig_param_string(sig_param, expected_class):
    """Test SuperGaussian initialization with string sig_param names"""
    sg = beamfit.SuperGaussian(sig_param=sig_param)
    assert isinstance(sg.sig_param, expected_class)
    assert sg.sig_param.type == sig_param


def test_supergaussian_init_with_predfun_args():
    """Test SuperGaussian initialization with predfun_args"""
    sg = beamfit.SuperGaussian(
        predfun="GaussianProfile1D",
        predfun_args={"sigma_threshold": 2.5, "median_filter_size": 5},
    )
    assert isinstance(sg.predfun, beamfit.GaussianProfile1D)
    assert isinstance(sg.predfun.filters[0], SigmaThresholdFilter)
    assert sg.predfun.filters[0].sigma == 2.5
    assert isinstance(sg.predfun.filters[1], MedianFilter)
    assert sg.predfun.filters[1].kernel_size == 5


def test_supergaussian_result_serialization():
    """Test SuperGaussianResult serialization and deserialization"""
    # Create a result with test data
    mu = np.array([128.5, 256.3])
    sigma = np.array([[50.2, 5.1], [5.1, 40.8]])
    c = np.eye(8) * 0.1  # Simple covariance matrix

    result1 = beamfit.SuperGaussianResult(
        mu=mu, sigma=sigma, n=1.2, a=0.95, o=0.05, c=c
    )

    # Serialize to dict
    result_dict = result1.model_dump()

    # Verify dict structure
    assert "mu" in result_dict
    assert "sigma" in result_dict
    assert "n" in result_dict
    assert "a" in result_dict
    assert "o" in result_dict
    assert "c" in result_dict

    # Deserialize from dict
    result2 = beamfit.SuperGaussianResult.model_validate(result_dict)

    # Verify deserialized values match
    np.testing.assert_allclose(result2.mu, mu)
    np.testing.assert_allclose(result2.sigma, sigma)
    assert result2.n == 1.2
    assert result2.a == 0.95
    assert result2.o == 0.05
    np.testing.assert_allclose(result2.c, c)


def test_supergaussian_result_h_property():
    """Test SuperGaussianResult h property getter and setter"""
    # Create result using h array
    h = np.array([128, 256, 50**2, 10, 40**2, 1.0, 0.95, 0.05])
    result = beamfit.SuperGaussianResult(h=h, c=np.eye(8))

    # Verify h property returns correct array
    np.testing.assert_allclose(result.h, h)

    # Verify individual fields
    assert result.mu[0] == 128
    assert result.mu[1] == 256
    assert result.sigma[0, 0] == 50**2
    assert result.sigma[0, 1] == 10
    assert result.sigma[1, 1] == 40**2
    assert result.n == 1.0
    assert result.a == 0.95
    assert result.o == 0.05


def test_supergaussian_result_get_uncertainty_matrix():
    """Test that get_uncertainty_matrix runs and returns sensible output"""
    # Create a result with realistic-ish parameters
    mu = np.array([128.0, 256.0])
    sigma = np.array([[50.0, 5.0], [5.0, 40.0]])
    n = 1.2

    # Build a positive-definite 8x8 covariance matrix for the fit parameters
    rng = np.random.default_rng(42)
    L = rng.standard_normal((8, 8))
    c = L @ L.T + np.eye(8)  # guaranteed positive definite

    result = beamfit.SuperGaussianResult(mu=mu, sigma=sigma, n=n, a=0.95, o=0.05, c=c)
    cov = result.get_uncertainty_matrix()

    # Should be a 5x5 matrix: [mu_x, mu_y, sig_xx, sig_xy, sig_yy]
    assert cov.shape == (5, 5)

    # Should be symmetric
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)

    # Should be positive semi-definite (all eigenvalues >= 0)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals >= -1e-12)

    # Diagonal entries (variances) should be positive
    assert np.all(np.diag(cov) > 0)


def test_supergaussian_result_get_uncertainty_matrix_none():
    """Test that get_uncertainty_matrix returns None when c is not provided"""
    mu = np.array([128.0, 256.0])
    sigma = np.array([[50.0, 5.0], [5.0, 40.0]])
    result = beamfit.SuperGaussianResult(mu=mu, sigma=sigma)
    assert result.get_uncertainty_matrix() is None


def test_supergaussian_result_json_serialization():
    """Test SuperGaussianResult JSON serialization and deserialization"""
    # Create a result with test data
    mu = np.array([128.5, 256.3])
    sigma = np.array([[50.2, 5.1], [5.1, 40.8]])
    c = np.eye(8) * 0.1

    result1 = beamfit.SuperGaussianResult(
        mu=mu, sigma=sigma, n=1.2, a=0.95, o=0.05, c=c
    )

    # Serialize to JSON
    json_str = result1.model_dump_json()

    # Verify it's valid JSON
    assert isinstance(json_str, str)

    # Deserialize from JSON
    result2 = beamfit.SuperGaussianResult.model_validate_json(json_str)

    # Verify deserialized values match
    np.testing.assert_allclose(result2.mu, mu)
    np.testing.assert_allclose(result2.sigma, sigma)
    assert result2.n == 1.2
    assert result2.a == 0.95
    assert result2.o == 0.05
    np.testing.assert_allclose(result2.c, c)
