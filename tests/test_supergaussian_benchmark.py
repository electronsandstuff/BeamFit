import numpy as np
import pytest
from beamfit.methods.supergaussian.c_drivers import supergaussian, supergaussian_grad
from beamfit import factory


@pytest.fixture
def image_dataset():
    """Realistic 256x256 image dataset with masking"""
    m, n = np.mgrid[:256, :256]
    # Simulate a masked image with ~80% valid pixels
    mask = np.random.rand(256, 256) < 0.8
    x = np.vstack((m[mask], n[mask]))
    return x


@pytest.fixture
def theta_params():
    """Standard theta parameters for benchmarking"""
    # theta = [mu_x, mu_y, st[0], st[1], st[2], log(n), a, o]
    return np.array([128.0, 128.0, 3.5, 0.0, 3.5, 0.0, 1.0, 0.05])


def create_fitfun(sig_param):
    """Create fitfun closure matching supergaussian.py:88-89"""

    def theta_to_h(theta):
        mu = theta[:2]
        st = theta[2:5]
        nt = theta[5]
        a = theta[6]
        o = theta[7]
        sigma = sig_param.reverse(st)
        n = np.exp(nt)
        return np.array([mu[0], mu[1], sigma[0, 0], sigma[0, 1], sigma[1, 1], n, a, o])

    def fitfun(xdata, *theta):
        return supergaussian(xdata[0], xdata[1], *theta_to_h(theta))

    return fitfun


def create_fitfun_grad(sig_param):
    """Create fitfun_grad closure matching supergaussian.py:91-94"""

    def theta_to_h(theta):
        mu = theta[:2]
        st = theta[2:5]
        nt = theta[5]
        a = theta[6]
        o = theta[7]
        sigma = sig_param.reverse(st)
        n = np.exp(nt)
        return np.array([mu[0], mu[1], sigma[0, 0], sigma[0, 1], sigma[1, 1], n, a, o])

    def theta_to_h_grad(theta):
        st = theta[2:5]
        nt = theta[5]
        j = np.identity(8)
        j[2:5, 2:5] = sig_param.reverse_grad(st)
        j[5, 5] = np.exp(nt)
        return j

    def fitfun_grad(xdata, *theta):
        jacf = theta_to_h_grad(theta)
        jacg = supergaussian_grad(xdata[0], xdata[1], *theta_to_h(theta))
        return jacg @ jacf  # Chain rule

    return fitfun_grad


@pytest.mark.parametrize(
    "sig_param_name",
    ["Cholesky", "LogCholesky", "Spherical", "MatrixLogarithm", "Givens"],
)
def test_benchmark_fitfun(benchmark, sig_param_name, image_dataset, theta_params):
    """Benchmark fitfun with 256x256 image for each sigma parameterization"""
    sig_param = factory.create("sig_param", sig_param_name)
    fitfun = create_fitfun(sig_param)
    benchmark(fitfun, image_dataset, *theta_params)


@pytest.mark.parametrize(
    "sig_param_name",
    ["Cholesky", "LogCholesky", "Spherical", "MatrixLogarithm", "Givens"],
)
def test_benchmark_fitfun_grad(benchmark, sig_param_name, image_dataset, theta_params):
    """Benchmark fitfun_grad with 256x256 image for each sigma parameterization"""
    sig_param = factory.create("sig_param", sig_param_name)
    fitfun_grad = create_fitfun_grad(sig_param)
    benchmark(fitfun_grad, image_dataset, *theta_params)
