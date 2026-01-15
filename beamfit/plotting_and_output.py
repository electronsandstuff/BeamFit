import matplotlib.pyplot as plt
import numpy as np
from .methods.supergaussian.c_drivers import supergaussian

from .fit_param_conversion import get_mu_sigma_std, get_mu_sigma


def pretty_print_loc_and_size(h, C, pixel_size, pixel_size_std):
    """
    Print the fit parameters and standard deviations nicely.

    Parameters
    ----------
    h : np.ndarray
        Parameters of best fit for supergaussian model
    c : np.ndarray
        Covariance matrix of the parameters of best fit
    pixel_size : float
        Conversion from pixel size to the size of the image
    pixel_size_std : float
        Standard deviation in the pixel size
    """
    # Pull out the components
    mu, sigma = get_mu_sigma(h, pixel_size)
    mu_std, sigma_std = get_mu_sigma_std(h, C, pixel_size, pixel_size_std)

    # Print them
    np.set_printoptions(precision=3)
    print("Position:  ", end="")
    print(mu * 1e3, end="")
    print(" mm +/- ", end="")
    print(mu_std * 1e6, end="")
    print(" um")
    print("Spot Size: ", end="")
    print(np.sqrt(sigma.diagonal()) * 1e3, end="")
    print(" mm +/- ", end="")
    size_std = np.abs(0.5 / np.sqrt(sigma.diagonal())) * sigma_std.diagonal()
    print(size_std * 1e6, end="")
    print(" um")


def plot_residuals(image, h, sigma_threshold=2):
    """
    Plot the residuals of the image.

    Parameters
    ----------
    image : np.ndarray
        The image that was fit to
    h : np.ndarray
        Parameters of best fit for supergaussian model
    sigma_threshold : int, optional
        Mask image values below this level, by default 2
    """
    # Calculate the threshold
    threshold = np.exp(-1 * sigma_threshold**2 / 2)

    M, N = np.mgrid[: image.shape[0], : image.shape[1]]
    sg = supergaussian(M, N, *h)
    thresh_image = supergaussian(M, N, h[0], h[1], h[2], h[3], h[4], 1, 1, 0)

    residual = image - sg
    mask = thresh_image < threshold
    ma_residual = np.ma.masked_array(data=residual, mask=mask)
    plt.imshow(ma_residual, cmap="seismic")


def plot_beam_contours(image, h):
    """
    Plot the beam's contours from the fit.

    Parameters
    ----------
    image : np.ndarray
        The image that was fit to
    h : np.ndarray
        Parameters of best fit for supergaussian model
    """
    plt.imshow(image)
    M, N = np.mgrid[: image.shape[0], : image.shape[1]]
    gauss = supergaussian(M, N, *h)
    plt.contour(gauss, colors="r", levels=3)


def plot_threshold(image, sigma_threshold=4):
    """
    Plot a thresholded version of the image (clipped by both mask and image value).

    Parameters
    ----------
    image : np.ndarray
        The image that was fit to
    sigma_threshold : int, optional
        How many std. deviations to include assuming Gaussian beam, by default 4
    """
    # Calculate the threshold
    threshold = np.exp(-1 * sigma_threshold**2 / 2)

    # Get a median filtered image for thresholding
    # image_filtered = ndimage.median_filter(image, size=6)

    # Get the mask
    mask_from_image = ~image.mask

    # Find the peak and bottom
    masked_image = np.array(image)[mask_from_image].ravel()
    ind = np.argsort(masked_image)
    low = np.mean(masked_image[ind][2:12])
    high = np.mean(masked_image[ind][-12:-2])

    # Make the threshold mask
    mask_from_threshold = (image - low) / (high - low) > threshold
    mask_combined = np.logical_and(mask_from_image, mask_from_threshold)

    # Show it
    plt.imshow(mask_combined)
