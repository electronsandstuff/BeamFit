# BeamFit
This library contains methods to fit for the size of a beam in a noisy 2D image

## Installation and Testing
Install through pip.

```bash
pip install beamfit
```

## Fitting Quickstart

There's a couple of different options for fitting images which vary in how robust they are and how much information you need to feed them.  The most simple fit only requires the image itself and is run with the following code.  The beam we're fitting looks like this.

![The beam](assets/beam.png)

We'll also plot the residuals as a measure of goodness-of-fit.  This should be done in an interactive environment like jupyter to see the plot.

```python
# Load the image and run the fit
img = imageio.imread('beam.png')
h, C = beamfit.fit_supergaussian(img, image_weights=np.ones_like(img))

# The image and the fit
plt.subplot(1,2,1)
beamfit.plot_beam_contours(img, h)

# The residuals
plt.subplot(1,2,2)
beamfit.plot_residuals(img, h)
```

The fitting function should output the parameters of best fit (h) and an estimation of the variance-covariance matrix for the parameters (C).  The fit looks like the following.

![The fit](assets/fit.png)

The fit looks good, notice that there is a nearly uniform pattern of noise in the residual and that the contours of the fit in the plot to the left are nicely centered on the beam.  Sometimes the beam will be clipped on the edge of the screen.  To fix this, add a mask to the image which defines the location of the screen.  This image must be the same size as the beam images and should be black on the screen and white outside of the screen.  The mask can easily be generated from one of the beam images using gimp or another image editting tool.  Here is an example mask for this image.

![Mask](assets/the_mask.png)

Let's run a fit using the masked image.

```python
# Load the image and mask
img = imageio.imread('beam.png')
mask = imageio.imread('mask.png')

# Mask the image and fit it
img_masked = np.ma.masked_array(data=img, mask=mask)
h, C = beamfit.fit_supergaussian(img_masked, image_weights=np.ones_like(img))

# The image and the fit
plt.subplot(1,2,1)
beamfit.plot_beam_contours(img_masked, h)

# The residuals
plt.subplot(1,2,2)
beamfit.plot_residuals(img_masked, h)
```

This will give the following fits

![Masked fit](assets/mask_fit.png)

You can see that the beam is once again fit nicely and for beams close to the edge of the screen the mask can become important.

The final piece of information that can be used to improve the fit is a weighting of each pixel for how important it is for the least squares loss function.  When you are averaging multiple beam images (say from an experiment where you take many images of the same beam) you can compute the weighting from the standard error of the mean of each pixel.  Beamfit has a function that will do this for you already along with background subtraction.

```python
# Load the beam images
img1 = imageio.imread('beam1.png')
img2 = imageio.imread('beam2.png')
img3 = imageio.imread('beam3.png')

# Load the background images
bak1 = imageio.imread('background1.png')
bak2 = imageio.imread('background2.png')
bak3 = imageio.imread('background3.png')

# Load the mask, compute the image and weights
mask = imageio.imread('mask.png')
img, w = beamfit.get_image_and_weight([img1, img2, img3], [bak1, bak2, bak3], mask)

# Perform the fit
h, C = beamfit.fit_supergaussian(img_masked, image_weights=np.ones_like(img))
```

I'm not going to show the fits here because they will be the same as before.  For most applications you want the mean and variance of the fit distribution for interfacing with other code.  Beamfit has functions built in to convert from the internal parameters of best fit to the mean and variance.

```python
# Compute the mean and variance and their uncertainty
mu, sigma = beamfit.get_mu_sigma(h, 50e-6)
mu_std, sigma_std = beamfit.get_mu_sigma_std(h, C, 50e-6, 1e-9)

# Print the location and size for the user
beamfit.pretty_print_loc_and_size(h, C, 50e-6, 1e-9)
```

We also print the location for the user using the built in beamfit routine.

## Reference

##### fit_supergaussian(image, image_weights=None, prediction_func=None, sigma_threshold=3, sigma_threshold_guess=1)

The primary fitting routine.  It will perform a contrained non-linear fit of 2D supergaussian to the provided image.  This method by default will use a linear least squares fit to a regular 2D gaussian to form a prediction of the best fit parameters.  It then uses scipy's implementation of Levenberg-Marquardt fitting on a bounded set of parameters to hone in on the final non-linear fit.

###### Parameters

* image: array_like
      The image of the beam to fit.  Must be a 2D numpy array and is compatible with numpy
      masked arrays as a means of specifying which pixels are not part of the beam.
* image_weights: array_like
      The weight of each pixel in the least squares fit.  When the variance of each pixel is known,
      the weights will be w = 1/V(pixels).  Must be a 2D numpy array or None.
* prediction_func: function
      A function that returns a prediction of the supergaussian parameters before the non-linear
      fit.  Should accept an image (2D numpy array) and return the estimated parameters h
      (numpy array).
* sigma_threshold: float
      Only part of the image is passed to the non-linear fit based on a threshold of the initially
      estimated gaussian.  This parameter is the width of the region included in the threshold
      measured in the number of standard deviations of the distribution.
* sigma_threshold_guess: float
      The same as sigma_threshold, but for the linear least squares prediction function.

###### Returns

* h: array_like
      The parameters of best fit.  It is a 1D numpy array with the elements h = np.array([mux, muy,
      Vxx, Vxy, Vyy, n, A, O]).  These are the same parameters in the same order as in the numpy
      ufuncs.
* C: array_like
      The estimated variance-covariance matrix of the fit parameters.  A 2D numpy array where the
      rows/columns are labeled in the same order as h.

##### beamfit.get_image_and_weight(raw_images, dark_fields, mask)

Computes the mean background subtracted image and weights for use with beamfit.

###### Parameters

* raw_images: list of array_like
      A list of 2D numpy arrays representing the beam images.  Must be more than one image in the list.
* dark_fields: list of array_like
      A list of 2D numpy arrays representing the background images.  Should have the same size as
      the beam images
* mask: array_like
      The mask as a 2D boolean numpy array.  Should have the same dimensions as raw_images
      and dark_fields.

###### Returns

* image: array_like
      The averaged and background subtracted beam image.
* weights: array_like
      Weights compatible with beamfit computed from the variance of the images and backgrounds.

##### beamfit.get_mu_sigma(h, pixel_size)

Converts the parameters of best fit to the mean and variance of the distribution.

###### Parameters

* h: array_like
      The paremeters of best fit
* pixel_size: float
     A scaling factor for the pixels in units of meters.

###### Returns

* mu: array_like
      The mean of the distribution in meters
* sigma: array_like
      The variance of the distribution in meters^2

##### beamfit.get_mu_sigma_std(h, C, pixel_size, pixel_size_std)

Converts the parameters of best fit into the uncertainty of the mean and variance of the distribution.

###### Parameters

* h: array_like
      The paremeters of best fit
* C: array_like
      The variance-covariance matrix of the parameters
* pixel_size: float
      A scaling factor for the pixels in units of meters.
* pixel_size_std: float
      An estimate of the uncertainty in the pixel scale factor

###### Returns

* mu_std: array_like
      The uncertainty of the mean of the distribution in meters
* sigma_std: array_like
      The uncertainty of the variance of the distribution in meters^2

##### beamfit.pretty_print_loc_and_size(h, C, pixel_size, pixel_size_std)

Prints a nice version of the mean and variance of the distribution.

###### Parameters

* h: array_like
      The paremeters of best fit
* C: array_like
      The variance-covariance matrix of the parameters
* pixel_size: float
      A scaling factor for the pixels in units of meters.
* pixel_size_std: float
      An estimate of the uncertainty in the pixel scale factor

###### Returns

No return

##### beamfit.plot_residuals(image, h, sigma_threshold=2)

Plots the residuals of the fit using matplotlib.  Compatible with other matplotlib plotting functions through their stateful interface.

###### Parameters

* image: array_like
* h: array_like
      The paremeters of best fit
* sigma_threshold: float

###### Returns

No return

##### beamfit.plot_beam_contours(image, h)

Plots the beam image and a countour plot of the fit supergaussian on top of it using matplotlib functions.  Compatible with the matplotlib stateful interface.

###### Parameters

* image: array_like
* h: array_like
      The paremeters of best fit

###### Returns

No return
