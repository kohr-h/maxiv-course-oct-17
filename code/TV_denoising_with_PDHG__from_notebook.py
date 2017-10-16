"""Denoising with total variation using the PDHG method.

This is the raw code file for the TV ``TV_denoising_with_PDHG`` Jupyter
notebook. For more mathematical details, see the notebook.
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

import odl
from odl.contrib import fom

# %%  Soft shrinkage

x = np.linspace(-3, 3, 31)


def soft_shrink(x, sigma):
    res = np.zeros_like(x)
    res[x < -sigma] = x[x < -sigma] + sigma
    res[x > sigma] = x[x > sigma] - sigma
    return res


plt.plot(x, x, label='x')
plt.plot(x, soft_shrink(x, 1), label='Soft-shrinkage of x')
plt.show()


# %% Generate a test image from SciPy

image = scipy.misc.ascent().astype('float32')
plt.imshow(image, cmap='gray')

# The shape we need to use is the transposed of the original shape since
# ODL uses 'xy' axes while the image is stored with 'ij' axis convention.
shape = image.T.shape

X = odl.uniform_discr(min_pt=[0, 0], max_pt=shape, shape=shape)

print('Pixel size:', X.cell_sides)

# %% Create space elements x_true and y (noisy)

# The rotation converts from 'ij' to 'xy' axes
image /= image.max()
x_true = X.element(np.rot90(image, -1))
# To get predictable randomness, we explicitly seed the random number generator
with odl.util.NumpyRandomSeed(123):
    y = x_true + 0.1 * odl.phantom.white_noise(X)

x_true.show(title='Original image (x_true)', force_show=True)
y.show(title='Noisy image (y)', force_show=True)

# %% Define operators

grad = odl.Gradient(X)
print('Gradient domain X:', grad.domain)
print('Gradient range X^d:', grad.range)

I = odl.IdentityOperator(X)
K = odl.BroadcastOperator(I, grad)
print('Domain of K:', K.domain)
print('Range of K:', K.range)

# %% Define the functions in the optimization problem

# `.translated(y)` takes care of the `. - y` part in the function
f_1 = odl.solvers.L2NormSquared(X).translated(y)

# The regularization parameter `alpha` is multiplied with the L1 norm.
# The L1 norm must be defined on X^d, the range of the gradient.
alpha = 0.15
f_2 = alpha * odl.solvers.L1Norm(grad.range)
f = odl.solvers.SeparableSum(f_1, f_2)

# We can test whether everything makes sense by evaluating `f(K(x))`
# at some arbitrary `x` in `X`. It should produce a scalar.
print(f(K(X.zero())))

g = odl.solvers.IndicatorNonnegativity(X)

# %% Compute some method parameters

# Estimate operator norm of K. The iteration cannot start at a constant vector
# since the gradient would produce 0, which is invalid in the power iteration.
# The noisy image `y` should do.
K_norm = 1.1 * odl.power_method_opnorm(K, xstart=y, maxiter=20)

tau = 1.0 / K_norm
sigma = 1.0 / K_norm

print('||K|| =', K_norm)

# %% Solve the problem

# Starting point
x = X.zero()

# In contrast to the notebook version, we can use callbacks to see
# intermediate results
callback = (odl.solvers.CallbackShow(step=10) &
            odl.solvers.CallbackPrintIteration(step=10))

# Run PDHG method. The vector `x` is updated in-place.
odl.solvers.pdhg(x, f, g, K, tau=tau, sigma=sigma, niter=200,
                 callback=callback)

# %% Show results

x_true.show('True image', force_show=True)
y.show('Noisy image', force_show=True)
x.show('Denoised image', force_show=True)
(x_true - x).show('Difference true - denoised', force_show=True)


# %% Compute some image quality metrics

print('Noisy')
print('-----')
print('Mean squared error:', fom.mean_squared_error(y, x_true))
print('PSNR:', fom.psnr(y, x_true))
print('SSIM:', fom.ssim(y, x_true))
print('')

print('Denoised')
print('--------')
print('Mean squared error:', fom.mean_squared_error(x, x_true))
print('PSNR:', fom.psnr(x, x_true))
print('SSIM:', fom.ssim(x, x_true))
