"""Denoising with total variation using the Douglas-Rachford method.

This is the raw code file for the TV ``Using_a_different_solver`` Jupyter
notebook. For more mathematical details, see the notebook.
"""

import numpy as np
import scipy.misc

import odl
from odl.contrib import fom

# %% Create test data and space

# Generate test image
image = scipy.misc.ascent().astype('float32')

# Create reconstruction space
shape = image.T.shape
X = odl.uniform_discr(min_pt=[0, 0], max_pt=shape, shape=shape)

# Wrap image as space element, generate noisy variant and display
image /= image.max()
x_true = X.element(np.rot90(image, -1))
# To get predictable randomness, we explicitly seed the random number generator
with odl.util.NumpyRandomSeed(123):
    y = x_true + 0.1 * odl.phantom.white_noise(X)

x_true.show(title='Original image (x_true)', force_show=True)
y.show(title='Noisy image (y)', force_show=True)

# %% Set up problem components

ident = odl.IdentityOperator(X)
grad = odl.Gradient(X)  # need this here for L1Norm below

# Function without linear operator
f = odl.solvers.IndicatorNonnegativity(X)

# Functions to be composed with linear operators. L[i] applies to g[i].
alpha = 0.15
g = [odl.solvers.L2NormSquared(X).translated(y),
     alpha * odl.solvers.L1Norm(grad.range)]
L = [ident, grad]

# We check if everything makes sense by evaluating the total functional at 0
x = X.zero()
print(f(x) + sum(g[i](L[i](x)) for i in range(len(g))))

# %% Choose solver parameters

grad_norm = 1.1 * odl.power_method_opnorm(grad, xstart=y, maxiter=20)
opnorms = [1, grad_norm]  # identity has norm 1


def check_params(tau, sigmas):
    sum_part = sum(sigma * opnorm ** 2
                   for sigma, opnorm in zip(sigmas, opnorms))
    print('Sum evaluates to', sum_part)
    check_value = tau * sum_part

    assert check_value < 4, 'value must be < 4, got {}'.format(check_value)
    print('Values ok, check evaluates to {}, must be < 4'.format(check_value))


tau = 1.5
c = 3.0 / (len(opnorms) * tau)
sigmas = [c / opnorm ** 2 for opnorm in opnorms]
check_params(tau, sigmas)

# %% Solve the problem

# Starting point
x = X.zero()

# In contrast to the notebook version, we can use callbacks to see
# intermediate results
callback = (odl.solvers.CallbackShow(step=10) &
            odl.solvers.CallbackPrintIteration(step=10))

# Run PDHG method. The vector `x` is updated in-place.
odl.solvers.douglas_rachford_pd(x, f, g, L, tau, sigmas, niter=200,
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
