# From PyGLN project
# Adapted from https://fsix.github.io/mnist/

import numpy as np
from scipy.ndimage import interpolation


def moments(image):
    # A trick in numPy to create a mesh grid
    c0, c1 = np.mgrid[:image.shape[0], :image.shape[1]]
    totalImage = np.sum(image)  # sum of pixels
    m0 = np.sum(c0*image)/totalImage  # mu_x
    m1 = np.sum(c1*image)/totalImage  # mu_y
    m00 = np.sum((c0-m0)**2*image)/totalImage  # var(x)
    m11 = np.sum((c1-m1)**2*image)/totalImage  # var(y)
    m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage  # covariance(x,y)
    # Notice that these are \mu_x, \mu_y respectively
    mu_vector = np.array([m0, m1])
    # Do you see a similarity between the covariance matrix
    covariance_matrix = np.array([[m00, m01], [m01, m11]])
    return mu_vector, covariance_matrix


def deskew_fn(image):
    image = image[0].numpy()
    c, v = moments(image)
    alpha = v[0, 1]/v[0, 0]
    affine = np.array([[1, 0], [alpha, 1]])
    ocenter = np.array(image.shape)/2.0
    offset = c-np.dot(affine, ocenter)
    transformed = interpolation.affine_transform(image, affine, offset=offset)
    return np.expand_dims(transformed, 0)


def deskewAll(X):
    currents = []
    for i in range(len(X)):
        currents.append(deskew_fn(X[i].reshape(28, 28)).flatten())
    return np.array(currents)
