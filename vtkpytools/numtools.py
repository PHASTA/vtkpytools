"""Module of generic numerical tools """
import numpy as np


def getGeometricSeries(maxval, minval, growthrate, include_zero=True) -> np.ndarray:
    """Return geometric progression based on inputs.

    A geometric progression is defined as the change from n_{i+1} to n_{i} is
    equal to some multiplicative growth rate.

        dn_{i+1} = dn_{i} * r

    or equivalently

        dn_{i+1} = dn_{1} * r^{i-1}

    where r is the growth rate of the series and dn_{i} = n_{i} - n_{i-1}.
    Thus:

        n_i = sum_{j=1}^{i} dn_{1} r^{j-1}

    By assuming n_0 = 0, we use n_{1} = dn_{1} = minval. Whether n_{0} is
    included in the resulting array is determined by the include_zero
    parameter.

    Parameters
    ----------
    maxval : float
        Maximum value that should be reached by series
    minval : float
        Initial value of the series, also sets dn_{1}
    growthrate : float
        Growth rate of the series
    include_zero : bool, optional
        Whether the series should insert a 0.0 at the beginning of the series
        (default is True)

    Returns
    -------
    numpy.ndarray
    """
    # Calculate the number of points required to reach maxval
    npoints = np.log((maxval*(growthrate-1))/minval)/np.log(growthrate)
    npoints = np.floor(npoints).astype(np.int32)

    # Calculate the values of dn_i
    if include_zero:
        geomseries = np.zeros(npoints + 2)
        geomseries[1:-1] = minval*growthrate**np.arange(npoints)
    else:
        geomseries = np.zeros(npoints + 1)
        geomseries[:-1]  = minval*growthrate**np.arange(npoints)

    # Sum the dn_i together to get n_i
    geomseries = np.cumsum(geomseries)

    geomseries[-1] = maxval

    return geomseries


def symmetric2FullTensor(tensor_array) -> np.ndarray:
    """ Turn (n, 6) shape array of tensor entries into (n, 3, 3)

    Assumed that symmtetric entires are in XX YY ZZ XY XZ YZ order."""

    shaped_tensors = np.array([
                     tensor_array[:,0], tensor_array[:,3], tensor_array[:,4],
                     tensor_array[:,3], tensor_array[:,1], tensor_array[:,5],
                     tensor_array[:,4], tensor_array[:,5], tensor_array[:,2]
                    ]).T
    return shaped_tensors.reshape(shaped_tensors.shape[0], 3, 3)


def full2SymmetricTensor(tensor_array) -> np.ndarray:
    """ Turn (n, 3, 3) shape array of tensor entries into (n, 6)

    Symmetric entires are in XX YY ZZ XY XZ YZ order."""
    return np.array([
        tensor_array[:,0,0], tensor_array[:,1,1], tensor_array[:,2,2],
        tensor_array[:,0,1], tensor_array[:,0,2], tensor_array[:,1,2]
                    ]).T


def makeRotationTensor(rotation_axis, theta) -> np.ndarray:
    """Create rotation tensor from axis and angle of rotation

    Uses Rodrigues' rotation formula

    Parameters
    ----------
    rotation_axis : np.ndarray
        Axis to rotate around
    theta : float
        The angle of rotation about the rotation_axis in radians
    """

    eijk = np.zeros((3, 3, 3))
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1

    rotation_tensor = np.cos(theta) * np.identity(3) + \
                      (1-np.cos(theta)) * np.einsum('i,j->ij', rotation_axis, rotation_axis) + \
                      -np.sin(theta) * np.einsum('ijk,k->ij', eijk, rotation_axis)

    return rotation_tensor


def rotateTensor(tensor_array, rotation_tensor) -> np.ndarray:
    """Given rotation_tensor, rotate the "value" tensor

    This will infer the type of tensor input (vector, symmetric/full tensor,
    etc.) based on the shape of the array. tensor_array is assumed to be an
    array of the n-rank tensors to be rotated.

    Parameters
    ----------
    tensor_array : np.ndarray
        Array of tensors to be rotated. The type of tensor input is inferred
        from the shape of the array. Shape [n,3] is a vector, [n,6] is a
        symmetric tensor, and [n,9] and [n,3,3] are full rank 2 tensors.
    rotation_tensor : np.ndarray
        The single rank 2 tensor defining rotation.
    """

    def rank2Rotation(rot_tensor, shaped_tensors):
        return np.einsum('ik,ekl,jl->eij', rot_tensor, shaped_tensors, rot_tensor)

    if tensor_array.shape[1] == 3 and tensor_array.ndim == 2:
        return np.einsum('ij,ej->ei', rotation_tensor, tensor_array)

    elif (tensor_array.ndim == 3 and
          all(dimsize == 3 for dimsize in tensor_array.shape[1:]) ):
        return rank2Rotation(rotation_tensor, tensor_array)

    elif tensor_array.shape[1] == 6:
        # Assumed to be symmetric tensor in XX YY ZZ XY XZ YZ order
        shaped_tensors = symmetric2FullTensor(tensor_array)
        rotated_tensor = rank2Rotation(rotation_tensor, shaped_tensors)
        return full2SymmetricTensor(rotated_tensor)

    elif tensor_array.shape[1] == 9:
        shaped_tensors = tensor_array.reshape(tensor_array.shape[0], 3, 3)
        return rank2Rotation(rotation_tensor, shaped_tensors).reshape(tensor_array.shape[0], 9)
    else:
        raise ValueError('Did not find appropriate method'
                           ' for array of shape{}'.format(tensor_array.shape))


def calcStrainRate(velocity_gradient) -> np.ndarray:
    """Calculate strain rate from n velocity gradient tensors

    Interpreted as the symmetric tensor of the velocity gradient:
    1/2 (u_{i,j} + u_{j,i})

    Parameters
    ----------
    velocity_gradient : np.ndarray
        Assumed to be of shape (n,9) in the order XX XY XZ YX YY YZ ZX ZY ZZ

    Returns
    -------
    np.ndarray of shape (n,6) in the order XX YY ZZ XY XZ YZ

    """
    return np.array([
        velocity_gradient[:,0], velocity_gradient[:,4], velocity_gradient[:,8],
        0.5*(velocity_gradient[:,1] + velocity_gradient[:,3]),
        0.5*(velocity_gradient[:,2] + velocity_gradient[:,6]),
        0.5*(velocity_gradient[:,5] + velocity_gradient[:,7]),
                     ]).T


def pwlinRoots(x, y) -> np.ndarray:
    """Find roots of piecewise linear function

    Finds the roots of a piecewise-linear function defined by the nodes x and
    the values at the nodes y.

        f(x[i]) = y[i]    for all i in range(x.size)

    Code taken from: https://stackoverflow.com/a/46911822/7564988

    Parameters
    ----------
    x : (N) ndarray
        1-D array of values corresponding to nodes of the piecewise linear
    y : (N) ndarray
        1-D array of values corresponding to function value at the nodes

    Returns
    -------
    1-D ndarray of locations where y=0
    """

        # Get nodes where the function value changes sign
    s = np.abs(np.diff(np.sign(y))).astype(bool)
        # Perform linear interpolation between the nodes where function
            # changes sign
    return x[:-1][s] + np.diff(x)[s]/(np.abs(y[1:][s]/y[:-1][s])+1)
