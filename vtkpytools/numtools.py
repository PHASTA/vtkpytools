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


def seriesDiffLimiter(series: np.ndarray, dx=None, magnitude=None) -> np.ndarray:
    """Take a series and limit the difference between successive elements (DBSE)

    This takes in a 1-D array (whose difference between successive elements (DBSE) is
    increasing) and creates a new 1-D array that contains the same initial
    elements, up to a point decided by arguments, and then continues the series
    with constant DBSE.

    By specifying `dx`, the function will keep the first elements in the array
    such that the DBSE is less than `dx`. After that, the new points will be
    filled with intervals of `dx` until they reach the same maximum as the
    original series.


    By specifying `magnitude`, elements in the series that are less than
    magnitude will be duplicated. Elements greater than `magnitude` will be
    replaced such that the DBSE is equal to the spacing of the original series
    at `magnitude`.

    Assumes that the series is increasing and DBSE is increasing. ie. for dxs =
    np.diff(series), dxs[i] < dxs[i+1] for all i.

    Parameters
    ----------
    series : np.ndarray
        Series of floats representing sample points
    dx : float
        The resolution at which to stop growing and maintain resolution.
    magnitude : float
        The magnitude of the series at which to maintain the given resolution.

    Returns
    -------
    np.ndarray

    Examples
    -------

    >> input = [0, 1, 2, 4, 8]
    >> seriesDiffLimiter(input, dx=2.4)
        [0, 1, 2, 4, 6.4, 8]

    Note that the end point is preserved even though the last DBSE is less than
    `dx`.

    >> seriesDiffLimiter(input, magnitude=4.0)
        [0, 1, 2, 4, 6, 8]
    """
    if not (bool(dx) ^ bool(magnitude)): #xnor
        raise RuntimeError('Either dx or magnitude must be set')
    if dx:
        dxs = np.diff(series)
        index = np.ceil( pwlinRoots(np.arange(dxs.size), dxs - dx)[0] ).astype(np.int)
    else:
        index = np.ceil( pwlinRoots(np.arange(series.size), series - magnitude)[0] ).astype(np.int)
        dx = series[index] - series[index-1]

    fill_distance = np.max(series) - series[index]
    fill_size = np.ceil(fill_distance / dx).astype(np.int)
    fill_array = np.arange(1, fill_size+1)*dx + series[index]

    new_series = np.zeros(index + fill_size + 1)
    new_series[:index+1] = series[:index+1]
    new_series[index+1:] = fill_array
    new_series[new_series > series[-1]] = series[-1]

    return new_series


def symmetric2FullTensor(tensor_array) -> np.ndarray:
    """ Turn (n, 6) shape array of tensor entries into (n, 3, 3)

    Assumed that symmtetric entires are in XX YY ZZ XY XZ YZ order."""

    shaped_tensors = np.array([
                     tensor_array[:,0], tensor_array[:,3], tensor_array[:,4],
                     tensor_array[:,3], tensor_array[:,1], tensor_array[:,5],
                     tensor_array[:,4], tensor_array[:,5], tensor_array[:,2]
                    ]).T
    return shaped_tensors.reshape(shaped_tensors.shape[0], 3, 3)

def ReyStressTo3rdOrderTensor(tensor_array) -> np.ndarray:
    """ Turn (n, 18) shape array of tensor entries into (n, 3, 3, 3)

    Assumed that symmtetric entires are in XX YY ZZ XY XZ YZ order."""

    shaped_tensors = np.array([
                     tensor_array[:,0], tensor_array[:,9],tensor_array[:,12],
                     tensor_array[:,9], tensor_array[:,3],tensor_array[:,15],
                     tensor_array[:,12],tensor_array[:,15],tensor_array[:,6],
                     tensor_array[:,1], tensor_array[:,10],tensor_array[:,13],
                     tensor_array[:,10], tensor_array[:,4],tensor_array[:,16],
                     tensor_array[:,13],tensor_array[:,16],tensor_array[:,7],
                     tensor_array[:,2], tensor_array[:,11],tensor_array[:,14],
                     tensor_array[:,11], tensor_array[:,5],tensor_array[:,17],
                     tensor_array[:,14],tensor_array[:,17],tensor_array[:,8]
                    ]).T
    return shaped_tensors.reshape(shaped_tensors.shape[0], 3, 3, 3)


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

    def rank3Rotation(rot_tensor, shaped_tensors):
        return np.einsum('li,mj,nk,eijk->elmn', rot_tensor, rot_tensor, rot_tensor, shaped_tensors)

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

    elif tensor_array.shape[1] == 18:
        shaped_tensors = ReyStressTo3rdOrderTensor(tensor_array)
        return rank3Rotation(rotation_tensor, shaped_tensors).reshape(tensor_array.shape[0], 27)

    elif tensor_array.shape[1] == 27:
        shaped_tensors = tensor_array.reshape(tensor_array.shape[0], 3, 3, 3)
        return rank3Rotation(rotation_tensor, shaped_tensors).reshape(tensor_array.shape[0], 27)


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

def calcRotationRate(velocity_gradient) -> np.ndarray:
    """Calculate rotation rate from n velocity gradient tensors

    Interpreted as the symmetric tensor of the velocity gradient:
    1/2 (u_{i,j} + u_{j,i})

    Parameters
    ----------
    velocity_gradient : np.ndarray
        Assumed to be of shape (n,9) in the order XX XY XZ YX YY YZ ZX ZY ZZ

    Returns
    -------
    np.ndarray of shape (n,9) in the order XX XY XZ YX YY YZ ZX ZY ZZ

    """
    [nps,tmp] = np.shape(velocity_gradient)
    return np.array([
        np.zeros([nps]), 0.5*(velocity_gradient[:,1] - velocity_gradient[:,3]), 0.5*(velocity_gradient[:,2] - velocity_gradient[:,6]),
        0.5*(velocity_gradient[:,3] - velocity_gradient[:,1]), np.zeros([nps]), 0.5*(velocity_gradient[:,5] - velocity_gradient[:,7]),
        0.5*(velocity_gradient[:,6] - velocity_gradient[:,2]), 0.5*(velocity_gradient[:,5] - velocity_gradient[:,7]), np.zeros([nps]),
                     ]).T

def calcTKEProduction(VelGrad,ReyStress) -> np.ndarray:
    """Calculate tke production from velocity gradient and Reynolds stress tensor
    
    Interpreted as: tau_{ij} G_{ij}
    
    Parameters
    ----------
    VelGrad : np.ndarray, Assumed to of shape(n,9)
    ReyStress: np.ndarray, Assumed to of shape(n,6)
    """
    
    [nps,tmp] = np.shape(VelGrad)
    return np.array([ np.multiply(VelGrad[:,0],ReyStress[:,0]) + np.multiply(VelGrad[:,1],ReyStress[:,3]) + 
        np.multiply(VelGrad[:,2],ReyStress[:,4]) +  np.multiply(VelGrad[:,3],ReyStress[:,3]) + np.multiply(VelGrad[:,4],ReyStress[:,1]) +
        np.multiply(VelGrad[:,5],ReyStress[:,5]) + np.multiply(VelGrad[:,6],ReyStress[:,4]) + np.multiply(VelGrad[:,7],ReyStress[:,5]) +
        np.multiply(VelGrad[:,8],ReyStress[:,2])]).T


def calcDissip(StrainRate,stsbarKeq) -> np.ndarray:
    """Calculate tke production from velocity gradient and Reynolds stress tensor
    
    Interpreted as: 2*(S'_{ij} S'_{ij})
    
    Parameters
    ----------
    StrainRate : np.ndarray, Assumed to of shape(n,6)
    stsbarKeq: np.ndarray, Assumed to of shape(n,10)
    """
    
    duidxjsq_bar =  stsbarKeq[:,9]
    duidxj_barsq = (StrainRate[:,0]**2 + StrainRate[:,1]**2 + StrainRate[:,2]**2 +
                    2*(StrainRate[:,3]**2 + StrainRate[:,4]**2 + StrainRate[:,5]**2))
    dissip = 2*(duidxjsq_bar - duidxj_barsq)
        
    return dissip


def calcTKEAdvec(vel,dkdxi) -> np.ndarray:
    """Calculate advection of turbulent kinetic energy
    
    Interpreted as: u_i k{,i}
    
    Parameters
    ----------
    vel : np.ndarray, Assumed to of shape(n,3)
    dkdxi: np.ndarray, Assumed to of shape(n,3)
    """    

    advec = vel[:,0]*dkdxi[:,0] + vel[:,1]*dkdxi[:,1] + vel[:,2]*dkdxi[:,2]
    
    return advec
    

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
