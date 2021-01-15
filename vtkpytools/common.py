import vtk
import pyvista as pv
import numpy as np
from scipy.io import FortranFile
from pathlib import Path

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
    npoints = np.ceil(npoints).astype(np.int32)

    # Calculate the values of dn_i
    if include_zero:
        geomseries = np.zeros(npoints + 1)
        geomseries[1:] = minval*growthrate**np.arange(npoints)
    else:
        geomseries = minval*growthrate**np.arange(npoints)

    # Sum the dn_i together to get n_i
    geomseries = np.cumsum(geomseries)

    return geomseries

def unstructuredToPoly(unstructured_grid):
    """Convert vtk.UnstructruedGrid to vtk.PolyData"""
    geom = vtk.vtkGeometryFilter()
    geom.SetInputData(unstructured_grid)
    geom.Update()
    return pv.wrap(geom.GetOutput())

def orderPolyDataLine(polydata):
    """Put line PolyData points in order"""
    strip = vtk.vtkStripper()
    strip.SetInputData(polydata)
    strip.Update()
    return pv.wrap(strip.GetOutput())

def vCutter(input_data, cut_function):
    """Returns the intersection of input_data and cut_function

    Wrapper around vtkCutter filter. Output contains interpolated data from
    input_data to the intersection location. Note that cell data is NOT passed
    through.

    Parameters
    ----------
    input_data : pyvista.PointSet
        Data that will be cut by the cut_function. Intersected point will have
        data interpolated.
    cut_function : vtk.vtkImplicitFunction
        VTK function that cuts the input_data. Most common example would be
        vtkPlane.
    """

    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(cut_function)
    cutter.SetInputData(input_data)
    cutter.Update()
    return pv.wrap(cutter.GetOutput())

class Profile(pv.PolyData):
    """Wrap of pyvista.PolyData that includes walldata attribute

    Use case is for storage of wall local data (boundary layer metrics, Cf
    etc.) with profiles that correspond to that wall local data.

    """
    walldata = {}

    def setWallDataFromPolyDataPoint(self, PolyPoint):
        """Set walldata attribute from PolyData Point

        Primary use case is the using the output of vtkpytools.vCutter()
        """
        if PolyPoint.n_points != 1:
            raise RuntimeError('Profile should only have 1 wallpoint, {:d} given'.format(
                                                                    PolyPoint.n_points))
        self.walldata = dict(PolyPoint.point_arrays)
        self.walldata['Point'] = PolyPoint.points

def readBinaryArray(path, ncols) -> np.ndarray:
    """Get array from Fortran binary file.

    Parameters
    ----------

    path : Path
        Path to Fortran binary array.
    ncols : uint
        Number of columns in the binary file.
    """
    array = FortranFile(path, 'r')
    array = array.read_reals()
    nrows = int(array.shape[0]/ncols)
    array = np.reshape(array, (nrows, ncols))

    return array

def globFile(globstring, path: Path) -> Path:
    """ Glob for one file in directory, then return.

    If it finds more than one file matching the globstring, it will error out.

    Parameters
    ----------
    globstring : str
        String used to glob for file
    path : Path
        Path where file should be searched for
    """
    globlist = list(path.glob(globstring))
    if len(globlist) == 1:
        assert globlist[0].is_file()
        return globlist[0]
    elif len(globlist) > 1:
        raise RuntimeError('Found multiple files matching'
                           '"{}" in {}:\n\t{}'.format(globstring, path,
                                                  '\n\t'.join([x.as_posix() for x in globlist])))
    else:
        raise RuntimeError('Could not find file matching'
                            '"{}" in {}'.format(globstring, path))

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
