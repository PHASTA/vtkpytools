import numpy as np
import vtk
from .core import *
from ..common import vCutter, Profile, readBinaryArray
from scipy.io import FortranFile
import warnings

def binaryVelbar(velbar_path):
    """Get velbar array from binary file.

    Wrapping around vpt.readBinaryArray. Assumes that the number of columns in
    the array is 5.

    Parameters
    ----------

    velbar_path : Path
        Path to velbar file.
    """
    return readBinaryArray(velbar_path, 5)

def binaryStsbar(stsbar_path):
    """Get stsbar array from binary file.

    Wrapping around vpt.readBinaryArray. Assumes that the number of columns in
    the array is 6.

    Parameters
    ----------
    stsbar_path : Path
        Path to stsbar file.
    """
    return readBinaryArray(stsbar_path, 6)

def calcReynoldsStresses(stsbar_array, velbar_array, conservative_stresses=False):
    """Calculate Reynolds Stresses from velbar and stsbar data.

    Parameters
    ----------
    stsbar_array : ndarray
        Array of stsbar values
    velbar_array : ndarray
        Array of velbar values
    conservative_stresses : bool
        Whether the stsbar file used the
        'Conservative Stresses' option (default:False)

    Returns
    -------
    numpy.ndarray
    """
    if conservative_stresses:
        warnings.warn("Calculation of Reynolds Stresses when using the 'Conservative Stress' option for stsbar has not been validated.")
        ReyStrTensor = np.empty((stsbar_array.shape[0], 6))
        ReyStrTensor[:,0] = stsbar_array[:,3] - stsbar_array[:,0]**2
        ReyStrTensor[:,1] = stsbar_array[:,4] - stsbar_array[:,1]**2
        ReyStrTensor[:,2] = stsbar_array[:,5] - stsbar_array[:,2]**2
        ReyStrTensor[:,3] = stsbar_array[:,6] - stsbar_array[:,0]*stsbar_array[:,1]
        ReyStrTensor[:,4] = stsbar_array[:,7] - stsbar_array[:,1]*stsbar_array[:,2]
        ReyStrTensor[:,5] = stsbar_array[:,8] - stsbar_array[:,0]*stsbar_array[:,2]
        # ReyStrTensor[:,5] = np.zeros_like(ReyStrTensor[:,5])
    else:
        ReyStrTensor = np.empty((stsbar_array.shape[0], 6))

        ReyStrTensor[:,0] = stsbar_array[:,0] - velbar_array[:,1]**2
        ReyStrTensor[:,1] = stsbar_array[:,1] - velbar_array[:,2]**2
        ReyStrTensor[:,2] = stsbar_array[:,2] - velbar_array[:,3]**2
        ReyStrTensor[:,3] = stsbar_array[:,3] - velbar_array[:,1]*velbar_array[:,2]
        ReyStrTensor[:,4] = stsbar_array[:,4] - velbar_array[:,1]*velbar_array[:,3]
        ReyStrTensor[:,5] = stsbar_array[:,5] - velbar_array[:,2]*velbar_array[:,3]

    return ReyStrTensor

def calcWallShearGradient(wall):
    """Calcuate the shear gradient at the wall

    Wall shear gradient is defined as the gradient of the velocity tangent to
    the wall in the wall-normal direction. To calculate wall shear stress,
    multiply by mu.

    If the wall normal unit vector is taken to be n and the
    gradient of velocity is e_ij, then the wall shear gradient is:

        (delta_ik - n_k n_i) n_j e_kj

    where n_j e_kj is the "traction" vector at the wall. See post at
    https://www.jameswright.xyz/post/wall_shear_gradient_from_velocity_gradient/
    for derivation of method.

    Parameters
    ----------
    wall : pv.UnstructuredGrid
        Wall cells and points

    Returns
    -------
    numpy.ndarray
    """

    if 'Normals' not in wall.array_names:
        raise RuntimeError('The wall object must have a "Normals" field present.')
    if 'gradient' not in wall.array_names:
        raise RuntimeError('The wall object must have a "gradient" field present.')

        # reshape the gradient such that is is an array of rank 2 tensors
    grad_tensors = wall['gradient'].reshape(wall['gradient'].shape[0], 3, 3)

    traction_vector = np.einsum('pkj,pj->pk', grad_tensors, wall['Normals'])
    del_ik_nkni = np.identity(3) - np.einsum('pk,pi->pki', wall['Normals'], wall['Normals'])
    wall_shear_gradient = np.einsum('pik,pk->pi', del_ik_nkni, traction_vector)

    return wall_shear_gradient

def calcCf(wall, Uref, nu, rho, plane_normal='XY'):
    """Calcuate the Coefficient of Friction of the wall

    Uses vpt.calcWallShearGradient to get du/dn, then uses input values to
    calculate Cf using:

        C_f = T_w / (0.5 * rho * Uref^2)

    Parameters
    ----------
    wall : pv.UnstructuredGrid
        Wall cells and points
    Uref : float
        Reference velocity
    nu : float, optional
        Kinematic viscosity
    rho : float, optional
        Density
    plane_normal : {'XY', 'XZ', 'YZ'}, optional
        Plane that the wall lies on. The shear stress vector will be projected
        onto it. (default: 'XY')

    Returns
    -------
    numpy.ndarray
    """
    mu = nu * rho

    wall_shear_gradient = calcWallShearGradient(wall)

    if plane_normal.lower() == 'xy':
        plane_normal = np.array([0,0,1])
    elif plane_normal.lower() == 'xz':
        plane_normal = np.array([0,1,0])
    elif plane_normal.lower() == 'yz':
        plane_normal = np.array([1,0,0])

        # Project tangential gradient vector onto the chosen plane using n x (T_w x n)
    Tw = mu * np.cross(plane_normal[None,:],
                       np.cross(wall_shear_gradient, plane_normal[None,:]))
    Tw = np.linalg.norm(Tw, axis=1)

    Cf = Tw / (0.5*rho*Uref**2)
    return Cf

def compute_vorticity(dataset, scalars, vorticity_name='vorticity'):
    """Compute Vorticity, only needed till my PR gets merged"""
    alg = vtk.vtkGradientFilter()

    alg.SetComputeVorticity(True)
    alg.SetVorticityArrayName(vorticity_name)

    _, field = dataset.get_array(scalars, preference='point', info=True)
    # args: (idx, port, connection, field, name)
    alg.SetInputArrayToProcess(0, 0, 0, field.value, scalars)
    alg.SetInputData(dataset)
    alg.Update()
    return pv.filters._get_output(alg)

def sampleDataBlockProfile(dataBlock, line_walldists, pointid=None,
                           cutterobj=None) -> Profile:
    """Sample data block over a wall-normal profile

    Given a dataBlock containing a 'grid' and 'wall' block, this will return a
    PolyData object that samples 'grid' at the wall distances specified in
    line_walldists. This assumes that the 'wall' block has a field named
    'Normals' containing the wall-normal vectors.

    The location of the profile is defined by either the index of a point in
    the 'wall' block or by specifying a vtk implicit function (such as
    vtk.vtkPlane) that intersects the 'wall' object. The latter uses the
    vtk.vtkCutter filter to determine the intersection.

    Parameters
    ----------
    dataBlock : pv.MultiBlock
        MultiBlock containing the 'grid' and 'wall' objects
    line_walldists : numpy.ndarray
        The locations normal to the wall that should be sampled and returned.
        Locations are expected to be in order.
    pointid : int, optional
        Index of the point in 'wall' where the profile should be taken.
        (default: None)
    cutterobj : vtk.vtkPlane, optional
        VTK object that defines the profile location via intersection with the
        'wall'

    Returns
    -------
    vtkpytools.Profile
    """

    wall = dataBlock['wall']

    if 'Normals' not in wall.array_names:
        raise RuntimeError('The wall object must have a "Normals" field present.')
    if not pointid and not cutterobj:
        raise RuntimeError('Must provide either pointid or cutterobj.')

    if pointid:
        wallnormal = wall['Normals'][pointid,:]
        wallnormal = np.tile(wallnormal, (len(line_walldists),1))

        sample_points = line_walldists[:, None] * wallnormal
        sample_points += wall.points[pointid]

        sample_line = pv.lines_from_points(sample_points)
        sample_line = sample_line.sample(dataBlock['grid'])
        sample_line['WallDistance'] = line_walldists

    if cutterobj:
        cutterout = vCutter(wall, cutterobj)
        if cutterout.points.shape[0] != 1:
            raise RuntimeError('vCutter resulted in {:d} points instead of 1.'.format(
                cutterout.points.shape[0]))

        wallnormal = cutterout['Normals']

        sample_points = line_walldists[:, None] * wallnormal
        sample_points += cutterout.points

        sample_line = pv.lines_from_points(sample_points)
        sample_line = sample_line.sample(dataBlock['grid'])
        sample_line['WallDistance'] = line_walldists

        sample_line = Profile(sample_line)
        sample_line.setWallDataFromPolyDataPoint(cutterout)

    return sample_line
