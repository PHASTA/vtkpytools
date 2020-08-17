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

def calcCf(wall, Uref, nu=1.5E-5, rho=1, plane_normal='XY'):
    """Calcuate the Coefficient of Friction of the wall

    Parameters
    ----------
    wall : pv.UnstructuredGrid
        Wall cells and points
    Uref : float
        Reference velocity
    nu : float, optional
        Kinematic viscosity (default: 1.5E-5)
    rho : float, optional
        Density (default: 1)
    plane_normal : {'XY', 'XZ', 'YZ'}, optional
        Plane that the wall lies on. The shear stress vector will be projected
        onto it. (default: 'XY')

    Returns
    -------
    numpy.ndarray
    """

    if 'Normals' not in wall.array_names:
        raise RuntimeError('The wall object must have a "Normals" field present.')
    mu = nu * rho
    # streamwise_vectors = np.array((wall['Normals'][:,1],
    #                                 -wall['Normals'][:,0],
    #                                 np.zeros_like(wall['Normals'][:,0]))).T

        # reshape the gradient such that is is an array of rank 2 tensors
    grad_tensors = wall['gradient'].reshape(wall['gradient'].shape[0], 3, 3)
        # Compute gradient vector tangential to the wall
    tangentialVelocityGradient = np.einsum('ijk,ik->ij', grad_tensors, wall['Normals'])

    # Tw = np.einsum('ij,ij->i', tangential_e_ij, streamwise_vectors)*mu
    if plane_normal.lower() == 'xy':
        plane_normal = np.array([0,0,1])
    elif plane_normal.lower() == 'xz':
        plane_normal = np.array([0,1,0])
    elif plane_normal.lower() == 'yz':
        plane_normal = np.array([1,0,0])

        # Project tangential gradient vector onto the chosen plane using n x (T_w x n)
    Tw = mu * np.cross(plane_normal[None,:],
                       np.cross(tangentialVelocityGradient, plane_normal[None,:]))
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

def calcBoundaryLayerStats(dataBlock, line_walldists, dpercent=False,
                                dvortpercent=False, velocity_component=0, Uref=1, nu=1):

    if dpercent and isinstance(dpercent, bool): dpercent = 0.95
    if dvortpercent and isinstance(dvortpercent, bool): dvortpercent = 0.95

    wall = dataBlock['wall']
    delta_mom = np.zeros(wall.points.shape[0])
    Re_theta = np.zeros(wall.points.shape[0])
    delta_displace = np.zeros(wall.points.shape[0])
    for i, point in enumerate(wall.points):
        wallnormal = wall['Normals'][i,:]
        wallnormal = np.tile(wallnormal, (len(line_walldists),1))
        sample_points = line_walldists[:, None] * wallnormal
        sample_points += point

        sampled = False
        attempt = 0
        tolerance = 1E-8
        nudge_increment = 1E-8
        nudge_size = 0
        while not sampled:
            # Sample domain
            sample_line = pv.lines_from_points(sample_points)
            sample_line = sample_line.sample(dataBlock['grid'])

            # Nudge sample_line back and forth until line is within domain
                # Primarily needed at end points
            if np.abs(sample_line['Velocity']).max() < 1E-8:
                if attempt == 0:
                    print(f'Could not get a sample line for index {i}. Will attempt nudging')
                elif attempt == 8:
                    warnings.warn(f'Nudging failed for index {i}! The last nudge size was {nudge_size}.\n')
                    break

                orig_nudge_size = nudge_size
                attempt += 1
                if attempt % 2 == 0:
                    nudge_size = (attempt / 2) * nudge_increment - nudge_size
                else:
                    nudge_size = -np.ceil(attempt / 2) * nudge_increment - nudge_size
                sample_points[:,0] += nudge_size

            else:
                sampled = True
                if attempt > 0:
                    print(f'Nudging index {i} was successful after {attempt} attempts.')
                    print(f'\tThe nudge size was {nudge_size + orig_nudge_size}.')

        Ue = sample_line['Velocity'][-1,velocity_component]
        U = sample_line['Velocity'][:,velocity_component]

        integrand_displace = 1 - U/Ue
        integrand_mom = integrand_displace * (U/Ue)
        delta_displace[i] = np.trapz(integrand_displace, line_walldists)
        delta_mom[i] = np.trapz(integrand_mom, line_walldists)
        Re_theta[i] = delta_mom[i]*Uref/nu

        # U_vort = cumtrapz(sample_line['vorticity'][:,2], line_walldists, initial=0)
        # Uinf_vort = U_vort[-1]
        # delta_vortpercentIndex = line_walldists[U_vort > dvortpercent*U_vort[-1] ]
        # if not line_walldists.size > 0:
        #     warnings.warn('Could not find U_vort value. Try increasing the range of search or adjusting the dvortpercent value.')
        # else:
        # test = False

    dataBlock['wall']['delta_displace'] = delta_displace
    dataBlock['wall']['delta_mom'] = delta_mom
    dataBlock['wall']['Re_theta'] = Re_theta

    return dataBlock

