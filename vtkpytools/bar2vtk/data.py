import numpy as np
import vtk
from .core import *
from scipy.io import FortranFile
import warnings

def getBinaryVelbar(velbarPath, velbar_ncols=5):
    """Get velbar array from binary file.

    Args:
        velbarPath (Path): Path to velbar file.
        velbar_ncols (uint): Number of columns in the binary file. (default: 5)
    """
    velbar = FortranFile(velbarPath, 'r')
    velbarArray = velbar.read_reals()
    velbar_nrows = int(velbarArray.shape[0]/velbar_ncols)
    velbarArray = np.reshape(velbarArray, (velbar_nrows, velbar_ncols))

    return velbarArray

def getBinaryStsbar(stsbarPath, stsbar_ncols=6):
    """Get stsbar array from binary file.

    Args:
        stsbarPath (Path): Path to stsbar file.
        stsbar_ncols (uint): Number of columns in the binary file. (default: 6)
    """
    stsbar = FortranFile(stsbarPath, 'r')
    stsbarArray = stsbar.read_reals()
    stsbar_nrows = int(stsbarArray.shape[0]/stsbar_ncols)
    stsbarArray = np.reshape(stsbarArray, (stsbar_nrows, stsbar_ncols))
    return stsbarArray

def calcReynoldsStresses(stsbarArray, velbarArray, conservative_stresses=False):
    """Calculate Reynolds Stresses from velbar and stsbar data.

    Args:
        stsbarArray (ndarray): Array of stsbar values
        velbarArray (ndarray): Array of velbar values
        conservative_stresses (bool): Whether the stsbar file used the
            'Conservative Stresses' option (default:False)
    """
    if conservative_stresses:
        warnings.warn("Calculation of Reynolds Stresses when using the 'Conservative Stress' option for stsbar has not been validated.")
        ReyStrTensor = np.empty((stsbarArray.shape[0], 6))
        ReyStrTensor[:,0] = stsbarArray[:,3] - stsbarArray[:,0]**2
        ReyStrTensor[:,1] = stsbarArray[:,4] - stsbarArray[:,1]**2
        ReyStrTensor[:,2] = stsbarArray[:,5] - stsbarArray[:,2]**2
        ReyStrTensor[:,3] = stsbarArray[:,6] - stsbarArray[:,0]*stsbarArray[:,1]
        ReyStrTensor[:,4] = stsbarArray[:,7] - stsbarArray[:,1]*stsbarArray[:,2]
        ReyStrTensor[:,5] = stsbarArray[:,8] - stsbarArray[:,0]*stsbarArray[:,2]
        # ReyStrTensor[:,5] = np.zeros_like(ReyStrTensor[:,5])
    else:
        ReyStrTensor = np.empty((stsbarArray.shape[0], 6))

        ReyStrTensor[:,0] = stsbarArray[:,0] - velbarArray[:,1]**2
        ReyStrTensor[:,1] = stsbarArray[:,1] - velbarArray[:,2]**2
        ReyStrTensor[:,2] = stsbarArray[:,2] - velbarArray[:,3]**2
        ReyStrTensor[:,3] = stsbarArray[:,3] - velbarArray[:,1]*velbarArray[:,2]
        ReyStrTensor[:,4] = stsbarArray[:,4] - velbarArray[:,1]*velbarArray[:,3]
        ReyStrTensor[:,5] = stsbarArray[:,5] - velbarArray[:,2]*velbarArray[:,3]

    return ReyStrTensor

def calcCf(wall, Uref, nu=1.5E-5, rho=1, planeNormal='XY'):

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
    if planeNormal == 'XY'.lower():
        planeNormal = np.array([0,0,1])
    elif planeNormal == 'XZ'.lower():
        planeNormal = np.array([0,1,0])
    elif planeNormal == 'YZ'.lower():
        planeNormal = np.array([1,0,0])

        # Project tangential gradient vector onto the chosen plane using n x (T_w x n)
    Tw = mu * np.cross(planeNormal[None,:],
                       np.cross(tangentialVelocityGradient, planeNormal[None,:]))
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

def sampleDataBlockProfile(dataBlock, line_walldists, pointid=None, cutterObj=None):
    "Return a sampled line at the wall point index"

    wall = dataBlock['wall']

    if 'Normals' not in wall.array_names:
        raise RuntimeError('The wall object must have a "Normals" field present.')

    if pointid:
        wallnormal = wall['Normals'][pointid,:]
        wallnormal = np.tile(wallnormal, (len(line_walldists),1))

        sample_points = line_walldists[:, None] * wallnormal
        sample_points += wall.points[pointid]

        sample_line = linesFromPoints(sample_points)
        sample_line = sample_line.sample(dataBlock['grid'])
        sample_line['WallDistance'] = line_walldists

    if cutterObj:
        cutter = vtk.vtkCutter()
        cutter.SetCutFunction(cutterObj)
        cutter.SetInputData(wall)
        cutter.Update()
        cutterout = pv.wrap(cutter.GetOutput())

        if cutterout.points.shape[0] != 1:
            raise RuntimeError('vtkCutter resulted in %d points instead of 1.'.format(
                cutterout.points.shape[0]))

        wallnormal = cutterout['Normals']

        sample_points = line_walldists[:, None] * wallnormal
        sample_points += cutterout.points

        sample_line = linesFromPoints(sample_points)
        sample_line = sample_line.sample(dataBlock['grid'])
        sample_line['WallDistance'] = line_walldists

    return sample_line
