import numpy as np
import vtk
import pyvista as pv
from ..common import readBinaryArray, Profile, vCutter
from ..numtools import makeRotationTensor
import warnings
import vtkpytools as vpt

def binaryBarFiles(bar_path,ncols) -> np.ndarray:
    """ Get bar (vel,sts,stsKeq,SMR) files from binary file.

    Wrapping around vpt.readBinaryArray. Allows flexibility in selection of number
    of columns.

    Parameters
    ----------
    bar_path : Path
        Path to bar file.
        
    ncol : int
        Number of columns expected to be in the file
            
    """  

    return readBinaryArray(bar_path, ncols)    

def binaryVelbar(velbar_path) -> np.ndarray:
    """Get velbar array from binary file.

    Wrapping around vpt.readBinaryArray. Assumes that the number of columns in
    the array is 5.

    Parameters
    ----------
    velbar_path : Path
        Path to velbar file.
    """
    return readBinaryArray(velbar_path, 5)

def binaryVelbarSclr(velbar_path,nsclr) -> np.ndarray:
    """Get velbar array from binary file.

    Wrapping around vpt.readBinaryArray. Assumes that the number of columns in
    the array is 5+nsclr.

    Parameters
    ----------
    velbar_path : Path
        Path to velbar file.
    """
    return readBinaryArray(velbar_path, 5+nsclr)

def binaryStsbar(stsbar_path) -> np.ndarray:
    """Get stsbar array from binary file.

    Wrapping around vpt.readBinaryArray. Assumes that the number of columns in
    the array is 6.

    Parameters
    ----------
    stsbar_path : Path
        Path to stsbar file.
    """
    return readBinaryArray(stsbar_path, 6)

def binaryStsbarWithpp(stsbar_path) -> np.ndarray:
    """Get stsbar array from binary file.

    Wrapping around vpt.readBinaryArray. Assumes that the number of columns in
    the array is 7.

    Parameters
    ----------
    stsbar_path : Path
        Path to stsbar file.
    """
    return readBinaryArray(stsbar_path, 7)

def binaryStsbarWithConsvStress(stsbar_path) -> np.ndarray:
    """Get stsbar array from binary file.

    Wrapping around vpt.readBinaryArray. Assumes that the number of columns in
    the array is 16.

    Parameters
    ----------
    stsbar_path : Path
        Path to stsbar file.
    """
    return readBinaryArray(stsbar_path, 16)

def binaryIDDESbar(IDDESbar_path) -> np.ndarray:
    """Get IDDESbar array from binary file.

    Wrapping around vpt.readBinaryArray. Assumes that the number of columns in
    the array is 7.

    Parameters
    ----------
    iddesbar_path : Path
        Path to IDDESbar file.
    """
    return readBinaryArray(IDDESbar_path, 10)

def binaryStsbarKeq(StsbarKeq_path) -> np.ndarray:
    """Get StsbarKeq array from binary file.

    Wrapping around vpt.readBinaryArray. Assumes that the number of columns in
    the array is 10.

    Parameters
    ----------
    stsbarkeqs_path : Path
        Path to stsbarkeq file.
    """
    return readBinaryArray(StsbarKeq_path, 10)

def binarySMRbar(SMRbar_path) -> np.ndarray:
    """Get SMRbar array from binary file.

    Wrapping around vpt.readBinaryArray. Assumes that the number of columns in
    the array is 15.

    Parameters
    ----------
    SMRbar_path : Path
        Path to SMRbar file.
    """
    return readBinaryArray(SMRbar_path, 55)


def calcReynoldsStresses(stsbar_array, velbar_array, conservative_stresses=False) -> np.ndarray:
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

def calcPresVel(stsbarKeq_array,velbar_array) -> np.ndarray:
    """Calculate Pressure-velocity from velbar and stsbarKeq data.

    Parameters
    ----------
    stsbarKeq_array : ndarray
        Array of stsbarKeq values
    velbar_array : ndarray
        Array of velbar values

    Returns
    -------
    numpy.ndarray
    """
    PresVel = np.empty((velbar_array.shape[0],3))
    PresVel[:,0] = stsbarKeq_array[:,0] - (velbar_array[:,0]*velbar_array[:,1])
    PresVel[:,1] = stsbarKeq_array[:,1] - (velbar_array[:,0]*velbar_array[:,2])
    PresVel[:,2] = stsbarKeq_array[:,2] - (velbar_array[:,0]*velbar_array[:,3])
    
    return PresVel

def calcvelStrain(stsbarKeq_array,velbar_array,velgrad) -> np.ndarray:
    """Calculate tke production from velocity gradient and Reynolds stress tensor
    
    Interpreted as: (u'_{i} S'_{ij})
    
    Parameters
    ----------
    stsbarKeq_array : np.ndarray, Assymed to of shape(n,10)
    velgrad : np.ndarray, Assumed to of shape(n,6)
    velbar_array: np.ndarray, Assumed to of shape(n,3)
    """
    
    ui = velbar_array[:,1:4]   
    uiSij_bar =  stsbarKeq_array[:,6:9]
    
    Sij = np.empty((stsbarKeq_array.shape[0],6))
    Sij[:,0] = velgrad[:,0]
    Sij[:,1] = velgrad[:,4]
    Sij[:,2] = velgrad[:,8]
    Sij[:,3] = 0.5*(velgrad[:,1]+velgrad[:,3])
    Sij[:,4] = 0.5*(velgrad[:,2]+velgrad[:,6])
    Sij[:,5] = 0.5*(velgrad[:,5]+velgrad[:,7])
    
    uiSijfluc_bar = np.empty((stsbarKeq_array.shape[0],3))
    uiSijfluc_bar[:,0] = uiSij_bar[:,0] - (ui[:,0]*Sij[:,0] + ui[:,1]*Sij[:,3] + ui[:,2]*Sij[:,4])
    uiSijfluc_bar[:,1] = uiSij_bar[:,1] - (ui[:,0]*Sij[:,3] + ui[:,1]*Sij[:,1] + ui[:,2]*Sij[:,5])
    uiSijfluc_bar[:,2] = uiSij_bar[:,2] - (ui[:,0]*Sij[:,4] + ui[:,1]*Sij[:,5] + ui[:,2]*Sij[:,2])
        
    return uiSijfluc_bar

def calcTurbTrans(stsbarKeq_array,velbar_array,stsbar_array) -> np.ndarray:
    """Calculate Turbulence Transport from velbar and stsbarKeq data.

    Parameters
    ----------
    stsbarKeq_array : ndarray
        Array of stsbarKeq values
    velbar_array : ndarray
        Array of velbar values
    stsbar_array: ndarray
        Array of computed Reynolds stress values

    Returns
    -------
    numpy.ndarray
    """
    u1 = velbar_array[:,1]
    u2 = velbar_array[:,2]
    u3 = velbar_array[:,3]
    term1 = -2*(u1**2 + u2**2 + u3**2)
    term2 = stsbar_array[:,0] + stsbar_array[:,1] + stsbar_array[:,2]
    
    TurbTrans = np.empty((velbar_array.shape[0],3))    
    TurbTrans[:,0] = stsbarKeq_array[:,3] - ((u1*(term1+term2)) + 
                                             2*((u1*stsbar_array[:,0]) + (u2*stsbar_array[:,3]) +
                                                (u3*stsbar_array[:,4])))
    TurbTrans[:,1] = stsbarKeq_array[:,4] - ((u2*(term1+term2)) + 
                                             2*((u1*stsbar_array[:,3]) + (u2*stsbar_array[:,1]) +
                                                (u3*stsbar_array[:,5])))
    TurbTrans[:,2] = stsbarKeq_array[:,5] - ((u3*(term1+term2)) + 
                                             2*((u1*stsbar_array[:,4]) + (u2*stsbar_array[:,5]) +
                                                (u3*stsbar_array[:,2])))    
    
    return TurbTrans

def calcWallShearGradient(wall) -> np.ndarray:
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

def calcdpdx(wall,flag) -> np.ndarray:
    """Calcuate the Pressure gradient

    Parameters
    ----------
    wall : pv.PolyData
        Wall cells and points

    Returns
    -------
    numpy.ndarray
    """
    if(flag==0):
        pres = wall['Pressure']
    elif(flag==1):
        pres = wall['Pressure_sv']
    elif(flag==2):
        pres = wall['dpdx_s']
        
    x = wall.points[:,0]
    y = wall.points[:,1]

    [length,] = pres.shape
    
    x0 = x[:-1]
    x1 = x[1:]
    y0 = y[:-1]
    y1 = y[1:]
    ds = ((x1-x0)**2+(y1-y0)**2)**(1/2)
    
    dpdx = np.zeros([length]) 
    
    y0 = pres[:-2]
    y1 = pres[1:-1]
    y2 = pres[2:]
    ds21=ds[1:]
    ds10=ds[:-1]
    ds20=ds21+ds10
    f = ds21/ds20 #(x2 - x1)/(x2 - x0)
    dpdx[1:length-1] = (1-f)*(y2 - y1)/ds21 + f*(y1 - y0)/ds10
    dpdx[length-1] = dpdx[length-2]
    dpdx[0] = dpdx[1]
    
    # for i in range(1,length-2):
    #     ds = np.sqrt((x[i+1] -  x[i-1])**2 + (y[i+1] - y[i-1])**2)
    #     dpdx[i] = (pres[i+1]-(2*pres[i])+pres[i-1])/ds
    # dpdx[length-1] = dpdx[length-2]
    # dpdx[0] = dpdx[1]
        
    return dpdx

def calcCf(wall, Uref, nu, rho, plane_normal='XY') -> np.ndarray:
    """Calcuate the Coefficient of Friction of the wall

    Uses vpt.calcWallShearGradient to get du/dn, then uses input values to
    calculate Cf using:

        C_f = T_w / (0.5 * rho * Uref^2)

    Parameters
    ----------
    wall : pv.PolyData
        Wall cells and points
    Uref : float
        Reference velocity
    nu : float
        Kinematic viscosity
    rho : float
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
        streamwise_vectors = np.array([wall['Normals'][:,1],
                                       -wall['Normals'][:,0],
                                       np.zeros_like(-wall['Normals'][:,0])]).T
    elif plane_normal.lower() == 'xz':
        plane_normal = np.array([0,1,0])
        streamwise_vectors = np.array([wall['Normals'][:,2],
                                       np.zeros_like(-wall['Normals'][:,0]),
                                       -wall['Normals'][:,0]]).T
    elif plane_normal.lower() == 'yz':
        plane_normal = np.array([1,0,0])
        streamwise_vectors = np.array([wall['Normals'][:,2],
                                       np.zeros_like(-wall['Normals'][:,0]),
                                       -wall['Normals'][:,0]]).T

        # Project tangential gradient vector onto the chosen plane using n x (T_w x n)
    Tw = mu * np.cross(plane_normal[None,:],
                       np.cross(wall_shear_gradient, plane_normal[None,:]))
    Tw = np.einsum('ej,ej->e', streamwise_vectors, Tw)

    Cf = Tw / (0.5*rho*Uref**2)
    return Cf

def compute_vorticity(dataset, scalars, vorticity_name='vorticity'):
    """(DEPRECATED) Compute Vorticity, only needed till my PR gets merged

     .. deprecated::
         Use the `compute_derivative` method in pyvista's `UnstructuredGrid` class
     """

    warnings.warn("This function is deprecated. Use the 'compute_derivative'"
                  " method in pyvista's 'UnstructuredGrid' class" , FutureWarning)
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
                           cutterobj=None, normal=None) -> Profile:
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
    normal : numpy.ndarray, optional
        If given, use this vector as the wall normal.

    Returns
    -------
    vtkpytools.Profile
    """

    wall = dataBlock['wall']

    if 'Normals' not in wall.array_names:
        raise RuntimeError('The wall object must have a "Normals" field present.')
    if not (isinstance(pointid, int) ^ bool(cutterobj) ): #xnor
        raise RuntimeError('Must provide either pointid or cutterobj.')

    if isinstance(pointid, int):
        wallnormal = wall['Normals'][pointid,:] if normal is None else normal
        wallnormal = np.tile(wallnormal, (len(line_walldists),1))

        sample_points = line_walldists[:, None] * wallnormal
        sample_points += wall.points[pointid]

        sample_line = pv.lines_from_points(sample_points)
        sample_line = sample_line.sample(dataBlock['grid'])
        sample_line['WallDistance'] = line_walldists

    else:
        cutterout = vCutter(wall, cutterobj)
        if cutterout.points.shape[0] != 1:
            raise RuntimeError('vCutter resulted in {:d} points instead of 1.'.format(
                cutterout.points.shape[0]))

        wallnormal = cutterout['Normals'] if normal is None else normal

        sample_points = line_walldists[:, None] * wallnormal
        sample_points += cutterout.points

        sample_line = pv.lines_from_points(sample_points)
        sample_line = sample_line.sample(dataBlock['grid'])
        sample_line['WallDistance'] = line_walldists

        sample_line = Profile(sample_line)
        sample_line.setWallDataFromPolyDataPoint(cutterout)

    return sample_line

def wallAlignRotationTensor(wallnormal, cart_normal, plane='xy') -> np.ndarray:
    """Create rotation tensor for wall alligning quantities

    For 2D xy plane and streamwise x, cart_normal should be the y unit vector.

    Parameters
    ----------
    wallnormal : numpy.ndarray
        The normal vector at the wall
    cart_normal : numpy.ndarray
        The Cartesian equivalent to the wall normal. If the wall were "flat",
        this would be the wall normal vector.
    plane : str
        2D plane to rotate on, from 'xy', 'xz', 'yz'. (Default is 'xy')
    """

    if plane.lower() == 'xy':   rotation_axis = np.array([0, 0, 1])
    elif plane.lower() == 'xz': rotation_axis = np.array([0, 1, 0])
    elif plane.lower() == 'yz': rotation_axis = np.array([1, 0, 0])

    theta = np.arccos(np.dot(wallnormal, cart_normal))
    # Determine whether clockwise or counter-clockwise rotation
    rotation_cross = np.cross(cart_normal, cart_normal - wallnormal)
    if np.dot(rotation_cross, rotation_axis) < 0: theta *= -1

    return makeRotationTensor(rotation_axis, theta)

def wallAlignRotationTensor_walldata(wallnormal, cart_normal, plane='xy') -> np.ndarray:
    """Create rotation tensor for wall alligning quantities for all wall locations

    For 2D xy plane and streamwise x, cart_normal should be the y unit vector.

    Parameters
    ----------
    wallnormal : npts,numpy.ndarray
        The normal vector at the wall for all wall points
    cart_normal : numpy.ndarray
        The Cartesian equivalent to the wall normal. If the wall were "flat",
        this would be the wall normal vector.
    plane : str
        2D plane to rotate on, from 'xy', 'xz', 'yz'. (Default is 'xy')
    """

    if plane.lower() == 'xy':   rotation_axis = np.array([0, 0, 1])
    elif plane.lower() == 'xz': rotation_axis = np.array([0, 1, 0])
    elif plane.lower() == 'yz': rotation_axis = np.array([1, 0, 0])

    [npts,tmp] = np.shape(wallnormal)
    rotTensor = np.zeros([npts,3,3])

    for i in range(npts):
        wallnormal_pt = wallnormal[i]
        theta = np.arccos(np.dot(wallnormal_pt, cart_normal))
        # Determine whether clockwise or counter-clockwise rotation
        rotation_cross = np.cross(cart_normal, cart_normal - wallnormal_pt)
        if np.dot(rotation_cross, rotation_axis) < 0: theta *= -1
        rotTensor[i,:,:] = makeRotationTensor(rotation_axis, theta)

    return rotTensor

def calcdpds(pgrad,rotTensor) -> np.ndarray:
    """Calcuate the Pressure gradient

    Parameters
    ----------
    pgrad : numpy.ndarray
        Pressure gradient at the wall
    rotTensor: numpy.ndarray
        Rotation Tensor at the wall

    Returns
    -------
    numpy.ndarray
    """
    
    [npts,tmp] = np.shape(pgrad)
    dpds = np.zeros(np.shape(pgrad))
    
    for i in range(npts):
        rotation_tensor = rotTensor[i]
        dpds[i] =  np.dot(rotation_tensor,pgrad[i])

    return dpds

def calcarclength(normals,coord,xStart) -> np.ndarray:
    """Calcuate the Pressure gradient

    Parameters
    ----------
    normals : numpy.ndarray
        normal to the surface
    coord: numpy.ndarray
        coordinates of surface points
    xStart: 1
        starting x-value for integration

    Returns
    -------
    numpy.ndarray
    """
    
    normal_slope = normals[:,1]/normals[:,0]
    tangent = -1/normal_slope
    arg = np.sqrt(1 + tangent**2)
    npts = tangent.shape[0]
    arclength = np.zeros(npts)
    for i in range(npts):
        arclength[i] = np.trapz(arg[0:i],x=coord[0:i,0])

    return arclength

def calcdpdsInt(dpds,arclen,idStart,flag) -> np.ndarray:
    """Calcuate the Pressure gradient

    Parameters
    ----------
    normals : numpy.ndarray
        normal to the surface
    coord: numpy.ndarray
        coordinates of surface points
    xStart: 1
        starting x-value for integration

    Returns
    -------
    numpy.ndarray
    """
    
    npts = dpds.shape[0]
    dpds_int = np.zeros(npts)

    if flag < 2:
        for i in range(npts):
            if flag == 1:
                weight = 1 - (arclen[idStart:i]/(arclen[i]))
            else:
                weight = 1
                idStart = 0
            dpds_int[i] = np.trapz(weight*dpds[idStart:i],x=arclen[idStart:i])
            if(arclen[i] == 0):
                dpds_int[i] = dpds[i]
            else:
                dpds_int[i] = dpds_int[i]/arclen[i]
    else:
        for i in range(npts):
            if idStart == 0:
                raise(RuntimeError('Change mode as idStart=0'))
            
            if i-idStart <= 0:
                weight = 1
                dpds_int[i] = dpds[i]
            else:
                S = arclen[i]-arclen[i-idStart]
                weight = 1 - ((arclen[i-idStart:i] - arclen[i-idStart])/S)
                dpds_int[i] = np.trapz(weight*np.flip(dpds[i-idStart:i]),x=arclen[i-idStart:i]-arclen[i-idStart])
                denom = np.trapz(weight,x=arclen[i-idStart:i]-arclen[i-idStart])
                # print(dpds_int[i],denom)
                dpds_int[i] = dpds_int[i]/denom
                
    return dpds_int

def calcStreamlineRotation(vel) -> np.ndarray:
    """Calculate local rotation matrix based on streamlines
    
    Parameters
    ----------
    vel: numpy.ndarray
        Velocity vector
        
    Returns
    -------
    numpy.ndarray
    """
    
    npts = vel.shape[0]
    RotMatrix = np.zeros([npts,3,3])
    theta = np.arctan(np.divide(vel[:,1], vel[:,0], out=np.zeros_like(vel[:,1]), where=vel[:,0]!=0))
    RotMatrix[:,0,0] = np.cos(theta)
    RotMatrix[:,0,1] = np.sin(theta)
    RotMatrix[:,1,0] = -np.sin(theta)
    RotMatrix[:,1,1] = np.cos(theta)
    RotMatrix[:,2,2] = 1.
    
    RotMatrix = np.reshape(RotMatrix, [npts,9])
            
    return RotMatrix

def calcVelocityAlongStreamline(vel,RotMatrix) -> np.ndarray:
    """ Calculate velocity along the streamline coordinate system
    
    Parameters
    ----------
    vel: numpy.ndarray
        Velocity vector
        
    Returns
    -------
    numpy.ndarray
    """

    npts = vel.shape[0]
    velRot = np.zeros([npts,3])  
    RotMatrix = np.reshape(RotMatrix, [npts,3,3])
    
    for i in range(npts):
        velRot[i,:] = np.dot(RotMatrix[i,:,:],vel[i,:])
                             
    return velRot

    
def calcVelocityStreamGradient(dvelStreamdx, vel) -> np.ndarray:
    """ Calculate gradient of (velocity in streamline frame) along
    streamline coordinate system directions
    
    Parameters
    ----------
    dvelStreamdx: numpy.ndarray
        cartesian gradient of (velocity in streamline frame)
    vel: numpy.ndarray
        Velocity vector
        
    Returns
    -------
    numpy.ndarray
    """
    
    npts = vel.shape[0]
    theta = np.arctan(np.divide(vel[:,1], vel[:,0], out=np.zeros_like(vel[:,1]), where=vel[:,0]!=0))
    psi = np.zeros([npts,3])
    psi[:,0] = np.cos(theta)
    psi[:,1] = np.sin(theta)
    
    z = np.zeros([1,3])
    z[0,2] = 1
    phi = np.zeros([npts,3])
    for i in range(npts):
        phi[i,:] = np.cross(z,psi[i,:])
        
    dusds = np.zeros([npts,9])
    dusds[:,0] = np.einsum('ei,ei->e',dvelStreamdx[:,0:3],psi[:,:])
    dusds[:,3] = np.einsum('ei,ei->e',dvelStreamdx[:,3:6],psi[:,:])
    dusds[:,6] = np.einsum('ei,ei->e',dvelStreamdx[:,6:9],psi[:,:])
    dusds[:,1] = np.einsum('ei,ei->e',dvelStreamdx[:,0:3],phi[:,:])
    dusds[:,4] = np.einsum('ei,ei->e',dvelStreamdx[:,3:6],phi[:,:])
    dusds[:,7] = np.einsum('ei,ei->e',dvelStreamdx[:,6:9],phi[:,:])    
    dusds[:,2] = dvelStreamdx[:,2]
    dusds[:,5] = dvelStreamdx[:,5]
    dusds[:,8] = dvelStreamdx[:,8]    
    
    return(dusds)

def calcVelocityStreamHessian(d2usdsdx, vel) -> np.ndarray:
    """ Calculate hessian of (velocity in streamline frame) along
    streamline coordinate system directions
    
    Parameters
    ----------
    d2usdsdx: numpy.ndarray
        mixed cartesian and streamline hessian of (velocity in streamline frame)
    vel: numpy.ndarray
        Velocity vector
        
    Returns
    -------
    numpy.ndarray
    """
    
    npts = vel.shape[0]
    theta = np.arctan(np.divide(vel[:,1], vel[:,0], out=np.zeros_like(vel[:,1]), where=vel[:,0]!=0))
    psi = np.zeros([npts,3])
    psi[:,0] = np.cos(theta)
    psi[:,1] = np.sin(theta)
    
    z = np.zeros([1,3])
    z[0,2] = 1
    phi = np.zeros([npts,3])
    for i in range(npts):
        phi[i,:] = np.cross(z,psi[i,:])
        
    d2usds2 = np.zeros([npts,27])
    d2usds2[:,0] = np.einsum('ei,ei->e',d2usdsdx[:,0:3],psi[:,:])
    d2usds2[:,1] = np.einsum('ei,ei->e',d2usdsdx[:,0:3],phi[:,:])
    d2usds2[:,2] = d2usdsdx[:,2]
    d2usds2[:,3] = np.einsum('ei,ei->e',d2usdsdx[:,3:6],psi[:,:])
    d2usds2[:,4] = np.einsum('ei,ei->e',d2usdsdx[:,3:6],phi[:,:])
    d2usds2[:,5] = d2usdsdx[:,5]
    d2usds2[:,6] = np.einsum('ei,ei->e',d2usdsdx[:,6:9],psi[:,:])
    d2usds2[:,7] = np.einsum('ei,ei->e',d2usdsdx[:,6:9],phi[:,:])
    d2usds2[:,8] = d2usdsdx[:,8]
    d2usds2[:,9] = np.einsum('ei,ei->e',d2usdsdx[:,9:12],psi[:,:])
    d2usds2[:,10] = np.einsum('ei,ei->e',d2usdsdx[:,9:12],phi[:,:])
    d2usds2[:,11] = d2usdsdx[:,11]
    d2usds2[:,12] = np.einsum('ei,ei->e',d2usdsdx[:,12:15],psi[:,:])
    d2usds2[:,13] = np.einsum('ei,ei->e',d2usdsdx[:,12:15],phi[:,:])
    d2usds2[:,14] = d2usdsdx[:,14]
    d2usds2[:,15] = np.einsum('ei,ei->e',d2usdsdx[:,15:18],psi[:,:])
    d2usds2[:,16] = np.einsum('ei,ei->e',d2usdsdx[:,15:18],phi[:,:])
    d2usds2[:,17] = d2usdsdx[:,17]
    d2usds2[:,18] = np.einsum('ei,ei->e',d2usdsdx[:,18:21],psi[:,:])
    d2usds2[:,19] = np.einsum('ei,ei->e',d2usdsdx[:,18:21],phi[:,:])
    d2usds2[:,20] = d2usdsdx[:,20]
    d2usds2[:,21] = np.einsum('ei,ei->e',d2usdsdx[:,21:24],psi[:,:])
    d2usds2[:,22] = np.einsum('ei,ei->e',d2usdsdx[:,21:24],phi[:,:])
    d2usds2[:,23] = d2usdsdx[:,23]
    d2usds2[:,24] = np.einsum('ei,ei->e',d2usdsdx[:,24:27],psi[:,:])
    d2usds2[:,25] = np.einsum('ei,ei->e',d2usdsdx[:,24:27],phi[:,:])
    d2usds2[:,26] = d2usdsdx[:,26]

    
    return(d2usds2)


def calcPressureStreamGradient(dPdx, vel) -> np.ndarray:
    """ Calculate gradient of pressure along
    streamline coordinate system directions
    
    Parameters
    ----------
    dPdx: numpy.ndarray
        cartesian gradient of pressure
    vel: numpy.ndarray
        Velocity vector
        
    Returns
    -------
    numpy.ndarray
    """
    
    npts = vel.shape[0]
    theta = np.arctan(np.divide(vel[:,1], vel[:,0], out=np.zeros_like(vel[:,1]), where=vel[:,0]!=0))
    psi = np.zeros([npts,3])
    psi[:,0] = np.cos(theta)
    psi[:,1] = np.sin(theta)
    
    z = np.zeros([1,3])
    z[0,2] = 1
    phi = np.zeros([npts,3])
    for i in range(npts):
        phi[i,:] = np.cross(z,psi[i,:])
        
    dPds = np.zeros([npts,3])
    dPds[:,0] = np.einsum('ei,ei->e',dPdx,psi)
    dPds[:,1] = np.einsum('ei,ei->e',dPdx,phi)
    dPds[:,2] = dPdx[:,2]
    
    return(dPds)


def calcReynoldsStressAlongStreamline(RSS,RotMatrix) -> np.ndarray:
    """ Calculate Reynolds stress tensor along the streamline coordinate system
    
    Parameters
    ----------
    RSS: numpy.ndarray
        Reynolds stress tensor
        
    Returns
    -------
    numpy.ndarray
    """

    npts = RSS.shape[0]
    RotMatrix = np.reshape(RotMatrix, [npts,3,3])
    RSSRot = np.zeros([npts,6])  
    RSSi = np.zeros([3,3])
    for i in range(npts):
        RSSi[0,0] = RSS[i,0]
        RSSi[1,1] = RSS[i,1]
        RSSi[2,2] = RSS[i,2]
        RSSi[0,1] = RSS[i,3]
        RSSi[0,2] = RSS[i,4]
        RSSi[1,2] = RSS[i,5]
        RSSi[1,0] = RSS[i,3]
        RSSi[2,0] = RSS[i,4]
        RSSi[2,1] = RSS[i,5]
        RSSRottemp = np.dot(RotMatrix[i,:,:],np.dot(RSSi,np.transpose(RotMatrix[i,:,:])))
        RSSRot[i,0] = RSSRottemp[0,0]
        RSSRot[i,1] = RSSRottemp[1,1]
        RSSRot[i,2] = RSSRottemp[2,2]
        RSSRot[i,3] = RSSRottemp[0,1]
        RSSRot[i,4] = RSSRottemp[0,2]
        RSSRot[i,5] = RSSRottemp[1,2]
                             
    return RSSRot

    
def calcRSSGradientStreamline(dRSSstreamdx, vel) -> np.ndarray:
    """ Calculate gradient of (velocity in streamline frame) along
    streamline coordinate system directions
    
    Parameters
    ----------
    dRSSstreamdx: numpy.ndarray
        cartesian gradient of (Reynolds Stress Tensor in streamline frame)
    vel: numpy.ndarray
        Velocity vector
        
    Returns
    -------
    numpy.ndarray
    """
    
    npts = vel.shape[0]
    theta = np.arctan(np.divide(vel[:,1], vel[:,0], out=np.zeros_like(vel[:,1]), where=vel[:,0]!=0))
    psi = np.zeros([npts,3])
    psi[:,0] = np.cos(theta)
    psi[:,1] = np.sin(theta)
    
    z = np.zeros([1,3])
    z[0,2] = 1
    phi = np.zeros([npts,3])
    for i in range(npts):
        phi[i,:] = np.cross(z,psi[i,:])
        
    dRSSsds = np.zeros([npts,18])
    dRSSsds[:,0] = np.einsum('ei,ei->e',dRSSstreamdx[:,0:3],psi[:,:])
    dRSSsds[:,1] = np.einsum('ei,ei->e',dRSSstreamdx[:,0:3],phi[:,:])
    dRSSsds[:,2] = dRSSstreamdx[:,2]
    dRSSsds[:,3] = np.einsum('ei,ei->e',dRSSstreamdx[:,3:6],psi[:,:])
    dRSSsds[:,4] = np.einsum('ei,ei->e',dRSSstreamdx[:,3:6],phi[:,:])
    dRSSsds[:,5] = dRSSstreamdx[:,5]
    dRSSsds[:,6] = np.einsum('ei,ei->e',dRSSstreamdx[:,6:9],psi[:,:])
    dRSSsds[:,7] = np.einsum('ei,ei->e',dRSSstreamdx[:,6:9],phi[:,:])
    dRSSsds[:,8] = dRSSstreamdx[:,8]
    dRSSsds[:,9] = np.einsum('ei,ei->e',dRSSstreamdx[:,9:12],psi[:,:])
    dRSSsds[:,10] = np.einsum('ei,ei->e',dRSSstreamdx[:,9:12],phi[:,:])
    dRSSsds[:,11] = dRSSstreamdx[:,11]
    dRSSsds[:,12] = np.einsum('ei,ei->e',dRSSstreamdx[:,12:15],psi[:,:])
    dRSSsds[:,13] = np.einsum('ei,ei->e',dRSSstreamdx[:,12:15],phi[:,:])
    dRSSsds[:,14] = dRSSstreamdx[:,14]
    dRSSsds[:,15] = np.einsum('ei,ei->e',dRSSstreamdx[:,15:18],psi[:,:])
    dRSSsds[:,16] = np.einsum('ei,ei->e',dRSSstreamdx[:,15:18],phi[:,:])
    dRSSsds[:,17] = dRSSstreamdx[:,17]
    
    
    return(dRSSsds)    


def entirewallAlignRotationTensor(wallnormal, cart_normal, plane='xy') -> np.ndarray:
    """Create rotation tensor for wall alligning quantities

    For 2D xy plane and streamwise x, cart_normal should be the y unit vector.

    Parameters
    ----------
    wallnormal : numpy.ndarray
        The normal vector at the wall
    cart_normal : numpy.ndarray
        The Cartesian equivalent to the wall normal. If the wall were "flat",
        this would be the wall normal vector.
    plane : str
        2D plane to rotate on, from 'xy', 'xz', 'yz'. (Default is 'xy')
    """

    if plane.lower() == 'xy':   rotation_axis = np.array([0, 0, 1])
    elif plane.lower() == 'xz': rotation_axis = np.array([0, 1, 0])
    elif plane.lower() == 'yz': rotation_axis = np.array([1, 0, 0])

    theta = np.arccos(np.dot(wallnormal, cart_normal))
    # Determine whether clockwise or counter-clockwise rotation
    npts = wallnormal.shape[0]
    RotMatrix = np.zeros([npts,3,3])
    for i in range(npts):
        rotation_cross = np.cross(cart_normal, cart_normal - wallnormal[i,:])
        if np.dot(rotation_cross, rotation_axis) < 0: theta[i] *= -1
        RotMatrix[:,0,0] = np.cos(theta[i])
        RotMatrix[:,0,1] = np.sin(theta[i])
        RotMatrix[:,1,0] = -np.sin(theta[i])
        RotMatrix[:,1,1] = np.cos(theta[i])
        RotMatrix[:,2,2] = 1.
        

    return RotMatrix

def computewalltangent(wallnormal) -> np.ndarray:
    """Create wall tangent vector 

    Parameters
    ----------
    wallnormal : numpy.ndarray
        The normal vector at the wall
    """
    npts = wallnormal.shape[0]
    z = np.zeros([1,3])
    z[0,2] = 1
    walltan = np.zeros([npts,3])
    for i in range(npts):
        walltan[i,:] = np.cross(wallnormal[i,:],z)
                

    return walltan

def computemetrictensor(cellsize) -> np.ndarray:
    """Compute metric tensor locally

    Parameters
    ----------
    cellsize : numpy.ndarray
        Local size of the cells
    """
    npts = cellsize.shape[0]
    metricTensor = np.zeros([npts,6])
    metricTensor[:,0] = 4./(cellsize[:,0]*cellsize[:,0])
    metricTensor[:,1] = 4./(cellsize[:,1]*cellsize[:,1])
    metricTensor[:,2] = 4./(cellsize[:,2]*cellsize[:,2]) 
    metricTensor[:,3] = 4./(cellsize[:,0]*cellsize[:,1])
    metricTensor[:,4] = 4./(cellsize[:,0]*cellsize[:,2])
    metricTensor[:,5] = 4./(cellsize[:,1]*cellsize[:,2]) 
               
    return metricTensor

def computeTauM(vel,gijd,consts) -> np.ndarray:
    """Compute TauM 

    Parameters
    ----------
    velocity : numpy.ndarray
        Velocity 
    metricTensor : numpy.ndarray
        Metric tensor of the cells
    consts: 
        Set of constants used for computing tauM
    """
    c1 = consts[0]
    c2 = consts[1]
    f = consts[3]
    dt = consts[4]
    nu = consts[5]
    
    npts = vel.shape[0]
    timeterm = ((2*c1/dt)*np.ones([npts]))**2
    velterm = vel[:,0]*vel[:,0]*gijd[:,0] + vel[:,1]*vel[:,1]*gijd[:,1] + \
        vel[:,2]*vel[:,2]*gijd[:,2] + 2*(vel[:,0]*vel[:,1]*gijd[:,3] + \
        vel[:,0]*vel[:,2]*gijd[:,4] + vel[:,1]*vel[:,2]*gijd[:,5])
    viscterm = c2*f*(nu**2)*(gijd[:,0]*gijd[:,0] + gijd[:,1]*gijd[:,1] + \
        gijd[:,2]*gijd[:,2] + 2*(gijd[:,0]*gijd[:,1] + gijd[:,0]*gijd[:,2] + \
        gijd[:,1]*gijd[:,2]))
        
    tauM = (timeterm + velterm + viscterm)**(-1/2)
               
    return tauM


def computeTauC(gijd,tauM,consts) -> np.ndarray:
    """Compute TauM 

    Parameters
    ----------
    metricTensor : numpy.ndarray
        Metric tensor of the cells
    tauM : numpy.ndarray
        tauM used in stabilization
    consts: 
        Set of constants used for computing tauC
    """
    c1 = consts[0]
    c3 = consts[2]
    rho = consts[6]
    
    gii = gijd[:,0] + gijd[:,1] + gijd[:,2]
    tauC = c3*rho/(8*c1*gii*tauM)
              
    return tauC


def computeMomBalance(ui,duidxj,dpdxi,duidxjdxk,dRSSdxk,const):
    """ Compute momentum balance
    
    Parameters
    ----------
    ui : numpy.ndarray
        velocity
    duidxj : numpy.ndarray
        velocity gradient 
    dpdxi : numpy.ndarray
        pressure gradient
    duidxjdxk : numpy.ndarray
        velocity hessian
    dRSSdxk : numpy.ndarray
        gradient of Reynolds stress
    const:
        stabilization consts (also contains viscosity)
        
    """
    nu = const[5]
    rho = const[6]
    
    nps = ui.shape[0]
    balance = np.zeros([nps,4])
    
    balance[:,0] = duidxj[:,0] + duidxj[:,4]
    
    Advec1 = ui[:,0]*duidxj[:,0] + ui[:,1]*duidxj[:,1] + ui[:,2]*duidxj[:,2]
    pgrad = dpdxi[:,0]/rho
    visc = nu*(duidxjdxk[:,0] + duidxjdxk[:,4] + duidxjdxk[:,8])
    turb = (dRSSdxk[:,0] + dRSSdxk[:,10] + dRSSdxk[:,14])
    balance[:,1] = Advec1 + pgrad + turb - visc
    
    Advec1 = ui[:,0]*duidxj[:,3] + ui[:,1]*duidxj[:,4] + ui[:,2]*duidxj[:,5]
    pgrad = dpdxi[:,1]/rho
    visc = nu*(duidxjdxk[:,9] + duidxjdxk[:,13] + duidxjdxk[:,17])
    turb = -(dRSSdxk[:,4] + dRSSdxk[:,9] + dRSSdxk[:,17])
    balance[:,2] = Advec1 + pgrad + turb + visc

    Advec1 = ui[:,0]*duidxj[:,6] + ui[:,1]*duidxj[:,7] + ui[:,2]*duidxj[:,8]
    pgrad = dpdxi[:,2]/rho
    visc = nu*(duidxjdxk[:,18] + duidxjdxk[:,22] + duidxjdxk[:,26])
    turb = -(dRSSdxk[:,8] + dRSSdxk[:,12] + dRSSdxk[:,16])
    balance[:,3] = Advec1 + pgrad + turb + visc
    
    return(balance)
    

def computeTauSUPG(uj, Li, tauM) -> np.ndarray:
    """Compute TauM 

    Parameters
    ----------
    uj : numpy.ndarray
        Velocity vector
    Li : numpy.ndarray
        Residual vector
    tauM: 
        tauM stabilization time scale
    """
    
    nps = uj.shape[0]
    tauSUPG = np.zeros([nps,9])
    tauSUPG[:,0] = tauM*Li[:,0]*uj[:,0]
    tauSUPG[:,1] = tauM*Li[:,0]*uj[:,1]
    tauSUPG[:,2] = tauM*Li[:,0]*uj[:,2]
    tauSUPG[:,3] = tauM*Li[:,1]*uj[:,0]
    tauSUPG[:,4] = tauM*Li[:,1]*uj[:,1]
    tauSUPG[:,5] = tauM*Li[:,1]*uj[:,2]
    tauSUPG[:,6] = tauM*Li[:,2]*uj[:,0]
    tauSUPG[:,7] = tauM*Li[:,2]*uj[:,1]
    tauSUPG[:,8] = tauM*Li[:,2]*uj[:,2]
    
    return tauSUPG