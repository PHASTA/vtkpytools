import numpy as np
import pyvista as pv
from .barfiletools import *
from scipy.integrate import cumtrapz
from numpy import ndarray


def sampleAlongVectors(dataBlock, sample_dists, vectors, locations) -> pv.PolyData:
    """Sample dataBlock at sample_dists away from locations in vectors direction

    Each nth location in 'locations' corresponds with the nth vector in
    'vectors'. At each location, a sample is take at each sample_dist in
    'sample_dists' in the direction of the location's corresponding vector.

    In other words, a "profile" is taken at each location. The vector defines
    the direction the profile is taken (with respect to the location itself)
    and sample_dists defines sample locations along that direction.

    Parameters
    ----------
    dataBlock : pyvista.MultiBlock
        dataBlock from where data will be interpolated. Must contain 'grid' and 'wall' VTK objects.
    sample_dists : (S) ndarray
        The distances away from location along the vectors
    vectors : (L, 3) ndarray
        Unit vectors that define the direction the samples should be taken away
        from location.
    locations : (L, 3) ndarray
        Coordinates from which samples should start from

    Returns
    -------
    samples: pv.PolyData
        Contains L*S data points in S-major order (ie. [:S] contains the all
        the samples associated with location[0])
    """

    nlocations = locations.shape[0]
    nprofilesamples = sample_dists.size
    ntotsamples = nlocations*nprofilesamples

    profiles = np.einsum('i,jk->jik', sample_dists, vectors)
    profiles += locations[:, None, :]
    profile_pnts = profiles.reshape(ntotsamples, 3)

    sample_pnts = pv.wrap(profile_pnts)
    sample_pnts = sample_pnts.sample(dataBlock['grid'])
    sample_pnts['WallDistance'] = np.tile(sample_dists, locations.shape[0])
    return sample_pnts


def delta_vortInt(vorticity, wall_distance, nwallpnts: int, displace=False,
                  momentum=False, returnUvort=False, Uedge=None) -> dict:
    """Calculate vorticity-integrated BL thicknesses (momentum and displacement)

    Based on eqns 3.1-3.4 in "Numerical study of turbulent separation bubbles
    with varying pressure gradient and Reynolds number" Coleman et. al. 2018

    First, we define U as the cumulative integral of vorticity from the wall:

        U_i(y) = \int_0^y vorticity_i(j) dj

    where i indexes the wall point and y is distance from the wall. This is
    Uvort (which is returned by setting 'returnUvort' to True) and can be used
    to find delta_percent.

        result = vpt.delta_vortInt(...., returnUvort=True)
        delta_percent_vorticity = delta_percent(U=result['Uvort'], ....)

    Displacement thickness, d_i, is defined as:

       d_i = -1/Uedge_i \int y * vorticity_i(y) dy

    over the full profile height, with i indexing the wall point.

    Momentum thickness, m_i, is defined as:

       m_i = -2/Uedge_i^2 \int [y * U_i(y) * vorticity_i(y)]dy - d_i

    over the full profile height, with i indexing the wall point. Note that U_i
    is the vorticity-defined velocity defined above.

    Parameters
    ----------
    vorticity : [nwallpnts*nprofilesamples] ndarray
        Spanwise component of vorticity of the flow.
    wall_distance : [nwallpnts*nprofilesamples] ndarray
        Distance to wall for all the sample points
    nwallpnts : int
        Number of wall locations used in the sampling of U. The size of U and
        wall_distance must be evenly divisible by nwallpnts.
    displace : bool, optional
        Whether to calculate displacement thickness
    momentum : bool, optional
        Whether to calcualte momentum thickness
    Uedge : [nwallpnts] ndarray, optional
        Sets the values for the edge (vorticity-integrated) velocity used in calculating the boundary
        layer height. If not given, the last sample point for each wall point
        profile will be used (Default: None).

    Returns
    -------
    Dictionary with the following items optionally defined:

    delta_displace: ndarray, optional
        Displacement thickness. Not passed if displace=False

    delta_momentum: ndarray, optional
        Momentum thickness. Not passed if momentum=False

    Uvort: ndarray, optional
        Vorticity-integrated velocity. Not passed if returnUvort=False

    """

    if vorticity.size % nwallpnts != 0:
        raise ValueError('Number of data points ({}) not evenly divisible by '
                         'nwallpnts ({}). Cannot reshape array.'.format(vorticity.size, nwallpnts))

    vorticity = vorticity.reshape(nwallpnts, -1)
    wall_distance = wall_distance.reshape(nwallpnts, -1)

    U = -cumtrapz(vorticity, wall_distance, initial=0, axis=1)
    Uedge = U[:, -1] if Uedge is None else Uedge

    delta_displace_lambda = lambda: -(1/Uedge) * np.trapz(wall_distance*vorticity, wall_distance, axis=1)

    results = {}
    if displace:
        delta_displace = delta_displace_lambda()
        results['delta_displace'] = delta_displace
    if momentum:
        delta_momentum = -(2/Uedge**2) * np.trapz(U*wall_distance*vorticity, wall_distance, axis=1)
        delta_momentum -= delta_displace if displace else delta_displace_lambda()

        results['delta_momentum'] = delta_momentum
    if returnUvort:
        results['Uvort'] = U
    return results


def delta_velInt(U, wall_distance, nwallpnts: int,
                 displace: bool = False, momentum: bool = False, Uedge=None) -> dict:
    """Calculate velocity-integrated BL thicknesses (momentum and displacement)

    Displacement thickness, d_i,  defined as integrating:

        (1 - U_i(wall_distance)/Uedge_i)

    over the full profile height, with i indexing the wall point.

    Momentum thickness, m_i,  defined as integrating:

        (1 - U_i(wall_distance)/Uedge_i) * (U_i(wall_distance)/Uedge_i)

    over the full profile height, with i indexing the wall point.

    Parameters
    ----------
    U : [nwallpnts*nprofilesamples] ndarray
        Quantity to base boundary layer height on (generally streamwise
        velocity)
    wall_distance : [nwallpnts*nprofilesamples] ndarray
        Distance to wall for all the sample points
    nwallpnts : int
        Number of wall locations used in the sampling of U. The size of U and
        wall_distance must be evenly divisible by nwallpnts.
    displace : bool, optional
        Whether to calculate displacement thickness
    momentum : bool, optional
        Whether to calcualte momentum thickness
    Uedge : [nwallpnts] ndarray, optional
        Sets the values for the edge velocity used in calculating the boundary
        layer height. If not given, the last sample point for each wall point
        profile will be used (Default: None).

    Returns
    -------
    Dictionary with the following items optionally defined:

    delta_displace: ndarray, optional
        Displacement thickness. Not passed if displace=False

    delta_momentum: ndarray, optional
        Momentum thickness. Not passed if momentum=False
    """

    if U.size % nwallpnts != 0:
        raise ValueError('Number of data points ({}) not evenly divisible by '
                         'nwallpnts ({}). Cannot reshape array.'.format(U.size, nwallpnts))

    U = U.reshape(nwallpnts, -1)
    wall_distance = wall_distance.reshape(nwallpnts, -1)

    Uedge = U[:, -1] if Uedge is None else Uedge

    delta_displace_lambda = lambda : np.trapz((1 - U/Uedge[:,None]), wall_distance, axis=1)
    delta_momentum_lambda = lambda : np.trapz((1 - U/Uedge[:,None])*(U/Uedge[:,None]), wall_distance, axis=1)

    if displace and not momentum:
        return {'delta_displace': delta_displace_lambda()}
    elif not displace and momentum:
        return {'delta_momentum': delta_momentum_lambda()}
    elif displace and momentum:
        return {'delta_displace': delta_displace_lambda(), 'delta_momentum': delta_momentum_lambda()}
    else:
        return {}


def delta_percent(U, wall_distance, nwallpnts: int, percent: float, Uedge=None) -> ndarray:
    """Calculate the boundary layer height based on percentage of U

    Define the percent boundary layer thickness as h_i such that

        U_i(h_i) = percent*Uedge_i

    for U_i(wall_distance) and i indexing every wall point.

    Parameters
    ----------
    U : [nwallpnts*nprofilesamples] ndarray
        Quantity to base boundary layer height on (generally streamwise
        velocity)
    wall_distance : [nwallpnts*nprofilesamples] ndarray
        Distance to wall for all the sample points
    nwallpnts : int
        Number of wall locations used in the sampling of U. The size of U and
        wall_distance must be evenly divisible by nwallpnts.
    percent : float
        The percentage of the "edge" value that defines the boundary layer
        height.
    Uedge : [nwallpnts] ndarray, optional
        Sets the values for the edge velocity used in calculating the boundary
        layer height. If not given, the last sample point for each wall point
        profile will be used (Default: None).

    Returns
    -------
    delta_percent: [nwallpnts] ndarray
    """

    if U.size % nwallpnts != 0:
        raise ValueError('Number of data points ({}) not evenly divisible by '
                         'nwallpnts ({}). Cannot reshape array.'.format(U.size, nwallpnts))

    U = U.reshape(nwallpnts, -1)
    wall_distance = wall_distance.reshape(nwallpnts, -1)

    Uedge = U[:, -1] if Uedge is None else Uedge

    index = np.argmax(U > percent*Uedge[:, None], axis=1)
    W = np.arange(nwallpnts)
    slopes = (wall_distance[W, index] - wall_distance[W, index -1]) / (U[W, index] - U[W, index -1])

    return wall_distance[W, index-1] + (percent*Uedge - U[W, index-1])*slopes

def compute_Ue(U, wall_distance, deltaBL, nwallpnts: int) -> ndarray:
    """Calculate the velocity at the edge of BL based on computed BL height

    Parameters
    ----------
    U : [nwallpnts*nprofilesamples] ndarray
        Quantity to base boundary layer height on (generally streamwise
        velocity)
    wall_distance : [nwallpnts*nprofilesamples] ndarray
        Distance to wall for all the sample points
    deltaBL : [nprofilesamples] ndarray
        BL thickness         
    nwallpnts : int
        Number of wall locations used in the sampling of U. The size of U and
        wall_distance must be evenly divisible by nwallpnts.

    Returns
    -------
    delta_percent: [nwallpnts] ndarray
    """
    if U.size % nwallpnts != 0:
        raise ValueError('Number of data points ({}) not evenly divisible by '
                         'nwallpnts ({}). Cannot reshape array.'.format(U.size, nwallpnts))

    U = U.reshape(nwallpnts, -1)
    wall_distance = wall_distance.reshape(nwallpnts, -1)

    [nstream,tmp] = U.shape
    
    U_e = np.zeros(deltaBL.size)
    for i in range(nstream):
        ind = np.argmin(np.abs(wall_distance[i,:] - deltaBL[i]*np.ones([nwallpnts,1])))
        U_e[i] = U[i,ind]

    return U_e

def integratedVortBLThickness(vorticity, wall_distance, delta_percent=0.995,
                         delta_displace=False, delta_momentum=False) -> dict:
    """(DEPRECATED) Computes vorticity-integrated BL thickness for a profile"""
        # Use eqns 3.1-3.4 in Numerical study of turbulent separation bubbles with varying pressure gradient and Reynolds number
    U = -cumtrapz(vorticity, wall_distance, initial=0)
    Ue = U[-1]

    percent_index = np.argmax(U > delta_percent*Ue)
    delta_percent = np.interp(delta_percent*Ue, U[percent_index-1:percent_index+1],
                                    wall_distance[percent_index-1:percent_index+1])

    delta_displace_lambda = lambda: -(1/Ue) * np.trapz(wall_distance*vorticity, wall_distance)
    if delta_displace:
        delta_displace = delta_displace_lambda()
        if delta_momentum:
            delta_momentum = -(2/Ue**2) * np.trapz(U*wall_distance*vorticity, wall_distance)
            delta_momentum -= delta_displace
    elif delta_momentum:
        delta_momentum = -(2/Ue**2) * np.trapz(U*wall_distance*vorticity, wall_distance)
        delta_momentum -= delta_displace_lambda()

    return {'delta_percent':delta_percent, 'delta_displace':delta_displace, 'delta_momentum':delta_momentum}
