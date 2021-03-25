import numpy as np
import pyvista as pv
import warnings
from .barfiletools import *
from scipy.integrate import cumtrapz
from numpy import ndarray

# For debugging only
import IPython.core.debugger as ipdb
# import ipdb

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
    nprofilesamples = len(sample_dists)
    ntotsamples = nlocations*nprofilesamples

    profiles = np.einsum('i,jk->jik', sample_dists, vectors)
    profiles += locations[:, None, :]
    profile_pnts = profiles.reshape(ntotsamples, 3)

    sample_pnts = pv.wrap(profile_pnts)
    sample_pnts = sample_pnts.sample(dataBlock['grid'])
    sample_pnts['WallDistance'] = np.tile(sample_dists, locations.shape[0])
    # Tracer()()
    return sample_pnts

def integratedVortBLThickness(vorticity, wall_distance, delta_percent=0.995,
                         delta_displace=False, delta_momentum=False) -> dict:
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

# def cumulativeIntegratedVorticity(sample_pnts)
#

def delta_momentum(U: ndarray, wall_distance: ndarray, nwallpnts: int) -> ndarray:
    samples_per_wallpnt = int(U.size/nwallpnts)

    U = U.reshape(nwallpnts, samples_per_wallpnt)
    wall_distance = wall_distance.reshape(nwallpnts, samples_per_wallpnt)

    Uedge = U[:, -1]
    # Uedge = np.ones((U.shape[0]))

    delta_mom = np.trapz((1 - U/Uedge[:,None])*(U/Uedge[:,None]), wall_distance, axis=1)
    # ipdb.set_trace()
    return delta_mom

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
        raise RuntimeError('Number of data points ({}) not evenly divisible by '
                           'nwallpnts ({}). Cannot reshape array.'.format(U.size, nwallpnts))

    samples_per_wallpnt = int(U.size/nwallpnts)

    U = U.reshape(nwallpnts, samples_per_wallpnt)
    wall_distance = wall_distance.reshape(nwallpnts, samples_per_wallpnt)

    if not Uedge:
        Uedge = U[:,-1]

    index = np.argmax(U > percent*Uedge[:, None], axis=1)
    W = np.arange(nwallpnts)
    slopes = (wall_distance[W, index] - wall_distance[W, index -1]) / (U[W, index] - U[W, index -1])

    return wall_distance[W, index-1] + (percent*Uedge - U[W, index-1])*slopes


# def velocityBLThickness(U, wall_distance, delta_percent=0.995,
#                          delta_displace=False, delta_momentum=False) -> dict:

#     Uedge = U[-1]

#     percent_index = np.argmax(U > delta_percent*Uedge)
#     delta_percent = np.interp(delta_percent*Uedge, U[percent_index-1:percent_index+1],
#                                     wall_distance[percent_index-1:percent_index+1])

#     delta_displace = np.trapz(1 - U/Uedge, wall_distance) if delta_displace else None
#     delta_mom = np.trapz((1 - U/Uedge)*(U/Uedge), wall_distance) if delta_mom else None

