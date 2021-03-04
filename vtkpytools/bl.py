import numpy as np
import pyvista as pv
import warnings
from .barfiletools import *
from scipy.integrate import cumtrapz
from numpy import ndarray

# For debugging only
import IPython.core.debugger as ipdb
# import ipdb


def calcBoundaryLayerStats(dataBlock, line_walldists, bl_functions, velocity_name, normals):

    # Create all the line points at once
    # Create a PolyData object from JUST the points vtk.PolyData(points)
    # Sample to the points from the grid

    # wall = dataBlock['wall']
    # for i, point in enumerate(wall.points):
    #     profile = sampleDataBlockProfile(dataBlock, line_walldists, normal=normals[i])
    #     for function in bl_functions:
    #         outputs[name] = function(profile)

    # dataBlock['wall']['delta_displace'] = delta_displace
    # dataBlock['wall']['delta_mom'] = delta_mom
    # dataBlock['wall']['Re_theta'] = Re_theta

    return dataBlock

def sampleBLStuff(dataBlock, sample_dists, vectors, locations) -> pv.PolyData:

    nlocations = locations.shape[0]
    ntotsamples = nlocations*len(sample_dists)
    nprofilesamples = len(sample_dists)
    profiles = np.einsum('i,jk->jik', sample_dists, vectors)
    profiles += locations[:, None, :]
    profile_pnts = profiles.reshape(ntotsamples, 3)

    sample_pnts = pv.wrap(profile_pnts)
    sample_pnts = sample_pnts.sample(dataBlock['grid'])
    sample_pnts['WallDistance'] = np.repeat(sample_dists, locations.shape[0])
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


# def velocityBLThickness(U, wall_distance, delta_percent=0.995,
#                          delta_displace=False, delta_momentum=False) -> dict:

#     Uedge = U[-1]

#     percent_index = np.argmax(U > delta_percent*Uedge)
#     delta_percent = np.interp(delta_percent*Uedge, U[percent_index-1:percent_index+1],
#                                     wall_distance[percent_index-1:percent_index+1])

#     delta_displace = np.trapz(1 - U/Uedge, wall_distance) if delta_displace else None
#     delta_mom = np.trapz((1 - U/Uedge)*(U/Uedge), wall_distance) if delta_mom else None

