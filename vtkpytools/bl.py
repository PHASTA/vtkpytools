import numpy as np
import pyvista as pv
import warnings
from .barfiletools import *
from scipy.integrate import cumtrapz


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


# def velocityBLThickness(U, wall_distance, delta_percent=0.995,
#                          delta_displace=False, delta_momentum=False) -> dict:

#     Uedge = U[-1]

#     percent_index = np.argmax(U > delta_percent*Uedge)
#     delta_percent = np.interp(delta_percent*Uedge, U[percent_index-1:percent_index+1],
#                                     wall_distance[percent_index-1:percent_index+1])

#     delta_displace = np.trapz(1 - U/Uedge, wall_distance) if delta_displace else None
#     delta_mom = np.trapz((1 - U/Uedge)*(U/Uedge), wall_distance) if delta_mom else None

