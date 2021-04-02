import numpy as np
import vtk
from pathlib import Path
import pyvista as pv
import matplotlib.pyplot as plt

import vtkpytools as vpt

# matplotlib settings
plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': 'cmr10',
            'mathtext.fontset': 'cm',
            'axes.unicode_minus': False,
            'axes.labelsize': 14,
            'font.size': 11,
            'figure.dpi': 100,
            'lines.linewidth': 1.0,
            'axes.grid': True
})

# Universal flow parameters
Uref = 1.0
rho = 1
mu = 1.2502E-5
nu = mu/rho
delta = 0.009
L = 0.9144

#%% Load data

    # Dictionary of paths to data
paths = {
   'data': Path('./result/exampleMesh_10000.vtm'),
}

    # Load data from paths into dataBlocks dictionary
dataBlocks = {}
for key, path in paths.items():
    dataBlocks[key] = pv.MultiBlock(path.as_posix())

#%% Get profiles
    # Get profiles of the grid

    # Loading Cf, delta_nu, and other wall-based quantities before grabbing
    # profiles.  By calculating them before the profiles are generated, they
    # will be made available via `profile.walldata['Cf']`
for key, dataBlock in dataBlocks.items():
    dataBlocks[key]['wall']['Cf'] = vpt.calcCf(dataBlocks[key]['wall'], Uref, nu, rho)
    Tw = dataBlocks[key]['wall']['Cf'] * (0.5*rho*Uref**2)
    dataBlocks[key]['wall']['delta_nu'] = nu/np.sqrt(np.abs(Tw)/rho)


    # Generate points to sample the mesh.
wall_dists = vpt.getGeometricSeries(0.2, 5E-6, 1.02)

def getProfiles(dataBlocksDict, location, profile_wall_dists, Normal=None):
    """Helper function for getting profiles

    Parameters
    ----------
    dataBlocksDict : dictionary
        Dictionary containing vtm data blocks
    location : float
        X location to take profile from
    profile_wall_dists : array
        Distances from wall to sample
    Normal : array
        Override the wall-normal value
    """
    profiles = {}
    plane = vtk.vtkPlane()
    plane.SetNormal((1, 0, 0))
    plane.SetOrigin((location, 0, 0))
    for key, dataBlock in dataBlocks.items():
        profiles[key] = vpt.sampleDataBlockProfile(dataBlock, wall_dists, cutterobj=plane, normal=Normal)

    return profiles


    # Grab profiles at different locations and put in dictionary called profilesAtLocs
profilesAtLocs = {}
# profilesAtLocs[''] = getProfiles(dataBlocks, -0.548638, wall_dists, Normal=np.array([0, 1, 0]))
profilesAtLocs['-0.546'] = getProfiles(dataBlocks, -0.546, wall_dists)


    # Rotate the velocity and Reynolds stresses to be wall-aligned
for profkey, profiles in profilesAtLocs.items():
    for profile in profiles.values():
        rotation_tensor = vpt.wallAlignRotationTensor(profile.walldata['Normals'], np.array([0,1,0]))
        profile['Velocity_Wall'] = vpt.rotateTensor(profile['Velocity'], rotation_tensor)
        profile['ReynoldsStress_Wall'] = vpt.rotateTensor(profile['ReynoldsStress'], rotation_tensor)

#%% Plot profiles

dataBlockPlotKwargs = {
    'data':{'linestyle':'solid' ,'color':'red', 'label':r'Data'},
}
#%% Plot Velocity
for profilekey, profiles in profilesAtLocs.items():
    fig, ax = plt.subplots()

    for key, profile in profiles.items():
        ax.plot(profile['WallDistance']/profile.walldata['delta_nu'], profile['Velocity'][:,0]*nu/profile.walldata['delta_nu'], **dataBlockPlotKwargs[key])

    ax.legend()
    ax.set_xlabel(r'$n^+$')
    ax.set_ylabel(r"$u^+$")
    ax.set_xscale('log')
    ax.set_title(profilekey)
    plt.tight_layout()

#%% Plot stresses:

for profilekey, profiles in profilesAtLocs.items():
    fig, ax = plt.subplots()

    for key, profile in profiles.items():
        ax.plot(profile['WallDistance']/profile.walldata['delta_nu'], profile['ReynoldsStress_Wall'][:,3], **dataBlockPlotKwargs[key])
        # ax.plot(profile['WallDistance']/profile.walldata['delta_percent'], profile['ReynoldsStress_Wall'][:,3]/Uref**2, **dataBlockPlotKwargs[key])

    ax.legend()
    ax.set_xlabel(r'$n^+$')
    ax.set_ylabel(r"$\langle u'v' \rangle$")
    ax.set_xscale('log')
    ax.set_title(profilekey)
    plt.tight_layout()

#%% Plot Cf

fig, ax = plt.subplots()
removeindex = -1
for key in dataBlockPlotKwargs.keys():
    ax.plot(dataBlocks[key]['wall'].points[:removeindex,0], dataBlocks[key]['wall']['Cf'][:removeindex], **dataBlockPlotKwargs[key])

ax.set_axisbelow(True)
ax.legend()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r"$C_f$")
fig.tight_layout()
