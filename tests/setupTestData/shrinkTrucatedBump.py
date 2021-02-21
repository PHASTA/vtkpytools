import pyvista as pv
from pathlib import Path
import numpy as np
from scipy.io import FortranFile

#%% Script Inputs
exampleDirectory = Path('../../example/bar2vtk/')

# Original file Paths
origCrdPath = exampleDirectory / 'meshFiles/TruncBump2d_woNPoints.crd'
origVelbarPath = exampleDirectory / 'data/velbar.10000.1'
origStsbarPath = exampleDirectory / 'data/stsbar.10000.1'

# New file paths
newCrdPath = exampleDirectory / 'meshFiles/geom.crd'
newVelbarPath = exampleDirectory / 'datanew/velbar.10000.1'
newStsbarPath = exampleDirectory / 'datanew/stsbar.10000.1'

# Script Settings
nyPnts = 280 # property of the mesh
nxPnts = 20 # set by us to shrink the mesh

#%% Make new coordinate file
origCrd = np.loadtxt(origCrdPath)

newCrd = origCrd[0:nxPnts*nyPnts, :]

np.savetxt(newCrdPath, newCrd)

#%% Make new *bar files

origVelbarObj = FortranFile(origVelbarPath)
origVelbar = origVelbarObj.read_reals()
newVelbarObj = FortranFile(newVelbarPath, 'w')
newVelbarObj.write_record(origVelbar[0:5*nxPnts*nyPnts])

origStsbarObj = FortranFile(origStsbarPath)
origStsbar = origStsbarObj.read_reals()
newstsbarObj = FortranFile(newStsbarPath, 'w')
newstsbarObj.write_record(origStsbar[0:6*nxPnts*nyPnts])
