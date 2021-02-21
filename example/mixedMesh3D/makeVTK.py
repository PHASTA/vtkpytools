#!/usr/bin/env python
import numpy as np
import vtk
from pathlib import Path
import pyvista as pv

## ---- Set path to this git repository ----
# (so that you can use a "custom" version easily)
import sys
sys.path.insert(0, '../../')

import vtkpytools as vpt

#%% ---- File inputs ----
# Change meshtype to either "HexPyrTetWedge" or 'TetWedge' to run either example
meshtype='TetWedge'

coordsPath = Path('./mixedMesh3D_{}.crd'.format(meshtype))
connecPath = Path('./mixedMesh3D_{}.cnn'.format(meshtype))

vtkPath = Path('./result/mixedmesh3d_{}.vtk'.format(meshtype))

#%% ---- Read in Files ----
coords = np.loadtxt(coordsPath, skiprows=1)
coords = coords[:,:]

connec = np.loadtxt(connecPath, dtype=np.int64, skiprows=1)
connec = connec[:,:] - 1 # node index must be index-by-0, so subtracting one from my index-by-1 array

# The coords and connec arrays should be [nen, 3] and [nen, nnodes per element] in shape
# Feel free to check via 'coords.shape' or 'connect.shape'.

#%% ---- Building the VTK Grid object ----
    # Create the 2D grid from the given information
grid = vpt.form3DGrid(coords, connectivity_array=connec)
grid.save(vtkPath)
