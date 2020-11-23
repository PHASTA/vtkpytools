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
    # Existing files
coordsPath = Path('./mixedMesh3D.crd')
connecPath = Path('./mixedMesh3D.cnn')

vtkPath = Path('result/mixedmesh3d.vtk')

#%% ---- Building the VTK Grid object ----
coords = np.loadtxt(coordsPath, skiprows=1)
coords = coords[:,:]

connec = np.loadtxt(connecPath, dtype=np.int64, skiprows=1)
connec = connec[:,:] -1

    # Create the 2D grid from the given information
grid = vpt.form3DGrid(coords, connectivity_array=connec)
grid.save(vtkPath)

