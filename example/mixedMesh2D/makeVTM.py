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
coordsPath = Path('mixedmesh.crd')
connecPath = Path('mixedmesh.ien')

vtmPath = Path('result/mixedmesh.vtm')

#%% ---- Building the VTK Grid object ----
coords = np.loadtxt(coordsPath)
coords = coords[:,:2]

connec = np.loadtxt(connecPath, dtype=np.int64)
connec = connec[:,1:] -1

    # Create the 2D grid from the given information
grid = vpt.form2DGrid(coords, connectivity_array=connec)

    # Get wall edge from grid
featedges = grid.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                       feature_edges=False, manifold_edges=False)

    # Second input is point internal to domain, helping ensure that wall normals point inward
featedges = vpt.computeEdgeNormals(featedges, np.array([0.5, 0.75, 0]))

    # Extract out cells based on their normals, may need adjustment per case
wall = featedges.extract_cells(np.arange(featedges.n_cells)[featedges['Normals'][:,1] > 0.8])
wall = vpt.unstructuredToPoly(wall.cell_data_to_point_data())
wall = vpt.orderPolyDataLine(wall)

# At this point you have two objects:
#   - grid: 2D grid object for the mesh
#   - wall: A line representing the wall

dataBlock = pv.MultiBlock()
dataBlock['grid'] = grid
dataBlock['wall'] = wall
dataBlock.save(vtmPath)
