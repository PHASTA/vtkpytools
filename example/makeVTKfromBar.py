#!/usr/bin/env python
import numpy as np
import vtk
from pathlib import Path
import pyvista as pv

## ---- Set path wvelbar_patcoords_path----
# (so that you can use a "custom" version easily)
import sys
sys.path.insert(0, '../')

import vtkpytools as vpt

#%% ---- File inputs ----
    # Existing files
coordsPath = Path('meshFiles/TruncBump2d_woNPoints.crd')
connecPath = Path('meshFiles/TruncBump2d_Squares.cnn')
velbarPath = Path('data/velbar.10000.1')
stsbarPath = Path('data/stsbar.10000.1')

    # New file names/paths
vtuPath = Path('result/exampleMesh-10000.vtu')
vtmPath = Path('result/exampleMesh-10000.vtm')

#%% ---- Building the VTK Grid object ----
    # Create the 2D grid from the given information
grid = vpt.form2DGrid(coordsPath, connectivity_path=connecPath, connectivity_zero_base=True)

    # Get wall edge from grid
featedges = grid.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                       feature_edges=False, manifold_edges=False)

    # Second input is point internal to domain, helping ensure that wall normals point inward
featedges = vpt.computeEdgeNormals(featedges, np.array([-0.42, 0.2, 0]))

    # Extract out cells based on their normals, may need adjustment per case
wall = featedges.extract_cells(np.arange(featedges.n_cells)[featedges['Normals'][:,1] > 0.8])
wall = vpt.unstructuredToPoly(wall.cell_data_to_point_data())
wall = vpt.orderPolyDataLine(wall)

# At this point you have two objects:
#   - grid: 2D grid object for the mesh
#   - wall: A line representing the wall

#%% ---- Load data onto VTK Grid ----
velbarArray = vpt.binaryVelbar(velbarPath)
stsbarArray = vpt.binaryStsbar(stsbarPath)

ReyStrTensor = vpt.calcReynoldsStresses(stsbarArray, velbarArray)

grid['Pressure'] = velbarArray[:,0]
grid['Velocity'] = velbarArray[:,1:4]
grid['ReynoldsStress'] = ReyStrTensor
grid['TurbulentEnergyKinetic'] = 1/3*(np.sum(ReyStrTensor[:,0:3], axis=1))

## For some reason, calculating gradients from contructed UnstructuredGrid object fails.
#   Workaround is to save and reload file. See github.com/pyvista/pyvista-support/issues/204
grid.save(vtuPath)
grid = pv.UnstructuredGrid(vtuPath.as_posix())

print('compute_gradient')
grid = grid.compute_gradient(scalars='Velocity')
print('compute_vorticity')
grid = vpt.compute_vorticity(grid, scalars='Velocity')

## Copy data from grid to wall object
wall = wall.sample(grid)

#%% ---- Save grid and wall to vtkMultiBlock file
## Combine grid and wall data into single object
dataBlock = pv.MultiBlock()
dataBlock['grid'] = grid
dataBlock['wall'] = wall
dataBlock.save(vtmPath)

#%% ---- Extracting Profiles from vtkMultiBlock files
wall = dataBlock['wall']

## Calculate BL height
# Set sample line parameters
line_height = 0.1
line_growthrate = 1.05
line_initpoint = 2E-6
line_walldists = vpt.getGeometricSeries(0.1, 1.02, 2E-6)

## Extract profiles by pointid, using the closest wall point to a given location
example_profile = vpt.sampleDataBlockProfile(dataBlock, line_walldists,
                                         pointid=dataBlock['wall'].find_closest_point((-0.4,0,0)))

## Extract profiles by specifying an plane intersection using vtkCutter
plane = vtk.vtkPlane()
plane.SetNormal((1, 0, 0))
plane.SetOrigin((-0.4, 0, 0))
example_profile = vpt.sampleDataBlockProfile(dataBlock, line_walldists, cutterobj=plane)

# example_profile is a pv.PolyData object containing the sampled data from the grid
