#!/usr/bin/env python
import numpy as np
import vtk
from pathlib import Path
import pyvista as pv

## ---- Set path to this git repository ----
# (so that you can use a "custom" version easily)
import sys
sys.path.insert(0, '../')

import vtkpytools as vpt

# ---- File inputs ----
vtmPath = Path('result/exampleMesh_10000.vtm')

# Reading in data
dataBlock = pv.MultiBlock(vtmPath.as_posix())

# ---- Calculating Cf profile
Uref = 16.4
nu = 1.5E-5
rho = 1
    # Save Cf values to numpy array
Cf = vpt.calcCf(dataBlock['wall'], Uref=Uref, nu=nu, rho=rho)
    # Or save them directly to the wall
dataBlock['wall']['Cf'] = vpt.calcCf(dataBlock['wall'], Uref=Uref, nu=nu, rho=rho)

# ---- Extracting Profiles from dataBlock files

## Calculate BL height
# Set sample line parameters
line_height = 0.1
line_growthrate = 1.05
line_initpoint = 2E-6
line_walldists = vpt.getGeometricSeries(line_height, line_initpoint, line_growthrate)

## Extract profiles by pointid, using the closest wall point to a given location
example_profile = vpt.sampleDataBlockProfile(dataBlock, line_walldists,
                                         pointid=dataBlock['wall'].find_closest_point([-0.4,0,0]))

## Extract profiles by specifying an plane intersection using vtkCutter
plane = vtk.vtkPlane()
plane.SetNormal((1, 0, 0))
plane.SetOrigin((-0.4, 0, 0))
example_profile = vpt.sampleDataBlockProfile(dataBlock, line_walldists, cutterobj=plane)

# example_profile is a pv.PolyData object containing the sampled data from the grid
