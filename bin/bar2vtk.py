#!/usr/bin/env python
import argparse
from pathlib import Path
import os

Description=""" Create a new VTK file from the *bar files and an existing VTK file of
the mesh.

Examples:
\tbar2vtk.py blankDataBlock.vtm BinaryBars 10000
\tbar2vtk.py blankDataBlock.vtm BinaryBars 10000-20000 --ts0=500

Time Step Windows:
------------------
Submit a timestep argument with a '-' in it to request a timestep window be
generated. This requires a '--ts0' argument be provided as well for calculating
the windowed value."""

## Parsing script input
class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """To display defaults in help and have a multiline help description"""
    # Shamelessly ripped from https://stackoverflow.com/a/18462760/7564988
    pass

parser = argparse.ArgumentParser(description=Description,
                                 formatter_class=CustomFormatter,
                                 prog='bar2vtk')
parser.add_argument('vtkfile', help='MultiBlock VTK file that contains grid and wall', type=Path)
parser.add_argument('barfiledir', help='Path to *bar file directory', type=Path)
parser.add_argument('timestep', help='Timestep of the barfiles. May be range', type=str)
parser.add_argument('--ts0','--bar-average-start',
                    help='Starting timestep of the averaging process. Only used for generating windows.',
                    type=int, default=-1)
parser.add_argument('-f','--new-file-prefix',
                    help='Prefix for the new file. Will have timestep appended.',
                    type=str)
parser.add_argument('--velonly', help='Only process velocity', action='store_true')
parser.add_argument('--vptpath', help='Custom path to vtkpytools package', type=Path)

args = parser.parse_args()

import numpy as np
import vtk
import pyvista as pv

# import the package in this repository
import sys
if args.vptpath:
    vptPath = args.vptpath
else:
    vptPath = Path(__file__).resolve().parent.parent
sys.path.insert(0, vptPath.as_posix())

import vtkpytools as vpt

## ---- Process/check script arguments
assert args.vtkfile.is_file()
assert args.barfiledir.is_dir()

if args.new_file_prefix:
    vtmPath = Path(args.new_file_prefix + '_' + args.timestep + '.vtm')
else:
    vtmPath = Path(os.path.splitext(args.vtkfile)[0] + '_' + args.timestep + '.vtm')

## ---- Loading data arrays
if '-' in args.timestep:
# Create timestep windows
    if args.ts0 == -1:
        raise RuntimeError("Starting timestep of bar field averaging required (--ts0)")
    # raise NotImplementedError("Haven't implemented creating time windows yet")

    timesteps = [int(x) for x in args.timestep.split('-')]
    print(f'Creating timewindow between {timesteps[0]} and {timesteps[1]}')
    velbarPaths = []; stsbarPaths = []
    for timestep in timesteps:
        velbarPath = list(args.barfiledir.glob(f'velbar*{timestep}*'))
        if len(velbarPath) > 0:
            assert velbarPath[0].is_file()
            velbarPaths.append(velbarPath[0])
        else:
            raise RuntimeError(f'Could not find file matching "velbar*{timestep}*" in {args.barfiledir}')

        if not args.velonly:
            stsbarPath = list(args.barfiledir.glob(f'stsbar*{timestep}*'))
            if len(stsbarPath) > 0:
                assert stsbarPath[0].is_file()
                stsbarPaths.append(stsbarPath[0])
            else:
                raise RuntimeError(f'Could not find file matching "stsbar*{timestep}*" in {args.barfiledir}')

    print(f'Using data files:\n\t{velbarPaths[0]}\t{velbarPaths[1]}')
    if not args.velonly:
        print(f'\t{stsbarPaths[0]}\t{stsbarPaths[1]}')

    velbarArrays = []; stsbarArrays = []
    for i in range(2):
        velbarArrays.append(vpt.binaryVelbar(velbarPaths[i]))
        if not args.velonly:
            stsbarArrays.append(vpt.binaryStsbar(stsbarPaths[i]))

    velbarArray = (velbarArrays[1]*(timesteps[1] - args.ts0) -
                   velbarArrays[0]*(timesteps[0] - args.ts0)) / (timesteps[1] - timesteps[0])
    if not args.velonly:
        stsbarArray = (stsbarArrays[1]*(timesteps[1] - args.ts0) -
                       stsbarArrays[0]*(timesteps[0] - args.ts0)) / (timesteps[1] - timesteps[0])
    print('Finished computing timestep window')
else:
# Don't create timestep windows
    velbarPath = list(args.barfiledir.glob(f'velbar*{args.timestep}*'))
    if len(velbarPath) > 0:
        assert velbarPath[0].is_file()
    else:
        raise RuntimeError(f'Could not find file matching "velbar*{timestep}*" in {args.barfiledir}')
    print(f'Using data files:\n\t{velbarPath[0]}')
    velbarArray = vpt.binaryVelbar(velbarPath[0])

    if not args.velonly:
        stsbarPath = list(args.barfiledir.glob(f'stsbar*{args.timestep}*'))
        if len(stsbarPath) > 0:
            assert stsbarPath[0].is_file()
        else:
            raise RuntimeError(f'Could not find file matching "stsbar*{timestep}*" in {args.barfiledir}')
        print(f'\t{stsbarPath[0]}')
        stsbarArray = vpt.binaryStsbar(stsbarPath[0])

## ---- Load DataBlock
dataBlock = pv.MultiBlock(args.vtkfile.as_posix())
grid = dataBlock['grid']
wall = dataBlock['wall']

## ---- Load *bar data into dataBlock
grid['Pressure'] = velbarArray[:,0]
grid['Velocity'] = velbarArray[:,1:4]

if not args.velonly:
    ReyStrTensor = vpt.calcReynoldsStresses(stsbarArray, velbarArray)
    grid['ReynoldsStress'] = ReyStrTensor
    grid['TurbulentEnergyKinetic'] = (1/3)*(np.sum(ReyStrTensor[:,0:3], axis=1))

grid = grid.compute_gradient(scalars='Velocity')
grid = vpt.compute_vorticity(grid, scalars='Velocity')

## ---- Copy data from grid to wall object
wall = wall.sample(grid)

dataBlock['grid'] = grid
dataBlock['wall'] = wall
print(f'Saving dataBlock file to: {vtmPath}', end='')
dataBlock.save(vtmPath)
print('\tDone!')
