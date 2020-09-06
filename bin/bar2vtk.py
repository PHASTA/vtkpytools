#!/usr/bin/env python3
import argparse
from pathlib import Path
import os

Description=""" Create a new VTK file from the *bar files and an existing VTK file of
the mesh.

Examples:
\tbar2vtk.py blankDataBlock.vtm BinaryBars 10000
\tbar2vtk.py blankDataBlock.vtm BinaryBars 10000-20000 --ts0=500

The name of the output file will be the same as the blank VTM file suffixed
with the timestep requested. So in the first example above, the output would be
"blankDataBlock_10000.vtm".

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
                    help='Starting timestep of the averaging process. Only used'
                         ' for generating windows.',
                    type=int, default=-1)
parser.add_argument('-f','--new-file-prefix',
                    help='Prefix for the new file. Will have timestep appended.',
                    type=str)
parser.add_argument('--outpath', help='Custom path for the output VTM file.'
                                      ' vtkfile path used if not given', type=Path)
parser.add_argument('--velonly', help='Only process velbar file', action='store_true')
parser.add_argument('--debug', help='Load raw stsbar data into VTM', action='store_true')
parser.add_argument('--vptpath', help='Custom path to vtkpytools package', type=Path)
parser.add_argument('-a', '--ascii', help='Read *bar files as ASCII', action='store_true')
parser.add_argument('--velbar', help='Path to velbar file(s)', type=Path, nargs='+')
parser.add_argument('--stsbar', help='Path to stsbar file(s)', type=Path, nargs='+')

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

if args.debug and args.velonly:
    raise RuntimeError('--velonly counteracts the effect of --debug. Choose one or the other.')

if isinstance(args.velbar, list) and len(args.velbar) > 2:
    velbarStrings = '\n\t' + '\n\t'.join([x.as_posix() for x in args.velbar])
    raise ValueError('--velbar can only contain two paths max.'
                       ' The following were given:{}'.format(velbarStrings))
if isinstance(args.stsbar, list) and len(args.stsbar) > 2:
    stsbarStrings = '\n\t' + '\n\t'.join([x.as_posix() for x in args.stsbar])
    raise ValueError('--stsbar can only contain two paths max.'
                       ' The following were given:{}'.format(stsbarStrings))

if args.new_file_prefix:
    vtmName = Path(args.new_file_prefix + '_' + args.timestep + '.vtm')
else:
    vtmName = Path(os.path.splitext(args.vtkfile.name)[0] + '_' + args.timestep + '.vtm')

vtmPath = (args.outpath if args.outpath else args.vtkfile.parent) / vtmName

velbarReader = np.loadtxt if args.ascii else vpt.binaryVelbar
stsbarReader = np.loadtxt if args.ascii else vpt.binaryStsbar

## ---- Loading data arrays
if '-' in args.timestep:
# Create timestep windows
    if args.ts0 == -1:
        raise RuntimeError("Starting timestep of bar field averaging required (--ts0)")

    timesteps = [int(x) for x in args.timestep.split('-')]
    print('Creating timewindow between {} and {}'.format(timesteps[0], timesteps[1]))
    if not args.velbar:
        velbarPaths = []; stsbarPaths = []
        for timestep in timesteps:
            velbarPaths.append(vpt.globFile('velbar*.{}*'.format(timestep), args.barfiledir))

            if not args.velonly:
                stsbarPaths.append(vpt.globFile('stsbar*.{}*'.format(timestep), args.barfiledir))
    else:
        velbarPaths = args.velbar
        stsbarPaths = args.stsbar

    print('Using data files:\n\t{}\t{}'.format(velbarPaths[0], velbarPaths[1]))
    if not args.velonly:
        print('\t{}\t{}'.format(stsbarPaths[0], stsbarPaths[1]))

    velbarArrays = []; stsbarArrays = []
    for i in range(2):
        velbarArrays.append(velbarReader(velbarPaths[i]))
        if not args.velonly:
            stsbarArrays.append(stsbarReader(stsbarPaths[i]))

    velbarArray = (velbarArrays[1]*(timesteps[1] - args.ts0) -
                   velbarArrays[0]*(timesteps[0] - args.ts0)) / (timesteps[1] - timesteps[0])
    if not args.velonly:
        stsbarArray = (stsbarArrays[1]*(timesteps[1] - args.ts0) -
                       stsbarArrays[0]*(timesteps[0] - args.ts0)) / (timesteps[1] - timesteps[0])
    print('Finished computing timestep window')
else:
    velbarPath = args.velbar if args.velbar else \
        (vpt.globFile('velbar*.{}*'.format(args.timestep), args.barfiledir))
    print('Using data files:\n\t{}'.format(velbarPath))
    velbarArray = velbarReader(velbarPath)

    if not args.velonly:
        stsbarPath = args.stsbar if args.stsbar else \
            (vpt.globFile('stsbar*.{}*'.format(args.timestep), args.barfiledir))
        print('\t{}'.format(stsbarPath))
        stsbarArray = stsbarReader(stsbarPath)

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

if args.debug and not args.velonly:
    grid['stsbar'] = stsbarArray

grid = grid.compute_gradient(scalars='Velocity')
grid = vpt.compute_vorticity(grid, scalars='Velocity')

## ---- Copy data from grid to wall object
wall = wall.sample(grid)

dataBlock['grid'] = grid
dataBlock['wall'] = wall
print('Saving dataBlock file to: {}'.format(vtmPath), end='')
dataBlock.save(vtmPath)
print('\tDone!')
