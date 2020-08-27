#!/usr/bin/env python3
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

if args.new_file_prefix:
    vtmName = Path(args.new_file_prefix + '_' + args.timestep + '.vtm')
else:
    vtmName = Path(os.path.splitext(args.vtkfile.name)[0] + '_' + args.timestep + '.vtm')

if args.outpath:
    vtmPath = args.outpath / vtmName
else:
    vtmPath = args.vtkfile.parent / vtmName

if args.ascii:
    velbarReader = np.loadtxt
    stsbarReader = np.loadtxt
else:
    velbarReader = vpt.binaryVelbar
    stsbarReader = vpt.binaryStsbar


## ---- Loading data arrays
if '-' in args.timestep:
# Create timestep windows
    if args.ts0 == -1:
        raise RuntimeError("Starting timestep of bar field averaging required (--ts0)")
    # raise NotImplementedError("Haven't implemented creating time windows yet")

    timesteps = [int(x) for x in args.timestep.split('-')]
    print('Creating timewindow between {} and {}'.format(timesteps[0], timesteps[1]))
    velbarPaths = []; stsbarPaths = []
    for timestep in timesteps:
        velbarPath = list(args.barfiledir.glob('velbar*{}*'.format(timestep)))
        if len(velbarPath) > 0:
            assert velbarPath[0].is_file()
            velbarPaths.append(velbarPath[0])
        else:
            raise RuntimeError('Could not find file matching'
                               '"velbar*{}*" in {}'.format(timestep, args.barfiledir))

        if not args.velonly:
            stsbarPath = list(args.barfiledir.glob('stsbar*{}*'.format(timestep)))
            if len(stsbarPath) > 0:
                assert stsbarPath[0].is_file()
                stsbarPaths.append(stsbarPath[0])
            else:
                raise RuntimeError('Could not find file matching'
                                   '"stsbar*{}*" in {}'.format(timestep, args.barfiledir))

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
# Don't create timestep windows
    velbarPath = list(args.barfiledir.glob('velbar*{}*'.format(args.timestep)))
    if len(velbarPath) > 0:
        assert velbarPath[0].is_file()
    else:
        raise RuntimeError('Could not find file matching'
                           '"velbar*{}*" in {}'.format(timestep, args.barfiledir))
    print('Using data files:\n\t{}'.format(velbarPath[0]))
    velbarArray = velbarReader(velbarPath[0])

    if not args.velonly:
        stsbarPath = list(args.barfiledir.glob('stsbar*{}*'.format(args.timestep)))
        if len(stsbarPath) > 0:
            assert stsbarPath[0].is_file()
        else:
            raise RuntimeError('Could not find file matching'
                               '"stsbar*{}*" in {}'.format(timestep, args.barfiledir))
        print('\t{}'.format(stsbarPath[0]))
        stsbarArray = stsbarReader(stsbarPath[0])

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
