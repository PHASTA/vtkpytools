#!/usr/bin/env python
import argparse
from pathlib import Path
import os

Description=""" Create ASCII split window *bar file from binary *bar files"""


## Parsing script input
class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """To display defaults in help and have a multiline help description"""
    # Shamelessly ripped from https://stackoverflow.com/a/18462760/7564988
    pass

parser = argparse.ArgumentParser(description=Description,
                                 formatter_class=CustomFormatter,
                                 prog='bar2vtk')
parser.add_argument('barfiledir', help='Path to *bar file directory', type=Path)
parser.add_argument('timestep', help='Timestep range', type=str)
parser.add_argument('outpath', help='Path of the ouput files', type=Path)
parser.add_argument('--ts0','--bar-average-start',
                    help='Starting timestep of the averaging process. Only used for generating windows.',
                    type=int, default=-1)
parser.add_argument('--velonly', help='Only process velocity', action='store_true')
parser.add_argument('--vptpath', help='Custom path to vtkpytools package', type=Path)

args = parser.parse_args()

import numpy as np

# import the package in this repository
import sys
if args.vptpath:
    vptPath = args.vptpath
else:
    vptPath = Path(__file__).resolve().parent.parent
sys.path.insert(0, vptPath.as_posix())

from vtkpytools import binaryVelbar, binaryStsbar

## ---- Process/check script arguments
assert args.barfiledir.is_dir()
assert args.outpath.is_dir()

## ---- Loading data arrays
if args.ts0 == -1:
    raise RuntimeError("Starting timestep of bar field averaging required (--ts0)")

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
    velbarArrays.append(binaryVelbar(velbarPaths[i]))
    if not args.velonly:
        stsbarArrays.append(binaryStsbar(stsbarPaths[i]))

velbarArray = (velbarArrays[1]*(timesteps[1] - args.ts0) -
                velbarArrays[0]*(timesteps[0] - args.ts0)) / (timesteps[1] - timesteps[0])
if not args.velonly:
    stsbarArray = (stsbarArrays[1]*(timesteps[1] - args.ts0) -
                    stsbarArrays[0]*(timesteps[0] - args.ts0)) / (timesteps[1] - timesteps[0])
print('Finished computing timestep window')

np.savetxt(args.outpath / ('velbar.' + args.timestep + '.1'), velbarArray, fmt=r'%.14E')
np.savetxt(args.outpath / ('stsbar.' + args.timestep + '.1'), stsbarArray, fmt=r'%.14E')

print('\tDone!')
