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
parser.add_argument('--velbar', help='Path to velbar file(s)', type=Path, nargs='+', default=[])
parser.add_argument('--stsbar', help='Path to stsbar file(s)', type=Path, nargs='+', default=[])

args = parser.parse_args()

# import the package in this repository
import sys
if args.vptpath:
    vptPath = args.vptpath
else:
    vptPath = Path(__file__).resolve().parent.parent
sys.path.insert(0, vptPath.as_posix())

import vtkpytools as vpt

vpt.bar2vtk(vtkfile = args.vtkfile,
barfiledir = args.barfiledir,
timestep = args.timestep,
ts0 = args.ts0,
new_file_prefix = args.new_file_prefix,
outpath = args.outpath,
velonly = args.velonly,
debug = args.debug,
asciibool = args.ascii,
velbar = args.velbar,
stsbar = args.stsbar)
