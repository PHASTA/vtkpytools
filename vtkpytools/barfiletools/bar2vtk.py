import vtk, os, argparse
import pyvista as pv
import numpy as np
from pathlib import Path

from .data import binaryVelbar, binaryStsbar, calcReynoldsStresses, compute_vorticity
from ..common import globFile

def bar2vtk_parse():
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

    return parser.parse_args()

    # bar2vtk(vtkfile = args.vtkfile,
    # barfiledir = args.barfiledir,
    # timestep = args.timestep,
    # ts0 = args.ts0,
    # new_file_prefix = args.new_file_prefix,
    # outpath = args.outpath,
    # velonly = args.velonly,
    # debug = args.debug,
    # asciibool = args.ascii,
    # velbar = args.velbar,
    # stsbar = args.stsbar)

def bar2vtk(vtkfile: Path, barfiledir: Path, timestep: str, \
            ts0: int=-1,  new_file_prefix: str='', outpath: Path=None, \
            velonly=False, debug=False, asciibool=False, \
            velbar=[],     stsbar=[]):
    """Convert velbar and stsbar files into 2D vtk files

    See bar2vtk_commandline help documentation for more information.

    Parameters
    ----------
    vtkfile : Path
        vtkfile
    barfiledir : Path
        barfiledir
    timestep : str
        timestep
    ts0 : int
        ts0
    new_file_prefix : str
        new_file_prefix
    outpath : Path
        outpath
    velonly :
        velonly
    debug :
        debug
    asciibool :
        asciibool
    velbar :
        velbar
    stsbar :
        stsbar
    """

    ## ---- Process/check script arguments
    assert vtkfile.is_file()
    assert barfiledir.is_dir()

    if debug and velonly:
        raise RuntimeError('--velonly counteracts the effect of --debug. Choose one or the other.')

    if not len(velbar) == len(stsbar):
        raise ValueError('--velbar and --stsbar must be given same number of paths'
                        ', given {} and {}, respectively'.format(len(velbar), len(stsbar)))

    for flag, arg in {'--velbar':velbar, '--stsbar':stsbar}.items():
        if len(arg) > 2:
            pathStrings = '\n\t' + '\n\t'.join([x.as_posix() for x in arg])
            raise ValueError('{} can only contain two paths max.'
                            ' The following were given:{}'.format(flag, pathStrings))

        if len(arg) == 2 and not '-' in timestep:
            raise ValueError('{} was given two paths, but timestep was not given range.'.format(flag))

    if new_file_prefix:
        vtmName = Path(new_file_prefix + '_' + timestep + '.vtm')
    else:
        vtmName = Path(os.path.splitext(vtkfile.name)[0] + '_' + timestep + '.vtm')

    vtmPath = (outpath if outpath else vtkfile.parent) / vtmName

    velbarReader = np.loadtxt if asciibool else binaryVelbar
    stsbarReader = np.loadtxt if asciibool else binaryStsbar

    ## ---- Loading data arrays
    if '-' in timestep:
    # Create timestep windows
        if ts0 == -1:
            raise RuntimeError("Starting timestep of bar field averaging required (--ts0)")

        timesteps = [int(x) for x in timestep.split('-')]
        print('Creating timewindow between {} and {}'.format(timesteps[0], timesteps[1]))
        if not velbar:
            velbarPaths = []; stsbarPaths = []
            for timestep in timesteps:
                velbarPaths.append(globFile('velbar*.{}*'.format(timestep), barfiledir))

                if not velonly:
                    stsbarPaths.append(globFile('stsbar*.{}*'.format(timestep), barfiledir))
        else:
            velbarPaths = velbar
            stsbarPaths = stsbar

        print('Using data files:\n\t{}\t{}'.format(velbarPaths[0], velbarPaths[1]))
        if not velonly:
            print('\t{}\t{}'.format(stsbarPaths[0], stsbarPaths[1]))

        velbarArrays = []; stsbarArrays = []
        for i in range(2):
            velbarArrays.append(velbarReader(velbarPaths[i]))
            if not velonly:
                stsbarArrays.append(stsbarReader(stsbarPaths[i]))

        velbarArray = (velbarArrays[1]*(timesteps[1] - ts0) -
                    velbarArrays[0]*(timesteps[0] - ts0)) / (timesteps[1] - timesteps[0])
        if not velonly:
            stsbarArray = (stsbarArrays[1]*(timesteps[1] - ts0) -
                        stsbarArrays[0]*(timesteps[0] - ts0)) / (timesteps[1] - timesteps[0])
        print('Finished computing timestep window')
    else:
        velbarPath = velbar if velbar else \
            (globFile('velbar*.{}*'.format(timestep), barfiledir))
        print('Using data files:\n\t{}'.format(velbarPath))
        velbarArray = velbarReader(velbarPath)

        if not velonly:
            stsbarPath = stsbar if stsbar else \
                (globFile('stsbar*.{}*'.format(timestep), barfiledir))
            print('\t{}'.format(stsbarPath))
            stsbarArray = stsbarReader(stsbarPath)

    ## ---- Load DataBlock
    dataBlock = pv.MultiBlock(vtkfile.as_posix())
    grid = dataBlock['grid']
    wall = dataBlock['wall']

    ## ---- Load *bar data into dataBlock
    grid['Pressure'] = velbarArray[:,0]
    grid['Velocity'] = velbarArray[:,1:4]

    if not velonly:
        ReyStrTensor = calcReynoldsStresses(stsbarArray, velbarArray)
        grid['ReynoldsStress'] = ReyStrTensor

    if debug and not velonly:
        grid['stsbar'] = stsbarArray

    grid = grid.compute_gradient(scalars='Velocity')
    grid = compute_vorticity(grid, scalars='Velocity')

    ## ---- Copy data from grid to wall object
    wall = wall.sample(grid)

    dataBlock['grid'] = grid
    dataBlock['wall'] = wall
    print('Saving dataBlock file to: {}'.format(vtmPath), end='')
    dataBlock.save(vtmPath)
    print('\tDone!')
