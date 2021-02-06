import numpy as np
import vtk
import os
import pyvista as pv
from pathlib import Path
import vtkpytools as vpt
from .data import binaryVelbar, binaryStsbar, calcReynoldsStresses, compute_vorticity
from ..common import globFile

def bar2vtk(vtkfile: Path, barfiledir: Path, timestep: str, \
            ts0: int=-1,  new_file_prefix: str='', outpath: Path=None, \
            velonly=False, debug=False, asciibool=False, \
            velbar=[],     stsbar=[]):


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
