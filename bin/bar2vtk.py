#!/usr/bin/env python3
import vtkpytools as vpt

# vpt.bar2vtk_bin()
#

from pathlib import Path, PurePath
test = {'vtkfile': Path('result/exampleMesh.vtm'), 'barfiledir': Path('data'), 'timestep': '10000', 'ts0': -1, 'new_ file_prefix': None, 'outpath': None, 'velonly': False, 'debug': False, 'velbar': [], 'stsbar': [], 'asciidata': False,
        'list': [Path('~/gitRepos'), Path('/projects'), 'fjfj']}
convertArray = [(PurePath, lambda x: x.as_posix()),
                (type(None), lambda x: '')
                ]
vpt.barfiletools.bar2vtk._convertArray2TomlTypes(test, convertArray)
print(test)

# argsdict = vpt.bar2vtk_parse()

# argsdict['asciidata'] = argsdict.pop('ascii')
# for key in list(argsdict.keys()):
#     if key not in vpt.bar2vtk.__code__.co_varnames:
#         del argsdict[key]

# vpt.bar2vtk(**argsdict)

# vpt.bar2vtk(vtkfile = args.vtkfile,
# barfiledir = args.barfiledir,
# timestep = args.timestep,
# ts0 = args.ts0,
# new_file_prefix = args.new_file_prefix,
# outpath = args.outpath,
# velonly = args.velonly,
# debug = args.debug,
# asciidata = args.ascii,
# velbar = args.velbar,
# stsbar = args.stsbar)
