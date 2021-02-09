import pytest
from unittest import TestCase

import vtkpytools as vpt
from pathlib import Path, PurePath

def test_convertArray2TomlTypes():
    test = {'vtkfile': Path('result/exampleMesh.vtm'),
            'timestep': '10000', 'ts0': -1, 'new_ file_prefix': None,
            'velonly': False, 'velbar': [],
            'list': [Path('~/gitRepos'), Path('/projects'), 'fjfj']}
    convertArray = [(PurePath, lambda x: x.as_posix()),
                    (type(None), lambda x: '')
                    ]

    solution = {'vtkfile': 'result/exampleMesh.vtm',
            'timestep': '10000', 'ts0': -1, 'new_ file_prefix': '',
            'velonly': False, 'velbar': [],
            'list': ['~/gitRepos', '/projects', 'fjfj']}

    vpt.barfiletools.bar2vtk._convertArray2TomlTypes(test, convertArray)

    TestCase().assertDictEqual(test, solution)
