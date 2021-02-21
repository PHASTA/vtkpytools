import pytest
from unittest import TestCase
import shutil, os, sys

import vtkpytools as vpt
from pathlib import Path, PurePath

FIXTURE_DIR = Path(__file__).parent.resolve()

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

@pytest.fixture()
def fixture_bar2vtk_exampledata(tmp_path):
    shutil.copytree(FIXTURE_DIR / '../../example/bar2vtk/data', tmp_path / 'data')
    shutil.copytree(FIXTURE_DIR / '../../example/bar2vtk/meshFiles', tmp_path / 'meshFiles')
    shutil.copy(FIXTURE_DIR / '../../example/bar2vtk/makeVTM.py', tmp_path)

    cwd = os.getcwd()
    os.chdir(tmp_path)
    exec(open('makeVTM.py').read())
    yield
    os.chdir(cwd)

def test_bar2vtk_exampledata(fixture_bar2vtk_exampledata, tmp_path):
    sys.argv = 'bar2vtk cli result/exampleMesh.vtm data 10000'.split(' ')
    vpt.bar2vtk_bin()
