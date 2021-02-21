import pytest
import shutil, os, sys, subprocess

import vtkpytools as vpt
from pathlib import Path, PurePath

FIXTURE_DIR = Path(__file__).parent.resolve()

@pytest.fixture()
def fixture_bar2vtk(tmp_path):
    shutil.copytree(FIXTURE_DIR / '../../example/bar2vtk/data', tmp_path / 'data')
    shutil.copytree(FIXTURE_DIR / '../../example/bar2vtk/meshFiles', tmp_path / 'meshFiles')
    shutil.copy(FIXTURE_DIR / '../../example/bar2vtk/makeVTM.py', tmp_path)
    shutil.copy(FIXTURE_DIR / '../../example/bar2vtk/runbar2vtk.sh', tmp_path)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    yield
    os.chdir(cwd)

def test_makeVTM(fixture_bar2vtk):
    exec(open('makeVTM.py').read())
    assert Path('./result/exampleMesh.vtm').exists()

def test_runbar2vtk(fixture_bar2vtk):
    test_makeVTM(fixture_bar2vtk)

    process = subprocess.run('./runbar2vtk.sh')
    assert process.returncode == 0
