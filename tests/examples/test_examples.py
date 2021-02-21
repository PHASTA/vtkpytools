import pytest
import shutil, os, sys, subprocess, re

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

@pytest.fixture()
def fixture_mixedMesh2D(tmp_path):
    shutil.copy(FIXTURE_DIR / '../../example/mixedMesh2D/makeVTM.py', tmp_path)
    shutil.copy(FIXTURE_DIR / '../../example/mixedMesh2D/mixedmesh.cnn', tmp_path)
    shutil.copy(FIXTURE_DIR / '../../example/mixedMesh2D/mixedmesh.crd', tmp_path)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    yield
    os.chdir(cwd)

def test_mixedMesh2D(fixture_mixedMesh2D):
    exec(open('makeVTM.py').read())
    assert Path('./result/mixedmesh.vtm').exists()

@pytest.mark.parametrize("meshtype", ['HexPyrTetWedge', 'TetWedge'])
def test_mixedMesh3D(meshtype, tmp_path):
    shutil.copy(FIXTURE_DIR / '../../example/mixedMesh3D/makeVTK.py', tmp_path)
    shutil.copy(FIXTURE_DIR / '../../example/mixedMesh3D/mixedMesh3D_{}.cnn'.format(meshtype), tmp_path)
    shutil.copy(FIXTURE_DIR / '../../example/mixedMesh3D/mixedMesh3D_{}.crd'.format(meshtype), tmp_path)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    os.mkdir('result')
    with open('makeVTK.py', 'r') as file:
        filestring = ''
        for line in file:
            filestring += re.sub(r"meshtype='\w+'", r"meshtype='{}'".format(meshtype), line)

    print(filestring)
    exec(filestring)
    printDirectoryTree()
    assert Path('./result/mixedmesh3d_{}.vtk'.format(meshtype)).exists()

    os.chdir(cwd)


def printDirectoryTree():
    for dirname, dirnames, filenames in os.walk('.'):
        # print path to all subdirectories first.
        for subdirname in dirnames:
            print(os.path.join(dirname, subdirname))

        # print path to all filenames.
        for filename in filenames:
            print(os.path.join(dirname, filename))
