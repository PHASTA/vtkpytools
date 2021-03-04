import pytest
import numpy as np

import vtkpytools as vpt
from pathlib import Path


@pytest.fixture
def twoFullTensors():
    return np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                     [5, 8, 7, 3, 2, 8, 9, 5, 1]])
@pytest.fixture
def symAndFullTensor():
    """Fixture for symmetric <-> full tensor conversion functions"""
    full = np.array([[[1, 2, 3],
                      [2, 5, 6],
                      [3, 6, 9]],
                     [[5, 8, 7],
                      [8, 2, 8],
                      [7, 8, 1]]])
    sym = np.array([[1, 5, 9, 2, 3, 6],
                    [5, 2, 1, 8, 7, 8]])
    return full, sym

def test_calcStrainRate(twoFullTensors):
    solution = np.array([[1, 5, 9,   3, 5,   7],
                         [5, 2, 1, 5.5, 8, 6.5]])
    strainrate = vpt.calcStrainRate(twoFullTensors)

    assert np.allclose(strainrate, solution)
    assert strainrate.shape[0] == twoFullTensors.shape[0]

def test_symmetric2FullTensor(symAndFullTensor):
    (full, sym) = symAndFullTensor

    full_test = vpt.symmetric2FullTensor(sym)
    assert np.allclose(full_test, full)

def test_full2SymmetricTensor(symAndFullTensor):
    (full, sym) = symAndFullTensor

    sym_test = vpt.full2SymmetricTensor(full)
    assert np.allclose(sym_test, sym)

@pytest.fixture(params=['', '.1'])
def createGlobFiles(request, tmp_path):
    testStrings = ['velbar.20000',
                   'velbar.20000-30000',
                   'velbar.200',
                   'velbar.200-30000']
    for testString in testStrings:
        testPath = tmp_path / Path(testString + request.param)
        testPath.touch()

# Should fail
def test_globFile_glob(createGlobFiles, tmp_path):
    with pytest.raises(RuntimeError):
        vpt.globFile(r'velbar*.200*', tmp_path)

def test_globFile_regex(createGlobFiles, tmp_path):
    print(list(tmp_path.iterdir()))
    vpt.globFile(r'^velbar\.200(?![\d|-]).*$', tmp_path, regex=True)

