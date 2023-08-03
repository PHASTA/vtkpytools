import pytest
import numpy as np

import vtkpytools as vpt

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

class Test_symmetric2FullTensor():

    @staticmethod
    def test_func2d(symAndFullTensor):
        full, sym = symAndFullTensor

        full_test = vpt.symmetric2FullTensor(sym)
        assert np.allclose(full_test, full)

    @staticmethod
    def test_func_Nd(symAndFullTensor):
        """Test (n,m,6) -> (n,m,3,3)"""
        full, sym = symAndFullTensor

        sym = sym.reshape((1,-1,6))
        full = full.reshape((1,-1,3,3))
        full_test = vpt.symmetric2FullTensor(sym)
        assert full_test.shape == full.shape
        assert np.allclose(full_test, full)

        sym = sym.reshape((-1,1,6))
        full = full.reshape((-1,1,3,3))
        full_test = vpt.symmetric2FullTensor(sym)
        assert full_test.shape == full.shape
        assert np.allclose(full_test, full)

    @staticmethod
    def test_inputCheck():
        array = np.empty((6,4))
        with pytest.raises(Exception):
            vpt.symmetric2FullTensor(array)


class Test_full2SymmetricTensor():

    @staticmethod
    def test_func3d(symAndFullTensor):
        full, sym = symAndFullTensor

        sym_test = vpt.full2SymmetricTensor(full)
        assert sym_test.shape == sym.shape
        assert np.allclose(sym_test, sym)

    @staticmethod
    def test_func_Nd(symAndFullTensor):
        """Test (n,m,3,3) -> (n,m,6)"""
        full, sym = symAndFullTensor

        sym = sym.reshape((1,-1,6))
        full = full.reshape((1,-1,3,3))
        sym_test = vpt.full2SymmetricTensor(full)
        assert sym_test.shape == sym.shape
        assert np.allclose(sym_test, sym)

        sym = sym.reshape((-1,1,6))
        full = full.reshape((-1,1,3,3))
        sym_test = vpt.full2SymmetricTensor(full)
        assert sym_test.shape == sym.shape
        assert np.allclose(sym_test, sym)

    @staticmethod
    def test_inputCheck():
        array = np.empty((3,3,4))
        with pytest.raises(Exception):
            vpt.full2SymmetricTensor(array)


@pytest.mark.parametrize('offset,expected', [(1, [0, 1] ),
                                             (0, [0.25, 0.75])])
def test_pwlinRoots(offset, expected):
    x = np.array([ 0, 0.5,  1])
    y = np.array([-1,   1, -1])

    roots = vpt.pwlinRoots(x, y + offset)
    assert np.allclose(roots, expected)


@pytest.mark.parametrize('kwargs,expected', [({'dx': 2.4}, [0, 1, 2, 4, 6.4, 8]),
                                             ({'dx': 2.0}, [0, 1, 2, 4, 6,   8]),
                                             ({'magnitude': 4.0}, [0, 1, 2, 4, 6,   8]) ])
def test_seriesDiffLimiter(kwargs, expected):
    series = np.array([0, 1, 2, 4, 8])

    result = vpt.seriesDiffLimiter(series, **kwargs)
    assert result.size == len(expected)
    assert np.allclose(result, expected)

@pytest.mark.parametrize('kwargs', [{},
                                    {'dx': 2.0, 'magnitude': 4.0}])
def test_seriesDiffLimiter_error(kwargs):
    series = np.array([0, 1, 2, 4, 8])
    with pytest.raises(RuntimeError):
        vpt.seriesDiffLimiter(series, **kwargs)
