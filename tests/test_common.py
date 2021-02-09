import pytest
import numpy as np

import vtkpytools as vpt

def test_calcStrainRate():
    gradientarray = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                              [5, 8, 7, 3, 2, 8, 9, 5, 1]])
    solution = np.array([[1, 5, 9,   3, 5,   7],
                         [5, 2, 1, 5.5, 8, 6.5]])
    strainrate = vpt.calcStrainRate(gradientarray)

    assert np.allclose(strainrate, solution)
    assert strainrate.shape[0] == gradientarray.shape[0]


