import pytest
import numpy as np
import pyvista as pv

import vtkpytools as vpt
from pathlib import Path

@pytest.fixture
def loadedBLData():
    vtmPath = Path('./tests/testData/BL_10000.vtm')
    return pv.MultiBlock(vtmPath.as_posix())


@pytest.fixture
def fix_sampleAlongVectors(loadedBLData):
    S = 5
    sample_dists = np.linspace(0, 0.1, S)
    sample = vpt.sampleAlongVectors(loadedBLData, sample_dists,
                           loadedBLData['wall']['Normals'], loadedBLData['wall'].points)
    yield sample, S, sample_dists


def test_sampleAlongVectors_walldistance(fix_sampleAlongVectors):
    sample, S, sample_dists = fix_sampleAlongVectors
    assert(np.allclose(sample_dists, sample['WallDistance'][:S]))


def test_sampleAlongVectors_size(fix_sampleAlongVectors, loadedBLData):
    sample, S, _ = fix_sampleAlongVectors
    assert(sample.n_points == S*loadedBLData['wall'].n_points)
