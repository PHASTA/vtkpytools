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
    return sample, S, sample_dists

@pytest.fixture
def fix_delta():
    U = np.array([[0, 1], [0, 1]])
    wall_dists = np.array([[0, 1], [0, 1]])
    nwallpnts = 2
    percent = 0.5

    return U, wall_dists, nwallpnts, percent


def test_sampleAlongVectors_walldistance(fix_sampleAlongVectors):
    sample, S, sample_dists = fix_sampleAlongVectors
    assert(np.allclose(sample_dists, sample['WallDistance'][:S]))


def test_sampleAlongVectors_size(fix_sampleAlongVectors, loadedBLData):
    sample, S, _ = fix_sampleAlongVectors
    assert(sample.n_points == S*loadedBLData['wall'].n_points)

def test_delta_percent_base(fix_delta):
    U, wall_dists, nwallpnts, percent = fix_delta

    result = vpt.delta_percent(U, wall_dists, nwallpnts, percent)

    assert(result.size == nwallpnts)
    assert(np.all(result == np.ones(nwallpnts)*percent))

def test_delta_percent_Uedge(fix_delta):
    U, wall_dists, nwallpnts, percent = fix_delta

    result = vpt.delta_percent(U, wall_dists, nwallpnts, percent, Uedge=np.ones(2)*0.5)

    assert(result.size == nwallpnts)
    assert(np.all(result == np.ones(nwallpnts)*percent*0.5))

def test_delta_percent_notSameSize(fix_delta):
    U, wall_dists, nwallpnts, percent = fix_delta

    with pytest.raises(RuntimeError):
        # Raise error if U.size is not evenly divisible by nwallpnts
        vpt.delta_percent(U, wall_dists, nwallpnts+1, percent)

