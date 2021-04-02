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
    U = np.array([[0, 0.75, 1], [0, 0.75, 1]])
    wall_dists = np.array([[0, 0.75, 1], [0, 0.75, 1]])
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
    assert(np.all(result == percent))


def test_delta_percent_Uedge(fix_delta):
    U, wall_dists, nwallpnts, percent = fix_delta
    result = vpt.delta_percent(U, wall_dists, nwallpnts, percent, Uedge=np.ones(2)*0.5)

    assert(result.size == nwallpnts)
    assert(np.all(result == percent*0.5))


def test_delta_percent_notSameSize(fix_delta):
    U, wall_dists, nwallpnts, percent = fix_delta
    with pytest.raises(ValueError):
        # Raise error if U.size is not evenly divisible by nwallpnts
        vpt.delta_percent(U, wall_dists, nwallpnts+7, percent)


@pytest.mark.parametrize('displace', [True, False])
@pytest.mark.parametrize('momentum', [True, False])
@pytest.mark.parametrize('Uedge', [None, np.ones(2)*0.75])
def test_delta_velInt_base(fix_delta, displace, momentum, Uedge):
    if Uedge is None:
        correct = {'delta_displace':0.5, 'delta_momentum':3/32}
    else:
        correct = {'delta_displace':1/3, 'delta_momentum':-5/90}
    U, wall_dists, nwallpnts, _ = fix_delta

    result = vpt.delta_velInt(U, wall_dists, nwallpnts, displace, momentum, Uedge)

    for key, val in result.items():
        assert(val.size == nwallpnts)
        assert(np.allclose(val, correct[key]))


def test_delta_velInt_notSameSize(fix_delta):
    U, wall_dists, nwallpnts, _ = fix_delta
    with pytest.raises(ValueError):
        # Raise error if U.size is not evenly divisible by nwallpnts
        vpt.delta_velInt(U, wall_dists, nwallpnts+7)


@pytest.mark.parametrize('Uedge', [None, np.ones(2)*0.75])
@pytest.mark.parametrize('displace', [True, False])
@pytest.mark.parametrize('momentum', [True, False])
@pytest.mark.parametrize('returnUvort', [True, False])
def test_delta_vortInt_base(fix_delta, displace, momentum, returnUvort, Uedge):
    if Uedge is None:
        correct = {'delta_displace':13/16, 'delta_momentum':41/128,
                   'Uvort': np.tile([0, -9/32, -1/2], 2)}
    else:
        correct = {'delta_displace':-13/24, 'delta_momentum':301/288,
                   'Uvort': np.tile([0, -9/32, -1/2], 2)}
    U, wall_dists, nwallpnts, _ = fix_delta

    result = vpt.delta_vortInt(U, wall_dists, nwallpnts, displace,
                               momentum, returnUvort, Uedge)

    for key, val in result.items():
        if key == 'Uvort':
            assert(val.size == U.size)
            assert(np.allclose(val.flatten(), correct[key]))
        else:
            assert(val.size == nwallpnts)
            assert(np.allclose(val, correct[key]))

def test_delta_vortInt_notSameSize(fix_delta):
    U, wall_dists, nwallpnts, _ = fix_delta
    with pytest.raises(ValueError):
        # Raise error if U.size is not evenly divisible by nwallpnts
        vpt.delta_vortInt(U, wall_dists, nwallpnts+7)
