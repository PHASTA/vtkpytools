import pytest

import vtkpytools as vpt
from pathlib import Path

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

