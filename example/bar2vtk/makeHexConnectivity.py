import numpy as np
from pathlib import Path

nypnts = 280
nxpnts = 1247
connectivityFilePath = Path('TruncBump2d_Squares.cnn')

nxcells = nxpnts - 1
nycells = nypnts - 1

nCells = nxcells*nycells

connectivity = np.ones((nCells, 5))
connectivity[:,0] = np.arange(1, nCells+1)

for i in range(nxcells):
    for j in range(nycells):
        connectivity[i*nycells+j,1] = i*nypnts + j
        connectivity[i*nycells+j,2] = i*nypnts + j + 1
        connectivity[i*nycells+j,3] = (i+1)*nypnts + j + 1
        connectivity[i*nycells+j,4] = (i+1)*nypnts + j

np.savetxt(connectivityFilePath, connectivity, fmt='%d')
