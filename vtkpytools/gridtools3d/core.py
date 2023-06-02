import numpy as np
import vtk
from pathlib import Path
import pyvista as pv
from packaging import version


def form3DGrid(coords_array, connectivity_array) -> pv.UnstructuredGrid:
    """Create 3D VTK UnstructuredGrid from coordinates and connectivity

    Currently supports tet grids, mixed tet/pyramid/hex grids, and mixed
    tet/wedge topologies.

    Parameters
    ----------
    coords_array : numpy.ndarray
        Coordinates of the points. Shape in (nPoints,2).
    connectivity_array : numpy.ndarray
        Connectivity array of the cells. Shape is (nCells,PointsPerCell). The
        index of points should start from 0. The type of cell will be inferred
        based on the number of points per cell. Mixed meshes inferred by
        repeated node IDs in a single element.

    Returns
    -------
    pv.UnstructuredGrid
        Pyvista UnstructuredGrid object with the grid loaded.
    """
    nCells = connectivity_array.shape[0]
    nPnts = connectivity_array.shape[1]
    cell_type = None
    if nPnts == 4:
        cell_type = vtk.VTK_TETRA # ==int(10)
    elif nPnts == 6:
        tetCells = connectivity_array[:,3] == connectivity_array[:,4]
        pyrCells = (connectivity_array[:,4] == connectivity_array[:,5]) & ~tetCells
        wedCells = np.invert(tetCells + pyrCells)
        hexCells = 0
        if not np.any(pyrCells + tetCells): # not a mixed mesh
            cell_type = vtk.VTK_WEDGE # ==int(12)
    elif nPnts == 8:
        tetCells = connectivity_array[:,3] == connectivity_array[:,4]
        pyrCells = (connectivity_array[:,4] == connectivity_array[:,5]) & ~tetCells
        wedCells = (connectivity_array[:,5] == connectivity_array[:,6]) & ~(tetCells | pyrCells)
        hexCells = np.invert(tetCells + pyrCells + wedCells)
        if not np.any(pyrCells + tetCells + wedCells): # not a mixed mesh
            cell_type = vtk.VTK_HEXAHEDRON # ==int(12)
    else:
        raise ValueError('This connectivity file has the wrong number of points.'
                            ' Must be either 4, 6, or 8 points per cell, this has {}'.format(nPnts))

    if cell_type:
        connectivity_array = np.hstack((np.ones((nCells,1), dtype=np.int64)*nPnts, connectivity_array))
        offsets = np.arange(0, connectivity_array.size+1, nPnts+1, dtype=np.int64)
        cell_types = np.ones(nCells, dtype=np.int64) * cell_type
    else: # mixed mesh
        cell_types = tetCells*vtk.VTK_TETRA      + pyrCells*vtk.VTK_PYRAMID + \
                     hexCells*vtk.VTK_HEXAHEDRON + wedCells*vtk.VTK_WEDGE
        offset     = tetCells*np.int64(4)   + pyrCells*np.int64(5) +  \
                     hexCells*np.int64(8)   + wedCells*np.int64(6)

        connectivity_array[tetCells, 4:] = np.int64(-1)
        connectivity_array[pyrCells, 5:] = np.int64(-1)
        connectivity_array[wedCells, 6:] = np.int64(-1)
        connectivity_array = np.hstack((offset[:,None], connectivity_array))
        connectivity_array = connectivity_array[connectivity_array != -1]

        offsets = np.roll(np.cumsum(offset+1), 1)
        offsets[0] = 0

    if version.parse(pv.__version__) < version.parse('0.37.0'):
        grid = pv.UnstructuredGrid(offsets, connectivity_array, cell_types, coords_array)
    else:
        grid = pv.UnstructuredGrid(connectivity_array, cell_types, coords_array)

    return grid
