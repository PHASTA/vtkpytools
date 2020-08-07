import numpy as np
import vtk
from pathlib import Path
from scipy.spatial import Delaunay
import pyvista as pv

def form2DGrid(coords_array, connectivity_array=None) -> pv.UnstructuredGrid:
    """Create 2D VTK UnstructuredGrid from coordinates and connectivity

    Parameters
    ----------
    coords_array : numpy.ndarray
        Coordinates of the points. Shape in (nPoints,2).
    connectivity_array : numpy.ndarray, optional
        Connectivity array of the cells. Shape in (nCells,PointsPerCell). The
        index of points should start from 0. The type of cell will be inferred
        based on the number of points per cell.  If not given, will create mesh
        from the points given in the coords_path file. (default: None)

    Returns
    -------
    pv.UnstructuredGrid
        Pyvista UnstructuredGrid object with the grid loaded.
    """
    coords_array = np.hstack([coords_array, np.zeros((coords_array.shape[0], 1)) ])

    if type(connectivity_array) == type(None):
        # Generate mesh from points
        print('Generating Mesh...')
        mesh = Delaunay(coords_array[:,0:2])
        print('Finished Meshing!')

        nCells = mesh.simplices.shape[0]
            # mesh.simplices contains the connectivity array
        connectivity_array = mesh.simplices
        connectivity_array = np.hstack((np.ones((nCells,1), dtype=np.int64)*3, connectivity_array))

        offsets = np.arange(3, mesh.simplices.size+4, 3, dtype=np.int64)

            # vtk.VTK_TRIANGLE is just an integer
        cell_types = np.ones(nCells, dtype=np.int64) * vtk.VTK_TRIANGLE
    else:
        nCells = connectivity_array.shape[0]
        nPnts = connectivity_array.shape[1]
        if nPnts == 3:
            cell_type = vtk.VTK_TRIANGLE # ==int(5)
        elif nPnts == 4:
            cell_type = vtk.VTK_QUAD # ==int(9)
        else:
            raise ValueError(f'This connectivity file has the wrong number of points. Must be either 3 or 4 points per cell, this has {nPnts}')

        connectivity_array = np.hstack((np.ones((nCells,1), dtype=np.int64)*nPnts, connectivity_array))
        offsets = np.arange(0, connectivity_array.size+1, nPnts+1, dtype=np.int64)
        cell_types = np.ones(nCells, dtype=np.int64) * cell_type

    grid = pv.UnstructuredGrid(offsets, connectivity_array, cell_types, coords_array)

    return grid

def computeEdgeNormals(edges, domain_point) -> pv.PolyData:
    """Compute the normals of the edge assuming coplanar to XY plane

    Loops through every line (or cell) in the edges to calculate it's normal
    vector. Then it will ensure that the vectors face towards the inside of the
    domain using the following:

    1) Create vector from point on line segment to domain_point called "insideVector"

    2) Determine whether the current normal vector points in the direction as
        the insideVector using a dot product.

    3) Reverse the wall normal vector if it points outside the domain

        n = n * dot(n, insideVector / |insideVector|)

    Parameters
    ----------
    edges : pyvista.PolyData
        The edges from which the normals should be calculated.
    domain_point : np.ndarray
        A point inside the domain that determines whether the calculated normal
        vector points inside or outside the domain.

    """
    normals = np.zeros((edges.n_cells, 3))
    i = 0
        # Indices of 2 points forming line cell
    indices = edges.lines.reshape((edges.n_cells, 3))[:,1:3]
    pnts1 = edges.points[indices[:,0], :]
    pnts2 = edges.points[indices[:,1], :]

    normals[:, 0:2] = np.array([-(pnts1[:,1] - pnts2[:,1]), (pnts1[:,0] - pnts2[:,0])]).T

    inside_vector = domain_point - pnts1

        # Dot product of all the normal vectors with the inside-pointing vector
    inOrOut = np.einsum('ij,ij->i',normals,inside_vector)
        # Normalize (force to be = 1, but retain negative values)
    inOrOut /= np.abs(inOrOut)

    normals = np.einsum('ij,i->ij',normals,inOrOut)
    normals = np.einsum('ij,i->ij', normals, np.linalg.norm(normals, axis=1)**-1)

    edges.cell_arrays['Normals'] = normals
    return edges

