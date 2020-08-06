import numpy as np
import vtk
from pathlib import Path
from scipy.spatial import Delaunay
import pyvista as pv

def form2DGrid(coords_path, connectivity_path=None,
               connectivity_zero_base=False) -> pv.UnstructuredGrid:
    """Create 2D VTK UnstructuredGrid from coordinates and connectivity

    Parameters
    ----------
    coords_path : Path
        Path to coordinates file.
    connectivity_path : Path, optional
        Path to connectivity file. If not given, will create mesh from the
        points given in the coords_path file. (default: None)
    connectivity_zero_base : bool, optional
        If the given connectivity file indexs the mesh points starting at 0.
        Only used if connectivity_path is given.

    Returns
    -------
    pv.UnstructuredGrid
        Pyvista UnstructuredGrid object with the grid loaded.
    """
    coords = np.loadtxt(coords_path)
    coords[:,2] = np.zeros(coords.shape[0])

    if not connectivity_path:
        # Generate mesh from points
        print('Generating Mesh...')
        mesh = Delaunay(coords[:,0:2])
        print('Finished Meshing!')

        nCells = mesh.simplices.shape[0]
            # mesh.simplices contains the connectivity array
        connectivity = mesh.simplices
        connectivity = np.hstack((np.ones((nCells,1), dtype=np.int64)*3, connectivity))

        offsets = np.arange(3, mesh.simplices.size+4, 3, dtype=np.int64)

            # vtk.VTK_TRIANGLE is just an integer
        cell_types = np.ones(nCells, dtype=np.int64) * vtk.VTK_TRIANGLE
    else:
        cnnFile = np.loadtxt(connectivity_path, dtype=np.int64)
        connectivity = cnnFile[:,1:]
        nCells = connectivity.shape[0]
        nPnts = connectivity.shape[1]
        if nPnts == 3:
            cell_type = vtk.VTK_TRIANGLE # ==int(5)
        elif nPnts == 4:
            cell_type = vtk.VTK_QUAD # ==int(9)
        else:
            raise ValueError(f'This connectivity file has the wrong number of points. Must be either 3 or 4 points per cell, this has {nPnts}')

        connectivity = np.hstack((np.ones((nCells,1), dtype=np.int64)*nPnts, connectivity))
        offsets = np.arange(0, connectivity.size+1, nPnts+1, dtype=np.int64)
        cell_types = np.ones(nCells, dtype=np.int64) * cell_type

    grid = pv.UnstructuredGrid(offsets, connectivity, cell_types, coords)

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

def getGeometricSeries(maxval, minval, growthrate, include_zero=True):
    """Return geometric series based on inputs.

    A geometric series is defined as one where each successive point is the
    multiplicative increase of the previous point:

        n_{i+1} = n_{i} * r

    where r is the growth rate of the series.

    Parameters
    ----------
    maxval : float
        Maximum value that should be reached by series
    minval : float
        Initial value of the series
    growthrate : float
        Growth rate of the series
    include_zero : bool, optional
        Whether the series should insert a 0.0 at the beginning of the series
        (default is True)
    """
    npoints = np.log(maxval/minval)/np.log(growthrate)
    npoints = np.ceil(npoints)

    geomseries = np.geomspace(minval, maxval, npoints)
    if include_zero: geomseries = np.insert(geomseries, 0, 0.0)

    return geomseries
