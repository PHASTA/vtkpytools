import vtk
import pyvista as pv
import numpy as np

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

    Returns
    -------
    numpy.ndarray
    """
    npoints = np.log(maxval/minval)/np.log(growthrate)
    npoints = np.ceil(npoints)

    geomseries = np.geomspace(minval, maxval, npoints)
    if include_zero: geomseries = np.insert(geomseries, 0, 0.0)

    return geomseries

def unstructuredToPoly(unstructured_grid):
    """Convert vtk.UnstructruedGrid to vtk.PolyData"""
    geom = vtk.vtkGeometryFilter()
    geom.SetInputData(unstructured_grid)
    geom.Update()
    return pv.wrap(geom.GetOutput())

def orderPolyDataLine(polydata):
    """Put line PolyData points in order"""
    strip = vtk.vtkStripper()
    strip.SetInputData(polydata)
    strip.Update()
    return pv.wrap(strip.GetOutput())
