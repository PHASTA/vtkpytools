import vtk
import pyvista as pv
import numpy as np
from scipy.io import FortranFile
from pathlib import Path
import re
from typing import Union, Optional
from os import PathLike

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

def vCutter(input_data: pv.PolyData, cut_function):
    """Returns the intersection of input_data and cut_function

    Wrapper around vtkCutter filter. Output contains interpolated data from
    input_data to the intersection location. Note that cell data is NOT passed
    through.

    Parameters
    ----------
    input_data : pyvista.PointSet
        Data that will be cut by the cut_function. Intersected point will have
        data interpolated.
    cut_function : vtk.vtkImplicitFunction
        VTK function that cuts the input_data. Most common example would be
        vtkPlane.
    """

    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(cut_function)
    cutter.SetInputData(input_data)
    cutter.Update()
    return pv.wrap(cutter.GetOutput())

class Profile(pv.PolyData):
    """Wrap of pyvista.PolyData that includes walldata attribute

    Use case is for storage of wall local data (boundary layer metrics, Cf
    etc.) with profiles that correspond to that wall local data.

    Attributes
    ----------
    walldata : dict
        Dictionary of data from the wall corresponding to the profile location.
        Generated using `setWallDataFromPolyDataPoint`
    """
    walldata = {}

    def setWallDataFromPolyDataPoint(self, PolyPoint: pv.PolyData):
        """Set walldata attribute from PolyData Point

        Primary use case is the using the output of vtkpytools.vCutter()

        Parameters
        ----------
        PolyPoint : pv.PolyPoint
            PolyData object containing one point. The point arrays attached to
            the points are converted to a `dict` and set to `walldata`.
        """
        if PolyPoint.n_points != 1:
            raise RuntimeError('Profile should only have 1 wallpoint, {:d} given'.format(
                                                                    PolyPoint.n_points))
        self.walldata = dict(PolyPoint.point_arrays)
        self.walldata['Point'] = PolyPoint.points

def readBinaryArray(path: Union[str, PathLike], ncols: int) -> np.ndarray:
    """Get array from Fortran binary file.

    Parameters
    ----------
    path : Path
        Path to Fortran binary array.
    ncols : uint
        Number of columns in the binary file.
    """
    array = FortranFile(path, 'r')
    array = array.read_reals()
    nrows = int(array.shape[0]/ncols)
    array = np.reshape(array, (nrows, ncols))

    return array

def globFile(globstring, path: Path, regex=False) -> Optional[Path]:
    """ Glob for one file in directory, then return.

    If it finds more than one file matching the globstring, it will error out.

    Parameters
    ----------
    globstring : str
        String used to glob for file
    path : Path
        Path where file should be searched for
    regex : bool
        Whether globstring should be interpreted by Python regex (default is
        False)
    """
    if not regex:
        globlist = list(path.glob(globstring))
        if len(globlist) == 1:
            assert globlist[0].is_file()
            return globlist[0]
        elif len(globlist) > 1:
            raise RuntimeError('Found multiple files matching'
                            '"{}" in {}:\n\t{}'.format(globstring, path,
                                                    '\n\t'.join([x.as_posix() for x in globlist])))
        else:
            raise RuntimeError('Could not find file matching'
                                '"{}" in {}'.format(globstring, path))
    elif regex:
        filestrings = [x.name for x in path.iterdir()]
        globlist = list(filter(re.compile(globstring).match, filestrings))
        if len(globlist) == 1:
            filePath = path / Path(globlist[0])
            assert filePath.is_file()
            return filePath
        elif len(globlist) > 1:
            raise RuntimeError('Found multiple files matching'
                            '"{}" in {}:\n\t{}'.format(globstring, path,
                                                    '\n\t'.join(globlist)))
        else:
            raise RuntimeError('Could not find file matching'
                                '"{}" in {}'.format(globstring, path))

