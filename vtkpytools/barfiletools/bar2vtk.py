import vtk, os, argparse, datetime, platform, warnings
import pytomlpp
import pyvista as pv
import numpy as np
from pathlib import Path, PurePath

from .data import binaryVelbar, binaryStsbar, calcReynoldsStresses, compute_vorticity
from ..common import globFile
from .._version import __version__

def bar2vtk_main(args=None):
    """Function that runs the "binary" bar2vtk"""

    argsdict = bar2vtk_parse(args)

    if argsdict['subparser_name'] == 'cli':
        bar2vtkargs = argsdict.copy()
        bar2vtkargs['asciidata'] = bar2vtkargs.pop('ascii')
        for key in list(bar2vtkargs.keys()):
            if key not in bar2vtk_function.__code__.co_varnames:
                del bar2vtkargs[key]

    elif argsdict['subparser_name'] == 'toml':
        if argsdict['blank']:
            blankToml(argsdict['tomlfile'])
            return
        assert argsdict['tomlfile'].is_file()
        with argsdict['tomlfile'].open() as tomlfile:
            tomldict = pytomlpp.load(tomlfile)
        bar2vtkargs = tomldict['arguments']
        for key, val in bar2vtkargs.items():
            if key in ['barfiledir', 'outpath', 'blankvtmfile', 'velbar', 'stsbar']:
                if isinstance(val, list):
                    for i, item in enumerate(val):
                        val[i] = Path(item)
                else:
                    bar2vtkargs[key] = Path(val)

    tomlMetadata = bar2vtk_function(**bar2vtkargs, returnTomlMetadata=True)
    tomlReceipt(bar2vtkargs, tomlMetadata)

def bar2vtk_parse(args=None):
    GeneralDescription="""Tool for putting velbar and stsbar data onto a vtm grid."""

    ModeDescription="""There are two modes: cli and toml. Run:
    \tbar2vtk cli --help
    \tbar2vtk toml --help

    to get help information for each mode respectively."""

    CLIDescription="""Set bar2vtk settings via cli arguments and flags.

    Examples:
    \tbar2vtk.py blankDataBlock.vtm BinaryBars 10000
    \tbar2vtk.py blankDataBlock.vtm BinaryBars 10000-20000 --ts0=500

    The name of the output file will be the same as the blank VTM file suffixed
    with the timestep requested. So in the first example above, the output would be
    "blankDataBlock_10000.vtm".

    Time Step Windows:
    ------------------
    Submit a timestep argument with a '-' in it to request a timestep window be
    generated. This requires a '--ts0' argument be provided as well for calculating
    the windowed value."""

    TomlDescription="""Set bar2vtk settings via a toml configuration file.

    Examples:
    \tbar2vtk toml filledBar2vtkConfig.toml
    \tbar2vtk toml --blank  #outputs 'blankConfig.toml'
    \tbar2vtk toml --blank customName.toml #outputs 'customName.toml' """

    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        """To display defaults in help and have a multiline help description"""
        # Shamelessly ripped from https://stackoverflow.com/a/18462760/7564988
        pass

    ## Parsing script input
    parser = argparse.ArgumentParser(description=GeneralDescription,
                                    formatter_class=CustomFormatter,
                                    prog='bar2vtk')
    parser.add_argument('-v', '--version', action='version', version=__version__)
    subparser = parser.add_subparsers(title='Modes', description=ModeDescription, dest='subparser_name')

    # CLI Parser Setup
    cliparser = subparser.add_parser('cli', description=CLIDescription,
                                     formatter_class=CustomFormatter,
                                     help='Command Line Interface mode uses standard flags and cli arguments')

    cliparser.add_argument('blankvtmfile', help="MultiBlock VTK file that contains 'grid' and 'wall'", type=Path)
    cliparser.add_argument('barfiledir', help='Path to *bar file directory', type=Path)
    cliparser.add_argument('timestep', help='Timestep of the barfiles. May be range', type=str)
    cliparser.add_argument('--ts0','--bar-average-start',
                        help='Starting timestep of the averaging process. Only used'
                            ' for generating windows.',
                        type=int, default=-1)
    cliparser.add_argument('-f','--new-file-prefix',
                        help='Prefix for the new file. Will have timestep appended.',
                        type=str)
    cliparser.add_argument('--outpath', help='Custom path for the output VTM file.'
                                        ' Current directory used if not given', type=Path)
    cliparser.add_argument('--velonly', help='Only process velbar file', action='store_true')
    cliparser.add_argument('--debug', help='Load raw stsbar data into VTM', action='store_true')
    cliparser.add_argument('-a', '--ascii', help='Read *bar files as ASCII', action='store_true')
    cliparser.add_argument('--velbar', help='Path to velbar file(s)', type=Path, nargs='+', default=[])
    cliparser.add_argument('--stsbar', help='Path to stsbar file(s)', type=Path, nargs='+', default=[])

    # Toml Parser Setup
    tomlparser = subparser.add_parser('toml', description=TomlDescription, formatter_class=CustomFormatter,
                                      help='Toml mode uses configuration files in the toml format')
    tomlparser.add_argument('-b', '--blank', help='Create blank toml', action='store_true')
    tomlparser.add_argument('tomlfile', nargs='?', help='Run bar2vtk using toml config file',
                            type=Path)

    args = vars(parser.parse_args(args))

    return args

def bar2vtk_function(blankvtmfile: Path, barfiledir: Path, timestep: str, \
                     ts0: int=-1,  new_file_prefix: str='', outpath: Path=None, \
                     velonly=False, debug=False, asciidata=False, \
                     velbar=[],     stsbar=[], returnTomlMetadata=False):
    """Convert velbar and stsbar files into 2D vtk files

    See bar2vtk_commandline help documentation for more information.

    Parameters
    ----------
    blankvtmfile : Path
        Path to blank VTM file. Data is loaded onto this file and then saved to
        different file. VTM file should contain 'grid' and 'wall' blocks.
    barfiledir : Path
        Directory to where the velbar and stsbar files are located
    timestep : str
        String of which timestep the data should be loaded from. Can either be
        a single integer ('1000') or two integers delimited with a -
        ('1000-2000'). The later implies that a split timewindow should be
        calculated.
    ts0 : int
        The timestep number where the velbar and stsbar averaging started. Note
        this is only used if a split timewindow calculation is requested.
        (Default: -1).
    new_file_prefix : str
        If given, the newly created file will take the form of
        '{new_file_prefix}_{timestep}.vtm'. If not given, the file prefix will
        be the same as blankvtmfile. (Default: '').
    outpath : Path
        Path where the new file should be written. If not given, the new file
        will be placed in the current working directory.
    velonly : bool
        Do not include the stsbar file. (Default: False)
    debug : bool
        Adds raw stsbar array as data field. This is purely for debugging
        purposes. (Default: False)
    asciidata : bool
        Whether file paths are in ASCII format. (Default: False)
    velbar : List of Path
        Path(s) to velbar files. If doing time windows, must have two Paths. (Default: [])
    stsbar : List of Path
        Path(s) to stsbar files. If doing time windows, must have two Paths. (Default: [])
    """

    ## ---- Process/check script arguments
    assert blankvtmfile.is_file()
    assert barfiledir.is_dir()

    if debug and velonly:
        raise RuntimeError('velonly counteracts the effect of debug. Choose one or the other.')

    if not len(velbar) == len(stsbar):
        raise ValueError('velbar and stsbar must be given same number of paths'
                        ', given {} and {}, respectively'.format(len(velbar), len(stsbar)))

    for flag, arg in {'velbar':velbar, 'stsbar':stsbar}.items():
        if len(arg) > 2:
            pathStrings = '\n\t' + '\n\t'.join([x.as_posix() for x in arg])
            raise ValueError('{} can only contain two paths max.'
                            ' The following were given:{}'.format(flag, pathStrings))

        if len(arg) == 2 and not '-' in timestep:
            raise ValueError('{} was given two paths, but timestep was not given range.'.format(flag))

    if new_file_prefix:
        vtmName = Path(new_file_prefix + '_' + timestep + '.vtm')
    else:
        vtmName = Path(os.path.splitext(blankvtmfile.name)[0] + '_' + timestep + '.vtm')

    vtmPath = (outpath if outpath else os.getcwd()) / vtmName

    velbarReader = np.loadtxt if asciidata else binaryVelbar
    stsbarReader = np.loadtxt if asciidata else binaryStsbar

    ## ---- Loading data arrays
    if '-' in timestep and ts0 == -1:
        raise RuntimeError("Starting timestep of bar field averaging required (ts0)")
    print('Using data files:')
    velbarArray, velbarPaths = getBarData(velbar, timestep, barfiledir,
                                              velbarReader, ts0, 'velbar')
    if not velonly:
        stsbarArray, stsbarPaths = getBarData(stsbar, timestep, barfiledir,
                                                  stsbarReader, ts0, 'stsbar')

    ## ---- Load DataBlock
    dataBlock = pv.MultiBlock(blankvtmfile.as_posix())
    grid = dataBlock['grid']
    wall = dataBlock['wall']

    ## ---- Load *bar data into dataBlock
    grid['Pressure'] = velbarArray[:,0]
    grid['Velocity'] = velbarArray[:,1:4]

    if not velonly:
        ReyStrTensor = calcReynoldsStresses(stsbarArray, velbarArray)
        grid['ReynoldsStress'] = ReyStrTensor

    if debug and not velonly:
        grid['stsbar'] = stsbarArray

    grid = grid.compute_gradient(scalars='Velocity')
    grid = compute_vorticity(grid, scalars='Velocity')

    ## ---- Copy data from grid to wall object
    wall = wall.sample(grid)

    dataBlock['grid'] = grid
    dataBlock['wall'] = wall
    print('Saving dataBlock file to: {}'.format(vtmPath), end='')
    dataBlock.save(vtmPath)
    print('\tDone!')
    if returnTomlMetadata:
        tomlMetadata = {}
        tomlMetadata['vtmPath'] = vtmPath
        tomlMetadata['velbarPaths'] = velbarPaths
        if not velonly:
            tomlMetadata['stsbarPaths'] = stsbarPaths

        return tomlMetadata

def getBarData(_bar: list, timestep_str: str, barfiledir: Path, _barReader,
                   ts0: int, globname: str):
    """Get array of data from bar2vtk arguments"""
    if '-' in timestep_str:
        timesteps = [int(x) for x in timestep_str.split('-')]
        print('Creating timewindow between {} and {}'.format(timesteps[0], timesteps[1]))
        if not _bar:
            _barPaths = []
            for timestep in timesteps:
                _barPaths.append(globFile(r'^{}\.{}(?![\d|-]).*$'.format(globname, timestep), barfiledir, regex=True))
        else:
            _barPaths = _bar

        print('\t{}\t{}'.format(_barPaths[0], _barPaths[1]))
        _barArrays = []
        for i in range(2):
            _barArrays.append(_barReader(_barPaths[i]))

        _barArray = (_barArrays[1]*(timesteps[1] - ts0) -
                     _barArrays[0]*(timesteps[0] - ts0)) / (timesteps[1] - timesteps[0])

        print('Finished computing timestep window')
    else:
        _barPaths = _bar if _bar else \
            (globFile(r'^{}\.{}(?![\d|-]).*$'.format(globname, timestep_str), barfiledir, regex=True))
        print('\t{}'.format(_barPaths))
        _barArray = _barReader(_barPaths)

    return _barArray, _barPaths

def blankToml(tomlfilepath: Path, returndict=False):
    """Write blank toml file to tomlfilepath"""
    tomldict = { 'arguments': {
        'blankvtmfile': 'path/to/blankVTMFile (required)',
        'barfiledir': 'path/to/directory/of/*barfiles (required)',
        'timestep': 'timestep (range) of data (required)',
        'ts0': -1,
        'new_file_prefix': '',
        'outpath': 'path/to/different/output/directory',
        'velonly': False,
        'debug': False,
        'asciidata': False,
        'velbar': [],
        'stsbar': [],
    }}

    with tomlfilepath.open('w') as file:
        pytomlpp.dump(tomldict, file)
    if returndict: return tomldict

def tomlReceipt(args: dict, tomlMetadata: dict):
    """Creates a receipt of the created file

    Parameters
    ----------
    args : dict
        Arguments passed to bar2vtk_function
    tomlMetadata : dict
        Extra metadata given by bar2vtk_function
    """

    convertArray = [(PurePath, lambda x: x.as_posix()),
                    (type(None), lambda x: '')
                    ]

    _convertArray2TomlTypes(args, convertArray)

    vtmPath = tomlMetadata['vtmPath']
    _convertArray2TomlTypes(tomlMetadata, convertArray)

    tomldict = {'arguments': args}

    meta = {}
    meta['bar2vtk_vars'] = tomlMetadata
    meta['created'] = datetime.datetime.now()
    meta['vtkpytools_version'] = __version__
    meta['pyvista_version'] = pv.__version__
    meta['vtk_version'] = vtk.VTK_VERSION
    meta['python_version'] = platform.python_version()
    meta['uname'] = platform.uname()._asdict()

    meta['directory'] = os.getcwd()

    tomldict['metadata'] = meta

    vtmDir = Path(os.path.splitext(vtmPath)[0])
    if not vtmDir.is_dir():
        warnings.warn('Directory {} does not exist. '
                      'Cannot create toml receipt.'.format(vtmDir.as_posix()),
                      RuntimeWarning)
    else:
        tomlPath = vtmDir / Path('receipt.toml')
        with tomlPath.open(mode='w') as file:
            pytomlpp.dump(tomldict, file)

def _convertArray2TomlTypes(array, convertArray: list):
    """Recursively convert objects in array to toml compatible objects based on convertArray

    Parameters
    ----------
    array : dict or list
        Array of objects to be converted. Can be either dictionary (which will
        only convert values in the dictionary) or list.
    convertArray : list
        List of conversion rules. Each item should be a tuple/list whose first
        entry is the type to be converted and the second entry is the function
        that converts it. For example `(float, lambda x: str(x))` would convert
        types to strings.
    """
    if isinstance(array, dict):
        for key, val in array.items():
            for typeobj, convert in convertArray:
                if isinstance(val, typeobj):
                    array[key] = convert(val)
            if isinstance(val, (list, dict, tuple)):
                _convertArray2TomlTypes(val, convertArray)
    else:
        for i, item in enumerate(array):
            for typeobj, convert in convertArray:
                if isinstance(item, typeobj):
                    array[i] = convert(item)
            if isinstance(item, (list, dict, tuple)):
                _convertArray2TomlTypes(item, convertArray)
