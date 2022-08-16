import vtk, os, argparse, datetime, platform, warnings
import pytomlpp
import pyvista as pv
import numpy as np
from pathlib import Path, PurePath

from .data import binaryVelbar, binaryStsbar, binaryIDDESbar, binaryVelbarSclr, binaryStsbarKeq
from .data import binaryStsbarWithpp, binaryStsbarWithConsvStress, binarySMRbar
from .data import binaryBarFiles
from .data import calcReynoldsStresses, calcPresVel, calcTurbTrans, calcvelStrain
from .data import calcStreamlineRotation, calcVelocityAlongStreamline
from .data import calcVelocityStreamGradient, calcPressureStreamGradient, calcVelocityStreamHessian
from .data import calcReynoldsStressAlongStreamline, calcRSSGradientStreamline
from .data import entirewallAlignRotationTensor, computewalltangent
from .data import computeMomBalance
from .data import computemetrictensor, computeTauM, computeTauC
from .data import findWallDist
from ..common import globFile, readBinaryArray, writeBinaryArray
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
            if key in ['barfiledir', 'outpath', 'blankvtmfile', 'velbar', 'stsbar', 'IDDESbar']:
                if isinstance(val, list):
                    for i, item in enumerate(val):
                        val[i] = Path(item)
                else:
                    bar2vtkargs[key] = Path(val)

    tomlMetadata = bar2vtk_function(**bar2vtkargs, returnTomlMetadata=True)
    tomlReceipt(bar2vtkargs, tomlMetadata)


def bar2vtk_parse(args=None):
    GeneralDescription="""Tool for putting velbar, stsbar and IDDESbar data onto a vtm grid."""

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
    cliparser.add_argument('--IDDESbar', help='Path to IDDESbar file(s)', type=Path, nargs='+', default=[])
    cliparser.add_argument('--stsbarKeq', help='Path to stsbarKeq file(s)', type=Path, nargs='+', default=[])    
    cliparser.add_argument('--consrvstress',
                           help='Calculate Reynolds stress assuming conservative stsbar data (only consv stress in stsbar)',
                           type=bool, nargs=1, default=False)
    cliparser.add_argument('--periodicity',
                           help='Flag for accounting periodicity correctly while averaging',
                           type=bool, nargs=1, default=True) 
    cliparser.add_argument('--nsons',
                           help='Number of nsons used for averaging',
                           type=int, nargs=1, default=0)     
    cliparser.add_argument('--loadIDDES', help='Also process IDDESbar file', action='store_true')
    cliparser.add_argument('--loadstsbarKeq', help='Also process stsbarKeq file', action='store_true')    
    cliparser.add_argument('--nsclr',
                        help='Number of scalars in velbar',
                        type=int, default=0)    
    cliparser.add_argument('--streamcoord', help='Compute quantities in streamline coordinates', action='store_true')    
    cliparser.add_argument('--barformat', help='Select the format of bar files', type=int, default=1)    
    cliparser.add_argument('--loadSMRbar', help='Read SMRbar files', type=bool, default=False)    
    cliparser.add_argument('--loadSMRbar2', help='Read SMRbar2 files', type=bool, default=False)    
    cliparser.add_argument('--addpptostsbar', help='<pp> is added to stsbar', type=bool, default=False)  
    cliparser.add_argument('--addconsvstresstostsbar', help='Conservative stress are also added to stsbar', type=bool, default=False)
    cliparser.add_argument('--adduttovelbar', help='<u_{i,t}> is added to velbar', type=bool, default=False)  
    cliparser.add_argument('--ncolSMRbar', help='Number of columns in SMRbar file', type=int, default=42)
    cliparser.add_argument('--writeCombinedarray', help='Combine bar files to output a single bar file', type=bool, default=False) 
    cliparser.add_argument('--subtractBars', help='Subract bar files to create a sub-window', type=bool, default=False) 
    cliparser.add_argument('--d2Wall', help='Calculate distance to nearest wall point', type=bool, default=False) 

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
                     velbar=[],     stsbar=[],   IDDESbar=[], stsbarKeq=[], SMRbar=[], SMRbar2=[], returnTomlMetadata=False, \
                     consrvstress=False, periodicity=True, nsons=0, loadIDDES=False, \
                     loadstsbarKeq=True, nsclr=0, streamcoord=False, barformat=1, \
                     loadSMRbar=False, loadSMRbar2=False, addpptostsbar=False, \
                     addconsvstresstostsbar=False, adduttovelbar=False,ncolSMRbar=42, \
                     writeCombinedarray=False, subtractBars=False, d2Wall=False):
    """Convert velbar, stsbar, stsbarKeq and iddes files into 2D vtk files

    See bar2vtk_commandline help documentation for more information.

    Parameters
    ----------
    blankvtmfile : Path
        Path to blank VTM file. Data is loaded onto this file and then saved to
        different file. VTM file should contain 'grid' and 'wall' blocks.
    barfiledir : Path
        Directory to where the velbar, stsbar and iddes files are located
    timestep : str
        String of which timestep the data should be loaded from. Can either be
        a single integer ('1000') or two integers delimited with a -
        ('1000-2000'). The later implies that a split timewindow should be
        calculated.
    ts0 : int
        The timestep number where the velbar, stsbar and IDDESbar averaging started. Note
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
        Do not include the stsbar and IDDESbar file. (Default: False)
    debug : bool
        Adds raw stsbar array as data field. This is purely for debugging
        purposes. (Default: False)
    asciidata : bool
        Whether file paths are in ASCII format. (Default: False)
    velbar : List of Path
        Path(s) to velbar files. If doing time windows, must have two Paths. (Default: [])
    stsbar : List of Path
        Path(s) to stsbar files. If doing time windows, must have two Paths. (Default: [])
    IDDESbar : List of Path
        Path(s) to IDDESbar files. If doing time windows, must have two Paths. (Default: [])        
    stsbarKeq : List of Path
        Path(s) to stsbarKeq files. If doing time windows, must have two Paths. (Default: [])
    SMRbar : List of Path
        Path(s) to SMRbar files. If doing time windows, must have two Paths. (Default: [])    
    SMRbar2 : List of Path
        Path(s) to SMRbar2 files. If doing time windows, must have two Paths. (Default: [])            
    returnTomlMetadata : bool
        Whether to return the metadata required for writing a toml reciept file. (Default: False)
    consrvstress : bool
        Whether the data in the stsbar file is conservative or not (only consv 
        stress in stsbar -> stsbar should not contain both nodal and conservative fields). This
        affects both how the stsbar file is read and how the Reynolds stresses
        are calculated from them. See calcReynoldsStresses() for more
        information. (Default: False)
    periodicity : bool
        Whether division by nsons included the periodic node or not. This option is only 
        valid for structured grids for the moment.
    nsons : int
        Number of nson along periodic direction used for this case.
    loadIDDES : bool
        Include IDDESbar files.
    loadstsbarKeq : bool
        Include stsbarKeq files.    
    nsclr : int
        Number of scalar solution variables in velbar.
    streamcoord : bool
        Transform flow variables and gradients in streamline coordiate system
    barformat : int
        Select the format of bar files
    loadSMRbar : bool
        Read SMRbar files
    loadSMRbar2 : bool
        Read SMRbar2 files
    addpptostsbar : bool
        <pp> is added to stsbar
    addconsvstresstostsbar : bool
        Conservative stress are also added to stsbar
    adduttovelbar : bool
        <u_{i,t}> is added to velbar
    ncolSMRbar : int
        Number of columns in SMRbar file
    writeCombinedarray : bool
        Combine bar files to output a single bar file
    subtractBars : bool
        Subracts two (and only two) bar files to create a sub-window using the formatting
        timerange1 - timerange2 = subrange, or in practice:
            The input ['ts1-ts3','ts1-ts2'] ouputs the window ts2-ts3
            -> Only implemented for new bar file format
    d2Wall : bool
        Calculates distance fomr each grid point to the nearest wall point
        
    """

    ## ---- Process/check script arguments
    assert blankvtmfile.is_file()
    assert barfiledir.is_dir()

    if debug and velonly:
        raise RuntimeError('velonly counteracts the effect of debug. Choose one or the other.')

    if not len(velbar) == len(stsbar):
        raise ValueError('velbar and stsbar must be given same number of paths'
                        ', given {} and {}, respectively'.format(len(velbar), len(stsbar)))
        
    if not len(velbar) == len(IDDESbar):
        raise ValueError('velbar and IDDESbar must be given same number of paths'
                        ', given {} and {}, respectively'.format(len(velbar), len(IDDESbar))) 
        
    if not len(velbar) == len(stsbarKeq):
        raise ValueError('velbar and stsbarKeq must be given same number of paths'
                        ', given {} and {}, respectively'.format(len(velbar), len(stsbarKeq)))         
        
    if not len(velbar) == len(SMRbar):
        raise ValueError('velbar and SMRbar must be given same number of paths'
                        ', given {} and {}, respectively'.format(len(velbar), len(SMRbar)))          

    if not len(velbar) == len(SMRbar2):
        raise ValueError('velbar and SMRbar2 must be given same number of paths'
                        ', given {} and {}, respectively'.format(len(velbar), len(SMRbar2)))          


    for flag, arg in {'velbar':velbar, 'stsbar':stsbar, 'IDDESbar':IDDESbar, 'stsbarKeq':stsbarKeq, 'SMRbar':SMRbar}.items():
        if len(arg) > 2:
            pathStrings = '\n\t' + '\n\t'.join([x.as_posix() for x in arg])
            raise ValueError('{} can only contain two paths max.'
                            ' The following were given:{}'.format(flag, pathStrings))

        if len(arg) == 2 and not '-' in timestep:
            raise ValueError('{} was given two paths, but timestep was not given range.'.format(flag))

    if barformat == 1:
        if new_file_prefix:
            vtmName = Path(new_file_prefix + '_' + timestep + '.vtm')
        else:
            vtmName = Path(os.path.splitext(blankvtmfile.name)[0] + '_' + timestep + '.vtm')     
        if subtractBars:
            raise RuntimeError('Subracting bar files is not yet implemented for bar format 1')
    elif barformat == 2:
        print(timestep)
        skiplist = []
        for time in timestep:
            timestepArray = [int(x) for x in time.split('-')]
            if(np.size(timestepArray)==3):
                timeskip = timestepArray[2]
            else:
                timeskip = 1            
            skiplist.append(timeskip)
        if skiplist[:-1] == skiplist[1:]:
            skipstr=str(skiplist[0])
        else:
            skipstr = "MultipleSkip"
            print("WARNING! Multiple time skips being combined.")
        timestepint = timestep[-1]
        timestepEnd = [int(x) for x in timestepint.split('-')]
        timestepint = timestep[0]
        timestepStart = [int(x) for x in timestepint.split('-')]        
        print(timestepEnd[1])
        if new_file_prefix:
            vtmName = Path(new_file_prefix + '_' + str(timestepEnd[1]) + '.vtm')
        else:
            if subtractBars:
                if timestepEnd[1] > timestepStart[1]:
                    raise RuntimeError('Check your window ordering for subwindowing! Formatting should be: \n', \
                                       '[\'whole window\', \'subwindow to subtract\']')
                vtmName = Path(os.path.splitext(blankvtmfile.name)[0] + '_' + str(timestepEnd[1]) + '-' + str(timestepStart[1]) + '-' + skipstr + '.vtm')
            else:
                vtmName = Path(os.path.splitext(blankvtmfile.name)[0] + '_' + str(timestepStart[0]) + '-' + str(timestepEnd[1]) + '-' + skipstr + '.vtm')
            if writeCombinedarray:
                fileCombineSuffix = str(timestepStart[0]) + '.' + str(timestepEnd[1]) + '.' + skipstr
        

    vtmPath = (outpath if outpath else os.getcwd()) / vtmName


    ncolvelbar = 5+nsclr
    if adduttovelbar:
        ncolvelbar = ncolvelbar + 3
        
    ncolstsbar = 6
    if addpptostsbar:
        ncolstsbar = ncolstsbar + 1
    if addconsvstresstostsbar:
        ncolstsbar = ncolstsbar + 9
        
    ncolIDDESbar = 10
    ncolstsbarKeq = 10

    velbarReader = np.loadtxt if asciidata else binaryVelbar
    
    if (nsclr > 0):
        velbarReader = np.loadtxt if asciidata else binaryVelbarSclr
            
    barReader = np.loadtxt if asciidata else binaryBarFiles
    
    
    if asciidata:
        stsbarReader = np.loadtxt
    elif consrvstress:
        stsbarReader = lambda path: readBinaryArray(path, ncols=9)
    else:
        if addpptostsbar and not addconsvstresstostsbar:
            stsbarReader = binaryStsbarWithpp
        elif addconsvstresstostsbar:
            stsbarReader = binaryStsbarWithConsvStress
        else:
            stsbarReader = binaryStsbar

    if asciidata:
        stsbarKeqReader = np.loadtxt
    else:
        stsbarKeqReader = binaryStsbarKeq

        
    IDDESbarReader = np.loadtxt if asciidata else binaryIDDESbar
    
    SMRbarReader = np.loadtxt if asciidata else binarySMRbar
    
    ## ---- Loading data arrays
    if barformat == 1:
        if '-' in timestep and ts0 == -1:
            raise RuntimeError("Starting timestep of bar field averaging required (ts0)")
        print('Using data files:')
        velbarArray, velbarPaths = getBarData(velbar, timestep, barfiledir,
                                                  velbarReader, ts0, 'velbar')
        if not velonly:
            stsbarArray, stsbarPaths = getBarData(stsbar, timestep, barfiledir,
                                                      stsbarReader, ts0, 'stsbar')
            
            if loadIDDES:
                IDDESbarArray, IDDESbarPaths = getBarData(IDDESbar, timestep, barfiledir,
                                                      IDDESbarReader, ts0, 'IDDESbar')  
                
            if loadstsbarKeq:
                stsbarKeqArray, stsbarKeqPaths = getBarData(stsbarKeq, timestep, barfiledir,
                                                      stsbarKeqReader, ts0, 'stsbarKeq')         
                
            if loadSMRbar:
                SMRbarArray, SMRbarPaths = getBarData(SMRbar, timestep, barfiledir,
                                                      SMRbarReader, ts0, 'SMRbar')
                
            if loadSMRbar2:
                SMRbar2Array, SMRbar2Paths = getBarData(SMRbar2, timestep, barfiledir,
                                                      SMRbarReader, ts0, 'SMRbar2')
            
    elif barformat == 2:
        print('Using data files:')
        velbarArray, velbarPaths = getBarData_Mode2(velbar, timestep, barfiledir,
                                                  barReader, 'velbar', ncolvelbar, subtractBars)
        if writeCombinedarray:
            fname = 'velbar.' + fileCombineSuffix
            barwritePath = barfiledir / fname
            writeBinaryArray(barwritePath,velbarArray.T)
        
        if not velonly:
            stsbarArray, stsbarPaths = getBarData_Mode2(stsbar, timestep, barfiledir,
                                                      barReader, 'stsbar', ncolstsbar, subtractBars)
            
            if writeCombinedarray:
                fname = 'stsbar.' + fileCombineSuffix
                barwritePath = barfiledir / fname
                writeBinaryArray(barwritePath,stsbarArray.T)            
            
            if loadIDDES:
                IDDESbarArray, IDDESbarPaths = getBarData_Mode2(IDDESbar, timestep, barfiledir,
                                                      barReader, 'IDDESbar', ncolIDDESbar, subtractBars)  
                
                if writeCombinedarray:
                    fname = 'IDDESbar.' + fileCombineSuffix
                    barwritePath = barfiledir / fname
                    writeBinaryArray(barwritePath,IDDESbarArray.T)
                
            if loadstsbarKeq:
                stsbarKeqArray, stsbarKeqPaths = getBarData_Mode2(stsbarKeq, timestep, barfiledir,
                                                      barReader, 'stsbarKeq', ncolstsbarKeq, subtractBars)    
                
                if writeCombinedarray:
                    fname = 'stsbarKeq.' + fileCombineSuffix
                    barwritePath = barfiledir / fname
                    writeBinaryArray(barwritePath,stsbarKeqArray.T)                
                
            if loadSMRbar:
                SMRbarArray, SMRbarPaths = getBarData_Mode2(SMRbar, timestep, barfiledir,
                                                      barReader, 'SMRbar', ncolSMRbar, subtractBars)      
                if writeCombinedarray:
                    fname = 'SMRbar.' + fileCombineSuffix
                    barwritePath = barfiledir / fname
                    writeBinaryArray(barwritePath,SMRbarArray.T)                
            
            if loadSMRbar2:
                SMRbar2Array, SMRbar2Paths = getBarData_Mode2(SMRbar2, timestep, barfiledir,
                                                      barReader, 'SMRbar2', ncolSMRbar, subtractBars)    

                if writeCombinedarray:
                    fname = 'SMRbar2.' + fileCombineSuffix
                    barwritePath = barfiledir / fname
                    writeBinaryArray(barwritePath,SMRbar2Array.T)                           
                    

    ## ---- Load DataBlock
    dataBlock = pv.MultiBlock(blankvtmfile.as_posix())
    grid = dataBlock['grid']
    wall = dataBlock['wall']
    
    ## ---- Correct velbar for periodicity
    if(not(periodicity)):
        velbarArray = velbarArray*nsons/(nsons-1)
        stsbarArray = stsbarArray*nsons/(nsons-1)
        if loadIDDES:
            IDDESbarArray = IDDESbarArray*nsons/(nsons-1)
        if loadstsbarKeq:
            stsbarKeqArray = stsbarKeqArray*nsons/(nsons-1)


    ## ---- Load *bar data into dataBlock
    if debug:
        grid['velbar'] = velbarArray
        
    # velbarArraytmp = velbarArray[:,1:5]
    # velbarArray = velbarArraytmp
    grid['Pressure'] = velbarArray[:,0]
    grid['Velocity'] = velbarArray[:,1:4]
    
    if streamcoord:
        print('Computing velocity along streamlines!')
        grid['StreamlineRotation'] = calcStreamlineRotation(grid['Velocity'])
        grid['VelocityStreamline'] = calcVelocityAlongStreamline(grid['Velocity'],grid['StreamlineRotation'])
    
    if(nsclr == 1):
        grid['savar'] = velbarArray[:,5]
    elif (nsclr == 2):
        grid['k'] = velbarArray[:,5]
        grid['omega'] = velbarArray[:,6]
        

    if not velonly:
        print('Computing Reynolds stresses!')
        ReyStrTensor = calcReynoldsStresses(stsbarArray, velbarArray, consrvstress)
        grid['ReynoldsStress'] = ReyStrTensor
        if addconsvstresstostsbar:
            grid['ReynoldsStressConsv'] = calcReynoldsStresses(stsbarArray[:,7:16], velbarArray, addconsvstresstostsbar)
        if streamcoord:
            grid['ReynoldsStressStream'] = calcReynoldsStressAlongStreamline(grid['ReynoldsStressConsv'],grid['StreamlineRotation'])
        if loadIDDES:
            grid['IDDESbar'] = IDDESbarArray
        if loadstsbarKeq:
            grid['stsbarKeq'] = stsbarKeqArray
            grid['presvel'] = calcPresVel(stsbarKeqArray, velbarArray)     
            grid['turbtrans'] = calcTurbTrans(stsbarKeqArray,velbarArray,stsbarArray)
            grid['turbke'] = 0.5*(ReyStrTensor[:,0]+ReyStrTensor[:,1]+ReyStrTensor[:,2])


    if debug and not velonly:
        grid['stsbar'] = stsbarArray

    if loadSMRbar:
        grid['BudgetSMR'] = SMRbarArray
        
    if loadSMRbar2:
        grid['BudgetSMR2'] = SMRbar2Array
        
    print('Computing derivatives of flow quantities!')
    grid = grid.compute_derivative(scalars='Velocity', gradient='gradient', vorticity='vorticity')
    grid = grid.compute_derivative(scalars='gradient', gradient='Vel-SecondDeriv')
    grid = grid.compute_derivative(scalars='Pressure', gradient='Pres-gradient')
    grid = grid.compute_derivative(scalars='ReynoldsStress', gradient='ReyStress-gradient')
    if addconsvstresstostsbar:
        grid = grid.compute_derivative(scalars='ReynoldsStressConsv', gradient='ReyStressConsv-gradient')

    
    if streamcoord:
        print('Computing derivatives of velocity along streamlines!')
        grid = grid.compute_derivative(scalars='VelocityStreamline', gradient='dvelStreamdx', vorticity='vorticityStreamdx')
        grid['vorticityStream'] = calcPressureStreamGradient(grid['vorticityStreamdx'],grid['Velocity'])
        grid['VelocityGradientStreamline'] = calcVelocityStreamGradient(grid['dvelStreamdx'],grid['Velocity'])
        grid['PressureGradientStreamline'] = calcPressureStreamGradient(grid['Pres-gradient'],grid['Velocity'])
    
        print('Computing hessian of velocity along streamlines!')
        grid = grid.compute_derivative(scalars='VelocityGradientStreamline', gradient='d2usdsdx')
        grid['VelocityHessianStreamline'] = calcVelocityStreamHessian(grid['d2usdsdx'],grid['Velocity'])
        
        print('Computing Radius of curvature and e-folding distance!')
        LocalRadiusCurvatureTmp = (grid['vorticity'][:,2] + grid['VelocityGradientStreamline'][:,1])/grid['VelocityStreamline'][:,0]
        grid['LocalRadiusCurvature'] = 1/LocalRadiusCurvatureTmp
        grid['UoverR'] = grid['VelocityStreamline'][:,0]/LocalRadiusCurvatureTmp
        grid = grid.compute_derivative(scalars='UoverR', gradient='dUoverRdx')
        grid['dUoverRds'] = calcPressureStreamGradient(grid['dUoverRdx'],grid['Velocity'])
        
        efoldingDistanceTmp = grid['VelocityGradientStreamline'][:,0]/grid['VelocityStreamline'][:,0]
        grid['efoldingDistance'] = 1/efoldingDistanceTmp
    
        print('Computing derivatives of Reynolds stress along streamlines!')
        if not velonly:    
            grid = grid.compute_derivative(scalars='ReynoldsStressStream',gradient='dRSSstreamdx')
            grid['RSSGradientStreamline'] = calcRSSGradientStreamline(grid['dRSSstreamdx'],grid['Velocity'])    
        
            
    if not velonly:    
        if(loadstsbarKeq):           
            grid = grid.compute_derivative(scalars='presvel', gradient='PresDiff')
            grid = grid.compute_derivative(scalars='turbtrans', gradient='TurbTransDeriv')
            grid = grid.compute_derivative(scalars='turbke', gradient='TurbKEDeriv')
            grid = grid.compute_derivative(scalars='TurbKEDeriv', gradient='TurbKESecondDeriv')
            grid['VelStrain'] = calcvelStrain(stsbarKeqArray,velbarArray,grid['gradient'])             
            grid = grid.compute_derivative(scalars='VelStrain', gradient='VelStrainDeriv')

    if d2Wall:
        print('Computing d2Wall!')
        grid['d2Wall'] = findWallDist(grid, wall)
        
    print(grid.array_names)

        
    ## ---- Copy data from grid to wall object
    wall = wall.sample(grid)
    
    # if streamcoord:
    #     RotTensor = entirewallAlignRotationTensor(wall['Normals'], np.array([0,1,0]))
    #     wall['VelocityStreamline'] = calcVelocityAlongStreamline(wall['Velocity'],RotTensor)
    #     if not velonly:
    #         wall['ReynoldsStressStream'] = calcReynoldsStressAlongStreamline(wall['ReynoldsStress'],RotTensor)
            
    #     wall = wall.compute_derivative(scalars='VelocityStreamline', gradient='dvelStreamdx')
    #     tangentWall = computewalltangent(wall['Normals'])
    #     wall['VelocityGradientStreamline'] = calcVelocityStreamGradient(wall['dvelStreamdx'],tangentWall)
    #     wall['PressureGradientStreamline'] = calcPressureStreamGradient(wall['Pres-gradient'],tangentWall)
    
    #     wall = wall.compute_derivative(scalars='VelocityGradientStreamline', gradient='d2usdsdx')
    #     wall['VelocityHessianStreamline'] = calcVelocityStreamHessian(wall['d2usdsdx'],tangentWall)
        
    #     LocalRadiusCurvatureTmp = (wall['vorticity'][:,2] + wall['VelocityGradientStreamline'][:,1])/wall['VelocityStreamline'][:,0]
    #     wall['LocalRadiusCurvature'] = 1/LocalRadiusCurvatureTmp
    #     wall['UoverR'] = wall['VelocityStreamline'][:,0]/LocalRadiusCurvatureTmp
    #     wall = wall.compute_derivative(scalars='UoverR', gradient='dUoverRdx')
    #     wall['dUoverRds'] = calcPressureStreamGradient(wall['dUoverRdx'],tangentWall)
        
    #     efoldingDistanceTmp = wall['VelocityGradientStreamline'][:,0]/wall['VelocityStreamline'][:,0]
    #     wall['efoldingDistance'] = 1/efoldingDistanceTmp
    
    #     if not velonly:    
    #         wall = wall.compute_derivative(scalars='ReynoldsStressStream',gradient='dRSSstreamdx')
    #         wall['RSSGradientStreamline'] = calcRSSGradientStreamline(wall['dRSSstreamdx'],tangentWall)           
    

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
            if loadIDDES:
                tomlMetadata['IDDESbarPaths'] = IDDESbarPaths
            if loadstsbarKeq:
                tomlMetadata['stsbarKeqPaths'] = stsbarKeqPaths
            
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

def getBarData_Mode2(_bar: list, timestep_list: list, barfiledir: Path, _barReader,
                     globname: str, ncols: int, subtractBars: bool):
    """Get array of data from bar2vtk arguments"""
    _barPaths = []
    timeinterval = []
    count = 0
    if subtractBars:
        if len(timestep_list) != 2:
            raise RuntimeError('Only two timesteps can be subtracted')
            
    for timestep_str in timestep_list:
        if '-' in timestep_str:
            timesteptmp = [int(x) for x in timestep_str.split('-')]
            timesteps = timesteptmp[0:2]
            if(np.size(timesteptmp) == 3):
                timeskip = timesteptmp[2]
            else:
                timeskip = 1
            print('Using timewindows between {} and {} and skip={}'.format(timesteps[0], timesteps[1], timeskip))
            timeinterval.append(np.floor((timesteps[1]-timesteps[0])/timeskip + 0.4999))
            if not _bar:
                _barPaths.append(globFile(r'^{}\.{}.{}.{}(?![\d|-])*$'.format(globname, timesteps[0], timesteps[1], timeskip), barfiledir, regex=True))
            else:
                _barPaths.append(_bar[count])
        
            print('Finished computing timestep window')
        else:
            print('Please specify time interval in format ts0-ts1')
            exit(0)        
        count = count + 1
        
    for i in range(len(_barPaths)):
        print('\t{}'.format(_barPaths[i]))
    _barArrays = []
    _barArray = np.zeros(_barReader(_barPaths[i],ncols).shape)
    if subtractBars:
        for i in range(len(_barPaths)):
            _barArrays.append(_barReader(_barPaths[i],ncols))
        _barArray = _barArrays[0] - _barArrays[1]
        subtimeInterval = (timeinterval[0]-timeinterval[1])
        _barArray = _barArray/subtimeInterval
        print('{} - {}'.format(_barPaths[0], _barPaths[1]))
        print(subtimeInterval)
    else:
        for i in range(len(_barPaths)):
            _barArrays.append(_barReader(_barPaths[i],ncols))
            # if globname != 'SMRbar' or globname != 'SMRbar2':
            # _barArray = _barArray + (_barArrays[i]*timeinterval[i])
            # else:
            _barArray = _barArray + _barArrays[i]
            print(timeinterval[i])
        _barArray = _barArray/sum(timeinterval)

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
        'stsbarKeq': [],
        'consrvstress': False,
        'periodicity': True,
        'nsons': 0,
        'loadIDDES': False,
        'loadstsbarKeq': False,        
        'loadSMRbar': False,
        'loadSMRBar2': False,
        'nsclr': 0,
        'streamcoord': False,
        'barformat': 1,
        'addpptostsbar': False
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
