#!/usr/bin/env python3
import vtkpytools as vpt

args = vpt.bar2vtk_parse()

vpt.bar2vtk(vtkfile = args.vtkfile,
barfiledir = args.barfiledir,
timestep = args.timestep,
ts0 = args.ts0,
new_file_prefix = args.new_file_prefix,
outpath = args.outpath,
velonly = args.velonly,
debug = args.debug,
asciibool = args.ascii,
velbar = args.velbar,
stsbar = args.stsbar)
