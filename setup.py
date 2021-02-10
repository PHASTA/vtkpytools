import setuptools
import os

version = ''
with open('vtkpytools/_version.py') as f:
    for line in f.readlines():
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().replace("'", '')

# Symlink script names
SCRIPTS_PATH='./_scripts'
def scripts_hack(*scripts):
    ''' Hack around `pip install` temporary directories '''
    if not os.path.exists(SCRIPTS_PATH):
        os.makedirs(SCRIPTS_PATH)
    scripts_path = []
    for src_path, basename in scripts:
        dest_path = os.path.join(SCRIPTS_PATH, basename)
        if not os.path.exists(dest_path):
            dest_actual_path = os.path.abspath(os.path.join(dest_path, os.pardir))
            os.symlink(os.path.relpath(src_path, dest_actual_path), dest_path)
        scripts_path += [dest_path]
    return scripts_path

setuptools.setup(
    name = "vtkpytools",
    version = version,
    packages = setuptools.find_packages(),
    scripts = scripts_hack(
        ('./bin/bar2vtk.py', 'bar2vtk'),
        ('./bin/barsplit.py', 'barsplit')
    ),
    install_requires = [
        'pyvista',
        'numpy',
        'pytomlpp'
    ]
)
