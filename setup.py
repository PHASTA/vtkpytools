import setuptools

version = ''
with open('vtkpytools/__init__.py') as f:
    for line in f.readlines():
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().replace("'", '')

setuptools.setup(
    name = "vtkpytools",
    version = version,
    packages = setuptools.find_packages()
)
