# netcdf4-python
[Python](http://python.org)/[numpy](http://numpy.org) interface to the netCDF [C library](https://github.com/Unidata/netcdf-c).

[![Build Status](https://travis-ci.org/Unidata/netcdf4-python.svg?branch=master)](https://travis-ci.org/Unidata/netcdf4-python)
[![PyPI package](https://badge.fury.io/py/netCDF4.svg)](http://python.org/pypi/netCDF4)

## News
For the latest updates, see the [Changelog](https://github.com/Unidata/netcdf4-python/blob/master/Changelog).

9/23/2015: Version [1.2.0](https://pypi.python.org/pypi/netCDF4/1.2.0) released. New features:

* [get_variables_by_attributes](http://unidata.github.io/netcdf4-python/#netCDF4.Dataset.get_variables_by_attributes) 
``Dataset`` and ``Group`` method for retrieving variables that have matching attributes.

* Support for [Enum](http://unidata.github.io/netcdf4-python/#section12) data types.

* [isopen](http://unidata.github.io/netcdf4-python/#netCDF4.Dataset.isopen) `Dataset` method.

7/28/2015: Version [1.1.9](https://pypi.python.org/pypi/netCDF4/1.1.9) bugfix release. 

5/14/2015: Version [1.1.8](https://pypi.python.org/pypi/netCDF4/1.1.8) released. Unix-like paths can now be used in `createVariable` and `createGroup`.
```python
    v = nc.createVariable('/path/to/var1', ('xdim', 'ydim'), float)
```
will create a variable named 'var1', while also creating the groups
'path' and 'path/to' if they do not already exist.

Similarly, 
```python
    g = nc.createGroup('/path/to') 
```
now acts like `mkdir -p` in unix, creating groups 'path' and '/path/to',
if they don't already exist. Users who relied on `nc.createGroup(groupname)`
failing when the group already exists will have to modify their code, since 
`nc.createGroup` will now return the existing group instance.
`Dataset.__getitem__` was also added.  `nc['/path/to']`
now returns a group instance, and `nc['/path/to/var1']` now returns a variable instance.

3/19/2015: Version [1.1.7](https://pypi.python.org/pypi/netCDF4/1.1.7) released.  Global Interpreter Lock (GIL) now released when extension
module calls C library for read operations.  This speeds up concurrent reads when using threads.
Users who wish to use netcdf4-python inside threads should read http://www.hdfgroup.org/hdf5-quest.html#gconc 
regarding thread-safety in the HDF5 C library.  Fixes to `setup.py` now ensure that `pip install netCDF4`
with `export USE_NCCONFIG=0` will use environment variables to find paths to libraries and include files,
instead of relying exclusively on the nc-config utility.

## Quick Start
* Clone GitHub repository (`git clone https://github.com/Unidata/netcdf4-python.git`), or get source tarball from [PyPI](https://pypi.python.org/pypi/netCDF4). Links to Windows and OS X precompiled binary packages are also available on [PyPI](https://pypi.python.org/pypi/netCDF4).

* Make sure [numpy](http://www.numpy.org/) (required) and [Cython](http://cython.org/) (recommended) are
  installed and you have [Python](https://www.python.org) 2.5 or newer.

* Make sure [HDF5](http://www.h5py.org/) and netcdf-4 are installed, and the `nc-config` utility
  is in your Unix PATH. If `setup.cfg` does not exist, copy `setup.cfg.template`
  to `setup.cfg`, and make sure the line with `use_ncconfig=True` is un-commented.

* Run `python setup.py build`, then `python setup.py install` (with `sudo` if necessary).

* To run all the tests, execute `cd test && python run_all.py`.

## Documentation
See the online [docs](http://unidata.github.io/netcdf4-python) for more details.

## Usage
###### Sample [iPython](http://ipython.org/) notebooks available in the examples directory on [reading](http://nbviewer.ipython.org/github/Unidata/netcdf4-python/blob/master/examples/reading_netCDF.ipynb) and [writing](http://nbviewer.ipython.org/github/Unidata/netcdf4-python/blob/master/examples/writing_netCDF.ipynb) netCDF data with Python.
