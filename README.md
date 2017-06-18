# netcdf4-python
[Python](http://python.org)/[numpy](http://numpy.org) interface to the netCDF [C library](https://github.com/Unidata/netcdf-c).

[![Linux Build Status](https://travis-ci.org/Unidata/netcdf4-python.svg?branch=master)](https://travis-ci.org/Unidata/netcdf4-python)
[![Windows Build Status](https://ci.appveyor.com/api/projects/status/fl9taa9je4e6wi7n/branch/master?svg=true)](https://ci.appveyor.com/project/jswhit/netcdf4-python/branch/master)
[![PyPI package](https://badge.fury.io/py/netCDF4.svg)](http://python.org/pypi/netCDF4)

## News
For the latest updates, see the [Changelog](https://github.com/Unidata/netcdf4-python/blob/master/Changelog).

6/10/2017: Version [1.2.9](https://github.com/Unidata/netcdf4-python/releases) released. Fixes for auto-scaling
and masking when `_Unsigned` and/or `valid_min`, `valid_max` attributes present.  setup.py updated
so that `pip install` works if cython not installed.  Now requires [setuptools](https://pypi.python.org/pypi/setuptools)
version 18.0 or greater.

6/1/2017: Version [1.2.8](https://pypi.python.org/pypi/netCDF4/1.2.8) released.  From Changelog:
 * recognize `_Unsigned` attribute used by [netcdf-java](http://www.unidata.ucar.edu/software/thredds/current/netcdf-java/)
   to designate unsigned integer data stored with a signed integer type in netcdf-3 
   [issue #656](https://github.com/Unidata/netcdf4-python/issues/656).
 * add Dataset init memory parameter to allow loading a file from memory
   [pull request #652](https://github.com/Unidata/netcdf4-python/pull/652),
   [issue #406](https://github.com/Unidata/netcdf4-python/issues/406) and
   [issue #295](https://github.com/Unidata/netcdf4-python/issues/295).
 * fix for negative times in num2date [issue #659](https://github.com/Unidata/netcdf4-python/pull/659).
 * fix for failing tests in numpy 1.13 due to changes in `numpy.ma`
   [issue #662](https://github.com/Unidata/netcdf4-python/issues/662).
 * Checking for `_Encoding` attribute for `NC_STRING` variables, otherwise use
   'utf-8'. 'utf-8' is used everywhere else, 'default_encoding' global module
   variable is no longer used.  getncattr method now takes optional kwarg
   'encoding' (default 'utf-8') so encoding of attributes can be specified
   if desired. If `_Encoding` is specified for an `NC_CHAR` (`'S1'`) variable,
   the chartostring utility function is used to convert the array of
   characters to an array of strings with one less dimension (the last
   dimension is interpreted as the length of each string) when reading the
   data. When writing the data, stringtochar is used to convert a numpy 
   array of fixed length strings to an array of characters with one more
   dimension. chartostring and stringtochar now also have an 'encoding' kwarg.
   Automatic conversion to/from character to string arrays can be turned off
   via a new `set_auto_chartostring` Dataset and Variable method (default
   is `True`). Addresses [issue #654](https://github.com/Unidata/netcdf4-python/issues/654)
 * [Cython](http://cython.org) >= 0.19 now required, `_netCDF4.c` and `_netcdftime.c` removed from
   repository.

1/8/2017: Version [1.2.7](https://pypi.python.org/pypi/netCDF4/1.2.7) released. 
Python 3.6 compatibility, and fix for vector missing_values.

12/10/2016: Version [1.2.6](https://pypi.python.org/pypi/netCDF4/1.2.6) released. 
Bug fixes for Enum data type, and _FillValue/missing_value usage when data is stored
in non-native endian format. Add get_variables_by_attributes to MFDataset. Support for python 2.6 removed.

12/1/2016: Version [1.2.5](https://pypi.python.org/pypi/netCDF4/1.2.5) released.
See the [Changelog](https://github.com/Unidata/netcdf4-python/blob/master/Changelog) for changes.

4/15/2016: Version [1.2.4](https://pypi.python.org/pypi/netCDF4/1.2.4) released. 
Bugs in handling of variables with specified non-native "endian-ness" (byte-order) fixed ([issue #554]
(https://github.com/Unidata/netcdf4-python/issues/554)).  Build instructions updated and warning issued
to deal with potential backwards incompatibility introduced when using HDF5 1.10.x
(see [Unidata/netcdf-c/issue#250](https://github.com/Unidata/netcdf-c/issues/250)).

3/10/2016: Version [1.2.3](https://pypi.python.org/pypi/netCDF4/1.2.3) released. Various bug fixes.
All text attributes in ``NETCDF4`` formatted files are now written as type ``NC_CHAR``, unless they contain unicode characters that
cannot be encoded in ascii, in which case they are written as ``NC_STRING``.  Previously,
all unicode strings were written as ``NC_STRING``. This change preserves compatibility
with clients, like Matlab, that can't deal with ``NC_STRING`` attributes. 
A ``setncattr_string`` method was added to force attributes to be written as ``NC_STRING``.

1/1/2016: Version [1.2.2](https://pypi.python.org/pypi/netCDF4/1.2.2) released. Mostly bugfixes, but with two new features.

* support for the new ``NETCDF3_64BIT_DATA`` format introduced in netcdf-c 4.4.0.
Similar to ``NETCDF3_64BIT`` (now ``NETCDF3_64BIT_OFFSET``), but includes
64 bit dimension sizes (> 2 billion), plus unsigned and 64 bit integer data types.
Uses the classic (netcdf-3) data model, and does not use HDF5 as the underlying storage format.

* Dimension objects now have a ``size`` attribute, which is the current length
of the dimension (same as invoking ``len`` on the Dimension instance).

The minimum required python version has now been increased from 2.5 to 2.6.

10/15/2015: Version [1.2.1](https://pypi.python.org/pypi/netCDF4/1.2.1) released. Adds the ability
to slice Variables with unsorted integer sequences, and integer sequences with duplicates.

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

* Make sure [numpy](http://www.numpy.org/) and [Cython](http://cython.org/) are
  installed and you have [Python](https://www.python.org) 2.7 or newer.

* Make sure [HDF5](http://www.h5py.org/) and netcdf-4 are installed, and the `nc-config` utility
  is in your Unix PATH. If `setup.cfg` does not exist, copy `setup.cfg.template`
  to `setup.cfg`, and make sure the line with `use_ncconfig=True` is un-commented.

* Run `python setup.py build`, then `python setup.py install` (with `sudo` if necessary).

* To run all the tests, execute `cd test && python run_all.py`.

## Documentation
See the online [docs](http://unidata.github.io/netcdf4-python) for more details.

## Usage
###### Sample [iPython](http://ipython.org/) notebooks available in the examples directory on [reading](http://nbviewer.ipython.org/github/Unidata/netcdf4-python/blob/master/examples/reading_netCDF.ipynb) and [writing](http://nbviewer.ipython.org/github/Unidata/netcdf4-python/blob/master/examples/writing_netCDF.ipynb) netCDF data with Python.
