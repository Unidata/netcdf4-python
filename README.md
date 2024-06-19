# [netcdf4-python](http://unidata.github.io/netcdf4-python)
[Python](http://python.org)/[numpy](http://numpy.org) interface to the netCDF [C library](https://github.com/Unidata/netcdf-c).

[![Build status](https://github.com/Unidata/netcdf4-python/workflows/Build%20and%20Test/badge.svg)](https://github.com/Unidata/netcdf4-python/actions)
[![PyPI package](https://img.shields.io/pypi/v/netCDF4.svg)](http://python.org/pypi/netCDF4)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/netCDF4/badges/version.svg)](https://anaconda.org/conda-forge/netCDF4)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2592291.svg)](https://doi.org/10.5281/zenodo.2592290)


## News
For details on the latest updates, see the [Changelog](https://github.com/Unidata/netcdf4-python/blob/master/Changelog).

06/17/2024: Version [1.7.1](https://pypi.python.org/pypi/netCDF4/1.7.1) released. Fixes for wheels, no code changes.

06/13/2024: Version [1.7.0](https://pypi.python.org/pypi/netCDF4/1.7.0) released. Add support for complex numbers via `auto_complex` keyword to `Dataset` ([PR #1295](https://github.com/Unidata/netcdf4-python/pull/1295))

10/20/2023: Version [1.6.5](https://pypi.python.org/pypi/netCDF4/1.6.5) released. 
Fix for issue #1271 (mask ignored if bool MA assinged to uint8 var), 
support for python 3.12 (removal of python 3.7 support), more
informative error messages.

6/4/2023:  Version [1.6.4](https://pypi.python.org/pypi/netCDF4/1.6.4) released.  Now requires 
[certifi](https://github.com/certifi/python-certifi) to locate SSL certificates - this allows 
OpenDAP https URLs to work with linux wheels (issue [#1246](https://github.com/Unidata/netcdf4-python/issues/1246)).

3/3/2023:  Version [1.6.3](https://pypi.python.org/pypi/netCDF4/1.6.3) released.

11/15/2022:  Version [1.6.2](https://pypi.python.org/pypi/netCDF4/1.6.2) released. Fix for
compilation with netcdf-c < 4.9.0 (issue [#1209](https://github.com/Unidata/netcdf4-python/issues/1209)).  
Slicing multi-dimensional variables with an all False boolean index array
now returns an empty numpy array (instead of raising an exception - issue [#1197](https://github.com/Unidata/netcdf4-python/issues/1197)).

09/18/2022:  Version [1.6.1](https://pypi.python.org/pypi/netCDF4/1.6.1) released.  GIL now
released for all C lib calls, `set_alignment` and `get_alignment` module functions
added to modify/retrieve HDF5 data alignment properties. Added `Dataset` methods to 
query availability of optional compression filters.

06/24/2022:  Version [1.6.0](https://pypi.python.org/pypi/netCDF4/1.6.0) released.  Support
for quantization (bit-grooming and bit-rounding) functionality in netcdf-c 4.9.0 which can
dramatically improve compression.  Dataset.createVariable now accepts dimension instances (instead
of just dimension names). 'compression' kwarg added to Dataset.createVariable to support szip as
well as new compression algorithms available in netcdf-c 4.9.0 through compression plugins (such
as zstd, bzip2 and blosc). Working arm64 wheels for Apple M1 Silicon now available on pypi.

10/31/2021:  Version [1.5.8](https://pypi.python.org/pypi/netCDF4/1.5.8) released. Fix Enum bug, add binary wheels for aarch64 and python 3.10.

6/22/2021:  Version [1.5.7](https://pypi.python.org/pypi/netCDF4/1.5.7) released.
Fixed OverflowError on Windows when reading data with dimension sizes greater than 2**32-1.
Masked arrays no longer returned for vlens.

2/15/2021:  Version [1.5.6](https://pypi.python.org/pypi/netCDF4/1.5.6) released. Added `Dataset.fromcdl` and `Dataset.tocdl`, which require `ncdump` and `ncgen` utilities to be in `$PATH`. Removed python 2.7 support.

12/20/2020: Version [1.5.5.1](https://pypi.python.org/pypi/netCDF4/1.5.5.1) released.
Updated binary wheels for OSX and linux that link latest netcdf-c and hdf5 libs.

12/01/2020: Version [1.5.5](https://pypi.python.org/pypi/netCDF4/1.5.5) released.
Update license wording to be consistent with MIT license. 

07/23/2020: Version [1.5.4](https://pypi.python.org/pypi/netCDF4/1.5.4) released. 
Now requires numpy >= 1.9.
 
10/27/2019: Version [1.5.3](https://pypi.python.org/pypi/netCDF4/1.5.3) released. Fix for
[issue #972](https://github.com/Unidata/netcdf4-python/issues/972), plus binary wheels for
python 3.8.

09/03/2019: Version [1.5.2](https://pypi.python.org/pypi/netCDF4/1.5.2) released. Bugfixes, no new features.

05/06/2019: Version [1.5.1.2](https://pypi.python.org/pypi/netCDF4/1.5.1.2) released. Fixes another slicing
regression ([issue #922)](https://github.com/Unidata/netcdf4-python/issues/922)) introduced in the 1.5.1 release.

05/02/2019: Version [1.5.1.1](https://pypi.python.org/pypi/netCDF4/1.5.1.1) released. Fixes incorrect `__version__`
module variable in 1.5.1 release, plus a slicing bug ([issue #919)](https://github.com/Unidata/netcdf4-python/issues/919)).
 
04/30/2019: Version [1.5.1](https://pypi.python.org/pypi/netCDF4/1.5.1) released. Bugfixes, no new features.

04/02/2019: Version [1.5.0.1](https://pypi.python.org/pypi/netCDF4/1.5.0.1) released. Binary wheels for macos x
and linux rebuilt with netcdf-c 4.6.3 (instead of 4.4.1.1).   Added read-shared capability for faster reads
of NETCDF3 files (mode='rs').

03/24/2019: Version [1.5.0](https://pypi.python.org/pypi/netCDF4/1.5.0) released. Parallel IO support for classic
file formats added using the pnetcdf library (contribution from Lars Pastewka, [pull request #897](https://github.com/Unidata/netcdf4-python/pull/897)).

03/08/2019: Version [1.4.3.2](https://pypi.python.org/pypi/netCDF4/1.4.3.2) released. 
Include missing membuf.pyx file in source tarball. No need to update if you installed
1.4.3.1 from a binary wheel.

03/07/2019: Version [1.4.3.1](https://pypi.python.org/pypi/netCDF4/1.4.3.1) released. 
Fixes bug in implementation of NETCDF4_CLASSIC parallel IO support in 1.4.3.

03/05/2019: Version [1.4.3](https://pypi.python.org/pypi/netCDF4/1.4.3) released. Issues with netcdf-c 4.6.2 fixed (including broken parallel IO).  `set_ncstring_attrs()` method added, memoryview buffer now returned when an in-memory Dataset is closed.

10/26/2018: Version [1.4.2](https://pypi.python.org/pypi/netCDF4/1.4.2) released. Minor bugfixes, added `Variable.get_dims()` method and `master_file` kwarg for `MFDataset.__init__`.

08/10/2018: Version [1.4.1](https://pypi.python.org/pypi/netCDF4/1.4.1) released. The old slicing behavior
(numpy array returned unless missing values are present, otherwise masked array returned) is re-enabled
via `set_always_mask(False)`.

05/11/2018: Version [1.4.0](https://pypi.python.org/pypi/netCDF4/1.4.0) released. The netcdftime package is no longer
included, it is now a separate [package](https://pypi.python.org/pypi/cftime) dependency.  In addition to several
bug fixes, there are a few important changes to the default behaviour to note:
 * Slicing a netCDF variable will now always return masked array by default, even if there are no 
   masked values.  The result depended on the slice before, which was too surprising.
   If auto-masking is turned off (with `set_auto_mask(False)`) a numpy array will always
   be returned.
 * `_FillValue` is no longer treated as a valid_min/valid_max.  This was  too surprising, despite
   the fact the thet netcdf docs [attribute best practices](https://www.unidata.ucar.edu/software/netcdf/docs/attribute_conventions.html) suggests that
   clients should to this if `valid_min`, `valid_max` and `valid_range` are not set. 
 * Changed behavior of string attributes so that `nc.stringatt = ['foo','bar']`
   produces an vlen string array attribute in NETCDF4, instead of concatenating
   into a single string (`foobar`).  In NETCDF3/NETCDF4_CLASSIC, an IOError
   is now raised, instead of writing `foobar`.
 * Retrieved compound-type variable data now returned with character array elements converted to 
   numpy strings ([issue #773](https://github.com/Unidata/netcdf4-python/issues/773)).
   Works for assignment also.  Can be disabled using
   `set_auto_chartostring(False)`. Numpy structured
   array dtypes with `'SN'` string subtypes can now be used to
   define netcdf compound types in `createCompoundType` (they get converted to `('S1',N)`
   character array types automatically).
 * `valid_min`, `valid_max`, `_FillValue` and `missing_value` are now treated as unsigned
   integers if `_Unsigned` variable attribute is set (to mimic behaviour of netcdf-java).
   Conversion to unsigned type now occurs before masking and scale/offset
   operation ([issue #794](https://github.com/Unidata/netcdf4-python/issues/794))

11/01/2017: Version [1.3.1](https://pypi.python.org/pypi/netCDF4/1.3.1) released.  Parallel IO support with MPI!
Requires that netcdf-c and hdf5 be built with MPI support, and [mpi4py](http://mpi4py.readthedocs.io/en/stable).
To open a file for parallel access in a program running in an MPI environment
using mpi4py, just use `parallel=True` when creating
the `Dataset` instance.  See [`examples/mpi_example.py`](https://github.com/Unidata/netcdf4-python/blob/master/examples/mpi_example.py)
 for a demonstration.  For more info, see the tutorial [section](http://unidata.github.io/netcdf4-python/#section13).

9/25/2017: Version [1.3.0](https://pypi.python.org/pypi/netCDF4/1.3.0) released. Bug fixes
for `netcdftime` and optimizations for reading strided slices. `encoding` kwarg added to 
`Dataset.__init__` and `Dataset.filepath` to deal with oddball encodings in filename
paths (`sys.getfilesystemencoding()` is used by default to determine encoding).
Make sure numpy datatypes used to define CompoundTypes have `isalignedstruct` flag set
to avoid segfaults - which required bumping the minimum required numpy from 1.7.0 
to 1.9.0. In cases where `missing_value/valid_min/valid_max/_FillValue` cannot be
safely cast to the variable's dtype, they are no longer be used to automatically
mask the data and a warning message is issued.

6/10/2017: Version [1.2.9](https://pypi.python.org/pypi/netCDF4/1.2.9) released. Fixes for auto-scaling
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

## Installation
The easiest way to install is through pip:

```shell
pip install netCDF4
```

or, if you are a user of the Conda package manager,

```shell
conda install -c conda-forge netCDF4
```

## Development installation
* Clone GitHub repository (`git clone https://github.com/Unidata/netcdf4-python.git`)

* Make sure [numpy](http://www.numpy.org/) and [Cython](http://cython.org/) are
  installed and you have [Python](https://www.python.org) 3.8 or newer.

* Make sure [HDF5](http://www.h5py.org/) and netcdf-4 are installed, 
  and the `nc-config` utility is in your Unix PATH.

* Run `python setup.py build`, then `pip install -e .`.

* To run all the tests, execute `cd test && python run_all.py`.

## Documentation
See the online [docs](http://unidata.github.io/netcdf4-python) for more details.

## Usage
###### Sample [iPython](http://ipython.org/) notebooks available in the examples directory on [reading](http://nbviewer.ipython.org/github/Unidata/netcdf4-python/blob/master/examples/reading_netCDF.ipynb) and [writing](http://nbviewer.ipython.org/github/Unidata/netcdf4-python/blob/master/examples/writing_netCDF.ipynb) netCDF data with Python.
