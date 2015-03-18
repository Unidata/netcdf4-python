netcdf4-python
==============

[python](http://python.org)/[numpy](http://numpy.org) interface to the netCDF [C library](https://github.com/Unidata/netcdf-c).

[![Build Status](https://travis-ci.org/Unidata/netcdf4-python.svg?branch=master)](https://travis-ci.org/Unidata/netcdf4-python)
[![PyPI package](https://badge.fury.io/py/netCDF4.svg)](http://python.org/pypi/netCDF4)

News
----

For the latest updates, see the [Changelog](https://github.com/Unidata/netcdf4-python/blob/master/Changelog).

3/19/2015: Version 1.1.7 released.  Global Interpreter Lock (GIL) now released when extension
module calls C library for read operations.  This speeds up concurrent reads when using threads.
Users who wish to use netcdf4-python inside threads should read http://www.hdfgroup.org/hdf5-quest.html#gconc 
regarding thread-safety in the HDF5 C library.  Fixes to setup.py now ensure that `pip install netCDF4`
with `export USE_NCCONFIG=0` will use environment variables to find paths to libraries and include files,
instead of relying exclusively on the nc-config utility.

3/8/2015: Version 1.1.6 released.  Minor bug fixes for regressions introduced in 1.1.5, 
including incorrect handling of UTC offsets in units string by date2num/num2date. 
Datetime instances returned by num2date are now time-zone naive, so python-dateutil
is no longer required.

3/1/2015: Version 1.1.5 released.  Significant improvements to netcdftime and 
num2date/date2num - accuracy is now between a millisecond and a microsecond depending
on the time interval and calendar used. `use_ncconfig=True` is now the default
in setup.py, so the utility nc-config is used to find the library and
include file paths.  

Quick Start
-----------

* clone github repository, or get source tarball (or Windows binary installers) from
  [PyPI](https://pypi.python.org/pypi/netCDF4).

* make sure numpy and (required) and Cython (recommended) are
  installed and you have python 2.5 or newer.

* make sure HDF5 and netcdf-4 are installed, and the nc-config utility
  is in your Unix PATH. If setup.cfg does not exist, copy setup.cfg.template
  to setup.cfg, and make sure the line with `use_ncconfig=True` is 
  un-commented.

* run `python setup.py build`, then `python setup.py install` (with sudo
  if necessary).

* To run all the tests, execute `cd test; python run_all.py`.

See the online [docs](http://unidata.github.io/netcdf4-python) for more details.

Sample ipython notebooks available in the examples directory on [reading](http://nbviewer.ipython.org/github/Unidata/netcdf4-python/blob/master/examples/reading_netCDF.ipynb) and [writing](http://nbviewer.ipython.org/github/Unidata/netcdf4-python/blob/master/examples/writing_netCDF.ipynb) netcdf data with python.
