Python/numpy interface to the netCDF version 4 library.

News
----

3/1/2015: Version 1.1.5 released.  Significant improvements to netcdftime and 
num2date/date2num - accuracy is now between a millisecond and a microsecond depending
on the time interval and calendar used. use_ncconfig is now True by default
in setup.py, so the utility nc-config e used to find the library and
include file paths.  For other changes see https://github.com/Unidata/netcdf4-python/blob/master/Changelog.

Quick Start
-----------

* clone github repository, or get sourball (or Windows binary installers) from
  https://pypi.python.org/pypi/netCDF4

* make sure numpy and (required) and Cython (recommended) are
  installed and you have python 2.5 or newer.

* make sure HDF5 and netcdf-4 are installed, and the nc-config utility
  is in your Unix PATH. If setup.cfg does not exist, copy setup.cfg.template
  to setup.cfg, and make sure the line with 'use_ncconfig=True' is 
  un-commented.

* run 'python setup.py build, then 'python setup.py install' (with sudo
  if necessary).

* To run all the tests, execute 'cd test; python run_all.py'.

More detailed documentation is available at docs/index.html, or
http://unidata.github.io/netcdf4-python.
