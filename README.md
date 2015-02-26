Python/numpy interface to the netCDF version 4 library.

News
----

2/19/2015: Version 1.1.4 released. Fixes serious bug in netcdftime that caused errors in num2date and date2num when units contains "seconds since".  Users of 1.1.2 and 1.1.3 should upgrade.  For other changes see https://github.com/Unidata/netcdf4-python/blob/master/Changelog.

Quick Start
-----------

* clone github repository, or get source tarball (or Windows binary installers) from
  https://pypi.python.org/pypi/netCDF4.

* make sure numpy (required) and Cython (recommended) are installed and
  you have python 2.5 or newer.

* make sure HDF5 and netcdf-4 are installed, and the nc-config utility
  is in your Unix PATH. If setup.cfg does not exist, copy setup.cfg.template
  to setup.cfg, and make sure the line with 'use_ncconfig=True' is 
  un-commented.

* run 'python setup.py build, then 'python setup.py install' (with sudo
  if necessary).

* To run all the tests, execute 'cd test; python run_all.py'.

More detailed documentation is available at docs/index.html, or
http://unidata.github.io/netcdf4-python.
