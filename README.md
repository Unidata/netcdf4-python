Python/numpy interface to the netCDF version 4 library.

Quick Start
-----------

* clone github repository, or get source tarball (or Windows binary installers) from
  https://pypi.python.org/pypi/netCDF4.

* make sure numpy (required) and Cython (recommended) are installed and
  you have python 2.5 or newer.

* make sure HDF5 and netcdf-4 are installed.

* copy setup.cfg.template to setup.cfg, open with a text editor
  and follow the instructions in the comments for editing.

* run 'python setup.py build, then 'python setup.py install' (with sudo
  if necessary).

* To run all the tests, execute 'cd test; python run_all.py'.

More detailed documentation is available at docs/index.html, or
http://unidata.github.io/netcdf4-python.
