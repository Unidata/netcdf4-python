import os, subprocess
from numpy.distutils.core  import setup, Extension

NETCDF4_DIR=os.getenv('NETCDF4_DIR')
# if NETCDF4_DIR env var is set, look for nc-config in NETCDF4_DIR/bin.
if NETCDF4_DIR is not None:
    ncconfig = os.path.join(NETCDF4_DIR,'bin/nc-config')
else: # otherwise, just hope it's in the users PATH.
    ncconfig = 'nc-config'
print 'nc-config = %s' % ncconfig
dep=subprocess.Popen([ncconfig,'--libs'],stdout=subprocess.PIPE).communicate()[0]
libs = [l[2:] for l in dep.split() if l[0:2] == '-l' ]
print 'libs = %s' % libs
lib_dirs = [l[2:] for l in dep.split() if l[0:2] == '-L' ]
print 'lib_dirs = %s' % lib_dirs
dep=subprocess.Popen([ncconfig,'--includedir'],stdout=subprocess.PIPE).communicate()[0]
inc_dirs = [i for i in dep.split()]
print 'inc_dirs = %s' % inc_dirs

extensions = [Extension("netCDF4",["netCDF4.c"],libraries=libs,library_dirs=lib_dirs,include_dirs=inc_dirs,runtime_library_dirs=lib_dirs)]

setup(name = "netCDF4",
  version = "0.9.3",
  long_description = "netCDF version 4 has many features not found in earlier versions of the library, such as hierarchical groups, zlib compression, multiple unlimited dimensions, and new data types.  It is implemented on top of HDF5.  This module implements most of the new features, and can read and write netCDF files compatible with older versions of the library.  The API is modelled after Scientific.IO.NetCDF, and should be familiar to users of that module.\n\nThis project has a `Subversion repository <http://code.google.com/p/netcdf4-python/source>`_ where you may access the most up-to-date source.",
  author            = "Jeff Whitaker",
  author_email      = "jeffrey.s.whitaker@noaa.gov",
  url               = "http://netcdf4-python.googlecode.com/svn/trunk/docs/netCDF4-module.html",
  download_url      = "http://code.google.com/p/netcdf4-python/downloads/list",
  scripts           = ['utils/nc3tonc4','utils/nc4tonc3','utils/grib2nc4'],
  platforms         = ["any"],
  license           = "OSI Approved",
  description = "Provides an object-oriented python interface to the netCDF version 4 library.",
  keywords = ['numpy','netcdf','data','science','network','oceanography','meteorology','climate'],
  classifiers = ["Development Status :: 3 - Alpha",
		         "Intended Audience :: Science/Research", 
		         "License :: OSI Approved", 
		         "Topic :: Software Development :: Libraries :: Python Modules",
                 "Topic :: System :: Archiving :: Compression",
		         "Operating System :: OS Independent"],
  packages = ["netcdftime"],
  py_modules = ["netCDF4_utils"],
  ext_modules = extensions)
