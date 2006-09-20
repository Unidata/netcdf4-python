import os
from numpy.distutils.core  import setup, Extension

HDF5_dir = os.environ.get('HDF5_DIR')
netCDF4_dir = os.environ.get('NETCDF4_DIR')
if HDF5_dir is None or netCDF4_dir is None:
    raise KeyError, 'please specify the locations of netCDF4 and HDF5 with the NETCDF4_DIR and HDF5_DIR environment variables'

libs = ['netcdf','hdf5','hdf5_hl']
lib_dirs = [os.path.join(netCDF4_dir,'lib'),os.path.join(HDF5_dir,'lib')]
inc_dirs = [os.path.join(netCDF4_dir,'include'),os.path.join(HDF5_dir,'include')]
extensions = [Extension("netCDF4",["netCDF4.c"],libraries=libs,library_dirs=lib_dirs,include_dirs=inc_dirs,runtime_library_dirs=lib_dirs)]
extensions.append(Extension("netCDF4_classic",["netCDF4_classic.c"],libraries=libs,library_dirs=lib_dirs,include_dirs=inc_dirs,runtime_library_dirs=lib_dirs))

setup(name = "netCDF4",
  version = "0.5",
  summary = "Provides an object-oriented python interface to the netCDF version 4 library.",
  description = "netCDF version 4 has many features not found in earlier versions of the library, such as hierarchical groups, zlib compression, multiple unlimited dimensions, and new data types.  It is implemented on top of HDF5.  This module implements many of the new features, and can read netCDF files created with older versions of the library.  The API is modelled after Scientific.IO.NetCDF, and should be familiar to users of that module",
  author            = "Jeff Whitaker",
  author_email      = "jeffrey.s.whitaker@noaa.gov",
  url               = "http://www.cdc.noaa.gov/people/jeffrey.s.whitaker/python/netCDF4.html",
  download_url      = "http://www.cdc.noaa.gov/people/jeffrey.s.whitaker/python/netCDF4-0.4.4.tar.gz",
  scripts           = ['utils/nc3tonc4'],
  platforms         = ["any"],
  license           = ["OSI Approved"],
  classifiers       = ["Development Status :: 3 - Alpha",
		       "Intended Audience :: Science/Research", 
		       "License :: OSI Approved", 
		       "Topic :: Software Development :: Libraries :: Python Modules",
                       "Topic :: System :: Archiving :: Compression",
		       "Operating System :: OS Independent"],
  packages = ["netcdftime"],
  ext_modules = extensions)
