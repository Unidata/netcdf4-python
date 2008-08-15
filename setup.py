import os
from numpy.distutils.core  import setup, Extension

def check_hdf5version(hdf5_dir):
    try:
        f = open(os.path.join(hdf5_dir,'include/H5pubconf.h'))
    except IOError:
        try:
            f = open(os.path.join(hdf5_dir,'include/H5pubconf-64.h'))
        except IOError:
            return None
    hdf5_version = None
    for line in f:
        if line.startswith('#define H5_VERSION'):
            hdf5_version = line.split()[2]
    return hdf5_version

def check_ifnetcdf4(netcdf4_dir):
    try:
        f = open(os.path.join(netcdf4_dir,'include/netcdf.h'))
    except IOError:
        return False
    isnetcdf4 = False
    for line in f:
        if line.startswith('nc_inq_compound'):
            isnetcdf4 = True
    return isnetcdf4

HDF5_dir = os.environ.get('HDF5_DIR')
netCDF4_dir = os.environ.get('NETCDF4_DIR')
dirstosearch =  [os.path.expanduser('~'),'/usr/local','/sw','/opt','/opt/local','/usr']
if HDF5_dir is None:
    print """
HDF5_DIR environment variable not set, checking some standard locations ..,"""
    for direc in dirstosearch:
        print 'checking %s ...' % direc
        hdf5_version = check_hdf5version(direc)
        if hdf5_version is None or hdf5_version[1:6] < '1.8.0':
            continue
        else:
            HDF5_dir = direc
            print 'HDF5 found in %s' % HDF5_dir
            break
    if HDF5_dir is None:
        raise ValueError('did not find HDF5 headers and libraries')
else:
    hdf5_version = check_hdf5version(HDF5_dir)
    if hdf5_version is None:
        raise ValueError('did not find HDF5 headers and libraries in %s' % HDF5_dir)
    elif hdf5_version[1:6] < '1.8.0':
        raise ValueError('HDF5 version >= 1.8.0 is required')

if netCDF4_dir is None:
    print """
NETCDF4_DIR environment variable not set, checking some standard locations ..,"""
    for direc in dirstosearch:
        print 'checking %s ...' % direc
        isnetcdf4 = check_ifnetcdf4(direc)
        if not isnetcdf4:
            continue
        else:
            netCDF4_dir = direc
            print 'netCDF4 found in %s' % netCDF4_dir
            break
    if netCDF4_dir is None:
        raise ValueError('did not find netCDF version 4 headers and libs')
else:
    isnetcdf4 = check_ifnetcdf4(netCDF4_dir)
    if not isnetcdf4:
        raise ValueError('did not find netCDF version 4 headers and libs in %s' % netCDF4_dir)

libs = ['netcdf','hdf5','hdf5_hl','z']
lib_dirs = [os.path.join(netCDF4_dir,'lib'),os.path.join(HDF5_dir,'lib')]
inc_dirs = [os.path.join(netCDF4_dir,'include'),os.path.join(HDF5_dir,'include')]
# add szip to link if desired.
szip_dir = os.environ.get('SZIP_DIR')
if szip_dir is not None:
    lib_dirs.append(os.path.join(szip_dir,'lib'))
    inc_dirs.append(os.path.join(szip_dir,'include'))
    libs.append('sz')
extensions = [Extension("netCDF4",["netCDF4.c"],libraries=libs,library_dirs=lib_dirs,include_dirs=inc_dirs,runtime_library_dirs=lib_dirs)]

setup(name = "netCDF4",
  version = "0.7.6",
  description = "netCDF version 4 has many features not found in earlier versions of the library, such as hierarchical groups, zlib compression, multiple unlimited dimensions, and new data types.  It is implemented on top of HDF5.  This module implements many of the new features, and can read and write netCDF files compatible with older versions of the library.  The API is modelled after Scientific.IO.NetCDF, and should be familiar to users of that module.\n\nThis project has a `Subversion repository <http://code.google.com/p/netcdf4-python/source>`_ where you may access the most up-to-date source.",
  author            = "Jeff Whitaker",
  author_email      = "jeffrey.s.whitaker@noaa.gov",
  url               = "http://netcdf4-python.googlecode.com/svn/trunk/docs/netCDF4-module.html",
  download_url      = "http://code.google.com/p/netcdf4-python/downloads/list",
  scripts           = ['utils/nc3tonc4','utils/grib2nc4'],
  platforms         = ["any"],
  license           = ["OSI Approved"],
  summary = "Provides an object-oriented python interface to the netCDF version 4 library.",
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
