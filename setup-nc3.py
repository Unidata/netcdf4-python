import os
from numpy.distutils.core  import setup, Extension

def check_ifnetcdf3(netcdf3_dir):
    try:
        f = open(os.path.join(netcdf3_dir,'include/netcdf.h'))
    except IOError:
        return False
    isnetcdf3 = False
    for line in f:
        if line.startswith('#define NC_GEOX'):
            isnetcdf3 = True
    return isnetcdf3

netCDF3_dir = os.environ.get('NETCDF3_DIR')
dirstosearch =  ['/usr/local','/sw','/opt','/opt/local',os.path.expanduser('~')]
if netCDF3_dir is None:
    print """
NETCDF3_DIR environment variable not set, checking some standard locations ..,"""
    for direc in dirstosearch:
        print 'checking %s ...' % direc
        isnetcdf3 = check_ifnetcdf3(direc)
        if not isnetcdf3:
            continue
        else:
            netCDF3_dir = direc
            print 'netCDF3 found in %s' % netCDF3_dir
            break
    if netCDF3_dir is None:
        raise ValueError('did not find netCDF version 3 headers and libs')
else:
    isnetcdf3 = check_ifnetcdf3(netCDF3_dir)
    if not isnetcdf3:
        raise ValueError('did not find netCDF version 3 headers and libs in %s' % netCDF3_dir)

libs = ['netcdf']
lib_dirs = [os.path.join(netCDF3_dir,'lib')]
inc_dirs = [os.path.join(netCDF3_dir,'include')]
extensions = [Extension("netCDF3",["netCDF3.c"],libraries=libs,library_dirs=lib_dirs,include_dirs=inc_dirs,runtime_library_dirs=lib_dirs)]

setup(name = "netCDF3",
  version = "0.7.6",
  description = "python interface to netCDF version 3",
  author            = "Jeff Whitaker",
  author_email      = "jeffrey.s.whitaker@noaa.gov",
  url               = "http://netcdf4-python.googlecode.com/svn/trunk/docs/netCDF3-module.html",
  download_url      = "http://code.google.com/p/netcdf4-python/downloads/list",
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
