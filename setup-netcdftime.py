from distutils.core import setup

setup(name = "netcdftime",
  version = "0.7",
  author_email      = "jeffrey.s.whitaker@noaa.gov",
  url               = "http://netcdf4-python.googlecode.com/svn/trunk/docs/netCDF4-module.html",
  download_url      = "http://cheeseshop.python.org/pypi/netCDF4/",
  platforms         = ["any"],
  license           = ["OSI Approved"],
  summary = "Performs conversions of netCDF time coordinate data to/from datetime objects.",
  keywords = ['numpy','netcdf','data','science','network','oceanography','meteorology','climate'],
  classifiers = ["Development Status :: 3 - Alpha",
		         "Intended Audience :: Science/Research", 
		         "License :: OSI Approved", 
		         "Topic :: Software Development :: Libraries :: Python Modules",
                 "Topic :: System :: Archiving :: Compression",
		         "Operating System :: OS Independent"],
  packages = ["netcdftime"])
