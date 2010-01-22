from distutils.core import setup, Extension
import numpy

setup(name = "grib_api",
      version = "1.0",
      ext_modules = [Extension(
        "pygrib",
        ["pygrib.c"],
	include_dirs=[numpy.get_include(),'/Users/jwhitaker/include','/opt/local/include'],
        library_dirs=['/opt/local/lib','/Users/jwhitaker/lib'],
        libraries=["grib_api","jasper","openjpeg","png"]
      )])

