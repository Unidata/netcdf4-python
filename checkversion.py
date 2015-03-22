import netCDF4
import sys
sys.stdout.write('netcdf4-python version: %s\n' % netCDF4.__version__)
sys.stdout.write('HDF5 lib version:       %s\n' % netCDF4.__hdf5libversion__)
sys.stdout.write(
    'netcdf lib version:     %s\n' %
    netCDF4.__netcdf4libversion__)
