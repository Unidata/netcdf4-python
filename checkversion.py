import netCDF4, sys, numpy
sys.stdout.write('netcdf4-python version: %s\n'%netCDF4.__version__)
sys.stdout.write('HDF5 lib version:       %s\n'%netCDF4.__hdf5libversion__)
sys.stdout.write('netcdf lib version:     %s\n'%netCDF4.__netcdf4libversion__)
sys.stdout.write('numpy version           %s\n' % numpy.__version__)
