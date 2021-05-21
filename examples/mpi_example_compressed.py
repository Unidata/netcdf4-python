# to run: mpirun -np 4 python mpi_example.py
import sys
from mpi4py import MPI
import numpy as np
from netCDF4 import Dataset
rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)
nc = Dataset('parallel_test.nc', 'w', parallel=True)
d = nc.createDimension('dim',4)
v = nc.createVariable('var', np.int32, 'dim', zlib=True)
v[:] = np.arange(4)
nc.close()
# read compressed files in parallel, check the data, try to rewrite some data
nc = Dataset('parallel_test.nc', 'a', parallel=True)
v = nc['var']
assert rank==v[rank]
v.set_collective(True) # issue #1108
v[rank]=1
nc.close()
