# to run: mpirun -np 4 python mpi_example_compressed.py
import sys
from mpi4py import MPI
import numpy as np
from netCDF4 import Dataset
rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)
nc = Dataset('parallel_test_compressed.nc', 'w', parallel=True)
d = nc.createDimension('dim',4)
v = nc.createVariable('var', np.int32, 'dim', zlib=True)
v[:] = np.arange(4)
nc.close()
# read compressed files in parallel, check the data, try to rewrite some data
nc = Dataset('parallel_test_compressed.nc', 'a', parallel=True)
v = nc['var']
assert rank==v[rank]
v.set_collective(True) # issue #1108 (var must be in collective mode or write will fail)
v[rank]=2*rank
nc.close()
