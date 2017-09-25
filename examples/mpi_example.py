from mpi4py import MPI
import numpy as np
from netCDF4 import Dataset
rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)
nc = Dataset('parallel_test.nc', 'w', parallel=True, comm=MPI.COMM_WORLD,
        info=MPI.Info())
d = nc.createDimension('dim',4)
v = nc.createVariable('var', np.int, 'dim')
v[rank] = rank
nc.close()
nc = Dataset('parallel_test.nc', parallel=True, comm=MPI.COMM_WORLD,
        info=MPI.Info())
assert rank==nc['var'][rank]
nc.close()
nc = Dataset('parallel_test.nc', 'a',parallel=True, comm=MPI.COMM_WORLD,
        info=MPI.Info())
if rank == 3: v[rank] = 2*rank
nc.close()
nc = Dataset('parallel_test.nc', parallel=True, comm=MPI.COMM_WORLD,
        info=MPI.Info())
if rank == 3:
    assert 2*rank==nc['var'][rank]
else:
    assert rank==nc['var'][rank]
nc.close()
