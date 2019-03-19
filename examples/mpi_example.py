# to run: mpirun -np 4 python mpi_example.py
import sys
from mpi4py import MPI
import numpy as np
from netCDF4 import Dataset
if len(sys.argv) == 2:
	format = sys.argv[1]
else:
	format = 'NETCDF4_CLASSIC'
rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)
if rank == 0:
	print('Creating file with format {}'.format(format))
nc = Dataset('parallel_test.nc', 'w', parallel=True, comm=MPI.COMM_WORLD,
        info=MPI.Info(),format=format)
# below should work also - MPI_COMM_WORLD and MPI_INFO_NULL will be used.
#nc = Dataset('parallel_test.nc', 'w', parallel=True)
d = nc.createDimension('dim',4)
v = nc.createVariable('var', np.int32, 'dim')
v[rank] = rank
# switch to collective mode, rewrite the data.
v.set_collective(True)
v[rank] = rank
nc.close()
# reopen the file read-only, check the data
nc = Dataset('parallel_test.nc', parallel=True, comm=MPI.COMM_WORLD,
        info=MPI.Info())
assert rank==nc['var'][rank]
nc.close()
# reopen the file in append mode, modify the data on the last rank.
nc = Dataset('parallel_test.nc', 'a',parallel=True, comm=MPI.COMM_WORLD,
        info=MPI.Info())
if rank == 3: v[rank] = 2*rank
nc.close()
# reopen the file read-only again, check the data.
# leave out the comm and info kwargs to check that the defaults
# (MPI_COMM_WORLD and MPI_INFO_NULL) work.
nc = Dataset('parallel_test.nc', parallel=True)
if rank == 3:
    assert 2*rank==nc['var'][rank]
else:
    assert rank==nc['var'][rank]
nc.close()
