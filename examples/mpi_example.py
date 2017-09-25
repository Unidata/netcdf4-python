from mpi4py import MPI
from netCDF4 import Dataset
rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)
nc = Dataset('parallel_test.nc', 'w', parallel=True, comm=MPI.COMM_WORLD)
d = nc.createDimension('dim',4)
v = f.createVariable('var', np.int, 'dim')
v[rank] = rank
nc.close()