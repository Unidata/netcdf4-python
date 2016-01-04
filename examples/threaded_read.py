from __future__ import print_function
from netCDF4 import Dataset
from numpy.testing import assert_array_equal, assert_array_almost_equal
import numpy as np
import threading
import queue
import time

# demonstrate reading of different files from different threads.
# Releasing the Global Interpreter Lock (GIL) when calling the
# netcdf C library for read operations speeds up the reads
# when threads are used (issue 369).
# Test script contributed by Ryan May of Unidata.

# Make some files
nfiles = 4
fnames = []; datal = []
for i in range(nfiles):
    fname = 'test%d.nc' % i
    fnames.append(fname)
    nc = Dataset(fname, 'w')
    data = np.random.randn(500, 500, 500)
    datal.append(data)
    nc.createDimension('x', 500)
    nc.createDimension('y', 500)
    nc.createDimension('z', 500)
    var = nc.createVariable('grid', 'f', ('x', 'y', 'z'))
    var[:] = data
    nc.close()

# Queue them up
items = queue.Queue()
for data,fname in zip(datal,fnames):
    items.put(fname)

# Function for threads to use
def get_data(serial=None):
    if serial is None: # if not called from a thread
        fname = items.get()
    else:
        fname = fnames[serial]
    nc = Dataset(fname, 'r')
    data2 = nc.variables['grid'][:]
    # make sure the data is correct
    #assert_array_almost_equal(data2,datal[int(fname[4])])
    nc.close()
    if serial is None:
        items.task_done()

# Time it (no threading).
start = time.time()
for i in range(nfiles):
    get_data(serial=i)
end = time.time()
print('no threads, time = ',end - start)

# with threading.
start = time.time()
for i in range(nfiles):
    threading.Thread(target=get_data).start()
items.join()
end = time.time()
print('with threading, time = ',end - start)
