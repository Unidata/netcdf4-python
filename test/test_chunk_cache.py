import unittest, netCDF4, tempfile, os

file_name = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
cache_size = 10000
cache_nelems = 100
cache_preempt = 0.5
cache_size2 = 20000
cache_nelems2 = 200
cache_preempt2 = 1.0 

class RefCountTestCase(unittest.TestCase):

    def setUp(self):
        nc = netCDF4.Dataset(file_name, mode='w', format='NETCDF4')
        d = nc.createDimension('fred', 2000)
        # can only change cache size in createVariable (not nelems or preemption)
        # this change lasts only as long as file is open.
        v = nc.createVariable('frank','f',('fred',),chunk_cache=15000)
        size, nelems, preempt = v.get_var_chunk_cache()
        assert size==15000 
        self.file=file_name
        nc.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """testing methods for accessing and changing chunk cache"""
        # change cache parameters before opening fil.
        netCDF4.set_chunk_cache(cache_size, cache_nelems, cache_preempt)
        nc = netCDF4.Dataset(self.file, mode='r')
        # check to see that chunk cache parameters were changed.
        assert netCDF4.get_chunk_cache() == (cache_size, cache_nelems, cache_preempt) 
        # change cache parameters for variable, check
        nc['frank'].set_var_chunk_cache(cache_size2, cache_nelems2, cache_preempt2)
        assert nc['frank'].get_var_chunk_cache() == (cache_size2, cache_nelems2, cache_preempt2) 
        nc.close()

if __name__ == '__main__':
    unittest.main()
