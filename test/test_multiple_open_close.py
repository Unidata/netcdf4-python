import os
import tracemalloc
import unittest

import netCDF4


@unittest.skipUnless(
    os.getenv("MEMORY_LEAK_TEST"), "computationally intensive test not enabled"
)
class MultipleVariablesByAttributesCallsTests(unittest.TestCase):
    def test_multiple_calls(self):
        netcdf_file = os.path.join(os.path.dirname(__file__), "netcdf_dummy_file.nc")
        tracemalloc.start()
        snapshot = tracemalloc.take_snapshot()

        k_times = 10
        for _k in range(k_times):
            nc = netCDF4.Dataset(netcdf_file)
            
            vs = nc.get_variables_by_attributes(axis='Z')
            self.assertEqual(len(vs), 1)
            
            vs = nc.get_variables_by_attributes(units='m/s')
            self.assertEqual(len(vs), 4)

            vs = nc.get_variables_by_attributes(axis='Z', units='m')
            self.assertEqual(len(vs), 1)
            
            vs = nc.get_variables_by_attributes(axis=lambda v: v in ['X', 'Y', 'Z', 'T'])
            self.assertEqual(len(vs), 1)

            vs = nc.get_variables_by_attributes(grid_mapping=lambda v: v is not None)
            self.assertEqual(len(vs), 12)

            vs = nc.get_variables_by_attributes(grid_mapping=lambda v: v is not None, long_name=lambda v: v is not None and 'Upward (w) velocity' in v)
            self.assertEqual(len(vs), 1)

            vs = nc.get_variables_by_attributes(units='m/s', grid_mapping=lambda v: v is not None)
            self.assertEqual(len(vs), 4)

            vs = nc.get_variables_by_attributes(grid_mapping=lambda v: v is not None, long_name='Upward (w) velocity')
            self.assertEqual(len(vs), 1)
            nc.close()
        stats = tracemalloc.take_snapshot().compare_to(snapshot, 'filename')
        tracemalloc.stop()
        print("[ Top 10 differences ]")
        for stat in stats[:10]:
            print(stat)

if __name__ == '__main__':    
    unittest.main()
