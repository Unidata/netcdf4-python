import os
import unittest

import netCDF4

class VariablesByAttributesTests(unittest.TestCase):

    def setUp(self):
        netcdf_file = os.path.join(os.path.dirname(__file__), "netcdf_dummy_file.nc")
        self.nc = netCDF4.Dataset(netcdf_file)

    def test_find_variables_by_single_attribute(self):
        vs = self.nc.get_variables_by_attributes(axis='Z')
        self.assertEqual(len(vs), 1)

        vs = self.nc.get_variables_by_attributes(units='m/s')
        self.assertEqual(len(vs), 4)

    def test_find_variables_by_multiple_attribute(self):
        vs = self.nc.get_variables_by_attributes(axis='Z', units='m')
        self.assertEqual(len(vs), 1)

    def test_find_variables_by_single_lambda(self):
        vs = self.nc.get_variables_by_attributes(axis=lambda v: v in ['X', 'Y', 'Z', 'T'])
        self.assertEqual(len(vs), 1)

        vs = self.nc.get_variables_by_attributes(grid_mapping=lambda v: v is not None)
        self.assertEqual(len(vs), 12)

    def test_find_variables_by_multiple_lambdas(self):
        vs = self.nc.get_variables_by_attributes(grid_mapping=lambda v: v is not None,
                                                 long_name=lambda v: v is not None and 'Upward (w) velocity' in v)
        self.assertEqual(len(vs), 1)

    def test_find_variables_by_attribute_and_lambda(self):
        vs = self.nc.get_variables_by_attributes(units='m/s',
                                                 grid_mapping=lambda v: v is not None)
        self.assertEqual(len(vs), 4)

        vs = self.nc.get_variables_by_attributes(grid_mapping=lambda v: v is not None,
                                                 long_name='Upward (w) velocity')
        self.assertEqual(len(vs), 1)

if __name__ == '__main__':
    unittest.main()
