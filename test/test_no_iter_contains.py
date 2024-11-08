import os
import tempfile
import unittest

import netCDF4

FILE_NAME = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name


class TestNoIterNoContains(unittest.TestCase):
    def setUp(self) -> None:
        self.file = FILE_NAME
        with netCDF4.Dataset(self.file, "w") as dataset:
            # just create a simple variable
            dataset.createVariable("var1", int)

    def tearDown(self) -> None:
        os.remove(self.file)

    def test_no_iter(self) -> None:
        """Verify that iteration is explicitly not supported"""
        with netCDF4.Dataset(self.file, "r") as dataset:
            with self.assertRaises(TypeError):
                for _ in dataset:  # type: ignore  # type checker catches that this doesn't work
                    pass

    def test_no_contains(self) -> None:
        """Verify the membership operations are explicity not supported"""
        with netCDF4.Dataset(self.file, "r") as dataset:
            with self.assertRaises(TypeError):
                _ = "var1" in dataset

if __name__ == "__main__":
    unittest.main(verbosity=2)
