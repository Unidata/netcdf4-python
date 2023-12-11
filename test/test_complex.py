import netCDF4
import numpy as np
import pathlib
import tempfile
import unittest

complex_array = np.array([0 + 0j, 1 + 0j, 0 + 1j, 1 + 1j, 0.25 + 0.75j], dtype="c16")
np_dt = np.dtype([("r", np.float64), ("i", np.float64)])
complex_struct_array = np.array(
    [(r, i) for r, i in zip(complex_array.real, complex_array.imag)],
    dtype=np_dt,
)


class ComplexNumbersTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_path = pathlib.Path(tempfile.mkdtemp())

    def test_read_dim(self):
        filename = self.tmp_path / "test_read_dim.nc"

        with netCDF4.Dataset(filename, "w") as f:
            f.createDimension("x", size=len(complex_array))
            f.createDimension("ri", size=2)
            c_ri = f.createVariable("data_dim", np.float64, ("x", "ri"))
            as_dim_array = np.vstack((complex_array.real, complex_array.imag)).T
            c_ri[:] = as_dim_array

        with netCDF4.Dataset(filename, "r", auto_complex=True) as f:
            assert "data_dim" in f.variables
            data_dim = f["data_dim"]
            assert data_dim.shape == complex_array.shape
            data = data_dim[:]

        assert np.array_equal(data, complex_array)

    def test_read_struct(self):
        filename = self.tmp_path / "test_read_struct.nc"

        with netCDF4.Dataset(filename, "w") as f:
            f.createDimension("x", size=len(complex_array))
            nc_dt = f.createCompoundType(np_dt, "nc_complex")
            c_struct = f.createVariable("data_struct", nc_dt, ("x",))
            c_struct[:] = complex_struct_array

        with netCDF4.Dataset(filename, "r", auto_complex=True) as f:
            assert "data_struct" in f.variables
            data = f["data_struct"][:]

        assert np.array_equal(data, complex_array)

    def test_write(self):
        filename = self.tmp_path / "test_write.nc"
        with netCDF4.Dataset(filename, "w", auto_complex=True) as f:
            f.createDimension("x", size=len(complex_array))
            complex_var = f.createVariable("complex_data", "c16", ("x",))
            complex_var[:] = complex_array

        with netCDF4.Dataset(filename, "r") as f:
            assert "complex_data" in f.variables
            assert np.array_equal(f["complex_data"], complex_struct_array)

    def test_write_with_np_complex128(self):
        filename = self.tmp_path / "test_write_with_np_complex128.nc"
        with netCDF4.Dataset(filename, "w", auto_complex=True) as f:
            f.createDimension("x", size=len(complex_array))
            complex_var = f.createVariable("complex_data", np.complex128, ("x",))
            complex_var[:] = complex_array

        with netCDF4.Dataset(filename, "r") as f:
            assert "complex_data" in f.variables
            assert np.array_equal(f["complex_data"], complex_struct_array)

    def test_write_netcdf3(self):
        filename = self.tmp_path / "test_write_netcdf3.nc"
        with netCDF4.Dataset(
            filename, "w", format="NETCDF3_CLASSIC", auto_complex=True
        ) as f:
            f.createDimension("x", size=len(complex_array))
            complex_var = f.createVariable("complex_data", "c16", ("x",))
            complex_var[:] = complex_array

        with netCDF4.Dataset(filename, "r", auto_complex=True) as f:
            assert "complex_data" in f.variables
            assert np.array_equal(f["complex_data"][:], complex_array)


if __name__ == "__main__":
    unittest.main()
