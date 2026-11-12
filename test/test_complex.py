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
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.tmp_path = pathlib.Path(tmp_dir.name)

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

    def test_write_real_with_complex_dimension(self):
        # A new variable must retain its requested dtype and dimensions, even
        # when its on-disk layout matches a complex-number convention (#1404).
        for format in ("NETCDF4", "NETCDF4_CLASSIC", "NETCDF3_CLASSIC"):
            for dim_name in ("complex", "ri", "_pfnc_complex"):
                for dtype in ("f4", "f8", "i4"):
                    with self.subTest(format=format, dim_name=dim_name, dtype=dtype):
                        filename = self.tmp_path / "test_write_real.nc"
                        values = np.arange(12, dtype=dtype).reshape(6, 2)
                        with netCDF4.Dataset(
                            filename, "w", format=format, auto_complex=True
                        ) as f:
                            f.createDimension("x", 6)
                            f.createDimension(dim_name, 2)
                            var = f.createVariable("data", dtype, ("x", dim_name))
                            self.assertEqual(var.dtype, np.dtype(dtype))
                            self.assertEqual(var.dimensions, ("x", dim_name))
                            self.assertEqual(var.ndim, 2)
                            self.assertEqual(var.shape, values.shape)
                            var[:] = values
                            var[::2, 1] = -1
                            values[::2, 1] = -1
                            np.testing.assert_array_equal(var[:], values)
                            np.testing.assert_array_equal(var[::-2, ::-1], values[::-2, ::-1])

                        with netCDF4.Dataset(filename) as f:
                            self.assertEqual(f["data"].dimensions, ("x", dim_name))
                            np.testing.assert_array_equal(f["data"][:], values)

                        # Existing floating-point layouts must still decode
                        # as complex when reopening with auto_complex=True.
                        if dtype != "i4":
                            with netCDF4.Dataset(filename, auto_complex=True) as f:
                                var = f["data"]
                                self.assertEqual(var.dimensions, ("x",))
                                self.assertEqual(var.ndim, 1)
                                self.assertEqual(var.dtype, np.dtype("c8" if dtype == "f4" else "c16"))
                                np.testing.assert_array_equal(
                                    var[:], values[:, 0] + 1j * values[:, 1]
                                )

    def test_write_real_component_axis(self):
        # A component-like dimension can also be an ordinary leading axis,
        # or the only axis of a real variable.
        for dimensions in (("complex", "x"), ("complex",)):
            with self.subTest(dimensions=dimensions):
                filename = self.tmp_path / "test_write_real_axis.nc"
                shape = (2, 3) if len(dimensions) == 2 else (2,)
                values = np.arange(np.prod(shape), dtype="f8").reshape(shape)
                with netCDF4.Dataset(filename, "w", auto_complex=True) as f:
                    f.createDimension("complex", 2)
                    f.createDimension("x", 3)
                    var = f.createVariable("data", "f8", dimensions)
                    self.assertEqual(var.dimensions, dimensions)
                    self.assertEqual(var.ndim, len(dimensions))
                    var[:] = values
                    np.testing.assert_array_equal(var[:], values)
                with netCDF4.Dataset(filename) as f:
                    np.testing.assert_array_equal(f["data"][:], values)

    def test_write_complex_casts_real_values(self):
        # Both compound and dimension storage must cast real input to complex,
        # and reopened variables must keep the same complex I/O behavior.
        for format in ("NETCDF4", "NETCDF3_CLASSIC"):
            for dtype in ("c8", "c16"):
                with self.subTest(format=format, dtype=dtype):
                    filename = self.tmp_path / "test_write_complex_cast.nc"
                    values = complex_array.astype(dtype)
                    with netCDF4.Dataset(
                        filename, "w", format=format, auto_complex=True
                    ) as f:
                        f.createDimension("x", len(values))
                        var = f.createVariable("data", dtype, ("x",))
                        self.assertEqual(var.dimensions, ("x",))
                        var[:] = values.real
                        np.testing.assert_array_equal(var[:], values.real)
                        var[::2] = values[::2]
                        expected = values.real.astype(dtype)
                        expected[::2] = values[::2]
                        np.testing.assert_array_equal(var[:], expected)
                    with netCDF4.Dataset(filename, "r+", auto_complex=True) as f:
                        var = f["data"]
                        self.assertEqual(var.dtype, np.dtype(dtype))
                        np.testing.assert_array_equal(var[:], expected)
                        var[:] = values
                    with netCDF4.Dataset(filename) as f:
                        raw = f["data"]
                        if format == "NETCDF4":
                            self.assertEqual(raw.dimensions, ("x",))
                            np.testing.assert_array_equal(raw[:]["r"], values.real)
                            np.testing.assert_array_equal(raw[:]["i"], values.imag)
                        else:
                            self.assertEqual(len(raw.dimensions), 2)
                            self.assertEqual(raw.shape, (len(values), 2))
                            np.testing.assert_array_equal(raw[:, 0], values.real)
                            np.testing.assert_array_equal(raw[:, 1], values.imag)
                    with netCDF4.Dataset(filename, auto_complex=True) as f:
                        np.testing.assert_array_equal(f["data"][:], values)

    def test_write_explicit_complex_struct(self):
        filename = self.tmp_path / "test_write_explicit_struct.nc"
        with netCDF4.Dataset(filename, "w", auto_complex=True) as f:
            f.createDimension("x", len(complex_array))
            datatype = f.createCompoundType(np_dt, "nc_complex")
            var = f.createVariable("data", datatype, ("x",))
            var[:] = complex_struct_array
            np.testing.assert_array_equal(var[:], complex_struct_array)
        with netCDF4.Dataset(filename, auto_complex=True) as f:
            np.testing.assert_array_equal(f["data"][:], complex_array)

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
