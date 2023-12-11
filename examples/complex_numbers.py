import netCDF4
import numpy as np

complex_array = np.array([0 + 0j, 1 + 0j, 0 + 1j, 1 + 1j, 0.25 + 0.75j], dtype="c16")
np_dt = np.dtype([("r", np.float64), ("i", np.float64)])
complex_struct_array = np.array(
    [(r, i) for r, i in zip(complex_array.real, complex_array.imag)],
    dtype=np_dt,
)

print("\n**********")
print("Reading a file that uses a dimension for complex numbers")
filename = "complex_numbers_as_dimension.nc"

with netCDF4.Dataset(filename, "w") as f:
    f.createDimension("x", size=len(complex_array))
    f.createDimension("complex", size=2)
    c_ri = f.createVariable("data_dim", np.float64, ("x", "complex"))
    as_dim_array = np.vstack((complex_array.real, complex_array.imag)).T
    c_ri[:] = as_dim_array

with netCDF4.Dataset(filename, "r", auto_complex=True) as f:
    print(f["data_dim"])


print("\n**********")
print("Reading a file that uses a compound datatype for complex numbers")
filename = "complex_numbers_as_datatype.nc"

with netCDF4.Dataset(filename, "w") as f:
    f.createDimension("x", size=len(complex_array))
    nc_dt = f.createCompoundType(np_dt, "nc_complex")
    breakpoint()

    c_struct = f.createVariable("data_struct", nc_dt, ("x",))
    c_struct[:] = complex_struct_array

with netCDF4.Dataset(filename, "r", auto_complex=True) as f:
    print(f["data_struct"])

print("\n**********")
print("Writing complex numbers to a file")
filename = "writing_complex_numbers.nc"
with netCDF4.Dataset(filename, "w", auto_complex=True) as f:
    f.createDimension("x", size=len(complex_array))
    c_var = f.createVariable("data", np.complex128, ("x",))
    c_var[:] = complex_array
    print(c_var)

with netCDF4.Dataset(filename, "r", auto_complex=True) as f:
    print(f["data"])
