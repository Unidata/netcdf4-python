# init for netCDF4 package
# Docstring comes from extension module _netCDF4
from ._netCDF4 import *
# Need explicit imports for names beginning with underscores
from ._netCDF4 import __doc__
from ._netCDF4 import (__version__, __netcdf4libversion__, __hdf5libversion__,
                       __has_rename_grp__, __has_nc_inq_path__,
                       __has_nc_inq_format_extended__)
