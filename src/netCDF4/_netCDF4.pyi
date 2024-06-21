# The definitions are intendionally done in the __init__.
# This file only exists in case someone imports from netCDF4._netCDF4
from . import (
    Dataset, Variable, Dimension, Group, MFDataset, MFTime, CompoundType,
    VLType, date2num, num2date, date2index, stringtochar, chartostring,
    stringtoarr, getlibversion, EnumType, get_chunk_cache, set_chunk_cache,
    set_alignment, get_alignment, default_fillvals, default_encoding,
    NetCDF4MissingFeatureException, is_native_big, is_native_little, unicode_error,
    __version__, __netcdf4libversion__, __hdf5libversion__, __has_rename_grp__,
    __has_nc_inq_path__, __has_nc_inq_format_extended__, __has_nc_open_mem__,
    __has_nc_create_mem__, __has_cdf5_format__, __has_parallel4_support__,
    __has_pnetcdf_support__, __has_parallel_support__,
    __has_quantization_support__, __has_zstandard_support__,
    __has_bzip2_support__, __has_blosc_support__, __has_szip_support__,
    __has_set_alignment__, __has_ncfilter__
)