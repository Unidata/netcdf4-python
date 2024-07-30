"""_type_guards - Helpers for input type-checking"""
from typing import Literal, TYPE_CHECKING, TypeGuard

if TYPE_CHECKING:
    # in stubs only
    from netCDF4 import CompressionLevel, EndianType, CompressionType
    from netCDF4 import Format as NCFormat
else:
    CompressionLevel = EndianType = CompressionType = NCFormat = object

def valid_complevel(complevel) -> TypeGuard[CompressionLevel | None]:
    return complevel is None or isinstance(complevel, int) and 0 <= complevel <= 9

def valid_endian(endian) -> TypeGuard[EndianType]:
    return endian in {"native", "big", "little"}

def valid_comptype(comptype) -> TypeGuard[CompressionType | None]:
    return comptype is None or comptype in {
        "zlib",
        "szip",
        "zstd",
        "bzip2",
        "blosc_lz",
        "blosc_lz4",
        "blosc_lz4hc",
        "blosc_zlib",
        "blosc_zstd",
    }

def valid_bloscshuffle(bloscshuffle) -> TypeGuard[Literal[0, 1, 2]]:
    return bloscshuffle in {0, 1, 2}

def valid_ncformat(ncformat) -> TypeGuard[NCFormat]:
    return ncformat in [
        "NETCDF4",
        "NETCDF4_CLASSIC",
        "NETCDF3_CLASSIC",
        "NETCDF3_64BIT_OFFSET",
        "NETCDF3_64BIT_DATA"
    ]
