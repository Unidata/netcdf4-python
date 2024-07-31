"""type_guards.py - Helpers for static and runtime type-checking of initialization arguments
for Dataset and Variable"""

from typing import TYPE_CHECKING, Any, Literal, Type, TypeVar, Union, overload

import netCDF4
from typing_extensions import TypeGuard

if TYPE_CHECKING:
    # in stubs only
    from netCDF4 import (
        AccessMode,
        CalendarType,
        CompressionLevel,
        CompressionType,
        EndianType,
        QuantizeMode,
    )
    from netCDF4 import Format as NCFormat
else:
    AccessMode = Any
    CalendarType = Any
    CompressionLevel = Any
    DiskFormat = Any
    EndianType = Any
    CompressionType = Any
    NCFormat = Any
    QuantizeMode = Any


T = TypeVar("T")


def var_isof_type(
    var: netCDF4.Variable, type_: Type[T]
) -> TypeGuard[netCDF4.Variable[T]]:
    """Check that a variable is of some type. This function does not support CompoundType,
    EnumType, or VLType"""
    if isinstance(type_, (netCDF4.EnumType, netCDF4.VLType, netCDF4.CompoundType)):
        raise TypeError("This function is not valid for user-defined types.")
    return (type_ is str and var.dtype is type_) or var.dtype.type is type_


def valid_access_mode(mode) -> TypeGuard[AccessMode]:
    """Check for a valid `mode` argument for opening a Dataset"""
    return mode in {"r", "w", "r+", "a", "x", "rs", "ws", "r+s", "as"}


def valid_calendar(calendar) -> TypeGuard[CalendarType]:
    """Check for a valid `calendar` argument for cftime functions"""
    return calendar in {
        "standard",
        "gregorian",
        "proleptic_gregorian",
        "noleap",
        "365_day",
        "360_day",
        "julian",
        "all_leap",
        "366_day",
    }


def valid_complevel(complevel) -> TypeGuard[CompressionLevel]:
    """Check for a valid `complevel` argument for creating a Variable"""
    return isinstance(complevel, int) and 0 <= complevel <= 9


def valid_compression(compression) -> TypeGuard[CompressionType]:
    """Check for a valid `compression` argument for creating a Variable"""
    return compression in {
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


def valid_format(format) -> TypeGuard[NCFormat]:
    """Check for a valid `format` argument for opening a Dataset"""
    return format in {
        "NETCDF4",
        "NETCDF4_CLASSIC",
        "NETCDF3_CLASSIC",
        "NETCDF3_64BIT_OFFSET",
        "NETCDF3_64BIT_DATA",
    }


def valid_endian(endian) -> TypeGuard[EndianType]:
    """Check for a valid `endian` argument for creating a Variable"""
    return endian in {"native", "big", "little"}


def valid_bloscshuffle(blosc_shuffle) -> TypeGuard[Literal[0, 1, 2]]:
    """Check for a valid `blosc_shuffle` argument for creating a Variable"""
    return blosc_shuffle in {0, 1, 2}


def valid_quantize_mode(quantize_mode) -> TypeGuard[QuantizeMode]:
    """Check for a valid `quantize_mode` argument for creating a Variable"""
    return quantize_mode in {"BitGroom", "BitRound", "GranularBitRound"}


def valid_szip_coding(szip_coding) -> TypeGuard[Literal["nn", "ec"]]:
    """Check for a valid `szip_coding` argument for creating a Variable"""
    return szip_coding in {"nn", "ec"}


def valid_szip_pixels_per_block(
    szip_pixels_per_block,
) -> TypeGuard[Literal[4, 8, 16, 32]]:
    """Check for a valid `szip_pixels_per_block` argument for creating a Variable"""
    return szip_pixels_per_block in {4, 8, 16, 32}
