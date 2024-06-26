
import os
import sys
import datetime as dt
from typing import (Any, Callable, Final, Generic, Iterable, Literal, Mapping,
                    NoReturn, Self, Sequence, TypeAlias, TypedDict, TypeVar, Union, overload)
from typing_extensions import Buffer

import cftime
import numpy as np
import numpy.typing as npt

__all__ = [
    'Dataset', 'Variable', 'Dimension', 'Group', 'MFDataset', 'MFTime', 'CompoundType',
    'VLType', 'date2num', 'num2date', 'date2index', 'stringtochar', 'chartostring',
    'stringtoarr', 'getlibversion', 'EnumType', 'get_chunk_cache', 'set_chunk_cache',
    'set_alignment', 'get_alignment'
]
__pdoc__ = {'utils': False}


if sys.version_info >= (3, 10):
    from types import EllipsisType

    ellipsis = EllipsisType
elif not TYPE_CHECKING:
    ellipsis = type(Ellipsis)  # keeps ruff happy until ruff uses typeshed


_DatatypeStrOptions: TypeAlias = Literal[
    'S1', 'c', 'i1', 'b', 'B', 'u1', 'i2', 'h', 's', 'u2', 'i4',
    'i', 'l', 'u4', 'i8', 'u8', 'f4', 'f', 'f8', 'd', 'c8', 'c16'
]
_DatatypeNCOptions: TypeAlias = Union[CompoundType, VLType, EnumType]
DatatypeOptions: TypeAlias = Union[_DatatypeStrOptions, _DatatypeNCOptions, npt.DTypeLike]
T_Datatype = TypeVar("T_Datatype", bound=DatatypeOptions)
T_DatatypeNC = TypeVar("T_DatatypeNC", CompoundType, VLType, EnumType)

DimensionsOptions: TypeAlias = Union[str, bytes, Dimension, Iterable[Union[str, bytes, Dimension]]]
CompressionOptions: TypeAlias = Literal[
    'zlib', 'szip', 'zstd', 'blosc_lz','blosc_lz4', 
    'blosc_lz4hc', 'blosc_zlib', 'blosc_zstd'
]
CompressionLevelOptions: TypeAlias = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
AccessModeOptions: TypeAlias = Literal['r', 'w', 'r+', 'a', 'x', 'rs', 'ws', 'r+s', 'as']
FormatOptions: TypeAlias = Literal[
    'NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_CLASSIC',
    'NETCDF3_64BIT_OFFSET', 'NETCDF3_64BIT_DATA'
]
DiskFormatOptions: TypeAlias = Literal['NETCDF3', 'HDF5', 'HDF4', 'PNETCDF', 'DAP2', 'DAP4', 'UNDEFINED']
QuantizeOptions: TypeAlias = Literal['BitGroom', 'BitRound', 'GranularBitRound']
EndianOptions: TypeAlias = Literal['native', 'little', 'big']
CalendarOptions: TypeAlias = Literal[
    'standard', 'gregorian', 'proleptic_gregorian' 'noleap',
    '365_day', '360_day', 'julian', 'all_leap', '366_day'
]
BoolInt: TypeAlias = Literal[0, 1]

DateTimeArray: TypeAlias = npt.NDArray[np.object_]
"""numpy array of datetime.datetime or cftime.datetime"""

GetSetItemKey: TypeAlias = (
    int
    | slice
    | ellipsis
    | list[int | bool]
    | npt.NDArray[np.integer | np.bool_]
    | tuple[int | slice | ellipsis | Sequence[int | bool] | npt.NDArray[np.integer | np.bool_], ...]
)

__version__: str
__netcdf4libversion__: str
__hdf5libversion__: str
__has_rename_grp__: BoolInt
__has_nc_inq_path__: BoolInt
__has_nc_inq_format_extended__: BoolInt
__has_nc_open_mem__: BoolInt
__has_nc_create_mem__: BoolInt
__has_cdf5_format__: BoolInt
__has_parallel4_support__: BoolInt
__has_pnetcdf_support__: BoolInt
__has_parallel_support__: BoolInt
__has_quantization_support__: BoolInt
__has_zstandard_support__: BoolInt
__has_bzip2_support__: BoolInt
__has_blosc_support__: BoolInt
__has_szip_support__: BoolInt
__has_set_alignment__: BoolInt
__has_ncfilter__: BoolInt
is_native_little: bool
is_native_big: bool
default_encoding: Final = 'utf-8'
unicode_error: Final = 'replace'
default_fillvals: dict[str, int | float | str]


# date2index, date2num, and num2date are actually provided by cftime and if stubs for
# cftime are completed these should be removed.
def date2index(
    dates: dt.datetime | cftime.datetime | Sequence[dt.datetime | cftime.datetime] | DateTimeArray,
    nctime: Variable,
    calendar: CalendarOptions | None = None,
    select: Literal["exact", "before", "after", "nearest"] = "exact",
    has_year_zero: bool | None = None,
) -> int | npt.NDArray[np.int_]: ...
def date2num(
    dates: dt.datetime | cftime.datetime | Sequence[dt.datetime | cftime.datetime] | DateTimeArray,
    units: str,
    calendar: CalendarOptions | None = None,
    has_year_zero: bool | None = None,
    longdouble: bool = False,
) -> np.number | npt.NDArray[np.number]: ...
def num2date(
    times: Sequence[int | float | np.number] | npt.NDArray[np.number],
    units: str,
    calendar: CalendarOptions = "standard",
    only_use_cftime_datetimes: bool = True,
    only_use_python_datetimes: bool = False,
    has_year_zero: bool | None = None,
) -> dt.datetime | DateTimeArray: ...


class BloscInfo(TypedDict):
    compressor: Literal["blosc_lz", "blosc_lz4", "blosc_lz4hc", "blosc_zlib", "blosc_zstd"]
    shuffle: Literal[0, 1, 2]


class SzipInfo(TypedDict):
    coding: Literal["nn", "ec"]
    pixels_per_block: Literal[4, 8, 16, 32]


class FiltersDict(TypedDict):
    """Dict returned from netCDF4.Variable.filters()"""
    zlib: bool
    szip: Literal[False] | SzipInfo
    zstd: bool
    bzip2: bool
    blosc: Literal[False] | BloscInfo
    shuffle: bool
    complevel: int
    fletcher32: bool


class NetCDF4MissingFeatureException(Exception):
    def __init__(self, feature: str, version: str): ...


class Dataset:
    def __init__(
        self,
        filename: str | os.PathLike,
        mode: AccessModeOptions = 'r',
        clobber: bool = True,
        format: FormatOptions = 'NETCDF4',
        diskless: bool = False,
        persist: bool = False,
        keepweakref: bool = False,
        memory: Buffer | int | None = None,
        encoding: str | None = None,
        parallel: bool = False,
        comm: Any = None,
        info: Any = None,
        auto_complex: bool = False,
        **kwargs: Any
    ): ...

    @property
    def name(self) -> str: ...
    @property
    def groups(self) -> dict[str, Group]: ...
    @property
    def dimensions(self) -> dict[str, Dimension]: ...
    @property
    def variables(self) -> dict[str, Variable[Any]]: ...
    @property
    def cmptypes(self) -> dict[str, CompoundType]: ...
    @property
    def vltypes(self) -> dict[str, VLType]: ...
    @property
    def enumtypes(self) -> dict[str, EnumType]: ...
    @property
    def data_model(self) -> FormatOptions: ...
    @property
    def file_format(self) -> FormatOptions: ...
    @property
    def disk_format(self) -> DiskFormatOptions: ...
    @property
    def parent(self) -> Dataset | None: ...
    @property
    def path(self) -> str: ...
    @property
    def keepweakref(self) -> bool: ...
    @property
    def auto_complex(self) -> bool: ...
    @property
    def _ncstring_attrs__(self) -> bool: ...
    @property
    def __orthogonal_indexing__(self) -> bool: ...

    def filepath(self, encoding: str | None = None) -> str: ...
    def isopen(self) -> bool: ...
    def close(self) -> memoryview: ...  # only if writing and memory != None, but otherwise people ignore the return None anyway
    def sync(self) -> None: ...
    def set_fill_on(self) -> None: ...
    def set_fill_off(self) -> None: ...

    def createDimension(self, dimname: str, size: int | None = None) -> Dimension: ...
    def renameDimension( self, oldname: str, newname: str) -> None: ...
    @overload
    def createVariable(  # type: ignore
        self,
        varname: str,
        datatype: T_DatatypeNC,
        dimensions: DimensionsOptions = (),
        compression: CompressionOptions | None = None,
        zlib: bool = False,
        complevel: CompressionLevelOptions | None = 4,
        shuffle: bool = True,
        szip_coding: Literal['nn', 'ec'] = 'nn',
        szip_pixels_per_block: Literal[4, 8, 16, 32] = 8,
        blosc_shuffle: Literal[0, 1, 2] = 1,
        fletcher32: bool = False,
        contiguous: bool = False,
        chunksizes: int | None = None,
        endian: EndianOptions = 'native',
        least_significant_digit: int | None = None,
        significant_digits: int | None = None,
        quantize_mode: QuantizeOptions = 'BitGroom',
        fill_value: int | float | str | bytes | Literal[False] | None = None,
        chunk_cache: int | None = None
    ) -> Variable[T_DatatypeNC]: ...
    @overload
    def createVariable(
        self,
        varname: str,
        datatype: _DatatypeStrOptions | npt.DTypeLike,
        dimensions: DimensionsOptions = (),
        compression: CompressionOptions | None = None,
        zlib: bool = False,
        complevel: CompressionLevelOptions | None = 4,
        shuffle: bool = True,
        szip_coding: Literal['nn', 'ec'] = 'nn',
        szip_pixels_per_block: Literal[4, 8, 16, 32] = 8,
        blosc_shuffle: Literal[0, 1, 2] = 1,
        fletcher32: bool = False,
        contiguous: bool = False,
        chunksizes: int | None = None,
        endian: EndianOptions = 'native',
        least_significant_digit: int | None = None,
        significant_digits: int | None = None,
        quantize_mode: QuantizeOptions = 'BitGroom',
        fill_value: int | float | str | bytes | Literal[False] | None = None,
        chunk_cache: int | None = None
    ) -> Variable[np.dtype]: ...
    def renameVariable(self, oldname: str, newname: str) -> None: ...
    def createGroup(self, groupname: str) -> Group: ...

    def renameGroup(self, oldname: str, newname: str) -> None: ...
    def renameAttribute(self, oldname: str, newname: str) -> None: ...
    def createCompoundType(
        self, datatype: npt.DTypeLike | Sequence[tuple[str, npt.DTypeLike]], datatype_name: str
    ) -> CompoundType: ...
    def createVLType(self, datatype: npt.DTypeLike, datatype_name: str) -> VLType: ...
    def createEnumType(
        self, datatype: np.dtype[np.integer] | type[np.integer] | type[int], datatype_name: str, enum_dict: dict[str, int]
    ) -> EnumType: ...

    def ncattrs(self) -> list[str]: ...
    def setncattr_string(self, name: str, value: Any) -> None: ...
    def setncattr(self, name: str, value: Any) -> None: ...
    def setncatts(self, attdict: Mapping[str, Any]) -> None: ...
    def getncattr(self, name: str, encoding: str = 'utf-8') -> Any: ...
    def delncattr(self, name: str) -> None: ...
    def set_auto_chartostring(self, value: bool) -> None: ...
    def set_auto_maskandscale(self, value: bool) -> None: ...
    def set_auto_mask(self, value: bool) -> None: ...
    def set_auto_scale(self, value: bool) -> None: ...
    def set_always_mask(self, value: bool) -> None: ...
    def set_ncstring_attrs(self, value: bool) -> None: ...
    def get_variables_by_attributes(self, **kwargs: Callable[[Any], bool] | Any) -> list[Variable]: ...

    @staticmethod
    def fromcdl(
        cdlfilename: str,
        ncfilename: str | None = None,
        mode: AccessModeOptions = 'a',
        format: FormatOptions = 'NETCDF4'
    ) -> Dataset: ...
    @overload
    def tocdl(
        self,
        coordvars: bool = False,
        data: bool = False,
        outfile: None = None
    ) -> str: ...
    @overload
    def tocdl(
        self,
        coordvars: bool = False,
        data: bool = False,
        *,
        outfile: str | os.PathLike
    ) -> None: ...

    def has_blosc_filter(self) -> bool: ...
    def has_zstd_filter(self) -> bool: ...
    def has_bzip2_filter(self) -> bool: ...
    def has_szip_filter(self) -> bool: ...

    def __getitem__(self, elem: str) -> Any: ...  # should be Group | Variable, but this causes too many problems
    def __setattr__(self, name: str, value: Any) -> None: ...
    def __getattr__(self, name: str) -> Any: ...
    def __delattr__(self, name: str): ...
    def __dealloc(self) -> None: ...
    def __reduce__(self) -> NoReturn: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, atype, value, traceback) -> None: ...


class Group(Dataset):
    def __init__(self, parent: Dataset, name: str, **kwargs: Any) -> None: ...

    def close(self) -> NoReturn: ...


class Dimension:
    def __init__(self, grp: Dataset, name: str, size: int | None = None, **kwargs: Any) -> None: ...

    @property
    def name(self) -> str: ...
    @property
    def size(self) -> int: ...

    def group(self) -> Dataset: ...
    def isunlimited(self) -> bool: ...

    def __len__(self) -> int: ...


class Variable(Generic[T_Datatype]):

    @overload
    def __new__(  # type: ignore
        self,
        grp: Dataset,
        name: str,
        datatype: T_DatatypeNC,
        dimensions: DimensionsOptions = (),
        compression: CompressionOptions | None = None,
        zlib: bool = False,
        complevel: CompressionLevelOptions | None = 4,
        shuffle: bool = True,
        szip_coding: Literal['nn', 'ec'] = 'nn',
        szip_pixels_per_block: Literal[4, 8, 16, 32] = 8,
        blosc_shuffle: Literal[0, 1, 2] = 1,
        fletcher32: bool = False,
        contiguous: bool = False,
        chunksizes: Sequence[int] | None = None,
        endian: EndianOptions = 'native',
        least_significant_digit: int | None = None,
        significant_digits: int | None = None,
        quantize_mode: QuantizeOptions = 'BitGroom',
        fill_value: int | float | str | bytes | Literal[False] | None = None,
        chunk_cache: int | None = None,
        **kwargs: Any
    ) -> Variable[T_DatatypeNC]: ...

    @overload
    def __new__(
        self,
        grp: Dataset,
        name: str,
        datatype: _DatatypeStrOptions | npt.DTypeLike,
        dimensions: DimensionsOptions = (),
        compression: CompressionOptions | None = None,
        zlib: bool = False,
        complevel: CompressionLevelOptions | None = 4,
        shuffle: bool = True,
        szip_coding: Literal['nn', 'ec'] = 'nn',
        szip_pixels_per_block: Literal[4, 8, 16, 32] = 8,
        blosc_shuffle: Literal[0, 1, 2] = 1,
        fletcher32: bool = False,
        contiguous: bool = False,
        chunksizes: Sequence[int] | None = None,
        endian: EndianOptions = 'native',
        least_significant_digit: int | None = None,
        significant_digits: int | None = None,
        quantize_mode: QuantizeOptions = 'BitGroom',
        fill_value: int | float | str | bytes | Literal[False] | None = None,
        chunk_cache: int | None = None,
        **kwargs: Any
    ) -> Variable[np.dtype]: ...


    def __init__(
        self,
        grp: Dataset,
        name: str,
        datatype: T_Datatype,
        dimensions: DimensionsOptions = (),
        compression: CompressionOptions | None = None,
        zlib: bool = False,
        complevel: CompressionLevelOptions | None = 4,
        shuffle: bool = True,
        szip_coding: Literal['nn', 'ec'] = 'nn',
        szip_pixels_per_block: Literal[4, 8, 16, 32] = 8,
        blosc_shuffle: Literal[0, 1, 2] = 1,
        fletcher32: bool = False,
        contiguous: bool = False,
        chunksizes: Sequence[int] | None = None,
        endian: EndianOptions = 'native',
        least_significant_digit: int | None = None,
        significant_digits: int | None = None,
        quantize_mode: QuantizeOptions = 'BitGroom',
        fill_value: int | float | str | bytes | Literal[False] | None = None,
        chunk_cache: int | None = None,
        **kwargs: Any
    ) -> None: ...

    @property
    def name(self) -> str: ...
    @property
    def dtype(self) -> np.dtype: ...
    @property
    def datatype(self) -> T_Datatype: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def size(self) -> int: ...
    @property
    def dimensions(self) -> tuple[str, ...]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def scale(self) -> bool: ...
    @property
    def mask(self) -> bool: ...
    @property
    def chartostring(self) -> bool: ...
    @property
    def always_mask(self) -> bool: ...
    @property
    def __orthogonal_indexing__(self) -> bool: ...

    def group(self) -> Dataset: ...
    def ncattrs(self) -> list[str]: ...
    def setncattr(self, name: str, value: Any) -> None: ...
    def setncattr_string(self, name: str, value: Any) -> None: ...
    def setncatts(self, attdict: Mapping[str, Any]) -> None: ...
    def getncattr(self, name: str, encoding='utf-8'): ...
    def delncattr(self, name: str) -> None: ...
    def filters(self) -> FiltersDict: ...
    def quantization(self) -> tuple[int, QuantizeOptions] | None: ...
    def endian(self) -> EndianOptions: ...
    def chunking(self) -> Literal['contiguous'] | list[int]: ...
    def get_var_chunk_cache(self) -> tuple[int, int, float]: ...
    def set_var_chunk_cache(
        self,
        size: int | None = None,
        nelems: int | None = None,
        preemption: float | None = None
    ) -> None: ...
    def renameAttribute(self, oldname: str, newname: str) -> None: ...
    def assignValue(self, val: Any) -> None: ...
    def getValue(self) -> Any: ...
    def set_auto_chartostring(self, chartostring: bool) -> None: ...
    def use_nc_get_vars(self, use_nc_get_vars: bool) -> None: ...
    def set_auto_maskandscale(self, maskandscale: bool) -> None: ...
    def set_auto_scale(self, scale: bool) -> None: ...
    def set_auto_mask(self, mask: bool) -> None: ...
    def set_always_mask(self, always_mask: bool) -> None: ...
    def set_ncstring_attrs(self, ncstring_attrs: bool) -> None: ...
    def set_collective(self, value: bool) -> None: ...
    def get_dims(self) -> tuple[Dimension, ...]: ...

    def __delattr__(self, name: str) -> None: ...
    def __setattr__(self, name: str, value: Any) -> None: ...
    def __getattr__(self, name: str) -> Any: ...
    def __getitem__(self, elem: GetSetItemKey) -> np.ndarray: ...
    def __setitem__(self, elem: GetSetItemKey, data: npt.ArrayLike) -> None: ...
    def __array__(self) -> np.ndarray: ...
    def __len__(self) -> int: ...


class CompoundType:
    dtype: np.dtype
    dtype_view: np.dtype
    name: str

    def __init__(
        self, grp: Dataset, dt: npt.DTypeLike | Sequence[tuple[str, npt.DTypeLike]], dtype_name: str, **kwargs: Any
    ) -> None: ...

    def __reduce__(self) -> NoReturn: ...


class VLType:
    dtype: np.dtype
    name: str | None

    def __init__(self, grp: Dataset, dt: npt.DTypeLike, dtype_name: str, **kwargs: Any) -> None: ...

    def __reduce__(self) -> NoReturn: ...


class EnumType:
    dtype: np.dtype[np.integer]
    name: str
    enum_dict: Mapping[str, int]

    def __init__(
        self,
        grp: Dataset,
        dt: np.dtype[np.integer] | type[np.integer] | type[int] | str,
        dtype_name: str,
        enum_dict: Mapping[str, int],
        **kwargs: Any
    ) -> None: ...

    def __reduce__(self) -> NoReturn: ...


class MFDataset(Dataset):
    def __init__(
        self,
        files: str | Sequence[str | os.PathLike],
        check: bool = False,
        aggdim: str | None = None,
        exclude: Sequence[str] = [],
        master_file: str | os.PathLike | None = None
    ) -> None: ...

    @property
    def groups(self) -> dict[str, Group]: ...
    @property
    def dimensions(self) -> dict[str, Dimension]: ...  # this should be: dict[str, Dimension | _Dimension]
    @property
    def variables(self) -> dict[str, Variable[Any]]: ...  # this should be: dict[str, _Variable[Any] | _Variable]
    @property
    def data_model(self) -> FormatOptions: ...
    @property
    def file_format(self) -> FormatOptions: ...
    @property
    def disk_format(self) -> DiskFormatOptions: ...
    @property
    def path(self) -> str: ...


class _Dimension:
    dimlens: list[int]
    dimtolen: int

    def __init__(
        self, dimname: str, dim: Dimension, dimlens: list[int], dimtotlen: int
    ) -> None: ...

    def __len__(self) -> int: ...
    def isunlimited(self) -> Literal[True]: ...


class _Variable:
    dimensions: tuple[str, ...]
    dtype: np.dtype

    def __init__(self, dset: Dataset, varname: str, var, recdimname: str) -> None: ...

    def typecode(self) -> np.dtype: ...
    def ncattrs(self) -> list[str]: ...
    def _shape(self) -> tuple[int, ...]: ...
    def set_auto_chartostring(self, val: bool) -> None: ...
    def set_auto_maskandscale(self, val: bool) -> None: ...
    def set_auto_mask(self, val: bool) -> None: ...
    def set_auto_scale(self, val: bool) -> None: ...
    def set_always_mask(self, val: bool) -> None: ...

    def __getattr__(self, name: str): ...
    def __getitem__(self, elem: GetSetItemKey): ...
    def __len__(self) -> int: ...


class MFTime(_Variable):
    calendar: CalendarOptions | None
    units: str | None

    def __init__(
        self,
        time: Variable,
        units: str | None = None,
        calendar: CalendarOptions | None = None
    ): ...
    def __getitem__(self, elem: GetSetItemKey) -> np.ndarray: ...


def stringtoarr(string, NUMCHARS: int, dtype: Literal['S', 'U'] = 'S') -> np.ndarray: ...
def stringtochar(a, encoding: str | bytes | None = 'utf-8') -> np.ndarray: ...
def chartostring(b, encoding: str | bytes | None = 'utf-8') -> np.ndarray: ...

def getlibversion() -> str: ...

def set_alignment(threshold: int, alignment: int): ...
def get_alignment() -> tuple[int, int]: ...

def set_chunk_cache(
    size: int | None = None,
    nelems: int | None = None,
    preemption: float | None = None
) -> None: ...
def get_chunk_cache() -> tuple[int, int, float]: ...
