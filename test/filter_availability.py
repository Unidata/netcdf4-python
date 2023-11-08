from tempfile import NamedTemporaryFile
from netCDF4 import (
    Dataset,
    __has_zstandard_support__,
    __has_bzip2_support__,
    __has_blosc_support__,
    __has_szip_support__,
)
import os

# True if plugins have been disabled
no_plugins = os.getenv("NO_PLUGINS")


with NamedTemporaryFile(suffix=".nc", delete=False) as tf:
    with Dataset(tf.name, "w") as nc:
        has_zstd_filter = __has_zstandard_support__ and nc.has_zstd_filter()
        has_bzip2_filter = __has_bzip2_support__ and nc.has_bzip2_filter()
        has_blosc_filter = __has_blosc_support__ and nc.has_blosc_filter()
        has_szip_filter = __has_szip_support__ and nc.has_szip_filter()
