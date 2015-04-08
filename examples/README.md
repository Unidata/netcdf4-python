* `tutorial.py`:  code from introduction section of documentation.
* `json_att.py`:  shows to to use json to serialize python objects, save them as
  netcdf attributes, and then convert them back to python objects.
* `subset.py`: shows how to use 'orthogonal indexing' to select geographic regions.
* `reading_netcdf.ipynb`: ipython notebook from Unidata python workshop.
* `writing_netcdf.ipynb`: ipython notebook from Unidata python workshop.
* `threaded_read.py`:  test script for concurrent threaded reads.
* `bench.py`:  benchmarks for reading/writing using different formats.
* `bench_compress*.py``: benchmarks for reading/writing with compression.
* `bench_diskless.py`: benchmarks for 'diskless' IO.
* `test_stringarr.py`: test utilities for converting arrays of fixed-length strings
  to arrays of characters (with an extra dimension), and vice-versa.
  Useful since netcdf does not have a datatype for fixed-length string arrays.
