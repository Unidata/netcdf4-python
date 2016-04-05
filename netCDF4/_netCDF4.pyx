"""
Version 1.2.4
-------------
- - - 

Introduction
============

netcdf4-python is a Python interface to the netCDF C library.  

[netCDF](http://www.unidata.ucar.edu/software/netcdf/) version 4 has many features
not found in earlier versions of the library and is implemented on top of
[HDF5](http://www.hdfgroup.org/HDF5). This module can read and write
files in both the new netCDF 4 and the old netCDF 3 format, and can create
files that are readable by HDF5 clients. The API modelled after
[Scientific.IO.NetCDF](http://dirac.cnrs-orleans.fr/ScientificPython/),
and should be familiar to users of that module.

Most new features of netCDF 4 are implemented, such as multiple
unlimited dimensions, groups and zlib data compression.  All the new
numeric data types (such as 64 bit and unsigned integer types) are
implemented. Compound (struct), variable length (vlen) and
enumerated (enum) data types are supported, but not the opaque data type.
Mixtures of compound, vlen and enum data types (such as
compound types containing enums, or vlens containing compound
types) are not supported.

Download
========

 - Latest bleeding-edge code from the 
   [github repository](http://github.com/Unidata/netcdf4-python).
 - Latest [releases](https://pypi.python.org/pypi/netCDF4)
   (source code and windows installers).

Requires
========

 - Python 2.6 or later (python 3 works too).
 - [numpy array module](http://numpy.scipy.org), version 1.7.0 or later.
 - [Cython](http://cython.org), version 0.19 or later, is optional - if it is installed setup.py will
   use it to recompile the Cython source code into C, using conditional compilation
   to enable features in the netCDF API that have been added since version 4.1.1.  If
   Cython is not installed, these features (such as the ability to rename Group objects)
   will be disabled to preserve backward compatibility with older versions of the netCDF
   library.
 - For python < 2.7, the [ordereddict module](http://python.org/pypi/ordereddict).
 - The HDF5 C library version 1.8.4-patch1 or higher (1.8.8 or higher
 recommended) from [](ftp://ftp.hdfgroup.org/HDF5/current/src).
 Be sure to build with `--enable-hl --enable-shared`.
 - [Libcurl](http://curl.haxx.se/libcurl), if you want
 [OPeNDAP](http://opendap.org) support.
 - [HDF4](http://www.hdfgroup.org/products/hdf4), if you want
 to be able to read HDF4 "Scientific Dataset" (SD) files.
 - The netCDF-4 C library from 
 [ftp://ftp.unidata.ucar.edu/pub/netcdf](ftp://ftp.unidata.ucar.edu/pub/netcdf).
 Version 4.1.1 or higher is required (4.2 or higher recommended).
 Be sure to build with `--enable-netcdf-4 --enable-shared`, and set
 `CPPFLAGS="-I $HDF5_DIR/include"` and `LDFLAGS="-L $HDF5_DIR/lib"`,
 where `$HDF5_DIR` is the directory where HDF5 was installed.
 If you want [OPeNDAP](http://opendap.org) support, add `--enable-dap`.
 If you want HDF4 SD support, add `--enable-hdf4` and add
 the location of the HDF4 headers and library to `$CPPFLAGS` and `$LDFLAGS`.


Install
=======

 - install the requisite python modules and C libraries (see above). It's
 easiest if all the C libs are built as shared libraries.
 - By default, the utility `nc-config`, installed with netcdf 4.1.2 or higher,
 will be run used to determine where all the dependencies live.
 - If `nc-config` is not in your default `$PATH`, rename the
 file `setup.cfg.template` to `setup.cfg`, then edit
 in a text editor (follow the instructions in the comments).
 In addition to specifying the path to `nc-config`,
 you can manually set the paths to all the libraries and their include files
 (in case `nc-config` does not do the right thing).
 - run `python setup.py build`, then `python setup.py install` (as root if
 necessary).
 - [`pip install`](https://pip.pypa.io/en/latest/reference/pip_install.html) can
   also be used, with library paths set with environment variables. To make
   this work, the `USE_SETUPCFG` environment variable must be used to tell 
   setup.py not to use `setup.cfg`.
   For example, `USE_SETUPCFG=0 HDF5_INCDIR=/usr/include/hdf5/serial
   HDF5_LIBDIR=/usr/lib/x86_64-linux-gnu/hdf5/serial pip install` has been
   shown to work on an Ubuntu/Debian linux system. Similarly, environment variables
   (all capitalized) can be used to set the include and library paths for
   `hdf5`, `netCDF4`, `hdf4`, `szip`, `jpeg`, `curl` and `zlib`. If the
   libraries are installed in standard places (e.g. `/usr` or `/usr/local`), 
   the environment variables do not need to be set.
 - run the tests in the 'test' directory by running `python run_all.py`.

Tutorial
========

1. [Creating/Opening/Closing a netCDF file.](#section1)
2. [Groups in a netCDF file.](#section2)
3. [Dimensions in a netCDF file.](#section3)
4. [Variables in a netCDF file.](#section4)
5. [Attributes in a netCDF file.](#section5)
6. [Writing data to and retrieving data from a netCDF variable.](#section6)
7. [Dealing with time coordinates.](#section7)
8. [Reading data from a multi-file netCDF dataset.](#section8)
9. [Efficient compression of netCDF variables.](#section9)
10. [Beyond homogeneous arrays of a fixed type - compound data types.](#section10)
11. [Variable-length (vlen) data types.](#section11)
12. [Enum data type.](#section12)


## <div id='section1'>1) Creating/Opening/Closing a netCDF file.

To create a netCDF file from python, you simply call the `netCDF4.Dataset`
constructor. This is also the method used to open an existing netCDF
file.  If the file is open for write access (`mode='w', 'r+'` or `'a'`), you may
write any type of data including new dimensions, groups, variables and
attributes.  netCDF files come in five flavors (`NETCDF3_CLASSIC,
NETCDF3_64BIT_OFFSET, NETCDF3_64BIT_DATA, NETCDF4_CLASSIC`, and `NETCDF4`). 
`NETCDF3_CLASSIC` was the original netcdf binary format, and was limited 
to file sizes less than 2 Gb. `NETCDF3_64BIT_OFFSET` was introduced
in version 3.6.0 of the library, and extended the original binary format
to allow for file sizes greater than 2 Gb. 
`NETCDF3_64BIT_DATA` is a new format that requires version 4.4.0 of
the C library - it extends the `NETCDF3_64BIT_OFFSET` binary format to
allow for unsigned/64 bit integer data types and 64-bit dimension sizes.
`NETCDF3_64BIT` is an alias for `NETCDF3_64BIT_OFFSET`.
`NETCDF4_CLASSIC` files use the version 4 disk format (HDF5), but omits features
not found in the version 3 API. They can be read by netCDF 3 clients
only if they have been relinked against the netCDF 4 library. They can
also be read by HDF5 clients. `NETCDF4` files use the version 4 disk
format (HDF5) and use the new features of the version 4 API.  The
`netCDF4` module can read and write files in any of these formats. When
creating a new file, the format may be specified using the `format`
keyword in the `Dataset` constructor.  The default format is
`NETCDF4`. To see how a given file is formatted, you can examine the
`data_model` attribute.  Closing the netCDF file is
accomplished via the `netCDF4.Dataset.close` method of the `netCDF4.Dataset`
instance.

Here's an example:

    :::python
    >>> from netCDF4 import Dataset
    >>> rootgrp = Dataset("test.nc", "w", format="NETCDF4")
    >>> print rootgrp.data_model
    NETCDF4
    >>> rootgrp.close()

Remote [OPeNDAP](http://opendap.org)-hosted datasets can be accessed for
reading over http if a URL is provided to the `netCDF4.Dataset` constructor instead of a
filename.  However, this requires that the netCDF library be built with
OPenDAP support, via the `--enable-dap` configure option (added in
version 4.0.1).


## <div id='section2'>2) Groups in a netCDF file.

netCDF version 4 added support for organizing data in hierarchical
groups, which are analogous to directories in a filesystem. Groups serve
as containers for variables, dimensions and attributes, as well as other
groups.  A `netCDF4.Dataset` defines creates a special group, called
the 'root group', which is similar to the root directory in a unix
filesystem.  To create `netCDF4.Group` instances, use the
`netCDF4.Dataset.createGroup` method of a `netCDF4.Dataset` or `netCDF4.Group`
instance. `netCDF4.Dataset.createGroup` takes a single argument, a
python string containing the name of the new group. The new `netCDF4.Group`
instances contained within the root group can be accessed by name using
the `groups` dictionary attribute of the `netCDF4.Dataset` instance.  Only
`NETCDF4` formatted files support Groups, if you try to create a Group
in a netCDF 3 file you will get an error message.

    :::python
    >>> rootgrp = Dataset("test.nc", "a")
    >>> fcstgrp = rootgrp.createGroup("forecasts")
    >>> analgrp = rootgrp.createGroup("analyses")
    >>> print rootgrp.groups
    OrderedDict([("forecasts", 
                  <netCDF4._netCDF4.Group object at 0x1b4b7b0>),
                 ("analyses", 
                  <netCDF4._netCDF4.Group object at 0x1b4b970>)])

Groups can exist within groups in a `netCDF4.Dataset`, just as directories
exist within directories in a unix filesystem. Each `netCDF4.Group` instance
has a `groups` attribute dictionary containing all of the group
instances contained within that group. Each `netCDF4.Group` instance also has a
`path` attribute that contains a simulated unix directory path to
that group.  To simplify the creation of nested groups, you can
use a unix-like path as an argument to `netCDF4.Dataset.createGroup`.

    :::python
    >>> fcstgrp1 = rootgrp.createGroup("/forecasts/model1")
    >>> fcstgrp2 = rootgrp.createGroup("/forecasts/model2")

If any of the intermediate elements of the path do not exist, they are created,
just as with the unix command `'mkdir -p'`. If you try to create a group
that already exists, no error will be raised, and the existing group will be 
returned.

Here's an example that shows how to navigate all the groups in a
`netCDF4.Dataset`. The function `walktree` is a Python generator that is used
to walk the directory tree. Note that printing the `netCDF4.Dataset` or `netCDF4.Group`
object yields summary information about it's contents.

    :::python
    >>> def walktree(top):
    >>>     values = top.groups.values()
    >>>     yield values
    >>>     for value in top.groups.values():
    >>>         for children in walktree(value):
    >>>             yield children
    >>> print rootgrp
    >>> for children in walktree(rootgrp):
    >>>      for child in children:
    >>>          print child
    <type "netCDF4._netCDF4.Dataset">
    root group (NETCDF4 file format):
        dimensions:
        variables:
        groups: forecasts, analyses
    <type "netCDF4._netCDF4.Group">
    group /forecasts:
        dimensions:
        variables:
        groups: model1, model2
    <type "netCDF4._netCDF4.Group">
    group /analyses:
        dimensions:
        variables:
        groups:
    <type "netCDF4._netCDF4.Group">
    group /forecasts/model1:
        dimensions:
        variables:
        groups:
    <type "netCDF4._netCDF4.Group">
    group /forecasts/model2:
        dimensions:
        variables:
        groups:

## <div id='section3'>3) Dimensions in a netCDF file.

netCDF defines the sizes of all variables in terms of dimensions, so
before any variables can be created the dimensions they use must be
created first. A special case, not often used in practice, is that of a
scalar variable, which has no dimensions. A dimension is created using
the `netCDF4.Dataset.createDimension` method of a `netCDF4.Dataset`
or `netCDF4.Group` instance. A Python string is used to set the name of the
dimension, and an integer value is used to set the size. To create an
unlimited dimension (a dimension that can be appended to), the size
value is set to `None` or 0. In this example, there both the `time` and
`level` dimensions are unlimited.  Having more than one unlimited
dimension is a new netCDF 4 feature, in netCDF 3 files there may be only
one, and it must be the first (leftmost) dimension of the variable.

    :::python
    >>> level = rootgrp.createDimension("level", None)
    >>> time = rootgrp.createDimension("time", None)
    >>> lat = rootgrp.createDimension("lat", 73)
    >>> lon = rootgrp.createDimension("lon", 144)


All of the `netCDF4.Dimension` instances are stored in a python dictionary.

    :::python
    >>> print rootgrp.dimensions
    OrderedDict([("level", <netCDF4._netCDF4.Dimension object at 0x1b48030>),
                 ("time", <netCDF4._netCDF4.Dimension object at 0x1b481c0>),
                 ("lat", <netCDF4._netCDF4.Dimension object at 0x1b480f8>),
                 ("lon", <netCDF4._netCDF4.Dimension object at 0x1b48a08>)])

Calling the python `len` function with a `netCDF4.Dimension` instance returns
the current size of that dimension.
The `netCDF4.Dimension.isunlimited` method of a `netCDF4.Dimension` instance
can be used to determine if the dimensions is unlimited, or appendable.

    :::python
    >>> print len(lon)
    144
    >>> print lon.isunlimited()
    False
    >>> print time.isunlimited()
    True

Printing the `netCDF4.Dimension` object
provides useful summary info, including the name and length of the dimension,
and whether it is unlimited.

    :::python
    >>> for dimobj in rootgrp.dimensions.values():
    >>>    print dimobj
    <type "netCDF4._netCDF4.Dimension"> (unlimited): name = "level", size = 0
    <type "netCDF4._netCDF4.Dimension"> (unlimited): name = "time", size = 0
    <type "netCDF4._netCDF4.Dimension">: name = "lat", size = 73
    <type "netCDF4._netCDF4.Dimension">: name = "lon", size = 144
    <type "netCDF4._netCDF4.Dimension"> (unlimited): name = "time", size = 0

`netCDF4.Dimension` names can be changed using the
`netCDF4.Datatset.renameDimension` method of a `netCDF4.Dataset` or
`netCDF4.Group` instance.

## <div id='section4'>4) Variables in a netCDF file.

netCDF variables behave much like python multidimensional array objects
supplied by the [numpy module](http://numpy.scipy.org). However,
unlike numpy arrays, netCDF4 variables can be appended to along one or
more 'unlimited' dimensions. To create a netCDF variable, use the
`netCDF4.Dataset.createVariable` method of a `netCDF4.Dataset` or
`netCDF4.Group` instance. The `netCDF4.Dataset.createVariable` method
has two mandatory arguments, the variable name (a Python string), and
the variable datatype. The variable's dimensions are given by a tuple
containing the dimension names (defined previously with
`netCDF4.Dataset.createDimension`). To create a scalar
variable, simply leave out the dimensions keyword. The variable
primitive datatypes correspond to the dtype attribute of a numpy array.
You can specify the datatype as a numpy dtype object, or anything that
can be converted to a numpy dtype object.  Valid datatype specifiers
include: `'f4'` (32-bit floating point), `'f8'` (64-bit floating
point), `'i4'` (32-bit signed integer), `'i2'` (16-bit signed
integer), `'i8'` (64-bit singed integer), `'i1'` (8-bit signed
integer), `'u1'` (8-bit unsigned integer), `'u2'` (16-bit unsigned
integer), `'u4'` (32-bit unsigned integer), `'u8'` (64-bit unsigned
integer), or `'S1'` (single-character string).  The old Numeric
single-character typecodes (`'f'`,`'d'`,`'h'`,
`'s'`,`'b'`,`'B'`,`'c'`,`'i'`,`'l'`), corresponding to
(`'f4'`,`'f8'`,`'i2'`,`'i2'`,`'i1'`,`'i1'`,`'S1'`,`'i4'`,`'i4'`),
will also work. The unsigned integer types and the 64-bit integer type
can only be used if the file format is `NETCDF4`.

The dimensions themselves are usually also defined as variables, called
coordinate variables. The `netCDF4.Dataset.createVariable`
method returns an instance of the `netCDF4.Variable` class whose methods can be
used later to access and set variable data and attributes.

    :::python
    >>> times = rootgrp.createVariable("time","f8",("time",))
    >>> levels = rootgrp.createVariable("level","i4",("level",))
    >>> latitudes = rootgrp.createVariable("latitude","f4",("lat",))
    >>> longitudes = rootgrp.createVariable("longitude","f4",("lon",))
    >>> # two dimensions unlimited
    >>> temp = rootgrp.createVariable("temp","f4",("time","level","lat","lon",))

To get summary info on a `netCDF4.Variable` instance in an interactive session, just print it.

    :::python
    >>> print temp
    <type "netCDF4._netCDF4.Variable">
    float32 temp(time, level, lat, lon)
        least_significant_digit: 3
        units: K
    unlimited dimensions: time, level
    current shape = (0, 0, 73, 144)

You can use a path to create a Variable inside a hierarchy of groups.

    :::python
    >>> ftemp = rootgrp.createVariable("/forecasts/model1/temp","f4",("time","level","lat","lon",))

If the intermediate groups do not yet exist, they will be created.

You can also query a `netCDF4.Dataset` or `netCDF4.Group` instance directly to obtain `netCDF4.Group` or 
`netCDF4.Variable` instances using paths.

    :::python
    >>> print rootgrp["/forecasts/model1"] # a Group instance
    <type "netCDF4._netCDF4.Group">
    group /forecasts/model1:
        dimensions(sizes):
        variables(dimensions): float32 temp(time,level,lat,lon)
        groups:
    >>> print rootgrp["/forecasts/model1/temp"] # a Variable instance
    <type "netCDF4._netCDF4.Variable">
    float32 temp(time, level, lat, lon)
    path = /forecasts/model1
    unlimited dimensions: time, level
    current shape = (0, 0, 73, 144)
    filling on, default _FillValue of 9.96920996839e+36 used

All of the variables in the `netCDF4.Dataset` or `netCDF4.Group` are stored in a
Python dictionary, in the same way as the dimensions:

    :::python
    >>> print rootgrp.variables
    OrderedDict([("time", <netCDF4.Variable object at 0x1b4ba70>),
                 ("level", <netCDF4.Variable object at 0x1b4bab0>),
                 ("latitude", <netCDF4.Variable object at 0x1b4baf0>),
                 ("longitude", <netCDF4.Variable object at 0x1b4bb30>),
                 ("temp", <netCDF4.Variable object at 0x1b4bb70>)])

`netCDF4.Variable` names can be changed using the
`netCDF4.Dataset.renameVariable` method of a `netCDF4.Dataset`
instance.


## <div id='section5'>5) Attributes in a netCDF file.

There are two types of attributes in a netCDF file, global and variable.
Global attributes provide information about a group, or the entire
dataset, as a whole. `netCDF4.Variable` attributes provide information about
one of the variables in a group. Global attributes are set by assigning
values to `netCDF4.Dataset` or `netCDF4.Group` instance variables. `netCDF4.Variable`
attributes are set by assigning values to `netCDF4.Variable` instances
variables. Attributes can be strings, numbers or sequences. Returning to
our example,

    :::python
    >>> import time
    >>> rootgrp.description = "bogus example script"
    >>> rootgrp.history = "Created " + time.ctime(time.time())
    >>> rootgrp.source = "netCDF4 python module tutorial"
    >>> latitudes.units = "degrees north"
    >>> longitudes.units = "degrees east"
    >>> levels.units = "hPa"
    >>> temp.units = "K"
    >>> times.units = "hours since 0001-01-01 00:00:00.0"
    >>> times.calendar = "gregorian"

The `netCDF4.Dataset.ncattrs` method of a `netCDF4.Dataset`, `netCDF4.Group` or
`netCDF4.Variable` instance can be used to retrieve the names of all the netCDF
attributes. This method is provided as a convenience, since using the
built-in `dir` Python function will return a bunch of private methods
and attributes that cannot (or should not) be modified by the user.

    :::python
    >>> for name in rootgrp.ncattrs():
    >>>     print "Global attr", name, "=", getattr(rootgrp,name)
    Global attr description = bogus example script
    Global attr history = Created Mon Nov  7 10.30:56 2005
    Global attr source = netCDF4 python module tutorial

The `__dict__` attribute of a `netCDF4.Dataset`, `netCDF4.Group` or `netCDF4.Variable`
instance provides all the netCDF attribute name/value pairs in a python
dictionary:

    :::python
    >>> print rootgrp.__dict__
    OrderedDict([(u"description", u"bogus example script"),
                 (u"history", u"Created Thu Mar  3 19:30:33 2011"),
                 (u"source", u"netCDF4 python module tutorial")])

Attributes can be deleted from a netCDF `netCDF4.Dataset`, `netCDF4.Group` or
`netCDF4.Variable` using the python `del` statement (i.e. `del grp.foo`
removes the attribute `foo` the the group `grp`).

## <div id='section6'>6) Writing data to and retrieving data from a netCDF variable.

Now that you have a netCDF `netCDF4.Variable` instance, how do you put data
into it? You can just treat it like an array and assign data to a slice.

    :::python
    >>> import numpy
    >>> lats =  numpy.arange(-90,91,2.5)
    >>> lons =  numpy.arange(-180,180,2.5)
    >>> latitudes[:] = lats
    >>> longitudes[:] = lons
    >>> print "latitudes =\\n",latitudes[:]
    latitudes =
    [-90.  -87.5 -85.  -82.5 -80.  -77.5 -75.  -72.5 -70.  -67.5 -65.  -62.5
     -60.  -57.5 -55.  -52.5 -50.  -47.5 -45.  -42.5 -40.  -37.5 -35.  -32.5
     -30.  -27.5 -25.  -22.5 -20.  -17.5 -15.  -12.5 -10.   -7.5  -5.   -2.5
       0.    2.5   5.    7.5  10.   12.5  15.   17.5  20.   22.5  25.   27.5
      30.   32.5  35.   37.5  40.   42.5  45.   47.5  50.   52.5  55.   57.5
      60.   62.5  65.   67.5  70.   72.5  75.   77.5  80.   82.5  85.   87.5
      90. ]

Unlike NumPy's array objects, netCDF `netCDF4.Variable`
objects with unlimited dimensions will grow along those dimensions if you
assign data outside the currently defined range of indices.

    :::python
    >>> # append along two unlimited dimensions by assigning to slice.
    >>> nlats = len(rootgrp.dimensions["lat"])
    >>> nlons = len(rootgrp.dimensions["lon"])
    >>> print "temp shape before adding data = ",temp.shape
    temp shape before adding data =  (0, 0, 73, 144)
    >>>
    >>> from numpy.random import uniform
    >>> temp[0:5,0:10,:,:] = uniform(size=(5,10,nlats,nlons))
    >>> print "temp shape after adding data = ",temp.shape
    temp shape after adding data =  (6, 10, 73, 144)
    >>>
    >>> # levels have grown, but no values yet assigned.
    >>> print "levels shape after adding pressure data = ",levels.shape
    levels shape after adding pressure data =  (10,)

Note that the size of the levels variable grows when data is appended
along the `level` dimension of the variable `temp`, even though no
data has yet been assigned to levels.

    :::python
    >>> # now, assign data to levels dimension variable.
    >>> levels[:] =  [1000.,850.,700.,500.,300.,250.,200.,150.,100.,50.]

However, that there are some differences between NumPy and netCDF
variable slicing rules. Slices behave as usual, being specified as a
`start:stop:step` triplet. Using a scalar integer index `i` takes the ith
element and reduces the rank of the output array by one. Boolean array and
integer sequence indexing behaves differently for netCDF variables
than for numpy arrays.  Only 1-d boolean arrays and integer sequences are
allowed, and these indices work independently along each dimension (similar
to the way vector subscripts work in fortran).  This means that

    :::python
    >>> temp[0, 0, [0,1,2,3], [0,1,2,3]]

returns an array of shape (4,4) when slicing a netCDF variable, but for a
numpy array it returns an array of shape (4,).
Similarly, a netCDF variable of shape `(2,3,4,5)` indexed
with `[0, array([True, False, True]), array([False, True, True, True]), :]`
would return a `(2, 3, 5)` array. In NumPy, this would raise an error since
it would be equivalent to `[0, [0,1], [1,2,3], :]`. When slicing with integer
sequences, the indices ***need not be sorted*** and ***may contain
duplicates*** (both of these are new features in version 1.2.1).
While this behaviour may cause some confusion for those used to NumPy's 'fancy indexing' rules,
it provides a very powerful way to extract data from multidimensional netCDF
variables by using logical operations on the dimension arrays to create slices.

For example,

    :::python
    >>> tempdat = temp[::2, [1,3,6], lats>0, lons>0]

will extract time indices 0,2 and 4, pressure levels
850, 500 and 200 hPa, all Northern Hemisphere latitudes and Eastern
Hemisphere longitudes, resulting in a numpy array of shape  (3, 3, 36, 71).

    :::python
    >>> print "shape of fancy temp slice = ",tempdat.shape
    shape of fancy temp slice =  (3, 3, 36, 71)

***Special note for scalar variables***: To extract data from a scalar variable
`v` with no associated dimensions, use `np.asarray(v)` or `v[...]`. The result
will be a numpy scalar array.

## <div id='section7'>7) Dealing with time coordinates.

Time coordinate values pose a special challenge to netCDF users.  Most
metadata standards (such as CF) specify that time should be
measure relative to a fixed date using a certain calendar, with units
specified like `hours since YY:MM:DD hh-mm-ss`.  These units can be
awkward to deal with, without a utility to convert the values to and
from calendar dates.  The function called `netCDF4.num2date` and `netCDF4.date2num` are
provided with this package to do just that.  Here's an example of how they
can be used:

    :::python
    >>> # fill in times.
    >>> from datetime import datetime, timedelta
    >>> from netCDF4 import num2date, date2num
    >>> dates = [datetime(2001,3,1)+n*timedelta(hours=12) for n in range(temp.shape[0])]
    >>> times[:] = date2num(dates,units=times.units,calendar=times.calendar)
    >>> print "time values (in units %s): " % times.units+"\\n",times[:]
    time values (in units hours since January 1, 0001):
    [ 17533056.  17533068.  17533080.  17533092.  17533104.]
    >>> dates = num2date(times[:],units=times.units,calendar=times.calendar)
    >>> print "dates corresponding to time values:\\n",dates
    dates corresponding to time values:
    [2001-03-01 00:00:00 2001-03-01 12:00:00 2001-03-02 00:00:00
     2001-03-02 12:00:00 2001-03-03 00:00:00]

`netCDF4.num2date` converts numeric values of time in the specified `units`
and `calendar` to datetime objects, and `netCDF4.date2num` does the reverse.
All the calendars currently defined in the
[CF metadata convention](http://cfconventions.org) are supported.
A function called `netCDF4.date2index` is also provided which returns the indices
of a netCDF time variable corresponding to a sequence of datetime instances.


## <div id='section8'>8) Reading data from a multi-file netCDF dataset.

If you want to read data from a variable that spans multiple netCDF files,
you can use the `netCDF4.MFDataset` class to read the data as if it were
contained in a single file. Instead of using a single filename to create
a `netCDF4.Dataset` instance, create a `netCDF4.MFDataset` instance with either a list
of filenames, or a string with a wildcard (which is then converted to
a sorted list of files using the python glob module).
Variables in the list of files that share the same unlimited
dimension are aggregated together, and can be sliced across multiple
files.  To illustrate this, let's first create a bunch of netCDF files with
the same variable (with the same unlimited dimension).  The files
must in be in `NETCDF3_64BIT_OFFSET`, `NETCDF3_64BIT_DATA`, `NETCDF3_CLASSIC` or
`NETCDF4_CLASSIC` format (`NETCDF4` formatted multi-file
datasets are not supported).

    :::python
    >>> for nf in range(10):
    >>>     f = Dataset("mftest%s.nc" % nf,"w")
    >>>     f.createDimension("x",None)
    >>>     x = f.createVariable("x","i",("x",))
    >>>     x[0:10] = numpy.arange(nf*10,10*(nf+1))
    >>>     f.close()

Now read all the files back in at once with `netCDF4.MFDataset`

    :::python
    >>> from netCDF4 import MFDataset
    >>> f = MFDataset("mftest*nc")
    >>> print f.variables["x"][:]
    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
     25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
     50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74
     75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99]

Note that `netCDF4.MFDataset` can only be used to read, not write, multi-file
datasets.

## <div id='section9'>9) Efficient compression of netCDF variables.

Data stored in netCDF 4 `netCDF4.Variable` objects can be compressed and
decompressed on the fly. The parameters for the compression are
determined by the `zlib`, `complevel` and `shuffle` keyword arguments
to the `netCDF4.Dataset.createVariable` method. To turn on
compression, set `zlib=True`.  The `complevel` keyword regulates the
speed and efficiency of the compression (1 being fastest, but lowest
compression ratio, 9 being slowest but best compression ratio). The
default value of `complevel` is 4. Setting `shuffle=False` will turn
off the HDF5 shuffle filter, which de-interlaces a block of data before
compression by reordering the bytes.  The shuffle filter can
significantly improve compression ratios, and is on by default.  Setting
`fletcher32` keyword argument to
`netCDF4.Dataset.createVariable` to `True` (it's `False` by
default) enables the Fletcher32 checksum algorithm for error detection.
It's also possible to set the HDF5 chunking parameters and endian-ness
of the binary data stored in the HDF5 file with the `chunksizes`
and `endian` keyword arguments to
`netCDF4.Dataset.createVariable`.  These keyword arguments only
are relevant for `NETCDF4` and `NETCDF4_CLASSIC` files (where the
underlying file format is HDF5) and are silently ignored if the file
format is `NETCDF3_CLASSIC`, `NETCDF3_64BIT_OFFSET` or `NETCDF3_64BIT_DATA`.

If your data only has a certain number of digits of precision (say for
example, it is temperature data that was measured with a precision of
0.1 degrees), you can dramatically improve zlib compression by
quantizing (or truncating) the data using the `least_significant_digit`
keyword argument to `netCDF4.Dataset.createVariable`. The least
significant digit is the power of ten of the smallest decimal place in
the data that is a reliable value. For example if the data has a
precision of 0.1, then setting `least_significant_digit=1` will cause
data the data to be quantized using `numpy.around(scale*data)/scale`, where
scale = 2**bits, and bits is determined so that a precision of 0.1 is
retained (in this case bits=4).  Effectively, this makes the compression
'lossy' instead of 'lossless', that is some precision in the data is
sacrificed for the sake of disk space.

In our example, try replacing the line

    :::python
    >>> temp = rootgrp.createVariable("temp","f4",("time","level","lat","lon",))

with

    :::python
    >>> temp = dataset.createVariable("temp","f4",("time","level","lat","lon",),zlib=True)

and then

    :::python
    >>> temp = dataset.createVariable("temp","f4",("time","level","lat","lon",),zlib=True,least_significant_digit=3)

and see how much smaller the resulting files are.

## <div id='section10'>10) Beyond homogeneous arrays of a fixed type - compound data types.

Compound data types map directly to numpy structured (a.k.a 'record'
arrays).  Structured arrays are akin to C structs, or derived types
in Fortran. They allow for the construction of table-like structures
composed of combinations of other data types, including other
compound types. Compound types might be useful for representing multiple
parameter values at each point on a grid, or at each time and space
location for scattered (point) data. You can then access all the
information for a point by reading one variable, instead of reading
different parameters from different variables.  Compound data types
are created from the corresponding numpy data type using the
`netCDF4.Dataset.createCompoundType` method of a `netCDF4.Dataset` or `netCDF4.Group` instance.
Since there is no native complex data type in netcdf, compound types are handy
for storing numpy complex arrays.  Here's an example:

    :::python
    >>> f = Dataset("complex.nc","w")
    >>> size = 3 # length of 1-d complex array
    >>> # create sample complex data.
    >>> datac = numpy.exp(1j*(1.+numpy.linspace(0, numpy.pi, size)))
    >>> # create complex128 compound data type.
    >>> complex128 = numpy.dtype([("real",numpy.float64),("imag",numpy.float64)])
    >>> complex128_t = f.createCompoundType(complex128,"complex128")
    >>> # create a variable with this data type, write some data to it.
    >>> f.createDimension("x_dim",None)
    >>> v = f.createVariable("cmplx_var",complex128_t,"x_dim")
    >>> data = numpy.empty(size,complex128) # numpy structured array
    >>> data["real"] = datac.real; data["imag"] = datac.imag
    >>> v[:] = data # write numpy structured array to netcdf compound var
    >>> # close and reopen the file, check the contents.
    >>> f.close(); f = Dataset("complex.nc")
    >>> v = f.variables["cmplx_var"]
    >>> datain = v[:] # read in all the data into a numpy structured array
    >>> # create an empty numpy complex array
    >>> datac2 = numpy.empty(datain.shape,numpy.complex128)
    >>> # .. fill it with contents of structured array.
    >>> datac2.real = datain["real"]; datac2.imag = datain["imag"]
    >>> print datac.dtype,datac # original data
    complex128 [ 0.54030231+0.84147098j -0.84147098+0.54030231j  -0.54030231-0.84147098j]
    >>>
    >>> print datac2.dtype,datac2 # data from file
    complex128 [ 0.54030231+0.84147098j -0.84147098+0.54030231j  -0.54030231-0.84147098j]

Compound types can be nested, but you must create the 'inner'
ones first. All of the compound types defined for a `netCDF4.Dataset` or `netCDF4.Group` are stored in a
Python dictionary, just like variables and dimensions. As always, printing
objects gives useful summary information in an interactive session:

    :::python
    >>> print f
    <type "netCDF4._netCDF4.Dataset">
    root group (NETCDF4 file format):
        dimensions: x_dim
        variables: cmplx_var
        groups:
    <type "netCDF4._netCDF4.Variable">
    >>> print f.variables["cmplx_var"]
    compound cmplx_var(x_dim)
    compound data type: [("real", "<f8"), ("imag", "<f8")]
    unlimited dimensions: x_dim
    current shape = (3,)
    >>> print f.cmptypes
    OrderedDict([("complex128", <netCDF4.CompoundType object at 0x1029eb7e8>)])
    >>> print f.cmptypes["complex128"]
    <type "netCDF4._netCDF4.CompoundType">: name = "complex128", numpy dtype = [(u"real","<f8"), (u"imag", "<f8")]

## <div id='section11'>11) Variable-length (vlen) data types.

NetCDF 4 has support for variable-length or "ragged" arrays.  These are arrays
of variable length sequences having the same type. To create a variable-length
data type, use the `netCDF4.Dataset.createVLType` method
method of a `netCDF4.Dataset` or `netCDF4.Group` instance.

    :::python
    >>> f = Dataset("tst_vlen.nc","w")
    >>> vlen_t = f.createVLType(numpy.int32, "phony_vlen")

The numpy datatype of the variable-length sequences and the name of the
new datatype must be specified. Any of the primitive datatypes can be
used (signed and unsigned integers, 32 and 64 bit floats, and characters),
but compound data types cannot.
A new variable can then be created using this datatype.

    :::python
    >>> x = f.createDimension("x",3)
    >>> y = f.createDimension("y",4)
    >>> vlvar = f.createVariable("phony_vlen_var", vlen_t, ("y","x"))

Since there is no native vlen datatype in numpy, vlen arrays are represented
in python as object arrays (arrays of dtype `object`). These are arrays whose
elements are Python object pointers, and can contain any type of python object.
For this application, they must contain 1-D numpy arrays all of the same type
but of varying length.
In this case, they contain 1-D numpy `int32` arrays of random length betwee
1 and 10.

    :::python
    >>> import random
    >>> data = numpy.empty(len(y)*len(x),object)
    >>> for n in range(len(y)*len(x)):
    >>>    data[n] = numpy.arange(random.randint(1,10),dtype="int32")+1
    >>> data = numpy.reshape(data,(len(y),len(x)))
    >>> vlvar[:] = data
    >>> print "vlen variable =\\n",vlvar[:]
    vlen variable =
    [[[ 1  2  3  4  5  6  7  8  9 10] [1 2 3 4 5] [1 2 3 4 5 6 7 8]]
     [[1 2 3 4 5 6 7] [1 2 3 4 5 6] [1 2 3 4 5]]
     [[1 2 3 4 5] [1 2 3 4] [1]]
     [[ 1  2  3  4  5  6  7  8  9 10] [ 1  2  3  4  5  6  7  8  9 10]
      [1 2 3 4 5 6 7 8]]]
    >>> print f
    <type "netCDF4._netCDF4.Dataset">
    root group (NETCDF4 file format):
        dimensions: x, y
        variables: phony_vlen_var
        groups:
    >>> print f.variables["phony_vlen_var"]
    <type "netCDF4._netCDF4.Variable">
    vlen phony_vlen_var(y, x)
    vlen data type: int32
    unlimited dimensions:
    current shape = (4, 3)
    >>> print f.VLtypes["phony_vlen"]
    <type "netCDF4._netCDF4.VLType">: name = "phony_vlen", numpy dtype = int32

Numpy object arrays containing python strings can also be written as vlen
variables,  For vlen strings, you don't need to create a vlen data type.
Instead, simply use the python `str` builtin (or a numpy string datatype
with fixed length greater than 1) when calling the
`netCDF4.Dataset.createVariable` method.

    :::python
    >>> z = f.createDimension("z",10)
    >>> strvar = rootgrp.createVariable("strvar", str, "z")

In this example, an object array is filled with random python strings with
random lengths between 2 and 12 characters, and the data in the object
array is assigned to the vlen string variable.

    :::python
    >>> chars = "1234567890aabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    >>> data = numpy.empty(10,"O")
    >>> for n in range(10):
    >>>     stringlen = random.randint(2,12)
    >>>     data[n] = "".join([random.choice(chars) for i in range(stringlen)])
    >>> strvar[:] = data
    >>> print "variable-length string variable:\\n",strvar[:]
    variable-length string variable:
    [aDy29jPt 5DS9X8 jd7aplD b8t4RM jHh8hq KtaPWF9cQj Q1hHN5WoXSiT MMxsVeq tdLUzvVTzj]
    >>> print f
    <type "netCDF4._netCDF4.Dataset">
    root group (NETCDF4 file format):
        dimensions: x, y, z
        variables: phony_vlen_var, strvar
        groups:
    >>> print f.variables["strvar"]
    <type "netCDF4._netCDF4.Variable">
    vlen strvar(z)
    vlen data type: <type "str">
    unlimited dimensions:
    current size = (10,)

It is also possible to set contents of vlen string variables with numpy arrays
of any string or unicode data type. Note, however, that accessing the contents
of such variables will always return numpy arrays with dtype `object`.

## <div id='section12'>12) Enum data type.

netCDF4 has an enumerated data type, which is an integer datatype that is
restricted to certain named values. Since Enums don't map directly to
a numpy data type, they are read and written as integer arrays.

Here's an example of using an Enum type to hold cloud type data. 
The base integer data type and a python dictionary describing the allowed
values and their names are used to define an Enum data type using
`netCDF4.Dataset.createEnumType`.

    :::python
    >>> nc = Dataset('clouds.nc','w')
    >>> # python dict with allowed values and their names.
    >>> enum_dict = {u'Altocumulus': 7, u'Missing': 255, 
    >>> u'Stratus': 2, u'Clear': 0,
    >>> u'Nimbostratus': 6, u'Cumulus': 4, u'Altostratus': 5,
    >>> u'Cumulonimbus': 1, u'Stratocumulus': 3}
    >>> # create the Enum type called 'cloud_t'.
    >>> cloud_type = nc.createEnumType(numpy.uint8,'cloud_t',enum_dict)
    >>> print cloud_type
    <type 'netCDF4._netCDF4.EnumType'>: name = 'cloud_t',
    numpy dtype = uint8, fields/values ={u'Cumulus': 4,
    u'Altocumulus': 7, u'Missing': 255,
    u'Stratus': 2, u'Clear': 0,
    u'Cumulonimbus': 1, u'Stratocumulus': 3,
    u'Nimbostratus': 6, u'Altostratus': 5}

A new variable can be created in the usual way using this data type.
Integer data is written to the variable that represents the named
cloud types in enum_dict. A `ValueError` will be raised if an attempt
is made to write an integer value not associated with one of the
specified names.

    :::python
    >>> time = nc.createDimension('time',None)
    >>> # create a 1d variable of type 'cloud_type'.
    >>> # The fill_value is set to the 'Missing' named value.
    >>> cloud_var =
    >>> nc.createVariable('primary_cloud',cloud_type,'time',
    >>> fill_value=enum_dict['Missing'])
    >>> # write some data to the variable.
    >>> cloud_var[:] = [enum_dict['Clear'],enum_dict['Stratus'],
    >>> enum_dict['Cumulus'],enum_dict['Missing'],
    >>> enum_dict['Cumulonimbus']]
    >>> nc.close()
    >>> # reopen the file, read the data.
    >>> nc = Dataset('clouds.nc')
    >>> cloud_var = nc.variables['primary_cloud']
    >>> print cloud_var
    <type 'netCDF4._netCDF4.Variable'>
    enum primary_cloud(time)
        _FillValue: 255
    enum data type: uint8
    unlimited dimensions: time
    current shape = (5,)
    >>> print cloud_var.datatype.enum_dict
    {u'Altocumulus': 7, u'Missing': 255, u'Stratus': 2,
    u'Clear': 0, u'Nimbostratus': 6, u'Cumulus': 4,
    u'Altostratus': 5, u'Cumulonimbus': 1,
    u'Stratocumulus': 3}
    >>> print cloud_var[:]
    [0 2 4 -- 1]
    >>> nc.close()

All of the code in this tutorial is available in `examples/tutorial.py`,
Unit tests are in the `test` directory.

**contact**: Jeffrey Whitaker <jeffrey.s.whitaker@noaa.gov>

**copyright**: 2008 by Jeffrey Whitaker.

**license**: Permission to use, copy, modify, and distribute this software and
its documentation for any purpose and without fee is hereby granted,
provided that the above copyright notice appear in all copies and that
both the copyright notice and this permission notice appear in
supporting documentation.
THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO
EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT OR
CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF
USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
- - -
"""

# Make changes to this file, not the c-wrappers that Cython generates.

# pure python utilities
from .utils import (_StartCountStride, _quantize, _find_dim, _walk_grps,
                    _out_array_shape, _sortbylist, _tostr)
# try to use built-in ordered dict in python >= 2.7
try:
    from collections import OrderedDict
except ImportError: # or else use drop-in substitute
    try:
        from ordereddict import OrderedDict
    except ImportError:
        raise ImportError('please install ordereddict (https://pypi.python.org/pypi/ordereddict)')
try:
    from itertools import izip as zip
except ImportError:
    # python3: zip is already python2's itertools.izip
    pass

__version__ = "1.2.4"

# Initialize numpy
import posixpath
import netcdftime
import numpy
import weakref
import sys
import warnings
from glob import glob
from numpy import ma
from libc.string cimport memcpy
from libc.stdlib cimport malloc, free
import_array()
include "netCDF4.pxi"
include "constants.pyx"

# check for required version of netcdf-4 and hdf5.

def _gethdf5libversion():
    majorvers = H5_VERS_MAJOR
    minorvers = H5_VERS_MINOR
    releasevers = H5_VERS_RELEASE
    patchstring = H5_VERS_SUBRELEASE.decode('ascii')
    if not patchstring:
        return '%d.%d.%d' % (majorvers,minorvers,releasevers)
    else:
        return '%d.%d.%d-%s' % (majorvers,minorvers,releasevers,patchstring)

def getlibversion():
    """
**`getlibversion()`**

returns a string describing the version of the netcdf library
used to build the module, and when it was built.
    """
    return (<char *>nc_inq_libvers()).decode('ascii')

__netcdf4libversion__ = getlibversion().split()[0]
__hdf5libversion__ = _gethdf5libversion()
__has_rename_grp__ = HAS_RENAME_GRP
__has_nc_inq_path__ = HAS_NC_INQ_PATH
__has_nc_inq_format_extended__ = HAS_NC_INQ_FORMAT_EXTENDED
__has_cdf5__ = HAS_CDF5_FORMAT
_needsworkaround_issue485 = __netcdf4libversion__ < "4.4.0" or \
               (__netcdf4libversion__.startswith("4.4.0") and \
                "-development" in __netcdf4libversion__)

# numpy data type <--> netCDF 4 data type mapping.
_nptonctype  = {'S1' : NC_CHAR,
                'i1' : NC_BYTE,
                'u1' : NC_UBYTE,
                'i2' : NC_SHORT,
                'u2' : NC_USHORT,
                'i4' : NC_INT,
                'u4' : NC_UINT,
                'i8' : NC_INT64,
                'u8' : NC_UINT64,
                'f4' : NC_FLOAT,
                'f8' : NC_DOUBLE}

# just integer types.
_intnptonctype  = {'i1' : NC_BYTE,
                   'u1' : NC_UBYTE,
                   'i2' : NC_SHORT,
                   'u2' : NC_USHORT,
                   'i4' : NC_INT,
                   'u4' : NC_UINT,
                   'i8' : NC_INT64,
                   'u8' : NC_UINT64}

# create dictionary mapping string identifiers to netcdf format codes
_format_dict  = {'NETCDF3_CLASSIC' : NC_FORMAT_CLASSIC,
                 'NETCDF4_CLASSIC' : NC_FORMAT_NETCDF4_CLASSIC,
                 'NETCDF4'         : NC_FORMAT_NETCDF4}
IF HAS_CDF5_FORMAT:
    # NETCDF3_64BIT deprecated, saved for compatibility.
    # use NETCDF3_64BIT_OFFSET instead.
    _format_dict['NETCDF3_64BIT_OFFSET'] = NC_FORMAT_64BIT_OFFSET
    _format_dict['NETCDF3_64BIT_DATA'] = NC_FORMAT_64BIT_DATA
ELSE:
    _format_dict['NETCDF3_64BIT'] = NC_FORMAT_64BIT
# invert dictionary mapping
_reverse_format_dict = dict((v, k) for k, v in _format_dict.iteritems())
# add duplicate entry (NETCDF3_64BIT == NETCDF3_64BIT_OFFSET)
IF HAS_CDF5_FORMAT:
    _format_dict['NETCDF3_64BIT'] = NC_FORMAT_64BIT_OFFSET
ELSE:
    _format_dict['NETCDF3_64BIT_OFFSET'] = NC_FORMAT_64BIT

# default fill_value to numpy datatype mapping.
default_fillvals = {#'S1':NC_FILL_CHAR,
                     'S1':'\0',
                     'i1':NC_FILL_BYTE,
                     'u1':NC_FILL_UBYTE,
                     'i2':NC_FILL_SHORT,
                     'u2':NC_FILL_USHORT,
                     'i4':NC_FILL_INT,
                     'u4':NC_FILL_UINT,
                     'i8':NC_FILL_INT64,
                     'u8':NC_FILL_UINT64,
                     'f4':NC_FILL_FLOAT,
                     'f8':NC_FILL_DOUBLE}

# logical for native endian type.
is_native_little = numpy.dtype('<f4').byteorder == '='
is_native_big = numpy.dtype('>f4').byteorder == '='

# hard code this here, instead of importing from netcdf.h
# so it will compile with versions <= 4.2.
NC_DISKLESS = 0x0008
# encoding used to convert strings to bytes when writing text data
# to the netcdf file, and for converting bytes to strings when reading
# from the netcdf file.
default_encoding = 'utf-8'
# unicode decode/encode error handling.  Replace bad chars with "?"
# can be set to 'strict' or 'ignore'.
unicode_error = 'replace'
python3 = sys.version_info[0] > 2

_nctonptype = {}
for _key,_value in _nptonctype.items():
    _nctonptype[_value] = _key
_supportedtypes = _nptonctype.keys()
# make sure NC_CHAR points to S1
_nctonptype[NC_CHAR]='S1'

# internal C functions.

cdef _get_att_names(int grpid, int varid):
    # Private function to get all the attribute names in a group
    cdef int ierr, numatts, n
    cdef char namstring[NC_MAX_NAME+1]
    if varid == NC_GLOBAL:
        with nogil:
            ierr = nc_inq_natts(grpid, &numatts)
    else:
        with nogil:
            ierr = nc_inq_varnatts(grpid, varid, &numatts)
    if ierr != NC_NOERR:
        raise AttributeError((<char *>nc_strerror(ierr)).decode('ascii'))
    attslist = []
    for n from 0 <= n < numatts:
        with nogil:
            ierr = nc_inq_attname(grpid, varid, n, namstring)
        if ierr != NC_NOERR:
            raise AttributeError((<char *>nc_strerror(ierr)).decode('ascii'))
        attslist.append(namstring.decode(default_encoding,unicode_error))
    return attslist

cdef _get_att(grp, int varid, name):
    # Private function to get an attribute value given its name
    cdef int ierr, n, _grpid
    cdef size_t att_len
    cdef char *attname
    cdef char *stratt
    cdef nc_type att_type
    cdef ndarray value_arr
    bytestr = _strencode(name)
    attname = bytestr
    _grpid = grp._grpid
    with nogil:
        ierr = nc_inq_att(_grpid, varid, attname, &att_type, &att_len)
    if ierr != NC_NOERR:
        raise AttributeError((<char *>nc_strerror(ierr)).decode('ascii'))
    # attribute is a character or string ...
    if att_type == NC_CHAR:
        value_arr = numpy.empty(att_len,'S1')
        with nogil:
            ierr = nc_get_att_text(_grpid, varid, attname, <char *>value_arr.data)
        if ierr != NC_NOERR:
            raise AttributeError((<char *>nc_strerror(ierr)).decode('ascii'))
        if name == '_FillValue' and python3:
            # make sure _FillValue for character arrays is a byte on python 3
            # (issue 271).
            pstring = bytes(value_arr)
        else:
            pstring =\
            value_arr.tostring().decode(default_encoding,unicode_error).replace('\x00','')
        return pstring
    elif att_type == NC_STRING:
        if att_len == 1:
            with nogil:
                ierr = nc_get_att_string(_grpid, varid, attname, &stratt)
            pstring = stratt.decode(default_encoding,unicode_error).replace('\x00','')
            ierr = nc_free_string(1, &stratt) # free memory in netcdf C lib
            return pstring
        else:
            raise KeyError('vlen string array attributes not supported')
    else:
    # a regular numeric or compound type.
        if att_type == NC_LONG:
            att_type = NC_INT
        try:
            type_att = _nctonptype[att_type] # see if it is a primitive type
            value_arr = numpy.empty(att_len,type_att)
        except KeyError:
            # check if it's a compound
            try:
                type_att = _read_compound(grp, att_type)
                value_arr = numpy.empty(att_len,type_att)
            except:
                # check if it's an enum
                try:
                    type_att = _read_enum(grp, att_type)
                    value_arr = numpy.empty(att_len,type_att.dtype)
                except:
                    raise KeyError('attribute %s has unsupported datatype' % attname)
        with nogil:
            ierr = nc_get_att(_grpid, varid, attname, value_arr.data)
        if ierr != NC_NOERR:
            raise AttributeError((<char *>nc_strerror(ierr)).decode('ascii'))
        if value_arr.shape == ():
            # return a scalar for a scalar array
            return value_arr.item()
        elif att_len == 1:
            # return a scalar for a single element array
            return value_arr[0]
        else:
            return value_arr

def _set_default_format(object format='NETCDF4'):
    # Private function to set the netCDF file format
    if format not in _format_dict:
        raise ValueError("unrecognized format requested")
    nc_set_default_format(_format_dict[format], NULL)

cdef _get_format(int grpid):
    # Private function to get the netCDF file format
    cdef int ierr, formatp
    with nogil:
        ierr = nc_inq_format(grpid, &formatp)
    if ierr != NC_NOERR:
        raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
    if formatp not in _reverse_format_dict:
        raise ValueError('format not supported by python interface')
    return _reverse_format_dict[formatp]

cdef _get_full_format(int grpid):
    # Private function to get the underlying disk format
    cdef int ierr, formatp, modep
    IF HAS_NC_INQ_FORMAT_EXTENDED:
        with nogil:
            ierr = nc_inq_format_extended(grpid, &formatp, &modep)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        if formatp == NC_FORMAT_NC3:
            return 'NETCDF3'
        elif formatp == NC_FORMAT_NC_HDF5:
            return 'HDF5'
        elif formatp == NC_FORMAT_NC_HDF4:
            return 'HDF4'
        elif formatp == NC_FORMAT_PNETCDF:
            return 'PNETCDF'
        elif formatp == NC_FORMAT_DAP2:
            return 'DAP2'
        elif formatp == NC_FORMAT_DAP4:
            return 'DAP4'
        elif formatp == NC_FORMAT_UNDEFINED:
            return 'UNDEFINED'
    ELSE:
        return 'UNDEFINED'

cdef _set_att(grp, int varid, name, value,\
              nc_type xtype=-99, force_ncstring=False):
    # Private function to set an attribute name/value pair
    cdef int i, ierr, lenarr, n
    cdef size_t att_len
    cdef nc_type att_type
    cdef char *attname
    cdef char *datstring
    cdef ndarray value_arr
    bytestr = _strencode(name)
    attname = bytestr
    # put attribute value into a numpy array.
    value_arr = numpy.array(value)
    # if array is 64 bit integers or
    # if 64-bit datatype not supported, cast to 32 bit integers.
    fmt = _get_format(grp._grpid)
    is_netcdf3 = fmt.startswith('NETCDF3') or fmt == 'NETCDF4_CLASSIC'
    if value_arr.dtype.str[1:] == 'i8' and ('i8' not in _supportedtypes or\
       is_netcdf3):
        value_arr = value_arr.astype('i4')
    # if array contains ascii strings, write a text attribute (stored as bytes).
    # if array contains unicode strings, and data model is NETCDF4, 
    # write as a string.  string arrays are concatenated into a single string.
    if value_arr.dtype.char in ['S','U']:
        if not value_arr.shape:
            dats = _strencode(value_arr.item())
        else:
            value_arr1 = value_arr.ravel()
            dats = _strencode(''.join(value_arr1.tolist()))
        lenarr = len(dats)
        datstring = dats
        if lenarr == 0:
            # write null byte
            lenarr=1; datstring = '\x00'
        if (force_ncstring or value_arr.dtype.char == 'U') and not is_netcdf3:
            # check to see if attribute already exists
            # and is NC_CHAR, if so delete it and re-create it
            # (workaround for issue #485). Fixed in C library
            # with commit 473259b7728120bb281c52359b1af50cca2fcb72,
            # which was included in 4.4.0-RC5.
            if _needsworkaround_issue485:
                ierr = nc_inq_att(grp._grpid, varid, attname, &att_type, &att_len)
                if ierr == NC_NOERR and att_type == NC_CHAR:
                    ierr = nc_del_att(grp._grpid, varid, attname)
                    if ierr != NC_NOERR:
                        raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
            # try to convert to ascii string, write as NC_CHAR 
            # else it's a unicode string, write as NC_STRING (if NETCDF4)
            try:
                if force_ncstring: raise UnicodeError
                dats_ascii = _to_ascii(dats) # try to encode bytes as ascii string
                ierr = nc_put_att_text(grp._grpid, varid, attname, lenarr, datstring)
            except UnicodeError:
                ierr = nc_put_att_string(grp._grpid, varid, attname, 1, &datstring)
        else:
            ierr = nc_put_att_text(grp._grpid, varid, attname, lenarr, datstring)
        if ierr != NC_NOERR:
            raise AttributeError((<char *>nc_strerror(ierr)).decode('ascii'))
    # a 'regular' array type ('f4','i4','f8' etc)
    else:
        if value_arr.dtype.kind == 'V': # compound attribute.
            xtype = _find_cmptype(grp,value_arr.dtype)
        elif value_arr.dtype.str[1:] not in _supportedtypes:
            raise TypeError, 'illegal data type for attribute, must be one of %s, got %s' % (_supportedtypes, value_arr.dtype.str[1:])
        elif xtype == -99: # if xtype is not passed in as kwarg.
            xtype = _nptonctype[value_arr.dtype.str[1:]]
        lenarr = PyArray_SIZE(value_arr)
        ierr = nc_put_att(grp._grpid, varid, attname, xtype, lenarr, value_arr.data)
        if ierr != NC_NOERR:
            raise AttributeError((<char *>nc_strerror(ierr)).decode('ascii'))

cdef _get_types(group):
    # Private function to create `netCDF4.CompoundType`,
    # `netCDF4.VLType` or `netCDF4.EnumType` instances for all the
    # compound, VLEN or Enum types in a `netCDF4.Group` or `netCDF4.Dataset`.
    cdef int ierr, ntypes, classp, n, _grpid
    cdef nc_type xtype
    cdef nc_type *typeids
    cdef char namstring[NC_MAX_NAME+1]
    _grpid = group._grpid
    # get the number of user defined types in this group.
    with nogil:
        ierr = nc_inq_typeids(_grpid, &ntypes, NULL)
    if ierr != NC_NOERR:
        raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
    if ntypes > 0:
        typeids = <nc_type *>malloc(sizeof(nc_type) * ntypes)
        with nogil:
            ierr = nc_inq_typeids(_grpid, &ntypes, typeids)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
    # create empty dictionary for CompoundType instances.
    cmptypes = OrderedDict()
    vltypes = OrderedDict()
    enumtypes = OrderedDict()
    if ntypes > 0:
        for n from 0 <= n < ntypes:
            xtype = typeids[n]
            with nogil:
                ierr = nc_inq_user_type(_grpid, xtype, namstring,
                                        NULL,NULL,NULL,&classp)
            if ierr != NC_NOERR:
                raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
            if classp == NC_COMPOUND: # a compound
                name = namstring.decode(default_encoding,unicode_error)
                # read the compound type info from the file,
                # create a CompoundType instance from it.
                try:
                    cmptype = _read_compound(group, xtype)
                except KeyError:
                    msg='WARNING: unsupported Compound type, skipping...'
                    warnings.warn(msg)
                    continue
                cmptypes[name] = cmptype
            elif classp == NC_VLEN: # a vlen
                name = namstring.decode(default_encoding,unicode_error)
                # read the VLEN type info from the file,
                # create a VLType instance from it.
                try:
                    vltype = _read_vlen(group, xtype)
                except KeyError:
                    msg='WARNING: unsupported VLEN type, skipping...'
                    warnings.warn(msg)
                    continue
                vltypes[name] = vltype
            elif classp == NC_ENUM: # an enum type
                name = namstring.decode(default_encoding,unicode_error)
                # read the Enum type info from the file,
                # create a EnumType instance from it.
                try:
                    enumtype = _read_enum(group, xtype)
                except KeyError:
                    msg='WARNING: unsupported Enum type, skipping...'
                    warnings.warn(msg)
                    continue
                enumtypes[name] = enumtype
        free(typeids)
    return cmptypes, vltypes, enumtypes

cdef _get_dims(group):
    # Private function to create `netCDF4.Dimension` instances for all the
    # dimensions in a `netCDF4.Group` or Dataset
    cdef int ierr, numdims, n, _grpid
    cdef int *dimids
    cdef char namstring[NC_MAX_NAME+1]
    # get number of dimensions in this Group.
    _grpid = group._grpid
    with nogil:
        ierr = nc_inq_ndims(_grpid, &numdims)
    if ierr != NC_NOERR:
        raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
    # create empty dictionary for dimensions.
    dimensions = OrderedDict()
    if numdims > 0:
        dimids = <int *>malloc(sizeof(int) * numdims)
        if group.data_model == 'NETCDF4':
            with nogil:
                ierr = nc_inq_dimids(_grpid, &numdims, dimids, 0)
            if ierr != NC_NOERR:
                raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        else:
            for n from 0 <= n < numdims:
                dimids[n] = n
        for n from 0 <= n < numdims:
            with nogil:
                ierr = nc_inq_dimname(_grpid, dimids[n], namstring)
            if ierr != NC_NOERR:
                raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
            name = namstring.decode(default_encoding,unicode_error)
            dimensions[name] = Dimension(group, name, id=dimids[n])
        free(dimids)
    return dimensions

cdef _get_grps(group):
    # Private function to create `netCDF4.Group` instances for all the
    # groups in a `netCDF4.Group` or Dataset
    cdef int ierr, numgrps, n, _grpid
    cdef int *grpids
    cdef char namstring[NC_MAX_NAME+1]
    # get number of groups in this Group.
    _grpid = group._grpid
    with nogil:
        ierr = nc_inq_grps(_grpid, &numgrps, NULL)
    if ierr != NC_NOERR:
        raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
    # create dictionary containing `netCDF4.Group` instances for groups in this group
    groups = OrderedDict()
    if numgrps > 0:
        grpids = <int *>malloc(sizeof(int) * numgrps)
        with nogil:
            ierr = nc_inq_grps(_grpid, NULL, grpids)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        for n from 0 <= n < numgrps:
            with nogil:
                ierr = nc_inq_grpname(grpids[n], namstring)
            if ierr != NC_NOERR:
                raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
            name = namstring.decode(default_encoding,unicode_error)
            groups[name] = Group(group, name, id=grpids[n])
        free(grpids)
    return groups

cdef _get_vars(group):
    # Private function to create `netCDF4.Variable` instances for all the
    # variables in a `netCDF4.Group` or Dataset
    cdef int ierr, numvars, n, nn, numdims, varid, classp, iendian, _grpid
    cdef int *varids
    cdef int *dimids
    cdef nc_type xtype
    cdef char namstring[NC_MAX_NAME+1]
    cdef char namstring_cmp[NC_MAX_NAME+1]
    # get number of variables in this Group.
    _grpid = group._grpid
    with nogil:
        ierr = nc_inq_nvars(_grpid, &numvars)
    if ierr != NC_NOERR:
        raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
    # create empty dictionary for variables.
    variables = OrderedDict()
    if numvars > 0:
        # get variable ids.
        varids = <int *>malloc(sizeof(int) * numvars)
        if group.data_model == 'NETCDF4':
            with nogil:
                ierr = nc_inq_varids(_grpid, &numvars, varids)
            if ierr != NC_NOERR:
                raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        else:
            for n from 0 <= n < numvars:
                varids[n] = n
        # loop over variables.
        for n from 0 <= n < numvars:
            varid = varids[n]
            # get variable name.
            with nogil:
                ierr = nc_inq_varname(_grpid, varid, namstring)
            if ierr != NC_NOERR:
                raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
            name = namstring.decode(default_encoding,unicode_error)
            if ierr != NC_NOERR:
                raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
            # get variable type.
            with nogil:
                ierr = nc_inq_vartype(_grpid, varid, &xtype)
            if ierr != NC_NOERR:
                raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
            # get endian-ness of variable.
            endianness = None
            # assume data returned by HDF4 lib has already been byte-swapped
            # (if needed) so that it is in native endian format (issue #391).
            if group.disk_format != 'HDF4':
                with nogil:
                    ierr = nc_inq_var_endian(_grpid, varid, &iendian)
                if ierr == NC_NOERR:
                    if iendian == NC_ENDIAN_LITTLE:
                        endianness = '<'
                    elif iendian == NC_ENDIAN_BIG:
                        endianness = '>'
            # check to see if it is a supported user-defined type.
            try:
                datatype = _nctonptype[xtype]
                if endianness is not None:
                    datatype = endianness + datatype
            except KeyError:
                if xtype == NC_STRING:
                    datatype = str
                else:
                    with nogil:
                        ierr = nc_inq_user_type(_grpid, xtype, namstring_cmp,
                                                NULL, NULL, NULL, &classp)
                    if classp == NC_COMPOUND: # a compound type
                        # create CompoundType instance describing this compound type.
                        try:
                            datatype = _read_compound(group, xtype, endian=endianness)
                        except KeyError:
                            msg="WARNING: variable '%s' has unsupported compound datatype, skipping .." % name
                            warnings.warn(msg)
                            continue
                    elif classp == NC_VLEN: # a compound type
                        # create VLType instance describing this compound type.
                        try:
                            datatype = _read_vlen(group, xtype, endian=endianness)
                        except KeyError:
                            msg="WARNING: variable '%s' has unsupported VLEN datatype, skipping .." % name
                            warnings.warn(msg)
                            continue
                    elif classp == NC_ENUM:
                        # create EnumType instance describing this compound type.
                        try:
                            datatype = _read_enum(group, xtype, endian=endianness)
                        except KeyError:
                            msg="WARNING: variable '%s' has unsupported Enum datatype, skipping .." % name
                            warnings.warn(msg)
                            continue
                    else:
                        msg="WARNING: variable '%s' has unsupported datatype, skipping .." % name
                        warnings.warn(msg)
                        continue
            # get number of dimensions.
            with nogil:
                ierr = nc_inq_varndims(_grpid, varid, &numdims)
            if ierr != NC_NOERR:
                raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
            dimids = <int *>malloc(sizeof(int) * numdims)
            # get dimension ids.
            with nogil:
                ierr = nc_inq_vardimid(_grpid, varid, dimids)
            if ierr != NC_NOERR:
                raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
            # loop over dimensions, retrieve names.
            # if not found in current group, look in parents.
            # QUESTION:  what if grp1 has a dimension named 'foo'
            # and so does it's parent - can a variable in grp1
            # use the 'foo' dimension from the parent?
            dimensions = []
            for nn from 0 <= nn < numdims:
                grp = group
                found = False
                while not found:
                    for key, value in grp.dimensions.items():
                        if value._dimid == dimids[nn]:
                            dimensions.append(key)
                            found = True
                            break
                    grp = grp.parent
            free(dimids)
            # create new variable instance.
            if endianness == '>':
                variables[name] = Variable(group, name, datatype, dimensions, id=varid, endian='big')
            elif endianness == '<':
                variables[name] = Variable(group, name, datatype, dimensions, id=varid, endian='little')
            else:
                variables[name] = Variable(group, name, datatype, dimensions, id=varid)
        free(varids) # free pointer holding variable ids.
    return variables

# these are class attributes that
# only exist at the python level (not in the netCDF file).

_private_atts =\
['_grpid','_grp','_varid','groups','dimensions','variables','dtype','data_model','disk_format',
 '_nunlimdim','path','parent','ndim','mask','scale','cmptypes','vltypes','enumtypes','_isprimitive',
 'file_format','_isvlen','_isenum','_iscompound','_cmptype','_vltype','_enumtype','name',
 '__orthogoral_indexing__','keepweakref','_has_lsd']
__pdoc__ = {}

cdef class Dataset:
    """
A netCDF `netCDF4.Dataset` is a collection of dimensions, groups, variables and
attributes. Together they describe the meaning of data and relations among
data fields stored in a netCDF file. See `netCDF4.Dataset.__init__` for more
details.

A list of attribute names corresponding to global netCDF attributes
defined for the `netCDF4.Dataset` can be obtained with the
`netCDF4.Dataset.ncattrs` method.
These attributes can be created by assigning to an attribute of the
`netCDF4.Dataset` instance. A dictionary containing all the netCDF attribute
name/value pairs is provided by the `__dict__` attribute of a
`netCDF4.Dataset` instance.

The following class variables are read-only and should not be
modified by the user.

**`dimensions`**: The `dimensions` dictionary maps the names of
dimensions defined for the `netCDF4.Group` or `netCDF4.Dataset` to instances of the
`netCDF4.Dimension` class.

**`variables`**: The `variables` dictionary maps the names of variables
defined for this `netCDF4.Dataset` or `netCDF4.Group` to instances of the 
`netCDF4.Variable` class.

**`groups`**: The groups dictionary maps the names of groups created for
this `netCDF4.Dataset` or `netCDF4.Group` to instances of the `netCDF4.Group` class (the
`netCDF4.Dataset` class is simply a special case of the `netCDF4.Group` class which
describes the root group in the netCDF4 file).

**`cmptypes`**: The `cmptypes` dictionary maps the names of
compound types defined for the `netCDF4.Group` or `netCDF4.Dataset` to instances of the
`netCDF4.CompoundType` class.

**`vltypes`**: The `vltypes` dictionary maps the names of
variable-length types defined for the `netCDF4.Group` or `netCDF4.Dataset` to instances 
of the `netCDF4.VLType` class.

**`enumtypes`**: The `enumtypes` dictionary maps the names of
Enum types defined for the `netCDF4.Group` or `netCDF4.Dataset` to instances 
of the `netCDF4.EnumType` class.

**`data_model`**: `data_model` describes the netCDF
data model version, one of `NETCDF3_CLASSIC`, `NETCDF4`,
`NETCDF4_CLASSIC`, `NETCDF3_64BIT_OFFSET` or `NETCDF3_64BIT_DATA`.

**`file_format`**: same as `data_model`, retained for backwards compatibility.

**`disk_format`**: `disk_format` describes the underlying
file format, one of `NETCDF3`, `HDF5`, `HDF4`,
`PNETCDF`, `DAP2`, `DAP4` or `UNDEFINED`. Only available if using
netcdf C library version >= 4.3.1, otherwise will always return
`UNDEFINED`.

**`parent`**: `parent` is a reference to the parent
`netCDF4.Group` instance. `None` for the root group or `netCDF4.Dataset`
instance.

**`path`**: `path` shows the location of the `netCDF4.Group` in
the `netCDF4.Dataset` in a unix directory format (the names of groups in the
hierarchy separated by backslashes). A `netCDF4.Dataset` instance is the root
group, so the path is simply `'/'`.

**`keepweakref`**: If `True`, child Dimension and Variables objects only keep weak 
references to the parent Dataset or Group.
    """
    cdef object __weakref__
    cdef public int _grpid
    cdef public int _isopen
    cdef public groups, dimensions, variables, disk_format, path, parent,\
    file_format, data_model, cmptypes, vltypes, enumtypes,  __orthogonal_indexing__, \
    keepweakref
    # Docstrings for class variables (used by pdoc).
    __pdoc__['Dataset.dimensions']=\
    """The `dimensions` dictionary maps the names of
    dimensions defined for the `netCDF4.Group` or `netCDF4.Dataset` to instances of the
    `netCDF4.Dimension` class."""
    __pdoc__['Dataset.variables']=\
    """The `variables` dictionary maps the names of variables
    defined for this `netCDF4.Dataset` or `netCDF4.Group` to instances of the `netCDF4.Variable`
    class."""
    __pdoc__['Dataset.groups']=\
    """The groups dictionary maps the names of groups created for
    this `netCDF4.Dataset` or `netCDF4.Group` to instances of the `netCDF4.Group` class (the
    `netCDF4.Dataset` class is simply a special case of the `netCDF4.Group` class which
    describes the root group in the netCDF4 file)."""
    __pdoc__['Dataset.cmptypes']=\
    """The `cmptypes` dictionary maps the names of
    compound types defined for the `netCDF4.Group` or `netCDF4.Dataset` to instances of the
    `netCDF4.CompoundType` class."""
    __pdoc__['Dataset.vltypes']=\
    """The `vltypes` dictionary maps the names of
    variable-length types defined for the `netCDF4.Group` or `netCDF4.Dataset` to instances of the
    `netCDF4.VLType` class."""
    __pdoc__['Dataset.enumtypes']=\
    """The `enumtypes` dictionary maps the names of
    Enum types defined for the `netCDF4.Group` or `netCDF4.Dataset` to instances of the
    `netCDF4.EnumType` class."""
    __pdoc__['Dataset.data_model']=\
    """`data_model` describes the netCDF
    data model version, one of `NETCDF3_CLASSIC`, `NETCDF4`,
    `NETCDF4_CLASSIC`, `NETCDF3_64BIT_OFFSET` or `NETCDF3_64BIT_DATA`."""
    __pdoc__['Dataset.file_format']=\
    """same as `data_model`, retained for backwards compatibility."""
    __pdoc__['Dataset.disk_format']=\
    """`disk_format` describes the underlying
    file format, one of `NETCDF3`, `HDF5`, `HDF4`,
    `PNETCDF`, `DAP2`, `DAP4` or `UNDEFINED`. Only available if using
    netcdf C library version >= 4.3.1, otherwise will always return
    `UNDEFINED`."""
    __pdoc__['Dataset.parent']=\
    """`parent` is a reference to the parent
    `netCDF4.Group` instance. `None` for the root group or `netCDF4.Dataset` instance"""
    __pdoc__['Dataset.path']=\
    """`path` shows the location of the `netCDF4.Group` in
    the `netCDF4.Dataset` in a unix directory format (the names of groups in the
    hierarchy separated by backslashes). A `netCDF4.Dataset` instance is the root
    group, so the path is simply `'/'`."""
    __pdoc__['Dataset.keepweakref']=\
    """If `True`, child Dimension and Variables objects only keep weak references to
    the parent Dataset or Group.""" 

    def __init__(self, filename, mode='r', clobber=True, format='NETCDF4',
                 diskless=False, persist=False, keepweakref=False, **kwargs):
        """
        **`__init__(self, filename, mode="r", clobber=True, diskless=False,
        persist=False, weakref=False, format='NETCDF4')`**

        `netCDF4.Dataset` constructor.

        **`filename`**: Name of netCDF file to hold dataset. Can also
	be a python 3 pathlib instance or the URL of an OpenDAP dataset.
        
        **`mode`**: access mode. `r` means read-only; no data can be
        modified. `w` means write; a new file is created, an existing file with
        the same name is deleted. `a` and `r+` mean append (in analogy with
        serial files); an existing file is opened for reading and writing.
        Appending `s` to modes `w`, `r+` or `a` will enable unbuffered shared
        access to `NETCDF3_CLASSIC`, `NETCDF3_64BIT_OFFSET` or
        `NETCDF3_64BIT_DATA` formatted files.
        Unbuffered acesss may be useful even if you don't need shared
        access, since it may be faster for programs that don't access data
        sequentially. This option is ignored for `NETCDF4` and `NETCDF4_CLASSIC`
        formatted files.
        
        **`clobber`**: if `True` (default), opening a file with `mode='w'`
        will clobber an existing file with the same name.  if `False`, an
        exception will be raised if a file with the same name already exists.
        
        **`format`**: underlying file format (one of `'NETCDF4',
        'NETCDF4_CLASSIC', 'NETCDF3_CLASSIC'`, `'NETCDF3_64BIT_OFFSET'` or
        `'NETCDF3_64BIT_DATA'`.
        Only relevant if `mode = 'w'` (if `mode = 'r','a'` or `'r+'` the file format
        is automatically detected). Default `'NETCDF4'`, which means the data is
        stored in an HDF5 file, using netCDF 4 API features.  Setting
        `format='NETCDF4_CLASSIC'` will create an HDF5 file, using only netCDF 3
        compatibile API features. netCDF 3 clients must be recompiled and linked
        against the netCDF 4 library to read files in `NETCDF4_CLASSIC` format.
        `'NETCDF3_CLASSIC'` is the classic netCDF 3 file format that does not
        handle 2+ Gb files. `'NETCDF3_64BIT_OFFSET'` is the 64-bit offset
        version of the netCDF 3 file format, which fully supports 2+ GB files, but
        is only compatible with clients linked against netCDF version 3.6.0 or
        later. `'NETCDF3_64BIT_DATA'` is the 64-bit data version of the netCDF 3
        file format, which supports 64-bit dimension sizes plus unsigned and
        64 bit integer data types, but is only compatible with clients linked against
        netCDF version 4.4.0 or later.
        
        **`diskless`**: If `True`, create diskless (in memory) file.  
        This is an experimental feature added to the C library after the
        netcdf-4.2 release.
        
        **`persist`**: if `diskless=True`, persist file to disk when closed
        (default `False`).

        **`keepweakref`**: if `True`, child Dimension and Variable instances will keep weak
        references to the parent Dataset or Group object.  Default is `False`, which
        means strong references will be kept.  Having Dimension and Variable instances
        keep a strong reference to the parent Dataset instance, which in turn keeps a
        reference to child Dimension and Variable instances, creates circular references.
        Circular references complicate garbage collection, which may mean increased
        memory usage for programs that create may Dataset instances with lots of
        Variables.  Setting `keepweakref=True` allows Dataset instances to be
        garbage collected as soon as they go out of scope, potential reducing memory
        usage.  However, in most cases this is not desirable, since the associated
        Variable instances may still be needed, but are rendered unusable when the
        parent Dataset instance is garbage collected.
        """
        cdef int grpid, ierr, numgrps, numdims, numvars
        cdef char *path
        cdef char namstring[NC_MAX_NAME+1]
        # flag to indicate that Variables in this Dataset support orthogonal indexing.
        self.__orthogonal_indexing__ = True
        if diskless and __netcdf4libversion__ < '4.2.1':
            #diskless = False # don't raise error, instead silently ignore
            raise ValueError('diskless mode requires netcdf lib >= 4.2.1, you have %s' % __netcdf4libversion__)
        bytestr = _strencode(str(filename))
        path = bytestr
        if mode == 'w':
            _set_default_format(format=format)
            if clobber:
                if diskless:
                    if persist:
                        ierr = nc_create(path, NC_WRITE | NC_CLOBBER | NC_DISKLESS , &grpid)
                    else:
                        ierr = nc_create(path, NC_CLOBBER | NC_DISKLESS , &grpid)
                else:
                    ierr = nc_create(path, NC_CLOBBER, &grpid)
            else:
                if diskless:
                    if persist:
                        ierr = nc_create(path, NC_WRITE | NC_NOCLOBBER | NC_DISKLESS , &grpid)
                    else:
                        ierr = nc_create(path, NC_NOCLOBBER | NC_DISKLESS , &grpid)
                else:
                    ierr = nc_create(path, NC_NOCLOBBER, &grpid)
            # reset default format to netcdf3 - this is a workaround
            # for issue 170 (nc_open'ing a DAP dataset after switching
            # format to NETCDF4). This bug should be fixed in version
            # 4.3.0 of the netcdf library (add a version check here?).
            _set_default_format(format='NETCDF3_64BIT_OFFSET')
        elif mode == 'r':
            if diskless:
                ierr = nc_open(path, NC_NOWRITE | NC_DISKLESS, &grpid)
            else:
                ierr = nc_open(path, NC_NOWRITE, &grpid)
        elif mode == 'r+' or mode == 'a':
            if diskless:
                ierr = nc_open(path, NC_WRITE | NC_DISKLESS, &grpid)
            else:
                ierr = nc_open(path, NC_WRITE, &grpid)
        elif mode == 'as' or mode == 'r+s':
            if diskless:
                ierr = nc_open(path, NC_SHARE | NC_DISKLESS, &grpid)
            else:
                ierr = nc_open(path, NC_SHARE, &grpid)
        elif mode == 'ws':
            if clobber:
                if diskless:
                    if persist:
                        ierr = nc_create(path, NC_WRITE | NC_SHARE | NC_CLOBBER | NC_DISKLESS , &grpid)
                    else:
                        ierr = nc_create(path, NC_SHARE | NC_CLOBBER | NC_DISKLESS , &grpid)
                else:
                    ierr = nc_create(path, NC_SHARE | NC_CLOBBER, &grpid)
            else:
                if diskless:
                    if persist:
                        ierr = nc_create(path, NC_WRITE | NC_SHARE | NC_NOCLOBBER | NC_DISKLESS , &grpid)
                    else:
                        ierr = nc_create(path, NC_SHARE | NC_NOCLOBBER | NC_DISKLESS , &grpid)
                else:
                    ierr = nc_create(path, NC_SHARE | NC_NOCLOBBER, &grpid)
        else:
            raise ValueError("mode must be 'w', 'r', 'a' or 'r+', got '%s'" % mode)
        if ierr != NC_NOERR:
            raise IOError((<char *>nc_strerror(ierr)).decode('ascii'))
        # data model and file format attributes
        self.data_model = _get_format(grpid)
        # data_model attribute used to be file_format (versions < 1.0.8), retain
        # file_format for backwards compatibility.
        self.file_format = self.data_model
        self.disk_format = _get_full_format(grpid)
        # diskless read access only works with NETCDF_CLASSIC (for now)
        #ncopen = mode.startswith('a') or mode.startswith('r')
        #if diskless and self.data_model != 'NETCDF3_CLASSIC' and ncopen:
        #    raise ValueError("diskless access only supported for NETCDF3_CLASSIC format")
        self._grpid = grpid
        self._isopen = 1
        self.path = '/'
        self.parent = None
        self.keepweakref = keepweakref
        # get compound, vlen and enum types in the root Group.
        self.cmptypes, self.vltypes, self.enumtypes = _get_types(self)
        # get dimensions in the root group.
        self.dimensions = _get_dims(self)
        # get variables in the root Group.
        self.variables = _get_vars(self)
        # get groups in the root Group.
        if self.data_model == 'NETCDF4':
            self.groups = _get_grps(self)
        else:
            self.groups = OrderedDict()

    # these allow Dataset objects to be used via a "with" statement.
    def __enter__(self):
        return self
    def __exit__(self,atype,value,traceback):
        self.close()

    def __getitem__(self, elem):
        # return variable or group defined in relative path.
        # split out group names in unix path.
        elem = posixpath.normpath(elem)
        # last name in path, could be a variable or group
        dirname, lastname = posixpath.split(elem)
        nestedgroups = dirname.split('/')
        group = self
        # iterate over groups in path.
        for g in nestedgroups:
            if g: group = group.groups[g]
        # return last one, either a group or a variable.
        if lastname in group.groups:
            return group.groups[lastname]
        elif lastname in group.variables:
            return group.variables[lastname]
        else:
            raise IndexError('%s not found in %s' % (lastname,group.path))

    def filepath(self):
        """
**`filepath(self)`**

Get the file system path (or the opendap URL) which was used to
open/create the Dataset. Requires netcdf >= 4.1.2"""
        cdef int ierr
        cdef size_t pathlen
        cdef char *c_path
        IF HAS_NC_INQ_PATH:
            with nogil:
                ierr = nc_inq_path(self._grpid, &pathlen, NULL)
            if ierr != NC_NOERR:
                raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
            c_path = <char *>malloc(sizeof(char) * pathlen)
            if not c_path:
                raise MemoryError()
            try:
                with nogil:
                    ierr = nc_inq_path(self._grpid, &pathlen, c_path)
                if ierr != NC_NOERR:
                    raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
                py_path = c_path[:pathlen] # makes a copy of pathlen bytes from c_string
            finally:
                free(c_path)
            return py_path.decode('ascii')
        ELSE:
            msg = """
filepath method not enabled.  To enable, install Cython, make sure you have
version 4.1.2 or higher of the netcdf C lib, and rebuild netcdf4-python."""
            raise ValueError(msg)

    def __repr__(self):
        if python3:
            return self.__unicode__()
        else:
            return unicode(self).encode(default_encoding)

    def __unicode__(self):
        ncdump = ['%r\n' % type(self)]
        dimnames = tuple([_tostr(dimname)+'(%s)'%len(self.dimensions[dimname])\
        for dimname in self.dimensions.keys()])
        varnames = tuple(\
        [_tostr(self.variables[varname].dtype)+' \033[4m'+_tostr(varname)+'\033[0m'+
        (((_tostr(self.variables[varname].dimensions)
        .replace("u'",""))\
        .replace("'",""))\
        .replace(", ",","))\
        .replace(",)",")") for varname in self.variables.keys()])
        grpnames = tuple([_tostr(grpname) for grpname in self.groups.keys()])
        if self.path == '/':
            ncdump.append('root group (%s data model, file format %s):\n' %
                    (self.data_model, self.disk_format))
        else:
            ncdump.append('group %s:\n' % self.path)
        attrs = ['    %s: %s\n' % (name,self.getncattr(name)) for name in\
                self.ncattrs()]
        ncdump = ncdump + attrs
        ncdump.append('    dimensions(sizes): %s\n' % ', '.join(dimnames))
        ncdump.append('    variables(dimensions): %s\n' % ', '.join(varnames))
        ncdump.append('    groups: %s\n' % ', '.join(grpnames))
        return ''.join(ncdump)

    def close(self):
        """
**`close(self)`**

Close the Dataset.
        """
        cdef int ierr
        ierr = nc_close(self._grpid)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        self._isopen = 0 # indicates file already closed, checked by __dealloc__

    def isopen(self):
        """
**`close(self)`**

is the Dataset open or closed?
        """
        return bool(self._isopen)

    def __dealloc__(self):
        # close file when there are no references to object left
        cdef int ierr
        if self._isopen:
            ierr = nc_close(self._grpid)

    def sync(self):
        """
**`sync(self)`**

Writes all buffered data in the `netCDF4.Dataset` to the disk file."""
        cdef int ierr
        ierr = nc_sync(self._grpid)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))

    def _redef(self):
        cdef int ierr
        ierr = nc_redef(self._grpid)

    def _enddef(self):
        cdef int ierr
        ierr = nc_enddef(self._grpid)

    def set_fill_on(self):
        """
**`set_fill_on(self)`**

Sets the fill mode for a `netCDF4.Dataset` open for writing to `on`.

This causes data to be pre-filled with fill values. The fill values can be
controlled by the variable's `_Fill_Value` attribute, but is usually
sufficient to the use the netCDF default `_Fill_Value` (defined
separately for each variable type). The default behavior of the netCDF
library correspongs to `set_fill_on`.  Data which are equal to the
`_Fill_Value` indicate that the variable was created, but never written
to."""
        cdef int ierr, oldmode
        ierr = nc_set_fill (self._grpid, NC_FILL, &oldmode)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))

    def set_fill_off(self):
        """
**`set_fill_off(self)`**

Sets the fill mode for a `netCDF4.Dataset` open for writing to `off`.

This will prevent the data from being pre-filled with fill values, which
may result in some performance improvements. However, you must then make
sure the data is actually written before being read."""
        cdef int ierr, oldmode
        ierr = nc_set_fill (self._grpid, NC_NOFILL, &oldmode)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))

    def createDimension(self, dimname, size=None):
        """
**`createDimension(self, dimname, size=None)`**

Creates a new dimension with the given `dimname` and `size`.

`size` must be a positive integer or `None`, which stands for
"unlimited" (default is `None`). Specifying a size of 0 also
results in an unlimited dimension. The return value is the `netCDF4.Dimension`
class instance describing the new dimension.  To determine the current
maximum size of the dimension, use the `len` function on the `netCDF4.Dimension`
instance. To determine if a dimension is 'unlimited', use the
`netCDF4.Dimension.isunlimited` method of the `netCDF4.Dimension` instance."""
        self.dimensions[dimname] = Dimension(self, dimname, size=size)
        return self.dimensions[dimname]

    def renameDimension(self, oldname, newname):
        """
**`renameDimension(self, oldname, newname)`**

rename a `netCDF4.Dimension` named `oldname` to `newname`."""
        cdef char *namstring
        bytestr = _strencode(newname)
        namstring = bytestr
        if self.data_model != 'NETCDF4': self._redef()
        try:
            dim = self.dimensions[oldname]
        except KeyError:
            raise KeyError('%s not a valid dimension name' % oldname)
        ierr = nc_rename_dim(self._grpid, dim._dimid, namstring)
        if self.data_model != 'NETCDF4': self._enddef()
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        # remove old key from dimensions dict.
        self.dimensions.pop(oldname)
        # add new key.
        self.dimensions[newname] = dim
        # Variable.dimensions is determined by a method that
        # looks in the file, so no need to manually update.

    def createCompoundType(self, datatype, datatype_name):
        """
**`createCompoundType(self, datatype, datatype_name)`**

Creates a new compound data type named `datatype_name` from the numpy
dtype object `datatype`.

***Note***: If the new compound data type contains other compound data types
(i.e. it is a 'nested' compound type, where not all of the elements
are homogeneous numeric data types), then the 'inner' compound types **must** be
created first.

The return value is the `netCDF4.CompoundType` class instance describing the new
datatype."""
        self.cmptypes[datatype_name] = CompoundType(self, datatype,\
                datatype_name)
        return self.cmptypes[datatype_name]

    def createVLType(self, datatype, datatype_name):
        """
**`createVLType(self, datatype, datatype_name)`**

Creates a new VLEN data type named `datatype_name` from a numpy
dtype object `datatype`.

The return value is the `netCDF4.VLType` class instance describing the new
datatype."""
        self.vltypes[datatype_name] = VLType(self, datatype, datatype_name)
        return self.vltypes[datatype_name]

    def createEnumType(self, datatype, datatype_name, enum_dict):
        """
**`createEnumType(self, datatype, datatype_name, enum_dict)`**

Creates a new Enum data type named `datatype_name` from a numpy
integer dtype object `datatype`, and a python dictionary
defining the enum fields and values.

The return value is the `netCDF4.EnumType` class instance describing the new
datatype."""
        self.enumtypes[datatype_name] = EnumType(self, datatype, datatype_name,
                enum_dict)
        return self.enumtypes[datatype_name]

    def createVariable(self, varname, datatype, dimensions=(), zlib=False,
            complevel=4, shuffle=True, fletcher32=False, contiguous=False,
            chunksizes=None, endian='native', least_significant_digit=None,
            fill_value=None, chunk_cache=None):
        """
**`createVariable(self, varname, datatype, dimensions=(), zlib=False,
complevel=4, shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,
endian='native', least_significant_digit=None, fill_value=None)`**

Creates a new variable with the given `varname`, `datatype`, and
`dimensions`. If dimensions are not given, the variable is assumed to be
a scalar.

If `varname` is specified as a path, using forward slashes as in unix to
separate components, then intermediate groups will be created as necessary 
For example, `createVariable('/GroupA/GroupB/VarC', float, ('x','y'))` will create groups `GroupA`
and `GroupA/GroupB`, plus the variable `GroupA/GroupB/VarC`, if the preceding
groups don't already exist.

The `datatype` can be a numpy datatype object, or a string that describes
a numpy dtype object (like the `dtype.str` attribue of a numpy array).
Supported specifiers include: `'S1' or 'c' (NC_CHAR), 'i1' or 'b' or 'B'
(NC_BYTE), 'u1' (NC_UBYTE), 'i2' or 'h' or 's' (NC_SHORT), 'u2'
(NC_USHORT), 'i4' or 'i' or 'l' (NC_INT), 'u4' (NC_UINT), 'i8' (NC_INT64),
'u8' (NC_UINT64), 'f4' or 'f' (NC_FLOAT), 'f8' or 'd' (NC_DOUBLE)`.
`datatype` can also be a `netCDF4.CompoundType` instance
(for a structured, or compound array), a `netCDF4.VLType` instance
(for a variable-length array), or the python `str` builtin
(for a variable-length string array). Numpy string and unicode datatypes with
length greater than one are aliases for `str`.

Data from netCDF variables is presented to python as numpy arrays with
the corresponding data type.

`dimensions` must be a tuple containing dimension names (strings) that
have been defined previously using `netCDF4.Dataset.createDimension`. The default value
is an empty tuple, which means the variable is a scalar.

If the optional keyword `zlib` is `True`, the data will be compressed in
the netCDF file using gzip compression (default `False`).

The optional keyword `complevel` is an integer between 1 and 9 describing
the level of compression desired (default 4). Ignored if `zlib=False`.

If the optional keyword `shuffle` is `True`, the HDF5 shuffle filter
will be applied before compressing the data (default `True`).  This
significantly improves compression. Default is `True`. Ignored if
`zlib=False`.

If the optional keyword `fletcher32` is `True`, the Fletcher32 HDF5
checksum algorithm is activated to detect errors. Default `False`.

If the optional keyword `contiguous` is `True`, the variable data is
stored contiguously on disk.  Default `False`. Setting to `True` for
a variable with an unlimited dimension will trigger an error.

The optional keyword `chunksizes` can be used to manually specify the
HDF5 chunksizes for each dimension of the variable. A detailed
discussion of HDF chunking and I/O performance is available
[here](http://www.hdfgroup.org/HDF5/doc/H5.user/Chunking.html).
Basically, you want the chunk size for each dimension to match as
closely as possible the size of the data block that users will read
from the file.  `chunksizes` cannot be set if `contiguous=True`.

The optional keyword `endian` can be used to control whether the
data is stored in little or big endian format on disk. Possible
values are `little, big` or `native` (default). The library
will automatically handle endian conversions when the data is read,
but if the data is always going to be read on a computer with the
opposite format as the one used to create the file, there may be
some performance advantage to be gained by setting the endian-ness.

The `zlib, complevel, shuffle, fletcher32, contiguous, chunksizes` and `endian`
keywords are silently ignored for netCDF 3 files that do not use HDF5.

The optional keyword `fill_value` can be used to override the default
netCDF `_FillValue` (the value that the variable gets filled with before
any data is written to it, defaults given in `netCDF4.default_fillvals`).
If fill_value is set to `False`, then the variable is not pre-filled.

If the optional keyword parameter `least_significant_digit` is
specified, variable data will be truncated (quantized). In conjunction
with `zlib=True` this produces 'lossy', but significantly more
efficient compression. For example, if `least_significant_digit=1`,
data will be quantized using `numpy.around(scale*data)/scale`, where
scale = 2**bits, and bits is determined so that a precision of 0.1 is
retained (in this case bits=4). From the 
[PSD metadata conventions](http://www.esrl.noaa.gov/psd/data/gridded/conventions/cdc_netcdf_standard.shtml):
"least_significant_digit -- power of ten of the smallest decimal place
in unpacked data that is a reliable value." Default is `None`, or no
quantization, or 'lossless' compression.

When creating variables in a `NETCDF4` or `NETCDF4_CLASSIC` formatted file,
HDF5 creates something called a 'chunk cache' for each variable.  The
default size of the chunk cache may be large enough to completely fill
available memory when creating thousands of variables.  The optional
keyword `chunk_cache` allows you to reduce (or increase) the size of
the default chunk cache when creating a variable.  The setting only
persists as long as the Dataset is open - you can use the set_var_chunk_cache
method to change it the next time the Dataset is opened.
Warning - messing with this parameter can seriously degrade performance.

The return value is the `netCDF4.Variable` class instance describing the new
variable.

A list of names corresponding to netCDF variable attributes can be
obtained with the `netCDF4.Variable` method `netCDF4.Variable.ncattrs`. A dictionary
containing all the netCDF attribute name/value pairs is provided by
the `__dict__` attribute of a `netCDF4.Variable` instance.

`netCDF4.Variable` instances behave much like array objects. Data can be
assigned to or retrieved from a variable with indexing and slicing
operations on the `netCDF4.Variable` instance. A `netCDF4.Variable` instance has six
Dataset standard attributes: `dimensions, dtype, shape, ndim, name` and
`least_significant_digit`. Application programs should never modify
these attributes. The `dimensions` attribute is a tuple containing the
names of the dimensions associated with this variable. The `dtype`
attribute is a string describing the variable's data type (`i4, f8,
S1,` etc). The `shape` attribute is a tuple describing the current
sizes of all the variable's dimensions. The `name` attribute is a
string containing the name of the Variable instance.
The `least_significant_digit`
attributes describes the power of ten of the smallest decimal place in
the data the contains a reliable value.  assigned to the `netCDF4.Variable`
instance. If `None`, the data is not truncated. The `ndim` attribute
is the number of variable dimensions."""
        # if varname specified as a path, split out group names.
        varname = posixpath.normpath(varname)
        dirname, varname = posixpath.split(varname) # varname is last.
        # create parent groups (like mkdir -p).
        if not dirname:
            group = self
        else:
            group = self.createGroup(dirname)
        # create variable.
        group.variables[varname] = Variable(group, varname, datatype,
        dimensions=dimensions, zlib=zlib, complevel=complevel, shuffle=shuffle,
        fletcher32=fletcher32, contiguous=contiguous, chunksizes=chunksizes,
        endian=endian, least_significant_digit=least_significant_digit,
        fill_value=fill_value, chunk_cache=chunk_cache)
        return group.variables[varname]

    def renameVariable(self, oldname, newname):
        """
**`renameVariable(self, oldname, newname)`**

rename a `netCDF4.Variable` named `oldname` to `newname`"""
        cdef char *namstring
        try:
            var = self.variables[oldname]
        except KeyError:
            raise KeyError('%s not a valid variable name' % oldname)
        bytestr = _strencode(newname)
        namstring = bytestr
        if self.data_model != 'NETCDF4': self._redef()
        ierr = nc_rename_var(self._grpid, var._varid, namstring)
        if self.data_model != 'NETCDF4': self._enddef()
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        # remove old key from dimensions dict.
        self.variables.pop(oldname)
        # add new key.
        self.variables[newname] = var

    def createGroup(self, groupname):
        """
**`createGroup(self, groupname)`**

Creates a new `netCDF4.Group` with the given `groupname`.

If `groupname` is specified as a path, using forward slashes as in unix to
separate components, then intermediate groups will be created as necessary 
(analogous to `mkdir -p` in unix).  For example,
`createGroup('/GroupA/GroupB/GroupC')` will create `GroupA`,
`GroupA/GroupB`, and `GroupA/GroupB/GroupC`, if they don't already exist.
If the specified path describes a group that already exists, no error is
raised.

The return value is a `netCDF4.Group` class instance."""
        # if group specified as a path, split out group names
        groupname = posixpath.normpath(groupname)
        nestedgroups = groupname.split('/')
        group = self
        # loop over group names, create parent groups if they do not already
        # exist.
        for g in nestedgroups:
            if not g: continue
            if g not in group.groups:
                group.groups[g] = Group(group, g)
            group = group.groups[g]
        # if group already exists, just return the group
        # (prior to 1.1.8, this would have raised an error)
        return group

    def ncattrs(self):
        """
**`ncattrs(self)`**

return netCDF global attribute names for this `netCDF4.Dataset` or `netCDF4.Group` in a list."""
        return _get_att_names(self._grpid, NC_GLOBAL)

    def setncattr(self,name,value):
        """
**`setncattr(self,name,value)`**

set a netCDF dataset or group attribute using name,value pair.
Use if you need to set a netCDF attribute with the
with the same name as one of the reserved python attributes."""
        if self.data_model != 'NETCDF4': self._redef()
        _set_att(self, NC_GLOBAL, name, value)
        if self.data_model !=  'NETCDF4': self._enddef()

    def setncattr_string(self,name,value):
        """
**`setncattr_string(self,name,value)`**

set a netCDF dataset or group string attribute using name,value pair.
Use if you need to ensure that a netCDF attribute is created with type
`NC_STRING` if the file format is `NETCDF4`."""
        cdef nc_type xtype
        xtype=-99
        if self.data_model != 'NETCDF4':
            msg='file format does not support NC_STRING attributes'
            raise IOError(msg)
        _set_att(self, NC_GLOBAL, name, str(value), xtype=xtype, force_ncstring=True)

    def setncatts(self,attdict):
        """
**`setncatts(self,attdict)`**

set a bunch of netCDF dataset or group attributes at once using a python dictionary.
This may be faster when setting a lot of attributes for a `NETCDF3`
formatted file, since nc_redef/nc_enddef is not called in between setting
each attribute"""
        if self.data_model != 'NETCDF4': self._redef()
        for name, value in attdict.items():
            _set_att(self, NC_GLOBAL, name, value)
        if self.data_model != 'NETCDF4': self._enddef()

    def getncattr(self,name):
        """
**`getncattr(self,name)`**

retrievel a netCDF dataset or group attribute.
Use if you need to get a netCDF attribute with the same 
name as one of the reserved python attributes."""
        return _get_att(self, NC_GLOBAL, name)

    def __delattr__(self,name):
        # if it's a netCDF attribute, remove it
        if name not in _private_atts:
            self.delncattr(name)
        else:
            raise AttributeError(
            "'%s' is one of the reserved attributes %s, cannot delete. Use delncattr instead." % (name, tuple(_private_atts)))

    def delncattr(self, name):
        """
**`delncattr(self,name,value)`**

delete a netCDF dataset or group attribute.  Use if you need to delete a
netCDF attribute with the same name as one of the reserved python
attributes."""
        cdef char *attname
        cdef int ierr
        bytestr = _strencode(name)
        attname = bytestr
        if self.data_model != 'NETCDF4': self._redef()
        ierr = nc_del_att(self._grpid, NC_GLOBAL, attname)
        if self.data_model != 'NETCDF4': self._enddef()
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))

    def __setattr__(self,name,value):
        # if name in _private_atts, it is stored at the python
        # level and not in the netCDF file.
        if name not in _private_atts:
            self.setncattr(name, value)
        elif not name.endswith('__'):
            if hasattr(self,name):
                raise AttributeError(
            "'%s' is one of the reserved attributes %s, cannot rebind. Use setncattr instead." % (name, tuple(_private_atts)))
            else:
                self.__dict__[name]=value

    def __getattr__(self,name):
        # if name in _private_atts, it is stored at the python
        # level and not in the netCDF file.
        if name.startswith('__') and name.endswith('__'):
            # if __dict__ requested, return a dict with netCDF attributes.
            if name == '__dict__':
                names = self.ncattrs()
                values = []
                for name in names:
                    values.append(_get_att(self, NC_GLOBAL, name))
                return OrderedDict(zip(names,values))
            else:
                raise AttributeError
        elif name in _private_atts:
            return self.__dict__[name]
        else:
            return self.getncattr(name)

    def renameAttribute(self, oldname, newname):
        """
**`renameAttribute(self, oldname, newname)`**

rename a `netCDF4.Dataset` or `netCDF4.Group` attribute named `oldname` to `newname`."""
        cdef int ierr
        cdef char *oldnamec
        cdef char *newnamec
        bytestr = _strencode(oldname)
        oldnamec = bytestr
        bytestr = _strencode(newname)
        newnamec = bytestr
        ierr = nc_rename_att(self._grpid, NC_GLOBAL, oldnamec, newnamec)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))

    def renameGroup(self, oldname, newname):
        """
**`renameGroup(self, oldname, newname)`**

rename a `netCDF4.Group` named `oldname` to `newname` (requires netcdf >= 4.3.1)."""
        cdef int ierr
        cdef char *newnamec
        IF HAS_RENAME_GRP:
            bytestr = _strencode(newname)
            newnamec = bytestr
            try:
                grp = self.groups[oldname]
            except KeyError:
                raise KeyError('%s not a valid group name' % oldname)
            ierr = nc_rename_grp(grp._grpid, newnamec)
            if ierr != NC_NOERR:
                raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
            # remove old key from groups dict.
            self.groups.pop(oldname)
            # add new key.
            self.groups[newname] = grp
        ELSE:
            msg = """
renameGroup method not enabled.  To enable, install Cython, make sure you have
version 4.3.1 or higher of the netcdf C lib, and rebuild netcdf4-python."""
            raise ValueError(msg)

    def set_auto_maskandscale(self, value):
        """
**`set_auto_maskandscale(self, True_or_False)`**

Call `netCDF4.Variable.set_auto_maskandscale` for all variables contained in this `netCDF4.Dataset` or
`netCDF4.Group`, as well as for all variables in all its subgroups.

**`True_or_False`**: Boolean determining if automatic conversion to masked arrays
and variable scaling shall be applied for all variables.

***Note***: Calling this function only affects existing variables. Variables created
after calling this function will follow the default behaviour.
        """

        for var in self.variables.values():
            var.set_auto_maskandscale(value)

        for groups in _walk_grps(self):
            for group in groups:
                for var in group.variables.values():
                    var.set_auto_maskandscale(value)


    def set_auto_mask(self, value):
        """
**`set_auto_mask(self, True_or_False)`**

Call `netCDF4.Variable.set_auto_mask` for all variables contained in this `netCDF4.Dataset` or
`netCDF4.Group`, as well as for all variables in all its subgroups.

**`True_or_False`**: Boolean determining if automatic conversion to masked arrays
shall be applied for all variables.

***Note***: Calling this function only affects existing variables. Variables created
after calling this function will follow the default behaviour.
        """

        for var in self.variables.values():
            var.set_auto_mask(value)

        for groups in _walk_grps(self):
            for group in groups:
                for var in group.variables.values():
                    var.set_auto_mask(value)

    def set_auto_scale(self, value):
        """
**`set_auto_scale(self, True_or_False)`**

Call `netCDF4.Variable.set_auto_scale` for all variables contained in this `netCDF4.Dataset` or
`netCDF4.Group`, as well as for all variables in all its subgroups.

**`True_or_False`**: Boolean determining if automatic variable scaling
shall be applied for all variables.

***Note***: Calling this function only affects existing variables. Variables created
after calling this function will follow the default behaviour.
        """

        for var in self.variables.values():
            var.set_auto_scale(value)

        for groups in _walk_grps(self):
            for group in groups:
                for var in group.variables.values():
                    var.set_auto_scale(value)

    def get_variables_by_attributes(self, **kwargs):
        """
**`get_variables_by_attribute(self, **kwargs)`**

Returns a list of variables that match specific conditions.

Can pass in key=value parameters and variables are returned that
contain all of the matches. For example, 

    :::python
    >>> # Get variables with x-axis attribute.
    >>> vs = nc.get_variables_by_attributes(axis='X')
    >>> # Get variables with matching "standard_name" attribute
    >>> vs = nc.get_variables_by_attributes(standard_name='northward_sea_water_velocity')

Can pass in key=callable parameter and variables are returned if the
callable returns True.  The callable should accept a single parameter,
the attribute value.  None is given as the attribute value when the
attribute does not exist on the variable. For example,

    :::python
    >>> # Get Axis variables
    >>> vs = nc.get_variables_by_attributes(axis=lambda v: v in ['X', 'Y', 'Z', 'T'])
    >>> # Get variables that don't have an "axis" attribute
    >>> vs = nc.get_variables_by_attributes(axis=lambda v: v is None)
    >>> # Get variables that have a "grid_mapping" attribute
    >>> vs = nc.get_variables_by_attributes(grid_mapping=lambda v: v is not None)
"""
        vs = []

        has_value_flag  = False
        for vname in self.variables:
            var = self.variables[vname]
            for k, v in kwargs.items():
                if callable(v):
                    has_value_flag = v(getattr(var, k, None))
                    if has_value_flag is False:
                        break
                elif hasattr(var, k) and getattr(var, k) == v:
                    has_value_flag = True
                else:
                    has_value_flag = False
                    break

            if has_value_flag is True:
                vs.append(self.variables[vname])

        return vs


cdef class Group(Dataset):
    """
Groups define a hierarchical namespace within a netCDF file. They are
analogous to directories in a unix filesystem. Each `netCDF4.Group` behaves like
a `netCDF4.Dataset` within a Dataset, and can contain it's own variables,
dimensions and attributes (and other Groups). See `netCDF4.Group.__init__`
for more details.

`netCDF4.Group` inherits from `netCDF4.Dataset`, so all the 
`netCDF4.Dataset` class methods and variables are available
to a `netCDF4.Group` instance (except the `close` method).

Additional read-only class variables:

**`name`**: String describing the group name.
    """
    # Docstrings for class variables (used by pdoc).
    __pdoc__['Group.name']=\
    """A string describing the name of the `netCDF4.Group`."""
    def __init__(self, parent, name, **kwargs):
        """
        **`__init__(self, parent, name)`**
        `netCDF4.Group` constructor.

        **`parent`**: `netCDF4.Group` instance for the parent group.  If being created
        in the root group, use a `netCDF4.Dataset` instance.

        **`name`**: - Name of the group.

        ***Note***: `netCDF4.Group` instances should be created using the
        `netCDF4.Dataset.createGroup` method of a `netCDF4.Dataset` instance, or
        another `netCDF4.Group` instance, not using this class directly.
        """
        cdef int ierr
        cdef char *groupname
        # flag to indicate that Variables in this Group support orthogonal indexing.
        self.__orthogonal_indexing__ = True
        # set data_model and file_format attributes.
        self.data_model = parent.data_model
        self.file_format = parent.file_format
        # full path to Group.
        self.path = posixpath.join(parent.path, name)
        # parent group.
        self.parent = parent
        # propagate weak reference setting from parent.
        self.keepweakref = parent.keepweakref
        if 'id' in kwargs:
            self._grpid = kwargs['id']
            # get compound, vlen and enum types in this Group.
            self.cmptypes, self.vltypes, self.enumtypes = _get_types(self)
            # get dimensions in this Group.
            self.dimensions = _get_dims(self)
            # get variables in this Group.
            self.variables = _get_vars(self)
            # get groups in this Group.
            self.groups = _get_grps(self)
        else:
            bytestr = _strencode(name)
            groupname = bytestr
            ierr = nc_def_grp(parent._grpid, groupname, &self._grpid)
            if ierr != NC_NOERR:
                raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
            self.cmptypes = OrderedDict()
            self.vltypes = OrderedDict()
            self.enumtypes = OrderedDict()
            self.dimensions = OrderedDict()
            self.variables = OrderedDict()
            self.groups = OrderedDict()

    def close(self):
        """
**`close(self)`**

overrides `netCDF4.Dataset` close method which does not apply to `netCDF4.Group`
instances, raises IOError."""
        raise IOError('cannot close a `netCDF4.Group` (only applies to Dataset)')

    def _getname(self):
        # private method to get name associated with instance.
        cdef int err
        cdef char namstring[NC_MAX_NAME+1]
        with nogil:
            ierr = nc_inq_grpname(self._grpid, namstring)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        return namstring.decode(default_encoding,unicode_error)

    property name:
        """string name of Group instance"""
        def __get__(self):
            return self._getname()
        def __set__(self,value):
            raise AttributeError("name cannot be altered")


cdef class Dimension:
    """
A netCDF `netCDF4.Dimension` is used to describe the coordinates of a `netCDF4.Variable`.
See `netCDF4.Dimension.__init__` for more details.

The current maximum size of a `netCDF4.Dimension` instance can be obtained by
calling the python `len` function on the `netCDF4.Dimension` instance. The
`netCDF4.Dimension.isunlimited` method of a `netCDF4.Dimension` instance can be used to
determine if the dimension is unlimited.

Read-only class variables:

**`name`**: String name, used when creating a `netCDF4.Variable` with
`netCDF4.Dataset.createVariable`.

**`size`**: Current `netCDF4.Dimension` size (same as `len(d)`, where `d` is a
`netCDF4.Dimension` instance).
    """
    cdef public int _dimid, _grpid
    cdef public _data_model, _name, _grp
    # Docstrings for class variables (used by pdoc).
    __pdoc__['Dimension.name']=\
    """A string describing the name of the `netCDF4.Dimension` - used when creating a
    `netCDF4.Variable` instance with `netCDF4.Dataset.createVariable`."""

    def __init__(self, grp, name, size=None, **kwargs):
        """
        **`__init__(self, group, name, size=None)`**

        `netCDF4.Dimension` constructor.

        **`group`**: `netCDF4.Group` instance to associate with dimension.

        **`name`**: Name of the dimension.

        **`size`**: Size of the dimension. `None` or 0 means unlimited. (Default `None`).

        ***Note***: `netCDF4.Dimension` instances should be created using the
        `netCDF4.Dataset.createDimension` method of a `netCDF4.Group` or
        `netCDF4.Dataset` instance, not using `netCDF4.Dimension.__init__` directly.
        """
        cdef int ierr
        cdef char *dimname
        cdef size_t lendim
        self._grpid = grp._grpid
        # make a weakref to group to avoid circular ref (issue 218)
        # keep strong reference the default behaviour (issue 251)
        if grp.keepweakref:
            self._grp = weakref.proxy(grp)
        else:
            self._grp = grp
        self._data_model = grp.data_model
        self._name = name
        if 'id' in kwargs:
            self._dimid = kwargs['id']
        else:
            bytestr = _strencode(name)
            dimname = bytestr
            if size is not None:
                lendim = size
            else:
                lendim = NC_UNLIMITED
            if grp.data_model != 'NETCDF4': grp._redef()
            ierr = nc_def_dim(self._grpid, dimname, lendim, &self._dimid)
            if grp.data_model != 'NETCDF4': grp._enddef()
            if ierr != NC_NOERR:
                raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))

    def _getname(self):
        # private method to get name associated with instance.
        cdef int err, _grpid
        cdef char namstring[NC_MAX_NAME+1]
        _grpid = self._grp._grpid
        with nogil:
            ierr = nc_inq_dimname(_grpid, self._dimid, namstring)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        return namstring.decode(default_encoding,unicode_error)

    property name:
        """string name of Dimension instance"""
        def __get__(self):
            return self._getname()
        def __set__(self,value):
            raise AttributeError("name cannot be altered")

    property size:
        """current size of Dimension (calls `len` on Dimension instance)"""
        def __get__(self):
            return len(self)
        def __set__(self,value):
            raise AttributeError("size cannot be altered")

    def __repr__(self):
        if python3:
            return self.__unicode__()
        else:
            return unicode(self).encode(default_encoding)

    def __unicode__(self):
        if not dir(self._grp):
            return 'Dimension object no longer valid'
        if self.isunlimited():
            return repr(type(self))+" (unlimited): name = '%s', size = %s\n" % (self._name,len(self))
        else:
            return repr(type(self))+": name = '%s', size = %s\n" % (self._name,len(self))

    def __len__(self):
        # len(`netCDF4.Dimension` instance) returns current size of dimension
        cdef int ierr
        cdef size_t lengthp
        with nogil:
            ierr = nc_inq_dimlen(self._grpid, self._dimid, &lengthp)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        return lengthp

    def group(self):
        """
**`group(self)`**

return the group that this `netCDF4.Dimension` is a member of."""
        return self._grp

    def isunlimited(self):
        """
**`isunlimited(self)`**

returns `True` if the `netCDF4.Dimension` instance is unlimited, `False` otherwise."""
        cdef int ierr, n, numunlimdims, ndims, nvars, ngatts, xdimid
        cdef int *unlimdimids
        if self._data_model == 'NETCDF4':
            ierr = nc_inq_unlimdims(self._grpid, &numunlimdims, NULL)
            if ierr != NC_NOERR:
                raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
            if numunlimdims == 0:
                return False
            else:
                unlimdimids = <int *>malloc(sizeof(int) * numunlimdims)
                dimid = self._dimid
                with nogil:
                    ierr = nc_inq_unlimdims(self._grpid, &numunlimdims, unlimdimids)
                if ierr != NC_NOERR:
                    raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
                unlimdim_ids = []
                for n from 0 <= n < numunlimdims:
                    unlimdim_ids.append(unlimdimids[n])
                free(unlimdimids)
                if dimid in unlimdim_ids:
                    return True
                else:
                    return False
        else: # if not NETCDF4, there is only one unlimited dimension.
            # nc_inq_unlimdims only works for NETCDF4.
            with nogil:
                ierr = nc_inq(self._grpid, &ndims, &nvars, &ngatts, &xdimid)
            if self._dimid == xdimid:
                return True
            else:
                return False

cdef class Variable:
    """
A netCDF `netCDF4.Variable` is used to read and write netCDF data.  They are
analogous to numpy array objects. See `netCDF4.Variable.__init__` for more
details.

A list of attribute names corresponding to netCDF attributes defined for
the variable can be obtained with the `netCDF4.Variable.ncattrs` method. These
attributes can be created by assigning to an attribute of the
`netCDF4.Variable` instance. A dictionary containing all the netCDF attribute
name/value pairs is provided by the `__dict__` attribute of a
`netCDF4.Variable` instance.

The following class variables are read-only:

**`dimensions`**: A tuple containing the names of the
dimensions associated with this variable.

**`dtype`**: A numpy dtype object describing the
variable's data type.

**`ndim`**: The number of variable dimensions.

**`shape`**: A tuple with the current shape (length of all dimensions).

**`scale`**: If True, `scale_factor` and `add_offset` are
applied. Default is `True`, can be reset using `netCDF4.Variable.set_auto_scale` and
`netCDF4.Variable.set_auto_maskandscale` methods.

**`mask`**: If True, data is automatically converted to/from masked 
arrays when missing values or fill values are present. Default is `True`, can be
reset using `netCDF4.Variable.set_auto_mask` and `netCDF4.Variable.set_auto_maskandscale`
methods.

**`least_significant_digit`**: Describes the power of ten of the 
smallest decimal place in the data the contains a reliable value.  Data is
truncated to this decimal place when it is assigned to the `netCDF4.Variable`
instance. If `None`, the data is not truncated.

**`__orthogonal_indexing__`**: Always `True`.  Indicates to client code
that the object supports 'orthogonal indexing', which means that slices
that are 1d arrays or lists slice along each dimension independently.  This
behavior is similar to Fortran or Matlab, but different than numpy.

**`datatype`**: numpy data type (for primitive data types) or VLType/CompoundType
 instance (for compound or vlen data types).

**`name`**: String name.

**`size`**: The number of stored elements.
    """
    cdef public int _varid, _grpid, _nunlimdim
    cdef public _name, ndim, dtype, mask, scale, _isprimitive, _iscompound,\
    _isvlen, _isenum, _grp, _cmptype, _vltype, _enumtype,\
    __orthogonal_indexing__, _has_lsd
    # Docstrings for class variables (used by pdoc).
    __pdoc__['Variable.dimensions'] = \
    """A tuple containing the names of the
    dimensions associated with this variable."""
    __pdoc__['Variable.dtype'] = \
    """A numpy dtype object describing the
    variable's data type."""
    __pdoc__['Variable.ndim'] = \
    """The number of variable dimensions."""
    __pdoc__['Variable.scale'] = \
    """if True, `scale_factor` and `add_offset` are
    applied. Default is `True`, can be reset using `netCDF4.Variable.set_auto_scale` and
    `netCDF4.Variable.set_auto_maskandscale` methods."""
    __pdoc__['Variable.mask'] = \
    """If True, data is automatically converted to/from masked 
    arrays when missing values or fill values are present. Default is `True`, can be
    reset using `netCDF4.Variable.set_auto_mask` and `netCDF4.Variable.set_auto_maskandscale`
    methods."""
    __pdoc__['Variable.least_significant_digit'] = \
    """Describes the power of ten of the 
    smallest decimal place in the data the contains a reliable value.  Data is
    truncated to this decimal place when it is assigned to the `netCDF4.Variable`
    instance. If `None`, the data is not truncated."""
    __pdoc__['Variable.__orthogonal_indexing__'] = \
    """Always `True`.  Indicates to client code
    that the object supports 'orthogonal indexing', which means that slices
    that are 1d arrays or lists slice along each dimension independently.  This
    behavior is similar to Fortran or Matlab, but different than numpy."""
    __pdoc__['Variable.datatype'] = \
     """numpy data type (for primitive data types) or
     VLType/CompoundType/EnumType instance (for compound, vlen or enum
     data types)."""
    __pdoc__['Variable.name'] = \
    """String name."""
    __pdoc__['Variable.shape'] = \
    """A tuple with the current shape (length of all dimensions)."""
    __pdoc__['Variable.size'] = \
    """The number of stored elements."""

    def __init__(self, grp, name, datatype, dimensions=(), zlib=False,
            complevel=4, shuffle=True, fletcher32=False, contiguous=False,
            chunksizes=None, endian='native', least_significant_digit=None,
            fill_value=None, chunk_cache=None, **kwargs):
        """
        **`__init__(self, group, name, datatype, dimensions=(), zlib=False,
        complevel=4, shuffle=True, fletcher32=False, contiguous=False,
        chunksizes=None, endian='native',
        least_significant_digit=None,fill_value=None)`**

        `netCDF4.Variable` constructor.

        **`group`**: `netCDF4.Group` or `netCDF4.Dataset` instance to associate with variable.

        **`name`**: Name of the variable.

        **`datatype`**: `netCDF4.Variable` data type. Can be specified by providing a
        numpy dtype object, or a string that describes a numpy dtype object.
        Supported values, corresponding to `str` attribute of numpy dtype
        objects, include `'f4'` (32-bit floating point), `'f8'` (64-bit floating
        point), `'i4'` (32-bit signed integer), `'i2'` (16-bit signed integer),
        `'i8'` (64-bit singed integer), `'i4'` (8-bit singed integer), `'i1'`
        (8-bit signed integer), `'u1'` (8-bit unsigned integer), `'u2'` (16-bit
        unsigned integer), `'u4'` (32-bit unsigned integer), `'u8'` (64-bit
        unsigned integer), or `'S1'` (single-character string).  From
        compatibility with Scientific.IO.NetCDF, the old Numeric single character
        typecodes can also be used (`'f'` instead of `'f4'`, `'d'` instead of
        `'f8'`, `'h'` or `'s'` instead of `'i2'`, `'b'` or `'B'` instead of
        `'i1'`, `'c'` instead of `'S1'`, and `'i'` or `'l'` instead of
        `'i4'`). `datatype` can also be a `netCDF4.CompoundType` instance
        (for a structured, or compound array), a `netCDF4.VLType` instance
        (for a variable-length array), or the python `str` builtin
        (for a variable-length string array). Numpy string and unicode datatypes with
        length greater than one are aliases for `str`.
        
        **`dimensions`**: a tuple containing the variable's dimension names
        (defined previously with `createDimension`). Default is an empty tuple
        which means the variable is a scalar (and therefore has no dimensions).
        
        **`zlib`**: if `True`, data assigned to the `netCDF4.Variable`
        instance is compressed on disk. Default `False`.
        
        **`complevel`**: the level of zlib compression to use (1 is the fastest,
        but poorest compression, 9 is the slowest but best compression). Default 4.
        Ignored if `zlib=False`.
        
        **`shuffle`**: if `True`, the HDF5 shuffle filter is applied
        to improve compression. Default `True`. Ignored if `zlib=False`.
        
        **`fletcher32`**: if `True` (default `False`), the Fletcher32 checksum
        algorithm is used for error detection.
        
        **`contiguous`**: if `True` (default `False`), the variable data is
        stored contiguously on disk.  Default `False`. Setting to `True` for
        a variable with an unlimited dimension will trigger an error.
        
        **`chunksizes`**: Can be used to specify the HDF5 chunksizes for each
        dimension of the variable. A detailed discussion of HDF chunking and I/O
        performance is available
        [here](http://www.hdfgroup.org/HDF5/doc/H5.user/Chunking.html).
        Basically, you want the chunk size for each dimension to match as
        closely as possible the size of the data block that users will read
        from the file. `chunksizes` cannot be set if `contiguous=True`.
        
        **`endian`**: Can be used to control whether the
        data is stored in little or big endian format on disk. Possible
        values are `little, big` or `native` (default). The library
        will automatically handle endian conversions when the data is read,
        but if the data is always going to be read on a computer with the
        opposite format as the one used to create the file, there may be
        some performance advantage to be gained by setting the endian-ness.
        For netCDF 3 files (that don't use HDF5), only `endian='native'` is allowed.
        
        The `zlib, complevel, shuffle, fletcher32, contiguous` and `chunksizes`
        keywords are silently ignored for netCDF 3 files that do not use HDF5.
        
        **`least_significant_digit`**: If specified, variable data will be
        truncated (quantized). In conjunction with `zlib=True` this produces
        'lossy', but significantly more efficient compression. For example, if
        `least_significant_digit=1`, data will be quantized using
        around(scale*data)/scale, where scale = 2**bits, and bits is determined
        so that a precision of 0.1 is retained (in this case bits=4). Default is
        `None`, or no quantization.
        
        **`fill_value`**:  If specified, the default netCDF `_FillValue` (the
        value that the variable gets filled with before any data is written to it)
        is replaced with this value.  If fill_value is set to `False`, then
        the variable is not pre-filled. The default netCDF fill values can be found
        in `netCDF4.default_fillvals`.

        ***Note***: `netCDF4.Variable` instances should be created using the
        `netCDF4.Dataset.createVariable` method of a `netCDF4.Dataset` or
        `netCDF4.Group` instance, not using this class directly.
        """
        cdef int ierr, ndims, icontiguous, ideflate_level, numdims, _grpid
        cdef char *varname
        cdef nc_type xtype
        cdef int *dimids
        cdef size_t sizep, nelemsp
        cdef size_t *chunksizesp
        cdef float preemptionp
        # flag to indicate that orthogonal indexing is supported
        self.__orthogonal_indexing__ = True
        # if complevel is set to zero, set zlib to False.
        if not complevel:
            zlib = False
        # if dimensions is a string, convert to a tuple
        # this prevents a common error that occurs when
        # dimensions = 'lat' instead of ('lat',)
        if type(dimensions) == str or type(dimensions) == bytes or type(dimensions) == unicode:
            dimensions = dimensions,
        self._grpid = grp._grpid
        # make a weakref to group to avoid circular ref (issue 218)
        # keep strong reference the default behaviour (issue 251)
        if grp.keepweakref:
            self._grp = weakref.proxy(grp)
        else:
            self._grp = grp
        user_type = isinstance(datatype, CompoundType) or \
                    isinstance(datatype, VLType) or \
                    isinstance(datatype, EnumType) or \
                    datatype == str
        # convert to a real numpy datatype object if necessary.
        if not user_type and type(datatype) != numpy.dtype:
            datatype = numpy.dtype(datatype)
        # convert numpy string dtype with length > 1
        # or any numpy unicode dtype into str
        if (isinstance(datatype, numpy.dtype) and
            ((datatype.kind == 'S' and datatype.itemsize > 1) or
              datatype.kind == 'U')):
            datatype = str
            user_type = True
        # check if endian keyword consistent with datatype specification.
        dtype_endian = getattr(datatype,'byteorder',None)
        if dtype_endian == '=': dtype_endian='native'
        if dtype_endian == '>': dtype_endian='big'
        if dtype_endian == '<': dtype_endian='little'
        if dtype_endian == '|': dtype_endian=None
        if dtype_endian is not None and dtype_endian != endian:
            if dtype_endian == 'native' and endian == sys.byteorder:
                pass
            else:
                # endian keyword prevails, issue warning
                msg = 'endian-ness of dtype and endian kwarg do not match, using endian kwarg'
                #msg = 'endian-ness of dtype and endian kwarg do not match, dtype over-riding endian kwarg'
                warnings.warn(msg)
                #endian = dtype_endian # dtype prevails
        # check validity of datatype.
        self._isprimitive = False
        self._iscompound = False
        self._isvlen = False
        self._isenum = False
        if user_type:
            if isinstance(datatype, CompoundType):
                self._iscompound = True
                self._cmptype = datatype
            if isinstance(datatype, VLType) or datatype==str:
                self._isvlen = True
                self._vltype = datatype
            if isinstance(datatype, EnumType):
                self._isenum = True
                self._enumtype = datatype
            if datatype==str:
                if grp.data_model != 'NETCDF4':
                    raise ValueError(
                        'Variable length strings are only supported for the '
                        'NETCDF4 format. For other formats, consider using '
                        'netCDF4.stringtochar to convert string arrays into '
                        'character arrays with an additional dimension.')
                datatype = VLType(self._grp, str, None)
                self._vltype = datatype
            xtype = datatype._nc_type
            # dtype variable attribute is a numpy datatype object.
            self.dtype = datatype.dtype
        elif datatype.str[1:] in _supportedtypes:
            self._isprimitive = True
            # find netCDF primitive data type corresponding to
            # specified numpy data type.
            xtype = _nptonctype[datatype.str[1:]]
            # dtype variable attribute is a numpy datatype object.
            self.dtype = datatype
        else:
            raise TypeError('illegal primitive data type, must be one of %s, got %s' % (_supportedtypes,datatype))
        if 'id' in kwargs:
            self._varid = kwargs['id']
        else:
            bytestr = _strencode(name)
            varname = bytestr
            ndims = len(dimensions)
            # find dimension ids.
            if ndims:
                dims = []
                dimids = <int *>malloc(sizeof(int) * ndims)
                for n from 0 <= n < ndims:
                    dimname = dimensions[n]
                    # look for dimension in this group, and if not
                    # found there, look in parent (and it's parent, etc, back to root).
                    dim = _find_dim(grp, dimname)
                    if dim is None:
                        raise KeyError("dimension %s not defined in group %s or any group in it's family tree" % (dimname, grp.path))
                    dimids[n] = dim._dimid
                    dims.append(dim)
            # go into define mode if it's a netCDF 3 compatible
            # file format.  Be careful to exit define mode before
            # any exceptions are raised.
            if grp.data_model != 'NETCDF4': grp._redef()
            # define variable.
            if ndims:
                ierr = nc_def_var(self._grpid, varname, xtype, ndims,
                                  dimids, &self._varid)
                free(dimids)
            else: # a scalar variable.
                ierr = nc_def_var(self._grpid, varname, xtype, ndims,
                                  NULL, &self._varid)
            # set chunk cache size if desired
            # default is 1mb per var, can cause problems when many (1000's)
            # of vars are created.  This change only lasts as long as file is
            # open.
            if grp.data_model.startswith('NETCDF4') and chunk_cache is not None:
                ierr = nc_get_var_chunk_cache(self._grpid, self._varid, &sizep,
                        &nelemsp, &preemptionp)
                if ierr != NC_NOERR:
                    raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
                # reset chunk cache size, leave other parameters unchanged.
                sizep = chunk_cache
                ierr = nc_set_var_chunk_cache(self._grpid, self._varid, sizep,
                        nelemsp, preemptionp)
                if ierr != NC_NOERR:
                    raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
            if ierr != NC_NOERR:
                if grp.data_model != 'NETCDF4': grp._enddef()
                raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
            # set zlib, shuffle, chunking, fletcher32 and endian
            # variable settings.
            # don't bother for NETCDF3* formats.
            # for NETCDF3* formats, the zlib,shuffle,chunking,
            # and fletcher32 are silently ignored. Only
            # endian='native' allowed for NETCDF3.
            if grp.data_model in ['NETCDF4','NETCDF4_CLASSIC']:
                # set zlib and shuffle parameters.
                if zlib and ndims: # don't bother for scalar variable
                    ideflate_level = complevel
                    if shuffle:
                        ierr = nc_def_var_deflate(self._grpid, self._varid, 1, 1, ideflate_level)
                    else:
                        ierr = nc_def_var_deflate(self._grpid, self._varid, 0, 1, ideflate_level)
                    if ierr != NC_NOERR:
                        if grp.data_model != 'NETCDF4': grp._enddef()
                        raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
                # set checksum.
                if fletcher32 and ndims: # don't bother for scalar variable
                    ierr = nc_def_var_fletcher32(self._grpid, self._varid, 1)
                    if ierr != NC_NOERR:
                        if grp.data_model != 'NETCDF4': grp._enddef()
                        raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
                # set chunking stuff.
                if ndims: # don't bother for scalar variable.
                    if contiguous:
                        icontiguous = NC_CONTIGUOUS
                        if chunksizes is not None:
                            raise ValueError('cannot specify chunksizes for a contiguous dataset')
                    else:
                        icontiguous = NC_CHUNKED
                    if chunksizes is None:
                        chunksizesp = NULL
                    else:
                        if len(chunksizes) != len(dimensions):
                            if grp.data_model != 'NETCDF4': grp._enddef()
                            raise ValueError('chunksizes must be a sequence with the same length as dimensions')
                        chunksizesp = <size_t *>malloc(sizeof(size_t) * ndims)
                        for n from 0 <= n < ndims:
                            if not dims[n].isunlimited() and \
                               chunksizes[n] > dims[n].size:
                                msg = 'chunksize cannot exceed dimension size'
                                raise ValueError(msg)
                            chunksizesp[n] = chunksizes[n]
                    if chunksizes is not None or contiguous:
                        ierr = nc_def_var_chunking(self._grpid, self._varid, icontiguous, chunksizesp)
                        free(chunksizesp)
                        if ierr != NC_NOERR:
                            if grp.data_model != 'NETCDF4': grp._enddef()
                            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
                # set endian-ness of variable
                if endian == 'little':
                    ierr = nc_def_var_endian(self._grpid, self._varid, NC_ENDIAN_LITTLE)
                elif endian == 'big':
                    ierr = nc_def_var_endian(self._grpid, self._varid, NC_ENDIAN_BIG)
                elif endian == 'native':
                    pass # this is the default format.
                else:
                    raise ValueError("'endian' keyword argument must be 'little','big' or 'native', got '%s'" % endian)
                if ierr != NC_NOERR:
                    if grp.data_model != 'NETCDF4': grp._enddef()
                    raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
            else:
                if endian != 'native':
                    msg="only endian='native' allowed for NETCDF3 files"
                    raise RuntimeError(msg)
            # set a fill value for this variable if fill_value keyword
            # given.  This avoids the HDF5 overhead of deleting and
            # recreating the dataset if it is set later (after the enddef).
            if fill_value is not None:
                if not fill_value and isinstance(fill_value,bool):
                    # no filling for this variable if fill_value==False.
                    if not self._isprimitive:
                        # no fill values for VLEN and compound variables
                        # anyway.
                        ierr = 0
                    else:
                        ierr = nc_def_var_fill(self._grpid, self._varid, 1, NULL)
                    if ierr != NC_NOERR:
                        if grp.data_model != 'NETCDF4': grp._enddef()
                        raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
                else:
                    # cast fill_value to type of variable.
                    if self._isprimitive or self._isenum:
                        fillval = numpy.array(fill_value, self.dtype)
                        _set_att(self._grp, self._varid, '_FillValue',\
                                 fillval, xtype=xtype)
                    else:
                        raise AttributeError("cannot set _FillValue attribute for VLEN or compound variable")
            if least_significant_digit is not None:
                self.least_significant_digit = least_significant_digit
            # leave define mode if not a NETCDF4 format file.
            if grp.data_model != 'NETCDF4': grp._enddef()
        # count how many unlimited dimensions there are.
        self._nunlimdim = 0
        for dimname in dimensions:
            # look in current group, and parents for dim.
            dim = _find_dim(self._grp, dimname)
            if dim.isunlimited(): self._nunlimdim = self._nunlimdim + 1
        # set ndim attribute (number of dimensions).
        with nogil:
            ierr = nc_inq_varndims(self._grpid, self._varid, &numdims)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        self.ndim = numdims
        self._name = name
        # default for automatically applying scale_factor and
        # add_offset, and converting to/from masked arrays is True.
        self.scale = True
        self.mask = True
        if 'least_significant_digit' in self.ncattrs():
            self._has_lsd = True

    def __array__(self):
        # numpy special method that returns a numpy array.
        # allows numpy ufuncs to work faster on Variable objects
        # (issue 216).
        return self[...]

    def __repr__(self):
        if python3:
            return self.__unicode__()
        else:
            return unicode(self).encode(default_encoding)

    def __unicode__(self):
        cdef int ierr, no_fill
        if not dir(self._grp):
            return 'Variable object no longer valid'
        ncdump_var = ['%r\n' % type(self)]
        dimnames = tuple([_tostr(dimname) for dimname in self.dimensions])
        attrs = ['    %s: %s\n' % (name,self.getncattr(name)) for name in\
                self.ncattrs()]
        if self._iscompound:
            ncdump_var.append('%s %s(%s)\n' %\
            ('compound',self._name,', '.join(dimnames)))
        elif self._isvlen:
            ncdump_var.append('%s %s(%s)\n' %\
            ('vlen',self._name,', '.join(dimnames)))
        elif self._isenum:
            ncdump_var.append('%s %s(%s)\n' %\
            ('enum',self._name,', '.join(dimnames)))
        else:
            ncdump_var.append('%s %s(%s)\n' %\
            (self.dtype,self._name,', '.join(dimnames)))
        ncdump_var = ncdump_var + attrs
        if self._iscompound:
            ncdump_var.append('compound data type: %s\n' % self.dtype)
        elif self._isvlen:
            ncdump_var.append('vlen data type: %s\n' % self.dtype)
        elif self._isenum:
            ncdump_var.append('enum data type: %s\n' % self.dtype)
        unlimdims = []
        for dimname in self.dimensions:
            dim = _find_dim(self._grp, dimname)
            if dim.isunlimited():
                unlimdims.append(dimname)
        if (self._grp.path != '/'): ncdump_var.append('path = %s\n' % self._grp.path)
        ncdump_var.append('unlimited dimensions: %s\n' % ', '.join(unlimdims))
        ncdump_var.append('current shape = %s\n' % repr(self.shape))
        with nogil:
            ierr = nc_inq_var_fill(self._grpid,self._varid,&no_fill,NULL)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        if self._isprimitive:
            if no_fill != 1:
                try:
                    fillval = self._FillValue
                    msg = 'filling on'
                except AttributeError:
                    fillval = default_fillvals[self.dtype.str[1:]]
                    if self.dtype.str[1:] in ['u1','i1']:
                        msg = 'filling on, default _FillValue of %s ignored\n' % fillval
                    else:
                        msg = 'filling on, default _FillValue of %s used\n' % fillval
                ncdump_var.append(msg)
            else:
                ncdump_var.append('filling off\n')


        return ''.join(ncdump_var)

    def _getdims(self):
        # Private method to get variables's dimension names
        cdef int ierr, numdims, n, nn
        cdef char namstring[NC_MAX_NAME+1]
        cdef int *dimids
        # get number of dimensions for this variable.
        with nogil:
            ierr = nc_inq_varndims(self._grpid, self._varid, &numdims)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        dimids = <int *>malloc(sizeof(int) * numdims)
        # get dimension ids.
        with nogil:
            ierr = nc_inq_vardimid(self._grpid, self._varid, dimids)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        # loop over dimensions, retrieve names.
        dimensions = ()
        for nn from 0 <= nn < numdims:
            with nogil:
                ierr = nc_inq_dimname(self._grpid, dimids[nn], namstring)
            if ierr != NC_NOERR:
                raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
            name = namstring.decode(default_encoding,unicode_error)
            dimensions = dimensions + (name,)
        free(dimids)
        return dimensions

    def _getname(self):
        # Private method to get name associated with instance
        cdef int err, _grpid
        cdef char namstring[NC_MAX_NAME+1]
        _grpid = self._grp._grpid
        with nogil:
            ierr = nc_inq_varname(_grpid, self._varid, namstring)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        return namstring.decode(default_encoding,unicode_error)

    property name:
        """string name of Variable instance"""
        def __get__(self):
            return self._getname()
        def __set__(self,value):
            raise AttributeError("name cannot be altered")

    property datatype:
        """numpy data type (for primitive data types) or
        VLType/CompoundType/EnumType instance 
        (for compound, vlen  or enum data types)"""
        def __get__(self):
            if self._iscompound:
                return self._cmptype
            elif self._isvlen:
                return self._vltype
            elif self._isenum:
                return self._enumtype
            elif self._isprimitive:
                return self.dtype

    property shape:
        """find current sizes of all variable dimensions"""
        def __get__(self):
            shape = ()
            for dimname in self._getdims():
                # look in current group, and parents for dim.
                dim = _find_dim(self._grp,dimname)
                shape = shape + (len(dim),)
            return shape
        def __set__(self,value):
            raise AttributeError("shape cannot be altered")

    property size:
        """Return the number of stored elements."""
        def __get__(self):
            return numpy.prod(self.shape)

    property dimensions:
        """get variables's dimension names"""
        def __get__(self):
            return self._getdims()
        def __set__(self,value):
            raise AttributeError("dimensions cannot be altered")


    def group(self):
        """
**`group(self)`**

return the group that this `netCDF4.Variable` is a member of."""
        return self._grp

    def ncattrs(self):
        """
**`ncattrs(self)`**

return netCDF attribute names for this `netCDF4.Variable` in a list."""
        return _get_att_names(self._grpid, self._varid)

    def setncattr(self,name,value):
        """
**`setncattr(self,name,value)`**

set a netCDF variable attribute using name,value pair.  Use if you need to set a
netCDF attribute with the same name as one of the reserved python
attributes."""
        if self._grp.data_model != 'NETCDF4': self._grp._redef()
        _set_att(self._grp, self._varid, name, value)
        if self._grp.data_model != 'NETCDF4': self._grp._enddef()

    def setncattr_string(self,name,value):
        """
**`setncattr_string(self,name,value)`**

set a netCDF variable string attribute using name,value pair.
Use if you need to ensure that a netCDF attribute is created with type
`NC_STRING` if the file format is `NETCDF4`."""
        cdef nc_type xtype
        xtype=-99
        if self._grp.data_model != 'NETCDF4':
            msg='file format does not support NC_STRING attributes'
            raise IOError(msg)
        _set_att(self._grp, self._varid, name, str(value), xtype=xtype, force_ncstring=True)

    def setncatts(self,attdict):
        """
**`setncatts(self,attdict)`**

set a bunch of netCDF variable attributes at once using a python dictionary.
This may be faster when setting a lot of attributes for a `NETCDF3`
formatted file, since nc_redef/nc_enddef is not called in between setting
each attribute"""
        if self._grp.data_model != 'NETCDF4': self._grp._redef()
        for name, value in attdict.items():
            _set_att(self._grp, self._varid, name, value)
        if self._grp.data_model != 'NETCDF4': self._grp._enddef()

    def getncattr(self,name):
        """
**`getncattr(self,name)`**

retrievel a netCDF variable attribute.  Use if you need to set a
netCDF attribute with the same name as one of the reserved python
attributes."""
        return _get_att(self._grp, self._varid, name)

    def delncattr(self, name):
        """
**`delncattr(self,name,value)`**

delete a netCDF variable attribute.  Use if you need to delete a
netCDF attribute with the same name as one of the reserved python
attributes."""
        cdef char *attname
        bytestr = _strencode(name)
        attname = bytestr
        if self._grp.data_model != 'NETCDF4': self._grp._redef()
        ierr = nc_del_att(self._grpid, self._varid, attname)
        if self._grp.data_model != 'NETCDF4': self._grp._enddef()
        if ierr != NC_NOERR:
            raise AttributeError((<char *>nc_strerror(ierr)).decode('ascii'))

    def filters(self):
        """
**`filters(self)`**

return dictionary containing HDF5 filter parameters."""
        cdef int ierr,ideflate,ishuffle,ideflate_level,ifletcher32
        filtdict = {'zlib':False,'shuffle':False,'complevel':0,'fletcher32':False}
        if self._grp.data_model not in ['NETCDF4_CLASSIC','NETCDF4']: return
        with nogil:
            ierr = nc_inq_var_deflate(self._grpid, self._varid, &ishuffle, &ideflate, &ideflate_level)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        with nogil:
            ierr = nc_inq_var_fletcher32(self._grpid, self._varid, &ifletcher32)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        if ideflate:
            filtdict['zlib']=True
            filtdict['complevel']=ideflate_level
        if ishuffle:
            filtdict['shuffle']=True
        if ifletcher32:
            filtdict['fletcher32']=True
        return filtdict

    def endian(self):
        """
**`endian(self)`**

return endian-ness (`little,big,native`) of variable (as stored in HDF5 file)."""
        cdef int ierr, iendian
        if self._grp.data_model not in ['NETCDF4_CLASSIC','NETCDF4']:
            return 'native'
        with nogil:
            ierr = nc_inq_var_endian(self._grpid, self._varid, &iendian)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        if iendian == NC_ENDIAN_LITTLE:
            return 'little'
        elif iendian == NC_ENDIAN_BIG:
            return 'big'
        else:
            return 'native'

    def chunking(self):
        """
**`chunking(self)`**

return variable chunking information.  If the dataset is
defined to be contiguous (and hence there is no chunking) the word 'contiguous'
is returned.  Otherwise, a sequence with the chunksize for
each dimension is returned."""
        cdef int ierr, icontiguous, ndims
        cdef size_t *chunksizesp
        if self._grp.data_model not in ['NETCDF4_CLASSIC','NETCDF4']: return None
        ndims = self.ndim
        chunksizesp = <size_t *>malloc(sizeof(size_t) * ndims)
        with nogil:
            ierr = nc_inq_var_chunking(self._grpid, self._varid, &icontiguous, chunksizesp)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        chunksizes=[]
        for n from 0 <= n < ndims:
            chunksizes.append(chunksizesp[n])
        free(chunksizesp)
        if icontiguous:
            return 'contiguous'
        else:
            return chunksizes

    def get_var_chunk_cache(self):
        """
**`get_var_chunk_cache(self)`**

return variable chunk cache information in a tuple (size,nelems,preemption).
See netcdf C library documentation for `nc_get_var_chunk_cache` for
details."""
        cdef int ierr
        cdef size_t sizep, nelemsp
        cdef float preemptionp
        with nogil:
            ierr = nc_get_var_chunk_cache(self._grpid, self._varid, &sizep,
                   &nelemsp, &preemptionp)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        size = sizep; nelems = nelemsp; preemption = preemptionp
        return (size,nelems,preemption)

    def set_var_chunk_cache(self,size=None,nelems=None,preemption=None):
        """
**`set_var_chunk_cache(self,size=None,nelems=None,preemption=None)`**

change variable chunk cache settings.
See netcdf C library documentation for `nc_set_var_chunk_cache` for
details."""
        cdef int ierr
        cdef size_t sizep, nelemsp
        cdef float preemptionp
        # reset chunk cache size, leave other parameters unchanged.
        size_orig, nelems_orig, preemption_orig = self.get_var_chunk_cache()
        if size is not None:
            sizep = size
        else:
            sizep = size_orig
        if nelems is not None:
            nelemsp = nelems
        else:
            nelemsp = nelems_orig
        if preemption is not None:
            preemptionp = preemption
        else:
            preemptionp = preemption_orig
        ierr = nc_set_var_chunk_cache(self._grpid, self._varid, sizep,
               nelemsp, preemptionp)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))

    def __delattr__(self,name):
        # if it's a netCDF attribute, remove it
        if name not in _private_atts:
            self.delncattr(name)
        else:
            raise AttributeError(
            "'%s' is one of the reserved attributes %s, cannot delete. Use delncattr instead." % (name, tuple(_private_atts)))

    def __setattr__(self,name,value):
        # if name in _private_atts, it is stored at the python
        # level and not in the netCDF file.
        if name not in _private_atts:
            # if setting _FillValue or missing_value, make sure value
            # has same type as variable.
            if name == '_FillValue':
                msg='_FillValue attribute must be set when variable is '+\
                'created (using fill_value keyword to createVariable)'
                raise AttributeError(msg)
                #if self._isprimitive:
                #    value = numpy.array(value, self.dtype)
                #else:
                #    msg="cannot set _FillValue attribute for "+\
                #    "VLEN or compound variable"
                #    raise AttributeError(msg)
            elif name == 'missing_value' and self._isprimitive:
                value = numpy.array(value, self.dtype)
            self.setncattr(name, value)
        elif not name.endswith('__'):
            if hasattr(self,name):
                raise AttributeError(
                "'%s' is one of the reserved attributes %s, cannot rebind. Use setncattr instead." % (name, tuple(_private_atts)))
            else:
                self.__dict__[name]=value

    def __getattr__(self,name):
        # if name in _private_atts, it is stored at the python
        # level and not in the netCDF file.
        if name.startswith('__') and name.endswith('__'):
            # if __dict__ requested, return a dict with netCDF attributes.
            if name == '__dict__':
                names = self.ncattrs()
                values = []
                for name in names:
                    values.append(_get_att(self._grp, self._varid, name))
                return OrderedDict(zip(names,values))
            else:
                raise AttributeError
        elif name in _private_atts:
            return self.__dict__[name]
        else:
            return self.getncattr(name)

    def renameAttribute(self, oldname, newname):
        """
**`renameAttribute(self, oldname, newname)`**

rename a `netCDF4.Variable` attribute named `oldname` to `newname`."""
        cdef int ierr
        cdef char *oldnamec
        cdef char *newnamec
        bytestr = _strencode(oldname)
        oldnamec = bytestr
        bytestr = _strencode(newname)
        newnamec = bytestr
        ierr = nc_rename_att(self._grpid, self._varid, oldnamec, newnamec)
        if ierr != NC_NOERR:
            raise AttributeError((<char *>nc_strerror(ierr)).decode('ascii'))

    def __getitem__(self, elem):
        # This special method is used to index the netCDF variable
        # using the "extended slice syntax". The extended slice syntax
        # is a perfect match for the "start", "count" and "stride"
        # arguments to the nc_get_var() function, and is much more easy
        # to use.
        start, count, stride, put_ind = _StartCountStride(elem,self.shape)
        datashape = _out_array_shape(count)
        if self._isvlen:
            data = numpy.empty(datashape, dtype='O')
        else:
            data = numpy.empty(datashape, dtype=self.dtype)

        # Determine which dimensions need to be
        # squeezed (those for which elem is an integer scalar).
        # The convention used is that for those cases,
        # put_ind for this dimension is set to -1 by _StartCountStride.
        squeeze = data.ndim * [slice(None),]
        for i,n in enumerate(put_ind.shape[:-1]):
            if n == 1 and put_ind[...,i].ravel()[0] == -1:
                squeeze[i] = 0

        # Reshape the arrays so we can iterate over them.
        start = start.reshape((-1, self.ndim or 1))
        count = count.reshape((-1, self.ndim or 1))
        stride = stride.reshape((-1, self.ndim or 1))
        put_ind = put_ind.reshape((-1, self.ndim or 1))

        # Fill output array with data chunks.
        for (a,b,c,i) in zip(start, count, stride, put_ind):
            datout = self._get(a,b,c)
            if not hasattr(datout,'shape') or data.shape == datout.shape:
                data = datout
            else:
                shape = getattr(data[tuple(i)], 'shape', ())
                if self._isvlen and not len(self.dimensions):
                    # special case of scalar VLEN
                    data[0] = datout
                else:
                    data[tuple(i)] = datout.reshape(shape)

        # Remove extra singleton dimensions.
        if hasattr(data,'shape'):
            data = data[tuple(squeeze)]
        if hasattr(data,'ndim') and self.ndim == 0:
            # Make sure a numpy scalar array is returned instead of a 1-d array of
            # length 1.
            if data.ndim != 0: data = numpy.asarray(data[0])

        # if auto_scale mode set to True, (through
        # a call to set_auto_scale or set_auto_maskandscale),
        # perform automatic unpacking using scale_factor/add_offset.
        # if auto_mask mode is set to True (through a call to
        # set_auto_mask or set_auto_maskandscale), perform
        # automatic conversion to masked array using
        # missing_value/_Fill_Value.
        # ignore for compound, vlen or enum datatypes.
        try: # check to see if scale_factor and add_offset is valid (issue 176).
            if hasattr(self,'scale_factor'): float(self.scale_factor)
            if hasattr(self,'add_offset'): float(self.add_offset)
            valid_scaleoffset = True
        except:
            valid_scaleoffset = False
            if self.scale:
                msg = 'invalid scale_factor or add_offset attribute, no unpacking done...'
                warnings.warn(msg)
        if self.mask and (self._isprimitive or self._isenum):
            data = self._toma(data)
        if self.scale and self._isprimitive and valid_scaleoffset:
            # if variable has scale_factor and add_offset attributes, rescale.
            if hasattr(self, 'scale_factor') and hasattr(self, 'add_offset') and\
            (self.add_offset != 0.0 or self.scale_factor != 1.0):
                data = data*self.scale_factor + self.add_offset
            # else if variable has only scale_factor attributes, rescale.
            elif hasattr(self, 'scale_factor') and self.scale_factor != 1.0:
                data = data*self.scale_factor
            # else if variable has only add_offset attributes, rescale.
            elif hasattr(self, 'add_offset') and self.add_offset != 0.0:
                data = data + self.add_offset
        return data

    def _toma(self,data):
        cdef int ierr, no_fill
        # private function for creating a masked array, masking missing_values
        # and/or _FillValues.
        totalmask = numpy.zeros(data.shape, numpy.bool)
        fill_value = None
        if hasattr(self, 'missing_value'):
            mval = numpy.array(self.missing_value, self.dtype)
            if (self.endian() == 'big' and is_native_little) or\
               (self.endian() == 'little' and is_native_big):
                mval.byteswap(True)
            if mval.shape == (): # mval a scalar.
                hasmval = data==mval
                # is scalar missing value a NaN?
                try:
                    mvalisnan = numpy.isnan(mval)
                except TypeError: # isnan fails on some dtypes (issue 206)
                    mvalisnan = False
            else: # mval a vector.
                hasmval = numpy.zeros(data.shape, numpy.bool)
                for m in mval:
                    m =  numpy.array(m)
                    hasmval += data == m
            if mval.shape == () and mvalisnan:
                mask = numpy.isnan(data)
            elif hasmval.any():
                mask = hasmval
            else:
                mask = None
            if mask is not None:
                fill_value = mval
                totalmask += mask
        if hasattr(self, '_FillValue'):
            fval = numpy.array(self._FillValue, self.dtype)
            # byte swap the _FillValue if endian-ness of the variable
            # is not native.
            if (self.endian() == 'big' and is_native_little) or\
               (self.endian() == 'little' and is_native_big):
                fval.byteswap(True)
            # is _FillValue a NaN?
            try:
                fvalisnan = numpy.isnan(fval)
            except: # isnan fails on some dtypes (issue 202)
                fvalisnan = False
            if fvalisnan:
                mask = numpy.isnan(data)
            elif (data == fval).any():
                mask = data==fval
            else:
                mask = None
            if mask is not None:
                if fill_value is None:
                    fill_value = fval
                totalmask += mask
        # issue 209: don't return masked array if variable filling
        # is disabled.
        else:
            with nogil:
                ierr = nc_inq_var_fill(self._grpid,self._varid,&no_fill,NULL)
            if ierr != NC_NOERR:
                raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
            # if no_fill is not 1, and not a byte variable, then use default fill value.
            # from http://www.unidata.ucar.edu/software/netcdf/docs/netcdf-c/Fill-Values.html#Fill-Values
            # "If you need a fill value for a byte variable, it is recommended
            # that you explicitly define an appropriate _FillValue attribute, as
            # generic utilities such as ncdump will not assume a default fill
            # value for byte variables."
            # Explained here too:
            # http://www.unidata.ucar.edu/software/netcdf/docs/known_problems.html#ncdump_ubyte_fill
            # "There should be no default fill values when reading any byte
            # type, signed or unsigned, because the byte ranges are too
            # small to assume one of the values should appear as a missing
            # value unless a _FillValue attribute is set explicitly."
            if no_fill != 1 and self.dtype.str[1:] not in ['u1','i1']:
                fillval = numpy.array(default_fillvals[self.dtype.str[1:]],self.dtype)
                # byte swap the _FillValue if endian-ness of the variable
                # is not native.
                if (self.endian() == 'big' and is_native_little) or\
                   (self.endian() == 'little' and is_native_big):
                    fillval.byteswap(True)
                has_fillval = data == fillval
                # if data is an array scalar, has_fillval will be a boolean.
                # in that case convert to an array.
                if type(has_fillval) == bool: has_fillval=numpy.asarray(has_fillval)
                if has_fillval.any():
                    mask=data==fillval
                    if fill_value is None:
                        fill_value = fillval
                    totalmask += mask
        # all values where data == missing_value or _FillValue are
        # masked.  fill_value set to missing_value if it exists,
        # otherwise _FillValue.
        if fill_value is not None:
            data = ma.masked_array(data,mask=totalmask,fill_value=fill_value)
            # issue 515 scalar array with mask=True should be converted
            # to numpy.ma.MaskedConstant to be consistent with slicing
            # behavior of masked arrays.
            if data.shape == () and data.mask.all():
                data = data[...]
        return data

    def _assign_vlen(self, elem, data):
        """private method to assign data to a single item in a VLEN variable"""
        cdef size_t *startp
        cdef size_t *countp
        cdef int ndims, n
        cdef nc_vlen_t *vldata
        cdef char **strdata
        cdef ndarray data2
        if not self._isvlen:
            raise TypeError('_assign_vlen method only for use with VLEN variables')
        ndims = self.ndim
        msg="data can only be assigned to VLEN variables using integer indices"
        # check to see that elem is a tuple of integers.
        # handle negative integers.
        if isinstance(elem, int):
            if ndims > 1:
                raise IndexError(msg)
            if elem < 0:
                if self.shape[0]+elem >= 0:
                    elem = self.shape[0]+elem
                else:
                    raise IndexError("Illegal index")
        elif isinstance(elem, tuple):
            if len(elem) != ndims:
                raise IndexError("Illegal index")
            elemnew = []
            for n,e in enumerate(elem):
                if not isinstance(e, int):
                    raise IndexError(msg)
                elif e < 0:
                    enew = self.shape[n]+e
                    if enew < 0:
                        raise IndexError("Illegal index")
                    else:
                        elemnew.append(self.shape[n]+e)
                else:
                    elemnew.append(e)
            elem = tuple(elemnew)
        else:
            raise IndexError(msg)
        # set start, count
        if isinstance(elem, tuple):
            start = list(elem)
        else:
            start = [elem]
        count = [1]*ndims
        startp = <size_t *>malloc(sizeof(size_t) * ndims)
        countp = <size_t *>malloc(sizeof(size_t) * ndims)
        for n from 0 <= n < ndims:
            startp[n] = start[n]
            countp[n] = count[n]
        if self.dtype == str: # VLEN string
            strdata = <char **>malloc(sizeof(char *))
            bytestr = _strencode(data)
            strdata[0] = bytestr
            ierr = nc_put_vara(self._grpid, self._varid,
                               startp, countp, strdata)
            if ierr != NC_NOERR:
                raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
            free(strdata)
        else: # regular VLEN
            if data.dtype != self.dtype:
                raise TypeError("wrong data type: should be %s, got %s" % (self.dtype,data.dtype))
            data2 = data
            vldata = <nc_vlen_t *>malloc(sizeof(nc_vlen_t))
            vldata[0].len = PyArray_SIZE(data2)
            vldata[0].p = data2.data
            ierr = nc_put_vara(self._grpid, self._varid,
                               startp, countp, vldata)
            if ierr != NC_NOERR:
                raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
            free(vldata)
        free(startp)
        free(countp)

    def __setitem__(self, elem, data):
        # This special method is used to assign to the netCDF variable
        # using "extended slice syntax". The extended slice syntax
        # is a perfect match for the "start", "count" and "stride"
        # arguments to the nc_put_var() function, and is much more easy
        # to use.

        if self._isvlen: # if vlen, should be object array (don't try casting)
            if self.dtype == str:
                # for string vars, if data is not an array
                # assume it is a python string and raise an error
                # if it is an array, but not an object array.
                if not hasattr(data,'ndim'):
                    # issue 458, allow Ellipsis to be used for scalar var
                    if type(elem) == type(Ellipsis) and not\
                       len(self.dimensions): elem = 0
                    self._assign_vlen(elem, data)
                    return
                elif data.dtype.kind in ['S', 'U']:
                    if ma.isMA(data):
                        msg='masked arrays cannot be assigned by VLEN str slices'
                        raise TypeError(msg)
                    data = data.astype(object)
                elif data.dtype.kind != 'O':
                    msg = ('only numpy string, unicode or object arrays can '
                           'be assigned to VLEN str var slices')
                    raise TypeError(msg)
            else:
                # for non-string vlen arrays, if data is not multi-dim, or
                # not an object array, assume it represents a single element
                # of the vlen var.
                if not hasattr(data,'ndim') or data.dtype.kind != 'O':
                    # issue 458, allow Ellipsis to be used for scalar var
                    if type(elem) == type(Ellipsis) and not\
                       len(self.dimensions): elem = 0
                    self._assign_vlen(elem, data)
                    return

        # A numpy array is needed. Convert if necessary.
        # assume it's a numpy or masked array if it has an 'ndim' attribute.
        if not hasattr(data,'ndim'):
            # if auto scaling is to be done, don't cast to an integer yet.
            if self.scale and self.dtype.kind in 'iu' and \
               hasattr(self, 'scale_factor') or hasattr(self, 'add_offset'):
                data = numpy.array(data,numpy.float)
            else:
                data = numpy.array(data,self.dtype)

        # for Enum variable, make sure data is valid.
        if self._isenum:
            test = numpy.zeros(data.shape,numpy.bool)
            for val in self.datatype.enum_dict.values():
                test += data == val
            if not numpy.all(test):
                msg="trying to assign illegal value to Enum variable"
                raise ValueError(msg)

        start, count, stride, put_ind =\
        _StartCountStride(elem,self.shape,self.dimensions,self._grp,datashape=data.shape,put=True)
        datashape = _out_array_shape(count)

        # if a numpy scalar, create an array of the right size
        # and fill with scalar values.
        if data.shape == ():
            data = numpy.tile(data,datashape)
        # reshape data array by adding extra singleton dimensions
        # if needed to conform with start,count,stride.
        if len(data.shape) != len(datashape):
            # create a view so shape in caller is not modified (issue 90)
            data = data.view()
            data.shape = tuple(datashape)

        # Reshape these arrays so we can iterate over them.
        start = start.reshape((-1, self.ndim or 1))
        count = count.reshape((-1, self.ndim or 1))
        stride = stride.reshape((-1, self.ndim or 1))
        put_ind = put_ind.reshape((-1, self.ndim or 1))

        # quantize data if least_significant_digit attribute
        # exists (improves compression).
        if self._has_lsd:
            data = _quantize(data,self.least_significant_digit)
        # if auto_scale mode set to True, (through
        # a call to set_auto_scale or set_auto_maskandscale),
        # perform automatic unpacking using scale_factor/add_offset.
        # if auto_mask mode is set to True (through a call to
        # set_auto_mask or set_auto_maskandscale), perform
        # automatic conversion to masked array using
        # missing_value/_Fill_Value.
        # ignore if not a primitive or enum data type (not compound or vlen).
        if self.mask and (self._isprimitive or self._isenum):
            # use missing_value as fill value.
            # if no missing value set, use _FillValue.
            if hasattr(self, 'scale_factor') or hasattr(self, 'add_offset'):
                # if not masked, create a masked array.
                if not ma.isMA(data): data = self._toma(data)
        if self.scale and self._isprimitive:
            # pack non-masked values using scale_factor and add_offset
            if hasattr(self, 'scale_factor') and hasattr(self, 'add_offset'):
                data = (data - self.add_offset)/self.scale_factor
                if self.dtype.kind in 'iu': data = numpy.around(data)
            elif hasattr(self, 'scale_factor'):
                data = data/self.scale_factor
                if self.dtype.kind in 'iu': data = numpy.around(data)
            elif hasattr(self, 'add_offset'):
                data = data - self.add_offset
                if self.dtype.kind in 'iu': data = numpy.around(data)
            if ma.isMA(data):
                # if underlying data in masked regions of masked array
                # corresponds to missing values, don't fill masked array -
                # just use underlying data instead
                if hasattr(self, 'missing_value') and \
                   numpy.all(numpy.in1d(data.data[data.mask],self.missing_value)):
                    data = data.data
                else:
                    if hasattr(self, 'missing_value'):
                        # if missing value is a scalar, use it as fill_value.
                        # if missing value is a vector, raise an exception
                        # since we then don't know how to fill in masked values.
                        if numpy.array(self.missing_value).shape == ():
                            fillval = self.missing_value
                        else:
                            msg="cannot assign fill_value for masked array when missing_value attribute is not a scalar"
                            raise RuntimeError(msg)
                        if numpy.array(fillval).shape != ():
                            fillval = fillval[0]
                    elif hasattr(self, '_FillValue'):
                        fillval = self._FillValue
                    else:
                        fillval = default_fillvals[self.dtype.str[1:]]
                    data = data.filled(fill_value=fillval)

        # Fill output array with data chunks.
        for (a,b,c,i) in zip(start, count, stride, put_ind):
            dataput = data[tuple(i)]
            if dataput.size == 0: continue # nothing to write
            # convert array scalar to regular array with one element.
            if dataput.shape == ():
                if self._isvlen:
                    dataput=numpy.array(dataput,'O')
                else:
                    dataput=numpy.array(dataput,dataput.dtype)
            self._put(dataput,a,b,c)


    def __len__(self):
        if not self.shape:
            raise TypeError('len() of unsized object')
        else:
            return self.shape[0]


    def assignValue(self,val):
        """
**`assignValue(self, val)`**

assign a value to a scalar variable.  Provided for compatibility with
Scientific.IO.NetCDF, can also be done by assigning to an Ellipsis slice ([...])."""
        if len(self.dimensions):
            raise IndexError('to assign values to a non-scalar variable, use a slice')
        self[:]=val

    def getValue(self):
        """
**`getValue(self)`**

get the value of a scalar variable.  Provided for compatibility with
Scientific.IO.NetCDF, can also be done by slicing with an Ellipsis ([...])."""
        if len(self.dimensions):
            raise IndexError('to retrieve values from a non-scalar variable, use slicing')
        return self[slice(None)]

    def set_auto_maskandscale(self,maskandscale):
        """
**`set_auto_maskandscale(self,maskandscale)`**

turn on or off automatic conversion of variable data to and
from masked arrays and automatic packing/unpacking of variable
data using `scale_factor` and `add_offset` attributes.

If `maskandscale` is set to `True`, when data is read from a variable
it is converted to a masked array if any of the values are exactly
equal to the either the netCDF _FillValue or the value specified by the
missing_value variable attribute. The fill_value of the masked array
is set to the missing_value attribute (if it exists), otherwise
the netCDF _FillValue attribute (which has a default value
for each data type).  When data is written to a variable, the masked
array is converted back to a regular numpy array by replacing all the
masked values by the missing_value attribute of the variable (if it
exists).  If the variable has no missing_value attribute, the _FillValue
is used instead.

If `maskandscale` is set to `True`, and the variable has a
`scale_factor` or an `add_offset` attribute, then data read
from that variable is unpacked using::

    data = self.scale_factor*data + self.add_offset

When data is written to a variable it is packed using::

    data = (data - self.add_offset)/self.scale_factor

If either scale_factor is present, but add_offset is missing, add_offset
is assumed zero.  If add_offset is present, but scale_factor is missing,
scale_factor is assumed to be one.
For more information on how `scale_factor` and `add_offset` can be
used to provide simple compression, see the
[PSD metadata conventions](http://www.esrl.noaa.gov/psd/data/gridded/conventions/cdc_netcdf_standard.shtml).

The default value of `maskandscale` is `True`
(automatic conversions are performed).
        """
        if maskandscale:
            self.scale = True
            self.mask = True
        else:
            self.scale = False
            self.mask = False

    def set_auto_scale(self,scale):
        """
**`set_auto_scale(self,scale)`**

turn on or off automatic packing/unpacking of variable
data using `scale_factor` and `add_offset` attributes.

If `scale` is set to `True`, and the variable has a
`scale_factor` or an `add_offset` attribute, then data read
from that variable is unpacked using::

    data = self.scale_factor*data + self.add_offset

When data is written to a variable it is packed using::

    data = (data - self.add_offset)/self.scale_factor

If either scale_factor is present, but add_offset is missing, add_offset
is assumed zero.  If add_offset is present, but scale_factor is missing,
scale_factor is assumed to be one.
For more information on how `scale_factor` and `add_offset` can be
used to provide simple compression, see the
[PSD metadata conventions](http://www.esrl.noaa.gov/psd/data/gridded/conventions/cdc_netcdf_standard.shtml).

The default value of `scale` is `True`
(automatic conversions are performed).
        """
        if scale:
            self.scale = True
        else:
            self.scale = False

    def set_auto_mask(self,mask):
        """
**`set_auto_mask(self,mask)`**

turn on or off automatic conversion of variable data to and
from masked arrays .

If `mask` is set to `True`, when data is read from a variable
it is converted to a masked array if any of the values are exactly
equal to the either the netCDF _FillValue or the value specified by the
missing_value variable attribute. The fill_value of the masked array
is set to the missing_value attribute (if it exists), otherwise
the netCDF _FillValue attribute (which has a default value
for each data type).  When data is written to a variable, the masked
array is converted back to a regular numpy array by replacing all the
masked values by the missing_value attribute of the variable (if it
exists).  If the variable has no missing_value attribute, the _FillValue
is used instead.

The default value of `mask` is `True`
(automatic conversions are performed).
        """
        if mask:
            self.mask = True
        else:
            self.mask = False


    def _put(self,ndarray data,start,count,stride):
        """Private method to put data into a netCDF variable"""
        cdef int ierr, ndims
        cdef npy_intp totelem
        cdef size_t *startp
        cdef size_t *countp
        cdef ptrdiff_t *stridep
        cdef char **strdata
        cdef void* elptr
        cdef char* databuff
        cdef ndarray dataarr
        cdef nc_vlen_t *vldata
        # rank of variable.
        ndims = len(self.dimensions)
        # make sure data is contiguous.
        # if not, make a local copy.
        if not PyArray_ISCONTIGUOUS(data):
            data = data.copy()
        # fill up startp,countp,stridep.
        totelem = 1
        negstride = 0
        sl = []
        startp = <size_t *>malloc(sizeof(size_t) * ndims)
        countp = <size_t *>malloc(sizeof(size_t) * ndims)
        stridep = <ptrdiff_t *>malloc(sizeof(ptrdiff_t) * ndims)
        for n from 0 <= n < ndims:
            count[n] = abs(count[n]) # make -1 into +1
            countp[n] = count[n]
            # for neg strides, reverse order (then flip that axis after data read in)
            if stride[n] < 0:
                negstride = 1
                stridep[n] = -stride[n]
                startp[n] = start[n]+stride[n]*(count[n]-1)
                stride[n] = -stride[n]
                sl.append(slice(None, None, -1)) # this slice will reverse the data
            else:
                startp[n] = start[n]
                stridep[n] = stride[n]
                sl.append(slice(None,None, 1))
            totelem = totelem*countp[n]
        # check to see that size of data array is what is expected
        # for slice given.
        dataelem = PyArray_SIZE(data)
        if totelem != dataelem:
            raise IndexError('size of data array does not conform to slice')
        if negstride:
            # reverse data along axes with negative strides.
            data = data[sl].copy() # make sure a copy is made.
        if self._isprimitive or self._iscompound or self._isenum:
            # primitive, enum or compound data type.
            # if data type of array doesn't match variable,
            # try to cast the data.
            if self.dtype != data.dtype:
                data = data.astype(self.dtype) # cast data, if necessary.
            # make sure byte-order of data matches byte-order of netcdf
            # variable.
            if self.endian() == 'native':
                if is_native_little and data.dtype.byteorder == '>':
                    data.byteswap(True)
                if is_native_big and data.dtype.byteorder == '<':
                    data.byteswap(True)
            if self.endian() == 'big':
                if is_native_big and data.dtype.byteorder not in ['=','|']:
                    data.byteswap(True)
                if is_native_little and data.dtype.byteorder == '=':
                    data.byteswap(True)
            if self.endian() == 'little':
                if is_native_little and data.dtype.byteorder not in ['=','|']:
                    data.byteswap(True)
                if is_native_big and data.dtype.byteorder == '=':
                    data.byteswap(True)
            # strides all 1 or scalar variable, use put_vara (faster)
            if sum(stride) == ndims or ndims == 0:
                ierr = nc_put_vara(self._grpid, self._varid,
                                   startp, countp, data.data)
            else:
                ierr = nc_put_vars(self._grpid, self._varid,
                                   startp, countp, stridep, data.data)
            if ierr != NC_NOERR:
                raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        elif self._isvlen:
            if data.dtype.char !='O':
                raise TypeError('data to put in string variable must be an object array containing Python strings')
            # flatten data array.
            data = data.flatten()
            if self.dtype == str:
                # convert all elements from strings to bytes
                for n in range(data.shape[0]):
                    data[n] = _strencode(data[n])
                # vlen string (NC_STRING)
                # loop over elements of object array, put data buffer for
                # each element in struct.
                # allocate struct array to hold vlen data.
                strdata = <char **>malloc(sizeof(char *)*totelem)
                for i from 0<=i<totelem:
                    strdata[i] = data[i]
                # strides all 1 or scalar variable, use put_vara (faster)
                if sum(stride) == ndims or ndims == 0:
                    ierr = nc_put_vara(self._grpid, self._varid,
                                       startp, countp, strdata)
                else:
                    raise IndexError('strides must all be 1 for string variables')
                    #ierr = nc_put_vars(self._grpid, self._varid,
                    #                   startp, countp, stridep, strdata)
                if ierr != NC_NOERR:
                    raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
                free(strdata)
            else:
                # regular vlen.
                # loop over elements of object array, put data buffer for
                # each element in struct.
                databuff = data.data
                # allocate struct array to hold vlen data.
                vldata = <nc_vlen_t *>malloc(<size_t>totelem*sizeof(nc_vlen_t))
                for i from 0<=i<totelem:
                    elptr = (<void**>databuff)[0]
                    dataarr = <ndarray>elptr
                    if self.dtype != dataarr.dtype.str[1:]:
                        #dataarr = dataarr.astype(self.dtype) # cast data, if necessary.
                        # casting doesn't work ?? just raise TypeError
                        raise TypeError("wrong data type in object array: should be %s, got %s" % (self.dtype,dataarr.dtype))
                    vldata[i].len = PyArray_SIZE(dataarr)
                    vldata[i].p = dataarr.data
                    databuff = databuff + data.strides[0]
                # strides all 1 or scalar variable, use put_vara (faster)
                if sum(stride) == ndims or ndims == 0:
                    ierr = nc_put_vara(self._grpid, self._varid,
                                       startp, countp, vldata)
                else:
                    raise IndexError('strides must all be 1 for vlen variables')
                    #ierr = nc_put_vars(self._grpid, self._varid,
                    #                   startp, countp, stridep, vldata)
                if ierr != NC_NOERR:
                    raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
                # free the pointer array.
                free(vldata)
        free(startp)
        free(countp)
        free(stridep)

    def _get(self,start,count,stride):
        """Private method to retrieve data from a netCDF variable"""
        cdef int ierr, ndims
        cdef size_t *startp
        cdef size_t *countp
        cdef ptrdiff_t *stridep
        cdef ndarray data, dataarr
        cdef void *elptr
        cdef char **strdata
        cdef nc_vlen_t *vldata
        # if one of the counts is negative, then it is an index
        # and not a slice so the resulting array
        # should be 'squeezed' to remove the singleton dimension.
        shapeout = ()
        squeeze_out = False
        for lendim in count:
            if lendim == -1:
                shapeout = shapeout + (1,)
                squeeze_out = True
            else:
                shapeout = shapeout + (lendim,)
        # rank of variable.
        ndims = len(self.dimensions)
        # fill up startp,countp,stridep.
        negstride = 0
        sl = []
        startp = <size_t *>malloc(sizeof(size_t) * ndims)
        countp = <size_t *>malloc(sizeof(size_t) * ndims)
        stridep = <ptrdiff_t *>malloc(sizeof(ptrdiff_t) * ndims)
        for n from 0 <= n < ndims:
            count[n] = abs(count[n]) # make -1 into +1
            countp[n] = count[n]
            # for neg strides, reverse order (then flip that axis after data read in)
            if stride[n] < 0:
                negstride = 1
                stridep[n] = -stride[n]
                startp[n] = start[n]+stride[n]*(count[n]-1)
                stride[n] = -stride[n]
                sl.append(slice(None, None, -1)) # this slice will reverse the data
            else:
                startp[n] = start[n]
                stridep[n] = stride[n]
                sl.append(slice(None,None, 1))
        if self._isprimitive or self._iscompound or self._isenum:
            data = numpy.empty(shapeout, self.dtype)
            # strides all 1 or scalar variable, use get_vara (faster)
            if sum(stride) == ndims or ndims == 0:
                with nogil:
                    ierr = nc_get_vara(self._grpid, self._varid,
                                       startp, countp, data.data)
            else:
                with nogil:
                    ierr = nc_get_vars(self._grpid, self._varid,
                                       startp, countp, stridep, data.data)
            if ierr == NC_EINVALCOORDS:
                raise IndexError
            elif ierr != NC_NOERR:
                raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        elif self._isvlen:
            # allocate array of correct primitive type.
            data = numpy.empty(shapeout, 'O')
            # flatten data array.
            data = data.flatten()
            totelem = PyArray_SIZE(data)
            if self.dtype == str:
                # vlen string (NC_STRING)
                # allocate pointer array to hold string data.
                strdata = <char **>malloc(sizeof(char *) * totelem)
                # strides all 1 or scalar variable, use get_vara (faster)
                if sum(stride) == ndims or ndims == 0:
                    with nogil:
                        ierr = nc_get_vara(self._grpid, self._varid,
                                           startp, countp, strdata)
                else:
                    # FIXME: is this a bug in netCDF4?
                    raise IndexError('strides must all be 1 for string variables')
                    #ierr = nc_get_vars(self._grpid, self._varid,
                    #                   startp, countp, stridep, strdata)
                if ierr == NC_EINVALCOORDS:
                    raise IndexError
                elif ierr != NC_NOERR:
                    raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
                # loop over elements of object array, fill array with
                # contents of strdata.
                for i from 0<=i<totelem:
                    data[i] = strdata[i].decode(default_encoding)
                # reshape the output array
                data = numpy.reshape(data, shapeout)
                # free string data internally allocated in netcdf C lib
                ierr = nc_free_string(totelem, strdata)
                # free the pointer array
                free(strdata)
            else:
                # regular vlen
                # allocate struct array to hold vlen data.
                vldata = <nc_vlen_t *>malloc(totelem*sizeof(nc_vlen_t))
                # strides all 1 or scalar variable, use get_vara (faster)
                if sum(stride) == ndims or ndims == 0:
                    with nogil:
                        ierr = nc_get_vara(self._grpid, self._varid,
                                           startp, countp, vldata)
                else:
                    raise IndexError('strides must all be 1 for vlen variables')
                    #ierr = nc_get_vars(self._grpid, self._varid,
                    #                   startp, countp, stridep, vldata)
                if ierr == NC_EINVALCOORDS:
                    raise IndexError
                elif ierr != NC_NOERR:
                    raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
                # loop over elements of object array, fill array with
                # contents of vlarray struct, put array in object array.
                for i from 0<=i<totelem:
                    arrlen  = vldata[i].len
                    dataarr = numpy.empty(arrlen, self.dtype)
                    #dataarr.data = <char *>vldata[i].p
                    memcpy(<void*>dataarr.data, vldata[i].p, dataarr.nbytes)
                    data[i] = dataarr
                # reshape the output array
                data = numpy.reshape(data, shapeout)
                # free vlen data internally allocated in netcdf C lib
                ierr = nc_free_vlens(totelem, vldata)
                # free the pointer array
                free(vldata)
        free(startp)
        free(countp)
        free(stridep)
        if negstride:
            # reverse data along axes with negative strides.
            data = data[sl].copy() # make a copy so data is contiguous.
        if not self.dimensions:
            return data[0] # a scalar
        elif squeeze_out:
            return numpy.squeeze(data)
        else:
            return data

# Compound datatype support.

cdef class CompoundType:
    """
A `netCDF4.CompoundType` instance is used to describe a compound data
type, and can be passed to the the `netCDF4.Dataset.createVariable` method of
a `netCDF4.Dataset` or `netCDF4.Group` instance. 
Compound data types map to numpy structured arrays.
See `netCDF4.CompoundType.__init__` for more details.

The instance variables `dtype` and `name` should not be modified by
the user.
    """
    cdef public nc_type _nc_type
    cdef public dtype, name
    __pdoc__['CompoundType.name'] = \
    """String name."""
    __pdoc__['CompoundType.dtype'] = \
    """A numpy dtype object describing the compound data type."""
    def __init__(self, grp, object dt, object dtype_name, **kwargs):
        """
        ***`__init__(group, datatype, datatype_name)`***

        CompoundType constructor.

        **`group`**: `netCDF4.Group` instance to associate with the compound datatype.

        **`datatype`**: A numpy dtype object describing a structured (a.k.a record)
        array.  Can be composed of homogeneous numeric or character data types, or
        other structured array data types.

        **`datatype_name`**: a Python string containing a description of the
        compound data type.

        ***Note 1***: When creating nested compound data types,
        the inner compound data types must already be associated with CompoundType
        instances (so create CompoundType instances for the innermost structures
        first).

        ***Note 2***: `netCDF4.CompoundType` instances should be created using the
        `netCDF4.Dataset.createCompoundType`
        method of a `netCDF4.Dataset` or `netCDF4.Group` instance, not using this class directly.
        """
        cdef nc_type xtype
        dt = numpy.dtype(dt,align=True)
        if 'typeid' in kwargs:
            xtype = kwargs['typeid']
        else:
            xtype = _def_compound(grp, dt, dtype_name)
        self._nc_type = xtype
        self.dtype = dt
        self.name = dtype_name

    def __repr__(self):
        if python3:
            return self.__unicode__()
        else:
            return unicode(self).encode(default_encoding)

    def __unicode__(self):
        return repr(type(self))+": name = '%s', numpy dtype = %s\n" %\
        (self.name,self.dtype)

cdef _def_compound(grp, object dt, object dtype_name):
    # private function used to construct a netcdf compound data type
    # from a numpy dtype object by CompoundType.__init__.
    cdef nc_type xtype, xtype_tmp
    cdef int ierr, ndims
    cdef size_t offset, size
    cdef char *namstring
    cdef char *nested_namstring
    cdef int *dim_sizes
    bytestr = _strencode(dtype_name)
    namstring = bytestr
    size = dt.itemsize
    ierr = nc_def_compound(grp._grpid, size, namstring, &xtype)
    if ierr != NC_NOERR:
        raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
    names = list(dt.fields.keys())
    formats = [v[0] for v in dt.fields.values()]
    offsets = [v[1] for v in dt.fields.values()]
    # make sure entries in lists sorted by offset.
    # (don't know why this is necessary, but it is for version 4.0.1)
    names = _sortbylist(names, offsets)
    formats = _sortbylist(formats, offsets)
    offsets.sort()
    for name, format, offset in zip(names, formats, offsets):
        bytestr = _strencode(name)
        namstring = bytestr
        if format.kind != 'V': # scalar primitive type
            try:
                xtype_tmp = _nptonctype[format.str[1:]]
            except KeyError:
                raise ValueError('Unsupported compound type element')
            ierr = nc_insert_compound(grp._grpid, xtype, namstring,
                                      offset, xtype_tmp)
            if ierr != NC_NOERR:
                raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        else:
            if format.shape ==  (): # nested scalar compound type
                # find this compound type in this group or it's parents.
                xtype_tmp = _find_cmptype(grp, format)
                bytestr = _strencode(name)
                nested_namstring = bytestr
                ierr = nc_insert_compound(grp._grpid, xtype,\
                                          nested_namstring,\
                                          offset, xtype_tmp)
                if ierr != NC_NOERR:
                    raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
            else: # array compound element
                ndims = len(format.shape)
                dim_sizes = <int *>malloc(sizeof(int) * ndims)
                for n from 0 <= n < ndims:
                    dim_sizes[n] = format.shape[n]
                if format.subdtype[0].str[1] != 'V': # primitive type.
                    try:
                        xtype_tmp = _nptonctype[format.subdtype[0].str[1:]]
                    except KeyError:
                        raise ValueError('Unsupported compound type element')
                    ierr = nc_insert_array_compound(grp._grpid,xtype,namstring,
                           offset,xtype_tmp,ndims,dim_sizes)
                    if ierr != NC_NOERR:
                        raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
                else: # nested array compound type.
                    # find this compound type in this group or it's parents.
                    xtype_tmp = _find_cmptype(grp, format.subdtype[0])
                    bytestr = _strencode(name)
                    nested_namstring = bytestr
                    ierr = nc_insert_array_compound(grp._grpid,xtype,\
                                                    nested_namstring,\
                                                    offset,xtype_tmp,\
                                                    ndims,dim_sizes)
                    if ierr != NC_NOERR:
                        raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
                free(dim_sizes)
    return xtype

cdef _find_cmptype(grp, dtype):
    # look for data type in this group and it's parents.
    # return datatype id when found, if not found, raise exception.
    cdef nc_type xtype
    match = False
    for cmpname, cmpdt in grp.cmptypes.items():
        xtype = cmpdt._nc_type
        names1 = dtype.names; names2 = cmpdt.dtype.names
        formats1 = [v[0] for v in dtype.fields.values()]
        formats2 = [v[0] for v in cmpdt.dtype.fields.values()]
        # match names, formats, but not offsets (they may be changed
        # by netcdf lib).
        if names1==names2 and formats1==formats2:
            match = True
            break
    if not match:
        try:
            parent_grp = grp.parent
        except AttributeError:
            raise ValueError("cannot find compound type in this group or parent groups")
        if parent_grp is None:
            raise ValueError("cannot find compound type in this group or parent groups")
        else:
            xtype = _find_cmptype(parent_grp,dtype)
    return xtype

cdef _read_compound(group, nc_type xtype, endian=None):
    # read a compound data type id from an existing file,
    # construct a corresponding numpy dtype instance,
    # then use that to create a CompoundType instance.
    # called by _get_vars, _get_types and _get_att.
    # Calls itself recursively for nested compound types.
    cdef int ierr, nf, numdims, ndim, classp, _grpid
    cdef size_t nfields, offset
    cdef nc_type field_typeid
    cdef int *dim_sizes
    cdef char field_namstring[NC_MAX_NAME+1]
    cdef char cmp_namstring[NC_MAX_NAME+1]
    # get name and number of fields.
    _grpid = group._grpid
    with nogil:
        ierr = nc_inq_compound(_grpid, xtype, cmp_namstring, NULL, &nfields)
    if ierr != NC_NOERR:
        raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
    name = cmp_namstring.decode(default_encoding,unicode_error)
    # loop over fields.
    names = []
    formats = []
    offsets = []
    for nf from 0 <= nf < nfields:
        with nogil:
            ierr = nc_inq_compound_field(_grpid,
                                         xtype,
                                         nf,
                                         field_namstring,
                                         &offset,
                                         &field_typeid,
                                         &numdims,
                                         NULL)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        dim_sizes = <int *>malloc(sizeof(int) * numdims)
        with nogil:
            ierr = nc_inq_compound_field(_grpid,
                                         xtype,
                                         nf,
                                         field_namstring,
                                         &offset,
                                         &field_typeid,
                                         &numdims,
                                         dim_sizes)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        field_name = field_namstring.decode(default_encoding,unicode_error)
        names.append(field_name)
        offsets.append(offset)
        # if numdims=0, not an array.
        field_shape = ()
        if numdims != 0:
            for ndim from 0 <= ndim < numdims:
                field_shape = field_shape + (dim_sizes[ndim],)
        free(dim_sizes)
        # check to see if this field is a nested compound type.
        try:
            field_type =  _nctonptype[field_typeid]
            if endian is not None:
                format = endian + format
        except KeyError:
            with nogil:
                ierr = nc_inq_user_type(_grpid,
                       field_typeid,NULL,NULL,NULL,NULL,&classp)
            if classp == NC_COMPOUND: # a compound type
                # recursively call this function?
                field_type = _read_compound(group, field_typeid, endian=endian)
            else:
                raise KeyError('compound field of an unsupported data type')
        if field_shape != ():
            formats.append((field_type,field_shape))
        else:
            formats.append(field_type)
    # make sure entries in lists sorted by offset.
    names = _sortbylist(names, offsets)
    formats = _sortbylist(formats, offsets)
    offsets.sort()
    # create a dict that can be converted into a numpy dtype.
    dtype_dict = {'names':names,'formats':formats,'offsets':offsets}
    return CompoundType(group, dtype_dict, name, typeid=xtype)

# VLEN datatype support.

cdef class VLType:
    """
A `netCDF4.VLType` instance is used to describe a variable length (VLEN) data
type, and can be passed to the the `netCDF4.Dataset.createVariable` method of
a `netCDF4.Dataset` or `netCDF4.Group` instance. See 
`netCDF4.VLType.__init__` for more details.

The instance variables `dtype` and `name` should not be modified by
the user.
    """
    cdef public nc_type _nc_type
    cdef public dtype, name
    __pdoc__['VLType.name'] = \
    """String name."""
    __pdoc__['VLType.dtype'] = \
    """A numpy dtype object describing the component type for the VLEN."""
    def __init__(self, grp, object dt, object dtype_name, **kwargs):
        """
        **`__init__(group, datatype, datatype_name)`**

        VLType constructor.

        **`group`**: `netCDF4.Group` instance to associate with the VLEN datatype.

        **`datatype`**: An numpy dtype object describing the component type for the
        variable length array.

        **`datatype_name`**: a Python string containing a description of the
        VLEN data type.

        ***`Note`***: `netCDF4.VLType` instances should be created using the
        `netCDF4.Dataset.createVLType`
        method of a `netCDF4.Dataset` or `netCDF4.Group` instance, not using this class directly.
        """
        cdef nc_type xtype
        if 'typeid' in kwargs:
            xtype = kwargs['typeid']
        else:
            xtype, dt = _def_vlen(grp, dt, dtype_name)
        self._nc_type = xtype
        self.dtype = dt
        if dt == str:
            self.name = None
        else:
            self.name = dtype_name

    def __repr__(self):
        if python3:
            return self.__unicode__()
        else:
            return unicode(self).encode(default_encoding)

    def __unicode__(self):
        if self.dtype == str:
            return repr(type(self))+': string type'
        else:
            return repr(type(self))+": name = '%s', numpy dtype = %s\n" %\
            (self.name, self.dtype)

cdef _def_vlen(grp, object dt, object dtype_name):
    # private function used to construct a netcdf VLEN data type
    # from a numpy dtype object or python str object by VLType.__init__.
    cdef nc_type xtype, xtype_tmp
    cdef int ierr, ndims
    cdef size_t offset, size
    cdef char *namstring
    cdef char *nested_namstring
    if dt == str: # python string, use NC_STRING
        xtype = NC_STRING
        # dtype_name ignored
    else: # numpy datatype
        bytestr = _strencode(dtype_name)
        namstring = bytestr
        dt = numpy.dtype(dt) # convert to numpy datatype.
        if dt.str[1:] in _supportedtypes:
            # find netCDF primitive data type corresponding to
            # specified numpy data type.
            xtype_tmp = _nptonctype[dt.str[1:]]
            ierr = nc_def_vlen(grp._grpid, namstring, xtype_tmp, &xtype);
            if ierr != NC_NOERR:
                raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        else:
            raise KeyError("unsupported datatype specified for VLEN")
    return xtype, dt

cdef _read_vlen(group, nc_type xtype, endian=None):
    # read a VLEN data type id from an existing file,
    # construct a corresponding numpy dtype instance,
    # then use that to create a VLType instance.
    # called by _get_types, _get_vars.
    cdef int ierr, _grpid
    cdef size_t vlsize
    cdef nc_type base_xtype
    cdef char vl_namstring[NC_MAX_NAME+1]
    _grpid = group._grpid
    if xtype == NC_STRING:
        dt = str
        name = None
    else:
        with nogil:
            ierr = nc_inq_vlen(_grpid, xtype, vl_namstring, &vlsize, &base_xtype)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        name = vl_namstring.decode(default_encoding,unicode_error)
        try:
            datatype = _nctonptype[base_xtype]
            if endian is not None: datatype = endian + datatype
            dt = numpy.dtype(datatype) # see if it is a primitive type
        except KeyError:
            raise KeyError("unsupported component type for VLEN")
    return VLType(group, dt, name, typeid=xtype)

# Enum datatype support.

cdef class EnumType:
    """
A `netCDF4.EnumType` instance is used to describe an Enum data
type, and can be passed to the the `netCDF4.Dataset.createVariable` method of
a `netCDF4.Dataset` or `netCDF4.Group` instance. See 
`netCDF4.EnumType.__init__` for more details.

The instance variables `dtype`, `name` and `enum_dict` should not be modified by
the user.
    """
    cdef public nc_type _nc_type
    cdef public dtype, name, enum_dict
    __pdoc__['EnumType.name'] = \
    """String name."""
    __pdoc__['EnumType.dtype'] = \
    """A numpy integer dtype object describing the base type for the Enum."""
    __pdoc__['EnumType.enum_dict'] = \
    """A python dictionary describing the enum fields and values."""
    def __init__(self, grp, object dt, object dtype_name, object enum_dict, **kwargs):
        """
        **`__init__(group, datatype, datatype_name, enum_dict)`**

        EnumType constructor.

        **`group`**: `netCDF4.Group` instance to associate with the VLEN datatype.

        **`datatype`**: An numpy integer dtype object describing the base type
        for the Enum.

        **`datatype_name`**: a Python string containing a description of the
        Enum data type.

        **`enum_dict`**: a Python dictionary containing the Enum field/value
        pairs.

        ***`Note`***: `netCDF4.EnumType` instances should be created using the
        `netCDF4.Dataset.createEnumType`
        method of a `netCDF4.Dataset` or `netCDF4.Group` instance, not using this class directly.
        """
        cdef nc_type xtype
        if 'typeid' in kwargs:
            xtype = kwargs['typeid']
        else:
            xtype, dt = _def_enum(grp, dt, dtype_name, enum_dict)
        self._nc_type = xtype
        self.dtype = dt
        self.name = dtype_name
        self.enum_dict = enum_dict

    def __repr__(self):
        if python3:
            return self.__unicode__()
        else:
            return unicode(self).encode(default_encoding)

    def __unicode__(self):
        return repr(type(self))+\
        ": name = '%s', numpy dtype = %s, fields/values =%s\n" %\
        (self.name, self.dtype, self.enum_dict)

cdef _def_enum(grp, object dt, object dtype_name, object enum_dict):
    # private function used to construct a netCDF Enum data type
    # from a numpy dtype object or python str object by EnumType.__init__.
    cdef nc_type xtype, xtype_tmp
    cdef int ierr, val
    cdef char *namstring
    bytestr = _strencode(dtype_name)
    namstring = bytestr
    dt = numpy.dtype(dt) # convert to numpy datatype.
    if dt.str[1:] in _intnptonctype.keys():
        # find netCDF primitive data type corresponding to
        # specified numpy data type.
        xtype_tmp = _intnptonctype[dt.str[1:]]
        ierr = nc_def_enum(grp._grpid, xtype_tmp, namstring, &xtype);
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
    else:
        msg="unsupported datatype specified for Enum (must be integer)"
        raise KeyError(msg)
    # insert named members into enum type.
    for field in enum_dict:
        value = enum_dict[field]
        bytestr = _strencode(field)
        namstring = bytestr
        val = value
        ierr = nc_insert_enum(grp._grpid, xtype, namstring, &val)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
    return xtype, dt

cdef _read_enum(group, nc_type xtype, endian=None):
    # read a Enum data type id from an existing file,
    # construct a corresponding numpy dtype instance,
    # then use that to create a EnumType instance.
    # called by _get_types, _get_vars.
    cdef int ierr, _grpid, nmem
    cdef char enum_val
    cdef nc_type base_xtype
    cdef char enum_namstring[NC_MAX_NAME+1]
    cdef size_t nmembers
    _grpid = group._grpid
    # get name, datatype, and number of members.
    with nogil:
        ierr = nc_inq_enum(_grpid, xtype, enum_namstring, &base_xtype, NULL,\
                &nmembers)
    if ierr != NC_NOERR:
        raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
    name = enum_namstring.decode(default_encoding,unicode_error)
    try:
        datatype = _nctonptype[base_xtype]
        if endian is not None: datatype = endian + datatype
        dt = numpy.dtype(datatype) # see if it is a primitive type
    except KeyError:
        raise KeyError("unsupported component type for VLEN")
    # loop over members, build dict.
    enum_dict = {}
    for nmem from 0 <= nmem < nmembers:
        with nogil:
            ierr = nc_inq_enum_member(_grpid, xtype, nmem, \
                                      enum_namstring, &enum_val)
        if ierr != NC_NOERR:
           raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        name = enum_namstring.decode(default_encoding,unicode_error)
        enum_dict[name] = int(enum_val)
    return EnumType(group, dt, name, enum_dict, typeid=xtype)

cdef _strencode(pystr,encoding=None):
    # encode a string into bytes.  If already bytes, do nothing.
    # uses default_encoding module variable for default encoding.
    if encoding is None:
        encoding = default_encoding
    try:
        return pystr.encode(encoding)
    except (AttributeError, UnicodeDecodeError):
        return pystr # already bytes or unicode?

def _to_ascii(bytestr):
    # encode a byte string to an ascii encoded string.
    if python3:
        return str(bytestr,encoding='ascii')
    else:
        return bytestr.encode('ascii')

#----------------------------------------
# extra utilities (formerly in utils.pyx)
#----------------------------------------
from datetime import timedelta, datetime, MINYEAR
from netcdftime import _parse_date, microsec_units, millisec_units,\
                       sec_units, min_units, hr_units, day_units

# start of the gregorian calendar
gregorian = datetime(1582,10,15)

def _dateparse(timestr):
    """parse a string of the form time-units since yyyy-mm-dd hh:mm:ss,
    return a datetime instance"""
    # same as version in netcdftime, but returns a timezone naive
    # python datetime instance with the utc_offset included.
    timestr_split = timestr.split()
    units = timestr_split[0].lower()
    if timestr_split[1].lower() != 'since':
        raise ValueError("no 'since' in unit_string")
    # parse the date string.
    n = timestr.find('since')+6
    isostring = timestr[n:]
    year, month, day, hour, minute, second, utc_offset =\
        _parse_date( isostring.strip() )
    if year >= MINYEAR:
        basedate = datetime(year, month, day, hour, minute, second)
        # add utc_offset to basedate time instance (which is timezone naive)
        basedate += timedelta(days=utc_offset/1440.)
    else:
        if not utc_offset:
            basedate = netcdftime.datetime(year, month, day, hour, minute, second)
        else:
            raise ValueError('cannot use utc_offset for reference years <= 0')
    return basedate

def stringtoarr(string,NUMCHARS,dtype='S'):
    """
**`stringtoarr(a, NUMCHARS,dtype='S')`**

convert a string to a character array of length `NUMCHARS`

**`a`**:  Input python string.

**`NUMCHARS`**:  number of characters used to represent string
(if len(a) < `NUMCHARS`, it will be padded on the right with blanks).

**`dtype`**:  type of numpy array to return.  Default is `'S'`, which
means an array of dtype `'S1'` will be returned.  If dtype=`'U'`, a
unicode array (dtype = `'U1'`) will be returned.

returns a rank 1 numpy character array of length NUMCHARS with datatype `'S1'`
(default) or `'U1'` (if dtype=`'U'`)"""
    if dtype not in ["S","U"]:
        raise ValueError("dtype must string or unicode ('S' or 'U')")
    arr = numpy.zeros(NUMCHARS,dtype+'1')
    arr[0:len(string)] = tuple(string)
    return arr

def stringtochar(a):
    """
**`stringtochar(a)`**

convert a string array to a character array with one extra dimension

**`a`**:  Input numpy string array with numpy datatype `'SN'` or `'UN'`, where N
is the number of characters in each string.  Will be converted to
an array of characters (datatype `'S1'` or `'U1'`) of shape `a.shape + (N,)`.

returns a numpy character array with datatype `'S1'` or `'U1'`
and shape `a.shape + (N,)`, where N is the length of each string in a."""
    dtype = a.dtype.kind
    if dtype not in ["S","U"]:
        raise ValueError("type must string or unicode ('S' or 'U')")
    b = numpy.array(tuple(a.tostring().decode(default_encoding)),dtype+'1')
    b.shape = a.shape + (a.itemsize,)
    return b

def chartostring(b):
    """
**`chartostring(b)`**

convert a character array to a string array with one less dimension.

**`b`**:  Input character array (numpy datatype `'S1'` or `'U1'`).
Will be converted to a array of strings, where each string has a fixed
length of `b.shape[-1]` characters.

returns a numpy string array with datatype `'SN'` or `'UN'` and shape
`b.shape[:-1]` where where `N=b.shape[-1]`."""
    dtype = b.dtype.kind
    if dtype not in ["S","U"]:
        raise ValueError("type must string or unicode ('S' or 'U')")
    bs = b.tostring().decode(default_encoding)
    slen = int(b.shape[-1])
    a = numpy.array([bs[n1:n1+slen] for n1 in range(0,len(bs),slen)],dtype+repr(slen))
    a.shape = b.shape[:-1]
    return a

def date2num(dates,units,calendar='standard'):
    """
**`date2num(dates,units,calendar='standard')`**

Return numeric time values given datetime objects. The units
of the numeric time values are described by the `netCDF4.units` argument
and the `netCDF4.calendar` keyword. The datetime objects must
be in UTC with no time-zone offset.  If there is a
time-zone offset in `units`, it will be applied to the
returned numeric values.

**`dates`**: A datetime object or a sequence of datetime objects.
The datetime objects should not include a time-zone offset.

**`units`**: a string of the form `<time units> since <reference time>`
describing the time units. `<time units>` can be days, hours, minutes,
seconds, milliseconds or microseconds. `<reference time>` is the time
origin.  Accuracy is somewhere between a millisecond and a microsecond,
depending on the time interval and the calendar used.

**`calendar`**: describes the calendar used in the time calculations.
All the values currently defined in the 
[CF metadata convention](http://cfconventions.org)
Valid calendars `'standard', 'gregorian', 'proleptic_gregorian'
'noleap', '365_day', '360_day', 'julian', 'all_leap', '366_day'`.
Default is `'standard'`, which is a mixed Julian/Gregorian calendar.

returns a numeric time value, or an array of numeric time values.
    """
    calendar = calendar.lower()
    basedate = _dateparse(units)
    unit = units.split()[0].lower()
    # real-world calendars limited to positive reference years.
    if calendar in ['julian', 'standard', 'gregorian', 'proleptic_gregorian']:
        if basedate.year == 0:
            msg='zero not allowed as a reference year, does not exist in Julian or Gregorian calendars'
            raise ValueError(msg)
        elif basedate.year < 0:
            msg='negative reference year in time units, must be >= 1'
            raise ValueError(msg)

    if (calendar == 'proleptic_gregorian' and basedate.year >= MINYEAR) or \
       (calendar in ['gregorian','standard'] and basedate > gregorian):
        # use python datetime module,
        isscalar = False
        try:
            dates[0]
        except:
            isscalar = True
        if isscalar:
            dates = numpy.array([dates])
        else:
            dates = numpy.array(dates)
            shape = dates.shape
        ismasked = False
        if hasattr(dates,'mask'):
            mask = dates.mask
            ismasked = True
        times = []
        for date in dates.flat:
            if ismasked and not date:
                times.append(None)
            else:
                td = date - basedate
                # total time in microseconds.
                totaltime = td.microseconds + (td.seconds + td.days * 24 * 3600) * 1.e6
                if unit in microsec_units:
                    times.append(totaltime)
                elif unit in millisec_units:
                    times.append(totaltime/1.e3)
                elif unit in sec_units:
                    times.append(totaltime/1.e6)
                elif unit in min_units:
                    times.append(totaltime/1.e6/60)
                elif unit in hr_units:
                    times.append(totaltime/1.e6/3600)
                elif unit in day_units:
                    times.append(totaltime/1.e6/3600./24.)
                else:
                    raise ValueError('unsupported time units')
        if isscalar:
            return times[0]
        else:
            return numpy.reshape(numpy.array(times), shape)
    else: # use netcdftime module for other calendars
        cdftime = netcdftime.utime(units,calendar=calendar)
        return cdftime.date2num(dates)

def num2date(times,units,calendar='standard'):
    """
**`num2date(times,units,calendar='standard')`**

Return datetime objects given numeric time values. The units
of the numeric time values are described by the `units` argument
and the `calendar` keyword. The returned datetime objects represent
UTC with no time-zone offset, even if the specified
`units` contain a time-zone offset.

**`times`**: numeric time values.

**`units`**: a string of the form `<time units> since <reference time>`
describing the time units. `<time units>` can be days, hours, minutes,
seconds, milliseconds or microseconds. `<reference time>` is the time
origin.  Accuracy is somewhere between a millisecond and a microsecond,
depending on the time interval and the calendar used.

**`calendar`**: describes the calendar used in the time calculations.
All the values currently defined in the 
[CF metadata convention](http://cfconventions.org)
Valid calendars `'standard', 'gregorian', 'proleptic_gregorian'
'noleap', '365_day', '360_day', 'julian', 'all_leap', '366_day'`.
Default is `'standard'`, which is a mixed Julian/Gregorian calendar.

returns a datetime instance, or an array of datetime instances.

***Note***: The datetime instances returned are 'real' python datetime
objects if `calendar='proleptic_gregorian'`, or
`calendar='standard'` or `'gregorian'`
and the date is after the breakpoint between the Julian and
Gregorian calendars (1582-10-15). Otherwise, they are 'phony' datetime
objects which support some but not all the methods of 'real' python
datetime objects. The datetime instances
do not contain a time-zone offset, even if the specified `units`
contains one.
    """
    calendar = calendar.lower()
    basedate = _dateparse(units)
    unit = units.split()[0].lower()
    # real-world calendars limited to positive reference years.
    if calendar in ['julian', 'standard', 'gregorian', 'proleptic_gregorian']:
        if basedate.year == 0:
            msg='zero not allowed as a reference year, does not exist in Julian or Gregorian calendars'
            raise ValueError(msg)
        elif basedate.year < 0:
            msg='negative reference year in time units, must be >= 1'
            raise ValueError(msg)

    if (calendar == 'proleptic_gregorian' and basedate.year >= MINYEAR) or \
       (calendar in ['gregorian','standard'] and basedate > gregorian):
        # use python datetime module,
        isscalar = False
        try:
            times[0]
        except:
            isscalar = True
        if isscalar:
            times = numpy.array([times],dtype='d')
        else:
            times = numpy.array(times, dtype='d')
            shape = times.shape
        ismasked = False
        if hasattr(times,'mask'):
            mask = times.mask
            ismasked = True
        dates = []
        for time in times.flat:
            if ismasked and not time:
                dates.append(None)
            else:
                # convert to total seconds
                if unit in microsec_units:
                    tsecs = time/1.e6
                elif unit in millisec_units:
                    tsecs = time/1.e3
                elif unit in sec_units:
                    tsecs = time
                elif unit in min_units:
                    tsecs = time*60.
                elif unit in hr_units:
                    tsecs = time*3600.
                elif unit in day_units:
                    tsecs = time*86400.
                else:
                    raise ValueError('unsupported time units')
                # compute time delta.
                days = tsecs // 86400.
                msecsd = tsecs*1.e6 - days*86400.*1.e6
                secs = msecsd // 1.e6
                msecs = numpy.round(msecsd - secs*1.e6)
                td = timedelta(days=days,seconds=secs,microseconds=msecs)
                # add time delta to base date.
                date = basedate + td
                dates.append(date)
        if isscalar:
            return dates[0]
        else:
            return numpy.reshape(numpy.array(dates), shape)
    else: # use netcdftime for other calendars
        cdftime = netcdftime.utime(units,calendar=calendar)
        return cdftime.num2date(times)

def date2index(dates, nctime, calendar=None, select='exact'):
    """
**`date2index(dates, nctime, calendar=None, select='exact')`**

Return indices of a netCDF time variable corresponding to the given dates.

**`dates`**: A datetime object or a sequence of datetime objects.
The datetime objects should not include a time-zone offset.

**`nctime`**: A netCDF time variable object. The nctime object must have a
`units` attribute.

**`calendar`**: describes the calendar used in the time calculations.
All the values currently defined in the 
[CF metadata convention](http://cfconventions.org)
Valid calendars `'standard', 'gregorian', 'proleptic_gregorian'
'noleap', '365_day', '360_day', 'julian', 'all_leap', '366_day'`.
Default is `'standard'`, which is a mixed Julian/Gregorian calendar.
If `calendar` is None, its value is given by `nctime.calendar` or
`standard` if no such attribute exists.

**`select`**: `'exact', 'before', 'after', 'nearest'`
The index selection method. `exact` will return the indices perfectly
matching the dates given. `before` and `after` will return the indices
corresponding to the dates just before or just after the given dates if
an exact match cannot be found. `nearest` will return the indices that
correspond to the closest dates.

returns an index (indices) of the netCDF time variable corresponding
to the given datetime object(s).
    """
    try: 
        nctime.units
    except AttributeError:
        raise AttributeError("netcdf time variable is missing a 'units' attribute")
    if calendar == None:
        calendar = getattr(nctime, 'calendar', 'standard')
    calendar = calendar.lower()
    basedate = _dateparse(nctime.units)
    # real-world calendars limited to positive reference years.
    if calendar in ['julian', 'standard', 'gregorian', 'proleptic_gregorian']:
        if basedate.year == 0:
            msg='zero not allowed as a reference year, does not exist in Julian or Gregorian calendars'
            raise ValueError(msg)
        elif basedate.year < 0:
            msg='negative reference year in time units, must be >= 1'
            raise ValueError(msg)

    if (calendar == 'proleptic_gregorian' and basedate.year >= MINYEAR) or \
       (calendar in ['gregorian','standard'] and basedate > gregorian):
        # use python datetime
        times = date2num(dates,nctime.units,calendar=calendar)
        return netcdftime.time2index(times, nctime, calendar, select)
    else: # use netcdftime module for other cases
        return netcdftime.date2index(dates, nctime, calendar, select)

class MFDataset(Dataset):
    """
Class for reading multi-file netCDF Datasets, making variables
spanning multiple files appear as if they were in one file.
Datasets must be in `NETCDF4_CLASSIC, NETCDF3_CLASSIC, NETCDF3_64BIT_OFFSET
or NETCDF3_64BIT_DATA` format (`NETCDF4` Datasets won't work).

Adapted from [pycdf](http://pysclint.sourceforge.net/pycdf) by Andre Gosselin.

Example usage (See `netCDF4.MFDataset.__init__` for more details):

    :::python
    >>> import numpy
    >>> # create a series of netCDF files with a variable sharing
    >>> # the same unlimited dimension.
    >>> for nf in range(10):
    >>>     f = Dataset("mftest%s.nc" % nf,"w")
    >>>     f.createDimension("x",None)
    >>>     x = f.createVariable("x","i",("x",))
    >>>     x[0:10] = numpy.arange(nf*10,10*(nf+1))
    >>>     f.close()
    >>> # now read all those files in at once, in one Dataset.
    >>> f = MFDataset("mftest*nc")
    >>> print f.variables["x"][:]
    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
     25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
     50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74
     75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99]
    """

    def __init__(self, files, check=False, aggdim=None, exclude=[]):
        """
        **`__init__(self, files, check=False, aggdim=None, exclude=[])`**

        Open a Dataset spanning multiple files, making it look as if it was a
        single file. Variables in the list of files that share the same
        dimension (specified with the keyword `aggdim`) are aggregated. If
        `aggdim` is not specified, the unlimited is aggregated.  Currently,
        `aggdim` must be the leftmost (slowest varying) dimension of each
        of the variables to be aggregated.
        
        **`files`**: either a sequence of netCDF files or a string with a
        wildcard (converted to a sorted list of files using glob)  The first file
        in the list will become the "master" file, defining all the
        variables with an aggregation dimension which may span
        subsequent files. Attribute access returns attributes only from "master"
        file. The files are always opened in read-only mode.
        
        **`check`**: True if you want to do consistency checking to ensure the
        correct variables structure for all of the netcdf files.  Checking makes
        the initialization of the MFDataset instance much slower. Default is
        False.
        
        **`aggdim`**: The name of the dimension to aggregate over (must
        be the leftmost dimension of each of the variables to be aggregated).
        If None (default), aggregate over the unlimited dimension.
        
        **`exclude`**: A list of variable names to exclude from aggregation.
        Default is an empty list.
       """

        # Open the master file in the base class, so that the CDFMF instance
        # can be used like a CDF instance.
        if isinstance(files, str):
            if files.startswith('http'):
                msg='cannot using file globbing for remote (OPeNDAP) datasets'
                raise ValueError(msg)
            else:
                files = sorted(glob(files))

        master = files[0]

        # Open the master again, this time as a classic CDF instance. This will avoid
        # calling methods of the CDFMF subclass when querying the master file.
        cdfm = Dataset(master)
        # copy attributes from master.
        for name, value in cdfm.__dict__.items():
            self.__dict__[name] = value

        # Make sure the master defines a dim with name aggdim,
        # or an unlimited dimension.
        aggDimId = None
        for dimname,dim in cdfm.dimensions.items():
            if aggdim is None:
                if dim.isunlimited():
                    aggDimId = dim
                    aggDimName = dimname
            else:
                if dimname == aggdim:
                    aggDimId = dim
                    aggDimName = dimname
        if aggDimId is None:
            raise IOError("master dataset %s does not have a aggregation dimension" % master)

        # Get info on all aggregation variables defined in the master.
        # Make sure the master defines at least one aggregation variable.
        masterRecVar = {}
        for vName,v in cdfm.variables.items():
            # skip variables specified in exclude list.
            if vName in exclude: continue
            dims = v.dimensions
            shape = v.shape
            dtype = v.dtype
            # Be carefull: we may deal with a scalar (dimensionless) variable.
            # Unlimited dimension always occupies index 0.
            if (len(dims) > 0 and aggDimName == dims[0]):
                masterRecVar[vName] = (dims, shape, dtype)
        if len(masterRecVar) == 0:
            raise IOError("master dataset %s does not have any variables to aggregate" % master)

        # Create the following:
        #   cdf       list of Dataset instances
        #   cdfVLen   list unlimited dimension lengths in each CDF instance
        #   cdfRecVar dictionary indexed by the aggregation var names; each key holds
        #             a list of the corresponding Variable instance, one for each
        #             cdf file of the file set
        cdf = [cdfm]
        self._cdf = cdf        # Store this now, because dim() method needs it
        cdfVLen = [len(aggDimId)]
        cdfRecVar = {}
        for v in masterRecVar.keys():
            cdfRecVar[v] = [cdfm.variables[v]]

        # Open each remaining file in read-only mode.
        # Make sure each file defines the same aggregation variables as the master
        # and that the variables are defined in the same way (name, shape and type)
        for f in files[1:]:
            part = Dataset(f)
            varInfo = part.variables
            for v in masterRecVar.keys():
                if check:
                    # Make sure master rec var is also defined here.
                    if v not in varInfo.keys():
                        raise IOError("aggregation variable %s not defined in %s" % (v, f))

                    #if not vInst.dimensions[0] != aggDimName:

                    masterDims, masterShape, masterType = masterRecVar[v][:3]
                    extDims = varInfo[v].dimensions
                    extShape = varInfo[v].shape
                    extType = varInfo[v].dtype
                    # Check that dimension names are identical.
                    if masterDims != extDims:
                        raise IOError("variable %s : dimensions mismatch between "
                                       "master %s (%s) and extension %s (%s)" %
                                       (v, master, masterDims, f, extDims))

                    # Check that the ranks are identical, and the dimension lengths are
                    # identical (except for that of the unlimited dimension, which of
                    # course may vary.
                    if len(masterShape) != len(extShape):
                        raise IOError("variable %s : rank mismatch between "
                                       "master %s (%s) and extension %s (%s)" %
                                       (v, master, len(masterShape), f, len(extShape)))
                    if masterShape[1:] != extShape[1:]:
                        raise IOError("variable %s : shape mismatch between "
                                       "master %s (%s) and extension %s (%s)" %
                                       (v, master, masterShape, f, extShape))

                    # Check that the data types are identical.
                    if masterType != extType:
                        raise IOError("variable %s : data type mismatch between "
                                       "master %s (%s) and extension %s (%s)" %
                                       (v, master, masterType, f, extType))

                    # Everythig ok.
                    vInst = part.variables[v]
                    cdfRecVar[v].append(vInst)
                else:
                    # No making sure of anything -- assume this is ok..
                    vInst = part.variables[v]
                    cdfRecVar[v].append(vInst)

            cdf.append(part)
            cdfVLen.append(len(part.dimensions[aggDimName]))

        # Attach attributes to the MFDataset instance.
        # A local __setattr__() method is required for them.
        self._files = files            # list of cdf file names in the set
        self._cdfVLen = cdfVLen              # list of unlimited lengths
        self._cdfTLen = sum(cdfVLen) # total length
        self._cdfRecVar = cdfRecVar          # dictionary of Variable instances for all
                                             # the aggregation variables
        self._dims = cdfm.dimensions
        self._grps = cdfm.groups
        for dimname, dim in self._dims.items():
            if dimname == aggDimName:
                self._dims[dimname] = _Dimension(dimname, dim, self._cdfVLen, self._cdfTLen)
        self._vars = cdfm.variables
        for varname,var in self._vars.items():
            if varname in self._cdfRecVar.keys():
                self._vars[varname] = _Variable(self, varname, var, aggDimName)
        self._file_format = []
        self._data_model = []
        self._disk_format = []
        for dset in self._cdf:
            if dset.file_format == 'NETCDF4' or dset.data_model == 'NETCDF4':
                raise ValueError('MFNetCDF4 only works with NETCDF3_* and NETCDF4_CLASSIC formatted files, not NETCDF4')
            self._file_format.append(dset.file_format)
            self._data_model.append(dset.data_model)
            self._disk_format.append(dset.disk_format)
        self._path = '/'

    def __setattr__(self, name, value):
        """override base class attribute creation"""
        self.__dict__[name] = value

    def __getattribute__(self, name):
        if name in ['variables','dimensions','file_format','groups',\
                    'data_model','disk_format','path']:
            if name == 'dimensions': return self._dims
            if name == 'variables': return self._vars
            if name == 'file_format': return self._file_format
            if name == 'data_model': return self._data_model
            if name == 'disk_format': return self._disk_format
            if name == 'path': return self._path
            if name == 'groups': return self._grps
        else:
            return Dataset.__getattribute__(self, name)

    def ncattrs(self):
        """
        **`ncattrs(self)`**

        return the netcdf attribute names from the master file.
        """
        return self._cdf[0].__dict__.keys()

    def close(self):
        """
        **`close(self)`**

        close all the open files.
        """
        for dset in self._cdf:
            dset.close()

    def __repr__(self):
        ncdump = ['%r\n' % type(self)]
        dimnames = tuple([str(dimname) for dimname in self.dimensions.keys()])
        varnames = tuple([str(varname) for varname in self.variables.keys()])
        grpnames = ()
        if self.path == '/':
            ncdump.append('root group (%s data model, file format %s):\n' %
                    (self.data_model[0], self.disk_format[0]))
        else:
            ncdump.append('group %s:\n' % self.path)
        attrs = ['    %s: %s\n' % (name,self.__dict__[name]) for name in\
                self.ncattrs()]
        ncdump = ncdump + attrs
        ncdump.append('    dimensions = %s\n' % str(dimnames))
        ncdump.append('    variables = %s\n' % str(varnames))
        ncdump.append('    groups = %s\n' % str(grpnames))
        return ''.join(ncdump)

class _Dimension(object):
    def __init__(self, dimname, dim, dimlens, dimtotlen):
        self.dimlens = dimlens
        self.dimtotlen = dimtotlen
        self._name = dimname
    def __len__(self):
        return self.dimtotlen
    def isunlimited(self):
        return True
    def __repr__(self):
        if self.isunlimited():
            return repr(type(self))+" (unlimited): name = '%s', size = %s\n" % (self._name,len(self))
        else:
            return repr(type(self))+": name = '%s', size = %s\n" % (self._name,len(self))

class _Variable(object):
    def __init__(self, dset, varname, var, recdimname):
        self.dimensions = var.dimensions
        self._dset = dset
        self._grp = dset
        self._mastervar = var
        self._recVar = dset._cdfRecVar[varname]
        self._recdimname = recdimname
        self._recLen = dset._cdfVLen
        self.dtype = var.dtype
        self._name = var._name
        # copy attributes from master.
        for name, value in var.__dict__.items():
            self.__dict__[name] = value
    def typecode(self):
        return self.dtype
    def ncattrs(self):
        return self._mastervar.__dict__.keys()
    def __getattr__(self,name):
        if name == 'shape': return self._shape()
        if name == 'ndim': return len(self._shape())
        try:
            return self.__dict__[name]
        except:
            raise AttributeError(name)
    def __repr__(self):
        ncdump_var = ['%r\n' % type(self)]
        dimnames = tuple([str(dimname) for dimname in self.dimensions])
        attrs = ['    %s: %s\n' % (name,self.__dict__[name]) for name in\
                self.ncattrs()]
        ncdump_var.append('%s %s%s\n' %\
        (self.dtype,self._name,dimnames))
        ncdump_var = ncdump_var + attrs
        unlimdims = []
        for dimname in self.dimensions:
            dim = _find_dim(self._grp, dimname)
            if dim.isunlimited():
                unlimdims.append(str(dimname))
        ncdump_var.append('unlimited dimensions = %s\n' % repr(tuple(unlimdims)))
        ncdump_var.append('current size = %s\n' % repr(self.shape))
        return ''.join(ncdump_var)
    def __len__(self):
        if not self._shape:
            raise TypeError('len() of unsized object')
        else:
            return self._shape()[0]
    def _shape(self):
        recdimlen = len(self._dset.dimensions[self._recdimname])
        return (recdimlen,) + self._mastervar.shape[1:]
    def set_auto_maskandscale(self,val):
        for v in self._recVar:
            v.set_auto_maskandscale(val)
    def __getitem__(self, elem):
        """Get records from a concatenated set of variables."""

        # This special method is used to index the netCDF variable
        # using the "extended slice syntax". The extended slice syntax
        # is a perfect match for the "start", "count" and "stride"
        # arguments to the nc_get_var() function, and is much more easy
        # to use.
        start, count, stride, put_ind =\
        _StartCountStride(elem, self.shape)
        datashape = _out_array_shape(count)
        data = ma.empty(datashape, dtype=self.dtype)

        # Determine which dimensions need to be squeezed
        # (those for which elem is an integer scalar).
        # The convention used is that for those cases,
        # put_ind for this dimension is set to -1 by _StartCountStride.
        squeeze = data.ndim * [slice(None),]
        for i,n in enumerate(put_ind.shape[:-1]):
            if n == 1 and put_ind[...,i].ravel()[0] == -1:
                squeeze[i] = 0

        # Reshape the arrays so we can iterate over them.
        strt = start.reshape((-1, self.ndim or 1))
        cnt = count.reshape((-1, self.ndim or 1))
        strd = stride.reshape((-1, self.ndim or 1))
        put_ind = put_ind.reshape((-1, self.ndim or 1))

        # Fill output array with data chunks.
        # Number of variables making up the MFVariable.Variable.
        nv = len(self._recLen)
        for (start,count,stride,ind) in zip(strt, cnt, strd, put_ind):
            # make sure count=-1 becomes count=1
            count = [abs(cnt) for cnt in count]
            if (numpy.array(stride) < 0).any():
                raise IndexError('negative strides not allowed when slicing MFVariable Variable instance')
            # Start, stop and step along 1st dimension, eg the unlimited
            # dimension.
            sta = start[0]
            step = stride[0]
            stop = sta + count[0] * step

            # Build a list representing the concatenated list of all records in
            # the MFVariable variable set. The list is composed of 2-elem lists
            # each holding:
            #  the record index inside the variables, from 0 to n
            #  the index of the Variable instance to which each record belongs
            idx = []    # list of record indices
            vid = []    # list of Variable indices
            for n in range(nv):
                k = self._recLen[n]     # number of records in this variable
                idx.extend(range(k))
                vid.extend([n] * k)

            # Merge the two lists to get a list of 2-elem lists.
            # Slice this list along the first dimension.
            lst = list(zip(idx, vid)).__getitem__(slice(sta, stop, step))

            # Rebuild the slicing expression for dimensions 1 and ssq.
            newSlice = [slice(None, None, None)]
            for n in range(1, len(start)):   # skip dimension 0
                s = slice(start[n],start[n] + count[n] * stride[n], stride[n])
                newSlice.append(s)

            # Apply the slicing expression to each var in turn, extracting records
            # in a list of arrays.
            lstArr = []
            ismasked = False
            for n in range(nv):
                # Get the list of indices for variable 'n'.
                idx = [i for i,numv in lst if numv == n]
                if idx:
                    # Rebuild slicing expression for dimension 0.
                    newSlice[0] = slice(idx[0], idx[-1] + 1, step)
                    # Extract records from the var, and append them to a list
                    # of arrays.
                    dat = Variable.__getitem__(self._recVar[n],tuple(newSlice))
                    if ma.isMA(dat) and not ismasked:
                        ismasked=True
                        fill_value = dat.fill_value
                    lstArr.append(dat)
            if ismasked:
                lstArr = ma.concatenate(lstArr)
            else:
                lstArr = numpy.concatenate(lstArr)
            if lstArr.dtype != data.dtype: data = data.astype(lstArr.dtype)
            # sometimes there are legitimate singleton dimensions, in which
            # case the array shapes won't conform. If so, a ValueError will
            # result, and no squeeze will be done.
            try:
                data[tuple(ind)] = lstArr.squeeze()
            except ValueError:
                data[tuple(ind)] = lstArr

        # Remove extra singleton dimensions.
        data = data[tuple(squeeze)]

        # if no masked elements, return numpy array.
        if ma.isMA(data) and not data.mask.any():
            data = data.filled()

        return data


class MFTime(_Variable):
    """
Class providing an interface to a MFDataset time Variable by imposing a unique common
time unit to all files. 

Example usage (See `netCDF4.MFTime.__init__` for more details):

    :::python
    >>> import numpy
    >>> f1 = Dataset("mftest_1.nc","w", format="NETCDF4_CLASSIC")
    >>> f2 = Dataset("mftest_2.nc","w", format="NETCDF4_CLASSIC")
    >>> f1.createDimension("time",None)
    >>> f2.createDimension("time",None)
    >>> t1 = f1.createVariable("time","i",("time",))
    >>> t2 = f2.createVariable("time","i",("time",))
    >>> t1.units = "days since 2000-01-01"
    >>> t2.units = "days since 2000-02-01"
    >>> t1.calendar = "standard"
    >>> t2.calendar = "standard"
    >>> t1[:] = numpy.arange(31)
    >>> t2[:] = numpy.arange(30)
    >>> f1.close()
    >>> f2.close()
    >>> # Read the two files in at once, in one Dataset.
    >>> f = MFDataset("mftest*nc")
    >>> t = f.variables["time"]
    >>> print t.units
    days since 2000-01-01
    >>> print t[32] # The value written in the file, inconsistent with the MF time units.
    1
    >>> T = MFTime(t)
    >>> print T[32]
    32
    """

    def __init__(self, time, units=None):
        """
        **`__init__(self, time, units=None)`**

        Create a time Variable with units consistent across a multifile
        dataset.
        
        **`time`**: Time variable from a `netCDF4.MFDataset`.
        
        **`units`**: Time units, for example, `days since 1979-01-01`. If None, use
        the units from the master variable.
        """
        import datetime
        self.__time = time

        # copy attributes from master time variable.
        for name, value in time.__dict__.items():
            self.__dict__[name] = value

        # make sure calendar attribute present in all files.
        for t in self._recVar:
            if not hasattr(t,'calendar'):
                raise ValueError('MFTime requires that the time variable in all files have a calendar attribute')

        # Check that calendar is the same in all files.
        if len(set([t.calendar for t in self._recVar])) > 1:
            raise ValueError('MFTime requires that the same time calendar is used by all files.')

        # Override units if units is specified.
        self.units = units or time.units

        # Reference date to compute the difference between different time units.
        ref_date = datetime.datetime(1900,1,1)
        ref_num = date2num(ref_date, self.units, self.calendar)

        # Create delta vector: delta = ref_num(ref_date) - num(ref_date)
        # So that ref_num(date) = num(date) + delta
        self.__delta = numpy.empty(len(self), time.dtype)

        i0 = 0; i1 = 0
        for i,v in enumerate(self._recVar):
            n = self._recLen[i] # Length of time vector.
            num = date2num(ref_date, v.units, self.calendar)
            i1 += n
            self.__delta[i0:i1] = ref_num - num
            i0 += n


    def __getitem__(self, elem):
        return self.__time[elem] + self.__delta[elem]
