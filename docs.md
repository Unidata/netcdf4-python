Module netCDF4
--------------
Introduction
============

netcdf4-python is a Python interface to the netCDF C library.  

[netCDF version 4](http://www.unidata.ucar.edu/software/netcdf/netcdf-4) has many features
not found in earlier versions of the library and is implemented on top of
[HDF5](http://www.hdfgroup.org/HDF5). This module can read and write
files in both the new netCDF 4 and the old netCDF 3 format, and can create
files that are readable by HDF5 clients. The API modelled after
[Scientific.IO.NetCDF](http://dirac.cnrs-orleans.fr/plone/software/scientificpython/),
and should be familiar to users of that module.

Most new features of netCDF 4 are implemented, such as multiple
unlimited dimensions, groups and zlib data compression.  All the new
numeric data types (such as 64 bit and unsigned integer types) are
implemented. Compound (struct) and variable length (vlen) data types are supported,
but the enum and opaque data types are not. Mixtures of compound and vlen
data types (compound types containing vlens, and vlens containing compound
types) are not supported.

Download
========

 - Latest bleeding-edge code from the 
   [github repository](http://github.com/Unidata/netcdf4-python).
 - Latest [releases](https://pypi.python.org/pypi/netCDF4)
   (source code and windows installers).

Requires
========

 - Python 2.5 or later (python 3 works too).
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
 - The netCDF-4 C library from [](ftp://ftp.unidata.ucar.edu/pub/netcdf).
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
 - run the tests in the 'test' directory by running `python run_all.py`.

Tutorial
========

1) Creating/Opening/Closing a netCDF file
-----------------------------------------

To create a netCDF file from python, you simply call the `netCDF4.Dataset`
constructor. This is also the method used to open an existing netCDF
file.  If the file is open for write access (`w, r+` or `a`, you may
write any type of data including new dimensions, groups, variables and
attributes.  netCDF files come in several flavors (`NETCDF3_CLASSIC,
NETCDF3_64BIT, NETCDF4_CLASSIC`, and `NETCDF4`). The first two flavors
are supported by version 3 of the netCDF library. `NETCDF4_CLASSIC`
files use the version 4 disk format (HDF5), but do not use any features
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

2) Groups in a netCDF file
--------------------------

netCDF version 4 added support for organizing data in hierarchical
groups, which are analagous to directories in a filesystem. Groups serve
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

3) Dimensions in a netCDF file
------------------------------

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

    >>> level = rootgrp.createDimension("level", None)
    >>> time = rootgrp.createDimension("time", None)
    >>> lat = rootgrp.createDimension("lat", 73)
    >>> lon = rootgrp.createDimension("lon", 144)

All of the `netCDF4.Dimension` instances are stored in a python dictionary.

    >>> print rootgrp.dimensions
    OrderedDict([("level", <netCDF4._netCDF4.Dimension object at 0x1b48030>),
                 ("time", <netCDF4._netCDF4.Dimension object at 0x1b481c0>),
                 ("lat", <netCDF4._netCDF4.Dimension object at 0x1b480f8>),
                 ("lon", <netCDF4._netCDF4.Dimension object at 0x1b48a08>)])

Calling the python `len` function with a `netCDF4.Dimension` instance returns
the current size of that dimension.
The `netCDF4.Dimension.isunlimited` method of a `netCDF4.Dimension` instance
can be used to determine if the dimensions is unlimited, or appendable.

    >>> print len(lon)
    144
    >>> print len.is_unlimited()
    False
    >>> print time.is_unlimited()
    True

Printing the `netCDF4.Dimension` object
provides useful summary info, including the name and length of the dimension,
and whether it is unlimited.

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

4) Variables in a netCDF file
-----------------------------

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

    >>> times = rootgrp.createVariable("time","f8",("time",))
    >>> levels = rootgrp.createVariable("level","i4",("level",))
    >>> latitudes = rootgrp.createVariable("latitude","f4",("lat",))
    >>> longitudes = rootgrp.createVariable("longitude","f4",("lon",))
    >>> # two dimensions unlimited
    >>> temp = rootgrp.createVariable("temp","f4",("time","level","lat","lon",))

To get summary info on a `netCDF4.Variable` instance in an interactive session, just print it.

    >>> print temp
    <type "netCDF4._netCDF4.Variable">
    float32 temp(time, level, lat, lon)
        least_significant_digit: 3
        units: K
    unlimited dimensions: time, level
    current shape = (0, 0, 73, 144)

You can use a path to create a Variable inside a hierarchy of groups.

    >>> ftemp = rootgrp.createVariable("/forecasts/model1/temp","f4",("time","level","lat","lon",))

If the intermediate groups do not yet exist, they will be created.

You can also query a `netCDF4.Dataset` or `netCDF4.Group` instance directly to obtain `netCDF4.Group` or 
`netCDF4.Variable` instances using paths.

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

    >>> print rootgrp.variables
    OrderedDict([("time", <netCDF4.Variable object at 0x1b4ba70>),
                 ("level", <netCDF4.Variable object at 0x1b4bab0>),
                 ("latitude", <netCDF4.Variable object at 0x1b4baf0>),
                 ("longitude", <netCDF4.Variable object at 0x1b4bb30>),
                 ("temp", <netCDF4.Variable object at 0x1b4bb70>)])

`netCDF4.Variable` names can be changed using the
`netCDF4.Dataset.renameVariable` method of a `netCDF4.Dataset`
instance.

5) Attributes in a netCDF file
------------------------------

There are two types of attributes in a netCDF file, global and variable.
Global attributes provide information about a group, or the entire
dataset, as a whole. `netCDF4.Variable` attributes provide information about
one of the variables in a group. Global attributes are set by assigning
values to `netCDF4.Dataset` or `netCDF4.Group` instance variables. `netCDF4.Variable`
attributes are set by assigning values to `netCDF4.Variable` instances
variables. Attributes can be strings, numbers or sequences. Returning to
our example,

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

    >>> for name in rootgrp.ncattrs():
    >>>     print "Global attr", name, "=", getattr(rootgrp,name)
    Global attr description = bogus example script
    Global attr history = Created Mon Nov  7 10.30:56 2005
    Global attr source = netCDF4 python module tutorial

The `__dict__` attribute of a `netCDF4.Dataset`, `netCDF4.Group` or `netCDF4.Variable`
instance provides all the netCDF attribute name/value pairs in a python
dictionary:

    >>> print rootgrp.__dict__
    OrderedDict([(u"description", u"bogus example script"),
                 (u"history", u"Created Thu Mar  3 19:30:33 2011"),
                 (u"source", u"netCDF4 python module tutorial")])

Attributes can be deleted from a netCDF `netCDF4.Dataset`, `netCDF4.Group` or
`netCDF4.Variable` using the python `del` statement (i.e. `del grp.foo`
removes the attribute `foo` the the group `grp`).

6) Writing data to and retrieving data from a netCDF variable
-------------------------------------------------------------

Now that you have a netCDF `netCDF4.Variable` instance, how do you put data
into it? You can just treat it like an array and assign data to a slice.

    >>> import numpy
    >>> lats =  numpy.arange(-90,91,2.5)
    >>> lons =  numpy.arange(-180,180,2.5)
    >>> latitudes[:] = lats
    >>> longitudes[:] = lons
    >>> print "latitudes =\n",latitudes[:]
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

    >>> temp[0, 0, [0,1,2,3], [0,1,2,3]]

returns an array of shape (4,4) when slicing a netCDF variable, but for a
numpy array it returns an array of shape (4,).
Similarly, a netCDF variable of shape `(2,3,4,5)` indexed
with `[0, array([True, False, True]), array([False, True, True, True]), :]`
would return a `(2, 3, 5)` array. In NumPy, this would raise an error since
it would be equivalent to `[0, [0,1], [1,2,3], :]`. When slicing with integer
sequences, the indices must be sorted in increasing order and contain no duplicates.
While this behaviour may cause some confusion for those used to NumPy's 'fancy indexing' rules,
it provides a very powerful way to extract data from multidimensional netCDF
variables by using logical operations on the dimension arrays to create slices.

For example,

    >>> tempdat = temp[::2, [1,3,6], lats>0, lons>0]

will extract time indices 0,2 and 4, pressure levels
850, 500 and 200 hPa, all Northern Hemisphere latitudes and Eastern
Hemisphere longitudes, resulting in a numpy array of shape  (3, 3, 36, 71).

    >>> print "shape of fancy temp slice = ",tempdat.shape
    shape of fancy temp slice =  (3, 3, 36, 71)

***Special note for scalar variables***: To extract data from a scalar variable
`v` with no associated dimensions, use `np.asarray(v)` or `v[...]`. The result
will be a numpy scalar array.

7) Dealing with time coordinates
--------------------------------

Time coordinate values pose a special challenge to netCDF users.  Most
metadata standards (such as CF and COARDS) specify that time should be
measure relative to a fixed date using a certain calendar, with units
specified like `hours since YY:MM:DD hh-mm-ss`.  These units can be
awkward to deal with, without a utility to convert the values to and
from calendar dates.  The functione called `netCDF4.num2date` and `netCDF4.date2num` are
provided with this package to do just that.  Here's an example of how they
can be used:

    >>> # fill in times.
    >>> from datetime import datetime, timedelta
    >>> from netCDF4 import num2date, date2num
    >>> dates = [datetime(2001,3,1)+n*timedelta(hours=12) for n in range(temp.shape[0])]
    >>> times[:] = date2num(dates,units=times.units,calendar=times.calendar)
    >>> print "time values (in units %s): " % times.units+"\n",times[:]
    time values (in units hours since January 1, 0001):
    [ 17533056.  17533068.  17533080.  17533092.  17533104.]
    >>> dates = num2date(times[:],units=times.units,calendar=times.calendar)
    >>> print "dates corresponding to time values:\n",dates
    dates corresponding to time values:
    [2001-03-01 00:00:00 2001-03-01 12:00:00 2001-03-02 00:00:00
     2001-03-02 12:00:00 2001-03-03 00:00:00]

`netCDF4.num2date` converts numeric values of time in the specified `units`
and `calendar` to datetime objects, and `netCDF4.date2num` does the reverse.
All the calendars currently defined in the
[CF metadata convention](http://cf-pcmdi.llnl.gov/documents/cf-conventions/) are supported.
A function called `netCDF4.date2index` is also provided which returns the indices
of a netCDF time variable corresponding to a sequence of datetime instances.

8) Reading data from a multi-file netCDF dataset.
-------------------------------------------------

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
must in be in `NETCDF3_64BIT`, `NETCDF3_CLASSIC` or
`NETCDF4_CLASSIC format` (`NETCDF4` formatted multi-file
datasets are not supported).

    >>> for nf in range(10):
    >>>     f = Dataset("mftest%s.nc" % nf,"w")
    >>>     f.createDimension("x",None)
    >>>     x = f.createVariable("x","i",("x",))
    >>>     x[0:10] = numpy.arange(nf*10,10*(nf+1))
    >>>     f.close()

Now read all the files back in at once with `netCDF4.MFDataset`

    >>> from netCDF4 import MFDataset
    >>> f = MFDataset("mftest*nc")
    >>> print f.variables["x"][:]
    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
     25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
     50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74
     75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99]

Note that `netCDF4.MFDataset` can only be used to read, not write, multi-file
datasets.

9) Efficient compression of netCDF variables
--------------------------------------------

Data stored in netCDF 4 `netCDF4.Variable` objects can be compressed and
decompressed on the fly. The parameters for the compression are
determined by the `zlib`, `complevel` and `shuffle` keyword arguments
to the `netCDF4.createVariable<Dataset.createVariable>` method. To turn on
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
format is `NETCDF3_CLASSIC` or `NETCDF3_64BIT`,

If your data only has a certain number of digits of precision (say for
example, it is temperature data that was measured with a precision of
0.1 degrees), you can dramatically improve zlib compression by
quantizing (or truncating) the data using the `least_significant_digit`
keyword argument to `netCDF4.createVariable<Dataset.createVariable>`. The least
significant digit is the power of ten of the smallest decimal place in
the data that is a reliable value. For example if the data has a
precision of 0.1, then setting `least_significant_digit=1` will cause
data the data to be quantized using `numpy.around(scale*data)/scale`, where
scale = 2**bits, and bits is determined so that a precision of 0.1 is
retained (in this case bits=4).  Effectively, this makes the compression
'lossy' instead of 'lossless', that is some precision in the data is
sacrificed for the sake of disk space.

In our example, try replacing the line

    >>> temp = rootgrp.createVariable("temp","f4",("time","level","lat","lon",))

with

    >>> temp = dataset.createVariable("temp","f4",("time","level","lat","lon",),zlib=True)

and then

    >>> temp = dataset.createVariable("temp","f4",("time","level","lat","lon",),zlib=True,least_significant_digit=3)

and see how much smaller the resulting files are.

10) Beyond homogenous arrays of a fixed type - compound data types
------------------------------------------------------------------

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

11) Variable-length (vlen) data types
-------------------------------------

NetCDF 4 has support for variable-length or "ragged" arrays.  These are arrays
of variable length sequences having the same type. To create a variable-length
data type, use the `netCDF4.Dataset.createVLType` method
method of a `netCDF4.Dataset` or `netCDF4.Group` instance.

    >>> f = Dataset("tst_vlen.nc","w")
    >>> vlen_t = f.createVLType(numpy.int32, "phony_vlen")

The numpy datatype of the variable-length sequences and the name of the
new datatype must be specified. Any of the primitive datatypes can be
used (signed and unsigned integers, 32 and 64 bit floats, and characters),
but compound data types cannot.
A new variable can then be created using this datatype.

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

    >>> import random
    >>> data = numpy.empty(len(y)*len(x),object)
    >>> for n in range(len(y)*len(x)):
    >>>    data[n] = numpy.arange(random.randint(1,10),dtype="int32")+1
    >>> data = numpy.reshape(data,(len(y),len(x)))
    >>> vlvar[:] = data
    >>> print "vlen variable =\n",vlvar[:]
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
`netCDF4.createVariable<Dataset.createVariable>` method.

    >>> z = f.createDimension("z",10)
    >>> strvar = rootgrp.createVariable("strvar", str, "z")

In this example, an object array is filled with random python strings with
random lengths between 2 and 12 characters, and the data in the object
array is assigned to the vlen string variable.

    >>> chars = "1234567890aabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    >>> data = numpy.empty(10,"O")
    >>> for n in range(10):
    >>>     stringlen = random.randint(2,12)
    >>>     data[n] = "".join([random.choice(chars) for i in range(stringlen)])
    >>> strvar[:] = data
    >>> print "variable-length string variable:\n",strvar[:]
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

Functions
---------
chartostring(...)
    **`chartostring(b)`**

    convert a character array to a string array with one less dimension.

    **`b`**:  Input character array (numpy datatype `'S1'` or `'U1'`).
    Will be converted to a array of strings, where each string has a fixed
    length of `b.shape[-1]` characters.

    returns a numpy string array with datatype `'SN'` or `'UN'` and shape
    `b.shape[:-1]` where where `N=b.shape[-1]`.

date2index(...)
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

date2num(...)
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

getlibversion(...)
    **`getlibversion()`**

    returns a string describing the version of the netcdf library
    used to build the module, and when it was built.

num2date(...)
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
    `calendar = 'standard'` or `'gregorian'`
    and the date is after the breakpoint between the Julian and
    Gregorian calendars (1582-10-15). Otherwise, they are 'phony' datetime
    objects which support some but not all the methods of 'real' python
    datetime objects. The datetime instances
    do not contain a time-zone offset, even if the specified `units`
    contains one.

stringtoarr(...)
    **`stringtoarr(a, NUMCHARS,dtype='S')`**

    convert a string to a character array of length `NUMCHARS`

    **`a`**:  Input python string.

    **`NUMCHARS`**:  number of characters used to represent string
    (if len(a) < `NUMCHARS`, it will be padded on the right with blanks).

    **`dtype`**:  type of numpy array to return.  Default is `'S'`, which
    means an array of dtype `'S1'` will be returned.  If dtype=`'U'`, a
    unicode array (dtype = `'U1'`) will be returned.

    returns a rank 1 numpy character array of length NUMCHARS with datatype `'S1'`
    (default) or `'U1'` (if dtype=`'U'`)

stringtochar(...)
    **`stringtochar(a)`**

    convert a string array to a character array with one extra dimension

    **`a`**:  Input numpy string array with numpy datatype `'SN'` or `'UN'`, where N
    is the number of characters in each string.  Will be converted to
    an array of characters (datatype `'S1'` or `'U1'`) of shape `a.shape + (N,)`.

    returns a numpy character array with datatype `'S1'` or `'U1'`
    and shape `a.shape + (N,)`, where N is the length of each string in a.

Classes
-------
CompoundType 
    A `netCDF4.CompoundType` instance is used to describe a compound data
    type, and can be passed to the the `netCDF4.Dataset.createVariable` method of
    a `netCDF4.Dataset` or `netCDF4.Group` instance.

    Compound data types map to numpy structured arrays.

    The instance variables `dtype` and `name` should not be modified by
    the user.

    Ancestors (in MRO)
    ------------------
    netCDF4.CompoundType
    __builtin__.object

    Class variables
    ---------------
    dtype
        A numpy dtype object describing the compound data type.

    name
        String name.

    Methods
    -------
    __init__(...)
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
        method of a Dataset or `netCDF4.Group` instance, not using this class directly.

Dataset 
    A netCDF `netCDF4.Dataset` is a collection of dimensions, groups, variables and
    attributes. Together they describe the meaning of data and relations among
    data fields stored in a netCDF file.

    A list of attribute names corresponding to global netCDF attributes
    defined for the `netCDF4.Dataset` can be obtained with the
    `netCDF4.Dataset.ncattrs` method.
    These attributes can be created by assigning to an attribute of the
    `netCDF4.Dataset` instance. A dictionary containing all the netCDF attribute
    name/value pairs is provided by the `__dict__` attribute of a
    `netCDF4.Dataset` instance.

    The class variables `dimensions, variables, groups, cmptypes, vltypes,
    data_model, disk_format` and `path` are read-only (and should not be modified by the
    user).

    Ancestors (in MRO)
    ------------------
    netCDF4.Dataset
    __builtin__.object

    Descendents
    -----------
    netCDF4.Group
    netCDF4.MFDataset

    Class variables
    ---------------
    cmptypes
        The `cmptypes` dictionary maps the names of
        compound types defined for the `netCDF4.Group` or `netCDF4.Dataset` to instances of the
        `netCDF4.CompoundType` class.

    data_model
        `data_model` describes the netCDF
        data model version, one of `NETCDF3_CLASSIC`, `NETCDF4`,
        `NETCDF4_CLASSIC` or `NETCDF3_64BIT`.

    dimensions
        The `dimensions` dictionary maps the names of
        dimensions defined for the `netCDF4.Group` or `netCDF4.Dataset` to instances of the
        `netCDF4.Dimension` class.

    disk_format
        `disk_format` describes the underlying
        file format, one of `NETCDF3`, `HDF5`, `HDF4`,
        `PNETCDF`, `DAP2`, `DAP4` or `UNDEFINED`. Only available if using
        netcdf C library version >= 4.3.1, otherwise will always return
        `UNDEFINED`.

    file_format
        same as `data_model`, retained for backwards compatibility.

    groups
        The groups dictionary maps the names of groups created for
        this `netCDF4.Dataset` or `netCDF4.Group` to instances of the `netCDF4.Group` class (the
        `netCDF4.Dataset` class is simply a special case of the `netCDF4.Group` class which
        describes the root group in the netCDF4.file).

    keepweakref
        If `True`, child Dimension and Variables objects only keep weak references to
        parent Dataset or Group.

    parent
        `parent` is a reference to the parent
        `netCDF4.Group` instance. `None` for a the root group or `netCDF4.Dataset` instance

    path
        `path` shows the location of the `netCDF4..Group` in
        the `netCDF4..Dataset` in a unix directory format (the names of groups in the
        hierarchy separated by backslashes). A `netCDF4..Dataset` instance is the root
        group, so the path is simply `'/'`.

    variables
        The `variables` dictionary maps the names of variables
        defined for this `netCDF4.Dataset` or `netCDF4.Group` to instances of the `netCDF4.Variable`
        class.

    vltypes
        The `vltypes` dictionary maps the names of
        variable-length types defined for the `netCDF4.Group` or `netCDF4.Dataset` to instances of the
        `netCDF4.VLType` class.

    Methods
    -------
    __init__(...)
        **`__init__(self, filename, mode="r", clobber=True, diskless=False,
        persist=False, weakref=False, format='NETCDF4')`**

        `netCDF4.Dataset` constructor.

        **`filename`**: Name of netCDF file to hold dataset.

        **`mode`**: access mode. `r` means read-only; no data can be
        modified. `w` means write; a new file is created, an existing file with
        the same name is deleted. `a` and `r+` mean append (in analogy with
        serial files); an existing file is opened for reading and writing.
        Appending `s` to modes `w`, `r+` or `a` will enable unbuffered shared
        access to `NETCDF3_CLASSIC` or `NETCDF3_64BIT` formatted files.
        Unbuffered acesss may be useful even if you don't need shared
        access, since it may be faster for programs that don't access data
        sequentially. This option is ignored for `NETCDF4` and `NETCDF4_CLASSIC`
        formatted files.

        **`clobber`**: if `True` (default), opening a file with `mode='w'`
        will clobber an existing file with the same name.  if `False`, an
        exception will be raised if a file with the same name already exists.

        **`format`**: underlying file format (one of `'NETCDF4',
        'NETCDF4_CLASSIC', 'NETCDF3_CLASSIC'` or `'NETCDF3_64BIT'`.  Only
        relevant if `mode = 'w'` (if `mode = 'r','a'` or `'r+'` the file format
        is automatically detected). Default `'NETCDF4'`, which means the data is
        stored in an HDF5 file, using netCDF 4 API features.  Setting
        `format='NETCDF4_CLASSIC'` will create an HDF5 file, using only netCDF 3
        compatibile API features. netCDF 3 clients must be recompiled and linked
        against the netCDF 4 library to read files in `NETCDF4_CLASSIC` format.
        `'NETCDF3_CLASSIC'` is the classic netCDF 3 file format that does not
        handle 2+ Gb files very well. `'NETCDF3_64BIT'` is the 64-bit offset
        version of the netCDF 3 file format, which fully supports 2+ GB files, but
        is only compatible with clients linked against netCDF version 3.6.0 or
        later.

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

    close(...)
        **`close(self)`**

        Close the Dataset.

    createCompoundType(...)
        **`createCompoundType(self, datatype, datatype_name)`**

        Creates a new compound data type named `datatype_name` from the numpy
        dtype object `datatype`.

        ***Note***: If the new compound data type contains other compound data types
        (i.e. it is a 'nested' compound type, where not all of the elements
        are homogenous numeric data types), then the 'inner' compound types **must** be
        created first.

        The return value is the `netCDF4.CompoundType` class instance describing the new
        datatype.

    createDimension(...)
        **`createDimension(self, dimname, size=None)`**

        Creates a new dimension with the given `dimname` and `size`.

        `size` must be a positive integer or `None`, which stands for
        "unlimited" (default is `None`). Specifying a size of 0 also
        results in an unlimited dimension. The return value is the `netCDF4.Dimension`
        class instance describing the new dimension.  To determine the current
        maximum size of the dimension, use the `len` function on the `netCDF4.Dimension`
        instance. To determine if a dimension is 'unlimited', use the
        `netCDF4.Dimension.isunlimited` method of the `netCDF4.Dimension` instance.

    createGroup(...)
        **`createGroup(self, groupname)`**

        Creates a new `netCDF4.Group` with the given `groupname`.

        If `groupname` is specified as a path, using forward slashes as in unix to
        separate components, then intermediate groups will be created as necessary 
        (analagous to `mkdir -p` in unix).  For example,
        `createGroup('/GroupA/GroupB/GroupC')` will create `GroupA`,
        `GroupA/GroupB`, and `GroupA/GroupB/GroupC`, if they don't already exist.
        If the specified path describes a group that already exists, no error is
        raised.

        The return value is a `netCDF4.Group` class instance.

    createVLType(...)
        **`createVLType(self, datatype, datatype_name)`**

        Creates a new VLEN data type named `datatype_name` from a numpy
        dtype object `datatype`.

        The return value is the `netCDF4.VLType` class instance describing the new
        datatype.

    createVariable(...)
        **`createVariable(self, varname, datatype, dimensions=(), zlib=False,
        complevel=4, shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,
        endian='native', least_significant_digit=None, fill_value=None)`**

        Creates a new variable with the given `varname`, `datatype`, and
        `dimensions`. If dimensions are not given, the variable is assumed to be
        a scalar.

        If `varname` is specified as a path, using forward slashes as in unix to
        separate components, then intermediate groups will be created as necessary 
        For example, `createVariable('/GroupA/GroupB/VarC'),('x','y'),float)` will create groups `GroupA`
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
        have been defined previously using `netCDF4.createDimension`. The default value
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
        any data is written to it, defaults given in netCDF4.default_fillvals).
        If fill_value is set to `False`, then the variable is not pre-filled.

        If the optional keyword parameter `least_significant_digit` is
        specified, variable data will be truncated (quantized). In conjunction
        with `zlib=True` this produces 'lossy', but significantly more
        efficient compression. For example, if `least_significant_digit=1`,
        data will be quantized using `numpy.around(scale*data)/scale`, where
        scale = 2**bits, and bits is determined so that a precision of 0.1 is
        retained (in this case bits=4). From
        [](http://www.cdc.noaa.gov/cdc/conventions/cdc_netcdf_standard.shtml):
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
        is the number of variable dimensions.

    delncattr(...)
        **`delncattr(self,name,value)`**

        delete a netCDF dataset or group attribute.  Use if you need to delete a
        netCDF attribute with the same name as one of the reserved python
        attributes.

    filepath(...)
        **`filepath(self)`**

        Get the file system path (or the opendap URL) which was used to
        open/create the Dataset. Requires netcdf >= 4.1.2

    getncattr(...)
        **`getncattr(self,name)`**

        retrievel a netCDF dataset or group attribute.
        Use if you need to get a netCDF attribute with the same 
        name as one of the reserved python attributes.

    ncattrs(...)
        **`ncattrs(self)`**

        return netCDF global attribute names for this `netCDF4.Dataset` or `netCDF4.Group` in a list.

    renameAttribute(...)
        **`renameAttribute(self, oldname, newname)`**

        rename a `netCDF4.Dataset` or `netCDF4.Group` attribute named `oldname` to `newname`.

    renameDimension(...)
        **`renameDimension(self, oldname, newname)`**

        rename a `netCDF4.Dimension` named `oldname` to `newname`.

    renameGroup(...)
        **`renameGroup(self, oldname, newname)`**

        rename a `netCDF4.Group` named `oldname` to `newname` (requires netcdf >= 4.3.1).

    renameVariable(...)
        **`renameVariable(self, oldname, newname)`**

        rename a `netCDF4.Variable` named `oldname` to `newname`

    set_auto_mask(...)
        **`set_auto_mask(self, True_or_False)`**

        Call `netCDF4.set_auto_mask` for all variables contained in this `netCDF4.Dataset` or
        `netCDF4.Group`, as well as for all variables in all its subgroups.

        **`True_or_False`**: Boolean determining if automatic conversion to masked arrays
        shall be applied for all variables.

        ***Note***: Calling this function only affects existing variables. Variables created
        after calling this function will follow the default behaviour.

    set_auto_maskandscale(...)
        **`set_auto_maskandscale(self, True_or_False)`**

        Call `netCDF4.set_auto_maskandscale` for all variables contained in this `netCDF4.Dataset` or
        `netCDF4.Group`, as well as for all variables in all its subgroups.

        **`True_or_False`**: Boolean determining if automatic conversion to masked arrays
        and variable scaling shall be applied for all variables.

        ***Note***: Calling this function only affects existing variables. Variables created
        after calling this function will follow the default behaviour.

    set_auto_scale(...)
        **`set_auto_scale(self, True_or_False)`**

        Call `netCDF4.set_auto_scale` for all variables contained in this `netCDF4.Dataset` or
        `netCDF4.Group`, as well as for all variables in all its subgroups.

        **`True_or_False`**: Boolean determining if automatic variable scaling
        shall be applied for all variables.

        ***Note***: Calling this function only affects existing variables. Variables created
        after calling this function will follow the default behaviour.

    set_fill_off(...)
        **`set_fill_off(self)`**

        Sets the fill mode for a `netCDF4.Dataset` open for writing to `off`.

        This will prevent the data from being pre-filled with fill values, which
        may result in some performance improvements. However, you must then make
        sure the data is actually written before being read.

    set_fill_on(...)
        **`set_fill_on(self)`**

        Sets the fill mode for a `netCDF4.Dataset` open for writing to `on`.

        This causes data to be pre-filled with fill values. The fill values can be
        controlled by the variable's `_Fill_Value` attribute, but is usually
        sufficient to the use the netCDF default `_Fill_Value` (defined
        separately for each variable type). The default behavior of the netCDF
        library correspongs to `set_fill_on`.  Data which are equal to the
        `_Fill_Value` indicate that the variable was created, but never written
        to.

    setncattr(...)
        **`setncattr(self,name,value)`**

        set a netCDF dataset or group attribute using name,value pair.
        Use if you need to set a netCDF attribute with the
        with the same name as one of the reserved python attributes.

    setncatts(...)
        **`setncatts(self,attdict)`**

        set a bunch of netCDF dataset or group attributes at once using a python dictionary.
        This may be faster when setting a lot of attributes for a `NETCDF3`
        formatted file, since nc_redef/nc_enddef is not called in between setting
        each attribute

    sync(...)
        **`sync(self)`**

        Writes all buffered data in the `netCDF4.Dataset` to the disk file.

Dimension 
    A netCDF `netCDF4.Dimension` is used to describe the coordinates of a `netCDF4.Variable`.

    The current maximum size of a `netCDF4.Dimension` instance can be obtained by
    calling the python `len` function on the `netCDF4.Dimension` instance. The
    `netCDF4.Dimension.isunlimited` method of a `netCDF4.Dimension` instance can be used to
    determine if the dimension is unlimited

    Ancestors (in MRO)
    ------------------
    netCDF4.Dimension
    __builtin__.object

    Class variables
    ---------------
    name
        A string describing the name of the `netCDF4.Dimension` - used when creating a
        `netCDF4.Variable` instance with `netCDF4.Dataset.createVariable`.

    Methods
    -------
    __init__(...)
        **`__init__(self, group, name, size=None)`**

        `netCDF4.Dimension` constructor.

        **`group`**: `netCDF4.Group` instance to associate with dimension.

        **`name`**: Name of the dimension.

        **`size`**: Size of the dimension. `None` or 0 means unlimited. (Default `None`).

        ***Note***: `netCDF4.Dimension` instances should be created using the
        `netCDF4.Dataset.createDimension` method of a `netCDF4.Group` or
        `netCDF4.Dataset` instance, not using `netCDF4.Dimension.__init__` directly.

    group(...)
        **`group(self)`**

        return the group that this `netCDF4.Dimension` is a member of.

    isunlimited(...)
        **`isunlimited(self)`**

        returns `True` if the `netCDF4.Dimension` instance is unlimited, `False` otherwise.

Group 
    Groups define a hierarchical namespace within a netCDF file. They are
    analagous to directories in a unix filesystem. Each `netCDF4.Group` behaves like
    a `netCDF4.Dataset` within a Dataset, and can contain it's own variables,
    dimensions and attributes (and other Groups).

    `netCDF4.Group` inherits from `netCDF4.Dataset`, so all the `netCDF4.Dataset` class methods and
    variables are available to a `netCDF4.Group` instance (except the `close`
    method).

    Ancestors (in MRO)
    ------------------
    netCDF4.Group
    netCDF4.Dataset
    __builtin__.object

    Class variables
    ---------------
    cmptypes
        The `cmptypes` dictionary maps the names of
        compound types defined for the `netCDF4.Group` or `netCDF4.Dataset` to instances of the
        `netCDF4.CompoundType` class.

    data_model
        `data_model` describes the netCDF
        data model version, one of `NETCDF3_CLASSIC`, `NETCDF4`,
        `NETCDF4_CLASSIC` or `NETCDF3_64BIT`.

    dimensions
        The `dimensions` dictionary maps the names of
        dimensions defined for the `netCDF4.Group` or `netCDF4.Dataset` to instances of the
        `netCDF4.Dimension` class.

    disk_format
        `disk_format` describes the underlying
        file format, one of `NETCDF3`, `HDF5`, `HDF4`,
        `PNETCDF`, `DAP2`, `DAP4` or `UNDEFINED`. Only available if using
        netcdf C library version >= 4.3.1, otherwise will always return
        `UNDEFINED`.

    file_format
        same as `data_model`, retained for backwards compatibility.

    groups
        The groups dictionary maps the names of groups created for
        this `netCDF4.Dataset` or `netCDF4.Group` to instances of the `netCDF4.Group` class (the
        `netCDF4.Dataset` class is simply a special case of the `netCDF4.Group` class which
        describes the root group in the netCDF4.file).

    keepweakref
        If `True`, child Dimension and Variables objects only keep weak references to
        parent Dataset or Group.

    name
        A string describing the name of the `netCDF4.Group`.

    parent
        `parent` is a reference to the parent
        `netCDF4.Group` instance. `None` for a the root group or `netCDF4.Dataset` instance

    path
        `path` shows the location of the `netCDF4..Group` in
        the `netCDF4..Dataset` in a unix directory format (the names of groups in the
        hierarchy separated by backslashes). A `netCDF4..Dataset` instance is the root
        group, so the path is simply `'/'`.

    variables
        The `variables` dictionary maps the names of variables
        defined for this `netCDF4.Dataset` or `netCDF4.Group` to instances of the `netCDF4.Variable`
        class.

    vltypes
        The `vltypes` dictionary maps the names of
        variable-length types defined for the `netCDF4.Group` or `netCDF4.Dataset` to instances of the
        `netCDF4.VLType` class.

    Methods
    -------
    __init__(...)
        **`__init__(self, parent, name)`**
        `netCDF4.Group` constructor.

        **`parent`**: `netCDF4.Group` instance for the parent group.  If being created
        in the root group, use a `netCDF4.Dataset` instance.

        **`name`**: - Name of the group.

        ***Note***: `netCDF4.Group` instances should be created using the
        `netCDF4.Dataset.createGroup` method of a `netCDF4.Dataset` instance, or
        another `netCDF4.Group` instance, not using this class directly.

    close(...)
        **`close(self)`**

        overrides `netCDF4.Dataset` close method which does not apply to `netCDF4.Group`
        instances, raises IOError.

    createCompoundType(...)
        **`createCompoundType(self, datatype, datatype_name)`**

        Creates a new compound data type named `datatype_name` from the numpy
        dtype object `datatype`.

        ***Note***: If the new compound data type contains other compound data types
        (i.e. it is a 'nested' compound type, where not all of the elements
        are homogenous numeric data types), then the 'inner' compound types **must** be
        created first.

        The return value is the `netCDF4.CompoundType` class instance describing the new
        datatype.

    createDimension(...)
        **`createDimension(self, dimname, size=None)`**

        Creates a new dimension with the given `dimname` and `size`.

        `size` must be a positive integer or `None`, which stands for
        "unlimited" (default is `None`). Specifying a size of 0 also
        results in an unlimited dimension. The return value is the `netCDF4.Dimension`
        class instance describing the new dimension.  To determine the current
        maximum size of the dimension, use the `len` function on the `netCDF4.Dimension`
        instance. To determine if a dimension is 'unlimited', use the
        `netCDF4.Dimension.isunlimited` method of the `netCDF4.Dimension` instance.

    createGroup(...)
        **`createGroup(self, groupname)`**

        Creates a new `netCDF4.Group` with the given `groupname`.

        If `groupname` is specified as a path, using forward slashes as in unix to
        separate components, then intermediate groups will be created as necessary 
        (analagous to `mkdir -p` in unix).  For example,
        `createGroup('/GroupA/GroupB/GroupC')` will create `GroupA`,
        `GroupA/GroupB`, and `GroupA/GroupB/GroupC`, if they don't already exist.
        If the specified path describes a group that already exists, no error is
        raised.

        The return value is a `netCDF4.Group` class instance.

    createVLType(...)
        **`createVLType(self, datatype, datatype_name)`**

        Creates a new VLEN data type named `datatype_name` from a numpy
        dtype object `datatype`.

        The return value is the `netCDF4.VLType` class instance describing the new
        datatype.

    createVariable(...)
        **`createVariable(self, varname, datatype, dimensions=(), zlib=False,
        complevel=4, shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,
        endian='native', least_significant_digit=None, fill_value=None)`**

        Creates a new variable with the given `varname`, `datatype`, and
        `dimensions`. If dimensions are not given, the variable is assumed to be
        a scalar.

        If `varname` is specified as a path, using forward slashes as in unix to
        separate components, then intermediate groups will be created as necessary 
        For example, `createVariable('/GroupA/GroupB/VarC'),('x','y'),float)` will create groups `GroupA`
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
        have been defined previously using `netCDF4.createDimension`. The default value
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
        any data is written to it, defaults given in netCDF4.default_fillvals).
        If fill_value is set to `False`, then the variable is not pre-filled.

        If the optional keyword parameter `least_significant_digit` is
        specified, variable data will be truncated (quantized). In conjunction
        with `zlib=True` this produces 'lossy', but significantly more
        efficient compression. For example, if `least_significant_digit=1`,
        data will be quantized using `numpy.around(scale*data)/scale`, where
        scale = 2**bits, and bits is determined so that a precision of 0.1 is
        retained (in this case bits=4). From
        [](http://www.cdc.noaa.gov/cdc/conventions/cdc_netcdf_standard.shtml):
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
        is the number of variable dimensions.

    delncattr(...)
        **`delncattr(self,name,value)`**

        delete a netCDF dataset or group attribute.  Use if you need to delete a
        netCDF attribute with the same name as one of the reserved python
        attributes.

    filepath(...)
        **`filepath(self)`**

        Get the file system path (or the opendap URL) which was used to
        open/create the Dataset. Requires netcdf >= 4.1.2

    getncattr(...)
        **`getncattr(self,name)`**

        retrievel a netCDF dataset or group attribute.
        Use if you need to get a netCDF attribute with the same 
        name as one of the reserved python attributes.

    ncattrs(...)
        **`ncattrs(self)`**

        return netCDF global attribute names for this `netCDF4.Dataset` or `netCDF4.Group` in a list.

    renameAttribute(...)
        **`renameAttribute(self, oldname, newname)`**

        rename a `netCDF4.Dataset` or `netCDF4.Group` attribute named `oldname` to `newname`.

    renameDimension(...)
        **`renameDimension(self, oldname, newname)`**

        rename a `netCDF4.Dimension` named `oldname` to `newname`.

    renameGroup(...)
        **`renameGroup(self, oldname, newname)`**

        rename a `netCDF4.Group` named `oldname` to `newname` (requires netcdf >= 4.3.1).

    renameVariable(...)
        **`renameVariable(self, oldname, newname)`**

        rename a `netCDF4.Variable` named `oldname` to `newname`

    set_auto_mask(...)
        **`set_auto_mask(self, True_or_False)`**

        Call `netCDF4.set_auto_mask` for all variables contained in this `netCDF4.Dataset` or
        `netCDF4.Group`, as well as for all variables in all its subgroups.

        **`True_or_False`**: Boolean determining if automatic conversion to masked arrays
        shall be applied for all variables.

        ***Note***: Calling this function only affects existing variables. Variables created
        after calling this function will follow the default behaviour.

    set_auto_maskandscale(...)
        **`set_auto_maskandscale(self, True_or_False)`**

        Call `netCDF4.set_auto_maskandscale` for all variables contained in this `netCDF4.Dataset` or
        `netCDF4.Group`, as well as for all variables in all its subgroups.

        **`True_or_False`**: Boolean determining if automatic conversion to masked arrays
        and variable scaling shall be applied for all variables.

        ***Note***: Calling this function only affects existing variables. Variables created
        after calling this function will follow the default behaviour.

    set_auto_scale(...)
        **`set_auto_scale(self, True_or_False)`**

        Call `netCDF4.set_auto_scale` for all variables contained in this `netCDF4.Dataset` or
        `netCDF4.Group`, as well as for all variables in all its subgroups.

        **`True_or_False`**: Boolean determining if automatic variable scaling
        shall be applied for all variables.

        ***Note***: Calling this function only affects existing variables. Variables created
        after calling this function will follow the default behaviour.

    set_fill_off(...)
        **`set_fill_off(self)`**

        Sets the fill mode for a `netCDF4.Dataset` open for writing to `off`.

        This will prevent the data from being pre-filled with fill values, which
        may result in some performance improvements. However, you must then make
        sure the data is actually written before being read.

    set_fill_on(...)
        **`set_fill_on(self)`**

        Sets the fill mode for a `netCDF4.Dataset` open for writing to `on`.

        This causes data to be pre-filled with fill values. The fill values can be
        controlled by the variable's `_Fill_Value` attribute, but is usually
        sufficient to the use the netCDF default `_Fill_Value` (defined
        separately for each variable type). The default behavior of the netCDF
        library correspongs to `set_fill_on`.  Data which are equal to the
        `_Fill_Value` indicate that the variable was created, but never written
        to.

    setncattr(...)
        **`setncattr(self,name,value)`**

        set a netCDF dataset or group attribute using name,value pair.
        Use if you need to set a netCDF attribute with the
        with the same name as one of the reserved python attributes.

    setncatts(...)
        **`setncatts(self,attdict)`**

        set a bunch of netCDF dataset or group attributes at once using a python dictionary.
        This may be faster when setting a lot of attributes for a `NETCDF3`
        formatted file, since nc_redef/nc_enddef is not called in between setting
        each attribute

    sync(...)
        **`sync(self)`**

        Writes all buffered data in the `netCDF4.Dataset` to the disk file.

MFDataset 
    Class for reading multi-file netCDF Datasets, making variables
    spanning multiple files appear as if they were in one file.

    Datasets must be in `NETCDF4_CLASSIC, NETCDF3_CLASSIC or NETCDF3_64BIT`
    format (`NETCDF4` Datasets won't work).

    Adapted from [pycdf](http://pysclint.sourceforge.net/pycdf) by Andre Gosselin.

    Example usage (See `netCDF4.MFDataset.__init__` for more details):

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

    Ancestors (in MRO)
    ------------------
    netCDF4.MFDataset
    netCDF4.Dataset
    __builtin__.object

    Class variables
    ---------------
    cmptypes
        The `cmptypes` dictionary maps the names of
        compound types defined for the `netCDF4.Group` or `netCDF4.Dataset` to instances of the
        `netCDF4.CompoundType` class.

    data_model
        `data_model` describes the netCDF
        data model version, one of `NETCDF3_CLASSIC`, `NETCDF4`,
        `NETCDF4_CLASSIC` or `NETCDF3_64BIT`.

    dimensions
        The `dimensions` dictionary maps the names of
        dimensions defined for the `netCDF4.Group` or `netCDF4.Dataset` to instances of the
        `netCDF4.Dimension` class.

    disk_format
        `disk_format` describes the underlying
        file format, one of `NETCDF3`, `HDF5`, `HDF4`,
        `PNETCDF`, `DAP2`, `DAP4` or `UNDEFINED`. Only available if using
        netcdf C library version >= 4.3.1, otherwise will always return
        `UNDEFINED`.

    file_format
        same as `data_model`, retained for backwards compatibility.

    groups
        The groups dictionary maps the names of groups created for
        this `netCDF4.Dataset` or `netCDF4.Group` to instances of the `netCDF4.Group` class (the
        `netCDF4.Dataset` class is simply a special case of the `netCDF4.Group` class which
        describes the root group in the netCDF4.file).

    keepweakref
        If `True`, child Dimension and Variables objects only keep weak references to
        parent Dataset or Group.

    parent
        `parent` is a reference to the parent
        `netCDF4.Group` instance. `None` for a the root group or `netCDF4.Dataset` instance

    path
        `path` shows the location of the `netCDF4..Group` in
        the `netCDF4..Dataset` in a unix directory format (the names of groups in the
        hierarchy separated by backslashes). A `netCDF4..Dataset` instance is the root
        group, so the path is simply `'/'`.

    variables
        The `variables` dictionary maps the names of variables
        defined for this `netCDF4.Dataset` or `netCDF4.Group` to instances of the `netCDF4.Variable`
        class.

    vltypes
        The `vltypes` dictionary maps the names of
        variable-length types defined for the `netCDF4.Group` or `netCDF4.Dataset` to instances of the
        `netCDF4.VLType` class.

    Methods
    -------
    __init__(...)
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

    close(...)
        **`close(self)`**

        Close the Dataset.

    createCompoundType(...)
        **`createCompoundType(self, datatype, datatype_name)`**

        Creates a new compound data type named `datatype_name` from the numpy
        dtype object `datatype`.

        ***Note***: If the new compound data type contains other compound data types
        (i.e. it is a 'nested' compound type, where not all of the elements
        are homogenous numeric data types), then the 'inner' compound types **must** be
        created first.

        The return value is the `netCDF4.CompoundType` class instance describing the new
        datatype.

    createDimension(...)
        **`createDimension(self, dimname, size=None)`**

        Creates a new dimension with the given `dimname` and `size`.

        `size` must be a positive integer or `None`, which stands for
        "unlimited" (default is `None`). Specifying a size of 0 also
        results in an unlimited dimension. The return value is the `netCDF4.Dimension`
        class instance describing the new dimension.  To determine the current
        maximum size of the dimension, use the `len` function on the `netCDF4.Dimension`
        instance. To determine if a dimension is 'unlimited', use the
        `netCDF4.Dimension.isunlimited` method of the `netCDF4.Dimension` instance.

    createGroup(...)
        **`createGroup(self, groupname)`**

        Creates a new `netCDF4.Group` with the given `groupname`.

        If `groupname` is specified as a path, using forward slashes as in unix to
        separate components, then intermediate groups will be created as necessary 
        (analagous to `mkdir -p` in unix).  For example,
        `createGroup('/GroupA/GroupB/GroupC')` will create `GroupA`,
        `GroupA/GroupB`, and `GroupA/GroupB/GroupC`, if they don't already exist.
        If the specified path describes a group that already exists, no error is
        raised.

        The return value is a `netCDF4.Group` class instance.

    createVLType(...)
        **`createVLType(self, datatype, datatype_name)`**

        Creates a new VLEN data type named `datatype_name` from a numpy
        dtype object `datatype`.

        The return value is the `netCDF4.VLType` class instance describing the new
        datatype.

    createVariable(...)
        **`createVariable(self, varname, datatype, dimensions=(), zlib=False,
        complevel=4, shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,
        endian='native', least_significant_digit=None, fill_value=None)`**

        Creates a new variable with the given `varname`, `datatype`, and
        `dimensions`. If dimensions are not given, the variable is assumed to be
        a scalar.

        If `varname` is specified as a path, using forward slashes as in unix to
        separate components, then intermediate groups will be created as necessary 
        For example, `createVariable('/GroupA/GroupB/VarC'),('x','y'),float)` will create groups `GroupA`
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
        have been defined previously using `netCDF4.createDimension`. The default value
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
        any data is written to it, defaults given in netCDF4.default_fillvals).
        If fill_value is set to `False`, then the variable is not pre-filled.

        If the optional keyword parameter `least_significant_digit` is
        specified, variable data will be truncated (quantized). In conjunction
        with `zlib=True` this produces 'lossy', but significantly more
        efficient compression. For example, if `least_significant_digit=1`,
        data will be quantized using `numpy.around(scale*data)/scale`, where
        scale = 2**bits, and bits is determined so that a precision of 0.1 is
        retained (in this case bits=4). From
        [](http://www.cdc.noaa.gov/cdc/conventions/cdc_netcdf_standard.shtml):
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
        is the number of variable dimensions.

    delncattr(...)
        **`delncattr(self,name,value)`**

        delete a netCDF dataset or group attribute.  Use if you need to delete a
        netCDF attribute with the same name as one of the reserved python
        attributes.

    filepath(...)
        **`filepath(self)`**

        Get the file system path (or the opendap URL) which was used to
        open/create the Dataset. Requires netcdf >= 4.1.2

    getncattr(...)
        **`getncattr(self,name)`**

        retrievel a netCDF dataset or group attribute.
        Use if you need to get a netCDF attribute with the same 
        name as one of the reserved python attributes.

    ncattrs(...)
        **`ncattrs(self)`**

        return netCDF global attribute names for this `netCDF4.Dataset` or `netCDF4.Group` in a list.

    renameAttribute(...)
        **`renameAttribute(self, oldname, newname)`**

        rename a `netCDF4.Dataset` or `netCDF4.Group` attribute named `oldname` to `newname`.

    renameDimension(...)
        **`renameDimension(self, oldname, newname)`**

        rename a `netCDF4.Dimension` named `oldname` to `newname`.

    renameGroup(...)
        **`renameGroup(self, oldname, newname)`**

        rename a `netCDF4.Group` named `oldname` to `newname` (requires netcdf >= 4.3.1).

    renameVariable(...)
        **`renameVariable(self, oldname, newname)`**

        rename a `netCDF4.Variable` named `oldname` to `newname`

    set_auto_mask(...)
        **`set_auto_mask(self, True_or_False)`**

        Call `netCDF4.set_auto_mask` for all variables contained in this `netCDF4.Dataset` or
        `netCDF4.Group`, as well as for all variables in all its subgroups.

        **`True_or_False`**: Boolean determining if automatic conversion to masked arrays
        shall be applied for all variables.

        ***Note***: Calling this function only affects existing variables. Variables created
        after calling this function will follow the default behaviour.

    set_auto_maskandscale(...)
        **`set_auto_maskandscale(self, True_or_False)`**

        Call `netCDF4.set_auto_maskandscale` for all variables contained in this `netCDF4.Dataset` or
        `netCDF4.Group`, as well as for all variables in all its subgroups.

        **`True_or_False`**: Boolean determining if automatic conversion to masked arrays
        and variable scaling shall be applied for all variables.

        ***Note***: Calling this function only affects existing variables. Variables created
        after calling this function will follow the default behaviour.

    set_auto_scale(...)
        **`set_auto_scale(self, True_or_False)`**

        Call `netCDF4.set_auto_scale` for all variables contained in this `netCDF4.Dataset` or
        `netCDF4.Group`, as well as for all variables in all its subgroups.

        **`True_or_False`**: Boolean determining if automatic variable scaling
        shall be applied for all variables.

        ***Note***: Calling this function only affects existing variables. Variables created
        after calling this function will follow the default behaviour.

    set_fill_off(...)
        **`set_fill_off(self)`**

        Sets the fill mode for a `netCDF4.Dataset` open for writing to `off`.

        This will prevent the data from being pre-filled with fill values, which
        may result in some performance improvements. However, you must then make
        sure the data is actually written before being read.

    set_fill_on(...)
        **`set_fill_on(self)`**

        Sets the fill mode for a `netCDF4.Dataset` open for writing to `on`.

        This causes data to be pre-filled with fill values. The fill values can be
        controlled by the variable's `_Fill_Value` attribute, but is usually
        sufficient to the use the netCDF default `_Fill_Value` (defined
        separately for each variable type). The default behavior of the netCDF
        library correspongs to `set_fill_on`.  Data which are equal to the
        `_Fill_Value` indicate that the variable was created, but never written
        to.

    setncattr(...)
        **`setncattr(self,name,value)`**

        set a netCDF dataset or group attribute using name,value pair.
        Use if you need to set a netCDF attribute with the
        with the same name as one of the reserved python attributes.

    setncatts(...)
        **`setncatts(self,attdict)`**

        set a bunch of netCDF dataset or group attributes at once using a python dictionary.
        This may be faster when setting a lot of attributes for a `NETCDF3`
        formatted file, since nc_redef/nc_enddef is not called in between setting
        each attribute

    sync(...)
        **`sync(self)`**

        Writes all buffered data in the `netCDF4.Dataset` to the disk file.

MFTime 
    Class providing an interface to a MFDataset time Variable by imposing a unique common
    time unit to all files.

    Example usage (See `netCDF4.MFTime.__init__` for more details):

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

    Ancestors (in MRO)
    ------------------
    netCDF4.MFTime
    netCDF4._netCDF4._Variable
    __builtin__.object

    Methods
    -------
    __init__(...)
        **`__init__(self, time, units=None)`**

        Create a time Variable with units consistent across a multifile
        dataset.

        **`time`**: Time variable from a `netCDF4.MFDataset`.

        **`units`**: Time units, for example, `days since 1979-01-01`. If None, use
        the units from the master variable.

    ncattrs(...)

    set_auto_maskandscale(...)

    typecode(...)

VLType 
    A `netCDF4.VLType` instance is used to describe a variable length (VLEN) data
    type, and can be passed to the the `netCDF4.Dataset.createVariable` method of
    a `netCDF4.Dataset` or `netCDF4.Group` instance.

    The instance variables `dtype` and `name` should not be modified by
    the user.

    Ancestors (in MRO)
    ------------------
    netCDF4.VLType
    __builtin__.object

    Class variables
    ---------------
    dtype
        A numpy dtype object describing the component type for the VLEN.

    name
        String name.

    Methods
    -------
    __init__(...)
        **`__init__(group, datatype, datatype_name)``**

        VLType constructor.

        **`group`**: `netCDF4.Group` instance to associate with the VLEN datatype.

        **`datatype`**: An numpy dtype object describing a the component type for the
        variable length array.

        **`datatype_name`**: a Python string containing a description of the
        VLEN data type.

        ***`Note`***: `netCDF4.VLType` instances should be created using the
        `netCDF4.Dataset.createVLType`
        method of a Dataset or `netCDF4.Group` instance, not using this class directly.

Variable 
    A netCDF `netCDF4.Variable` is used to read and write netCDF data.  They are
    analagous to numpy array objects.

    A list of attribute names corresponding to netCDF attributes defined for
    the variable can be obtained with the `netCDF4.Variable.ncattrs` method. These
    attributes can be created by assigning to an attribute of the
    `netCDF4.Variable` instance. A dictionary containing all the netCDF attribute
    name/value pairs is provided by the `__dict__` attribute of a
    `netCDF4.Variable` instance.

    The instance variables `dimensions, dtype, ndim, shape`
    and `least_significant_digit` are read-only (and
    should not be modified by the user).

    Ancestors (in MRO)
    ------------------
    netCDF4.Variable
    __builtin__.object

    Class variables
    ---------------
    datatype
        numpy data type (for primitive data types) or VLType/CompoundType
        instance (for compound or vlen data types).

    dimensions
        A tuple containing the names of the
        dimensions associated with this variable.

    dtype
        A numpy dtype object describing the
        variable's data type.

    mask
        If True, data is automatically converted to/from masked 
        arrays when missing values or fill values are present. Default is `True`, can be
        reset using `netCDF4.set_auto_mask` and `netCDF4.set_auto_maskandscale`
        methods.

    name
        String name.

    ndim
        The number of variable dimensions.

    scale
        if True, `scale_factor` and `add_offset` are
        applied. Default is `True`, can be reset using `netCDF4.set_auto_scale` and
        `netCDF4.set_auto_maskandscale` methods.

    shape
        A tuple with the current shape (length of all dimensions).

    size
        The number of stored elements.

    Methods
    -------
    __init__(...)
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

        **`dimensions`** - a tuple containing the variable's dimension names
        (defined previously with `createDimension`). Default is an empty tuple
        which means the variable is a scalar (and therefore has no dimensions).

        **`zlib`** - if `True`, data assigned to the `netCDF4.Variable`
        instance is compressed on disk. Default `False`.

        **`complevel`** - the level of zlib compression to use (1 is the fastest,
        but poorest compression, 9 is the slowest but best compression). Default 4.
        Ignored if `zlib=False`.

        **`shuffle`** - if `True`, the HDF5 shuffle filter is applied
        to improve compression. Default `True`. Ignored if `zlib=False`.

        **`fletcher32`** - if `True` (default `False`), the Fletcher32 checksum
        algorithm is used for error detection.

        **`contiguous`** - if `True` (default `False`), the variable data is
        stored contiguously on disk.  Default `False`. Setting to `True` for
        a variable with an unlimited dimension will trigger an error.

        **`chunksizes`** - Can be used to specify the HDF5 chunksizes for each
        dimension of the variable. A detailed discussion of HDF chunking and I/O
        performance is available U{here
        <http://www.hdfgroup.org/HDF5/doc/H5.user/Chunking.html>`.
        Basically, you want the chunk size for each dimension to match as
        closely as possible the size of the data block that users will read
        from the file. `chunksizes` cannot be set if `contiguous=True`.

        **`endian`** - Can be used to control whether the
        data is stored in little or big endian format on disk. Possible
        values are `little, big` or `native` (default). The library
        will automatically handle endian conversions when the data is read,
        but if the data is always going to be read on a computer with the
        opposite format as the one used to create the file, there may be
        some performance advantage to be gained by setting the endian-ness.
        For netCDF 3 files (that don't use HDF5), only `endian='native'` is allowed.

        The `zlib, complevel, shuffle, fletcher32, contiguous` and {chunksizes`
        keywords are silently ignored for netCDF 3 files that do not use HDF5.

        **`least_significant_digit`** - If specified, variable data will be
        truncated (quantized). In conjunction with `zlib=True` this produces
        'lossy', but significantly more efficient compression. For example, if
        `least_significant_digit=1`, data will be quantized using
        around(scale*data)/scale, where scale = 2**bits, and bits is determined
        so that a precision of 0.1 is retained (in this case bits=4). Default is
        `None`, or no quantization.

        **`fill_value`** - If specified, the default netCDF `_FillValue` (the
        value that the variable gets filled with before any data is written to it)
        is replaced with this value.  If fill_value is set to `False`, then
        the variable is not pre-filled. The default netCDF fill values can be found
        in netCDF4.default_fillvals.

        ***Note***: `netCDF4.Variable` instances should be created using the
        `netCDF4.Dataset.createVariable` method of a `netCDF4.Dataset` or
        `netCDF4.Group` instance, not using this class directly.

    assignValue(...)
        **`assignValue(self, val)`**

        assign a value to a scalar variable.  Provided for compatibility with
        Scientific.IO.NetCDF, can also be done by assigning to an Ellipsis slice ([...]).

    chunking(...)
        **`chunking(self)`**

        return variable chunking information.  If the dataset is
        defined to be contiguous (and hence there is no chunking) the word 'contiguous'
        is returned.  Otherwise, a sequence with the chunksize for
        each dimension is returned.

    delncattr(...)
        **`delncattr(self,name,value)`**

        delete a netCDF variable attribute.  Use if you need to delete a
        netCDF attribute with the same name as one of the reserved python
        attributes.

    endian(...)
        **`endian(self)`**

        return endian-ness (`little,big,native`) of variable (as stored in HDF5 file).

    filters(...)
        **`filters(self)`**

        return dictionary containing HDF5 filter parameters.

    getValue(...)
        **`getValue(self)`**

        get the value of a scalar variable.  Provided for compatibility with
        Scientific.IO.NetCDF, can also be done by slicing with an Ellipsis ([...]).

    get_var_chunk_cache(...)
        **`get_var_chunk_cache(self)`**

        return variable chunk cache information in a tuple (size,nelems,preemption).
        See netcdf C library documentation for `nc_get_var_chunk_cache` for
        details.

    getncattr(...)
        **`getncattr(self,name)`**

        retrievel a netCDF variable attribute.  Use if you need to set a
        netCDF attribute with the same name as one of the reserved python
        attributes.

    group(...)
        **`group(self)`**

        return the group that this `netCDF4.Variable` is a member of.

    ncattrs(...)
        **`ncattrs(self)`**

        return netCDF attribute names for this `netCDF4.Variable` in a list.

    renameAttribute(...)
        **`renameAttribute(self, oldname, newname)`**

        rename a `netCDF4.Variable` attribute named `oldname` to `newname`.

    set_auto_mask(...)
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
        masked values by the fill_value of the masked array.

        The default value of `mask` is `True`
        (automatic conversions are performed).

    set_auto_maskandscale(...)
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
        masked values by the fill_value of the masked array.

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
        used to provide simple compression, see
        [](http://www.cdc.noaa.gov/cdc/conventions/cdc_netcdf_standard.shtml).

        The default value of `maskandscale` is `True`
        (automatic conversions are performed).

    set_auto_scale(...)
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
        used to provide simple compression, see
        [](http://www.cdc.noaa.gov/cdc/conventions/cdc_netcdf_standard.shtml).

        The default value of `scale` is `True`
        (automatic conversions are performed).

    set_var_chunk_cache(...)
        **`set_var_chunk_cache(self,size=None,nelems=None,preemption=None)`**

        change variable chunk cache settings.
        See netcdf C library documentation for `nc_set_var_chunk_cache` for
        details.

    setncattr(...)
        **`setncattr(self,name,value)`**

        set a netCDF variable attribute using name,value pair.  Use if you need to set a
        netCDF attribute with the same name as one of the reserved python
        attributes.

    setncatts(...)
        **`setncatts(self,attdict)`**

        set a bunch of netCDF variable attributes at once using a python dictionary.
        This may be faster when setting a lot of attributes for a `NETCDF3`
        formatted file, since nc_redef/nc_enddef is not called in between setting
        each attribute
