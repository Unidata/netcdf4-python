"""
Introduction
============

Python interface to the netCDF version 4 library.  U{netCDF version 4 
<http://www.unidata.ucar.edu/software/netcdf/netcdf-4>} has many features 
not found in earlier versions of the library and is implemented on top of 
U{HDF5 <http://www.hdfgroup.org/HDF5>}. This module can read and write 
files in both the new netCDF 4 and the old netCDF 3 format, and can create 
files that are readable by HDF5 clients. The API modelled after 
U{Scientific.IO.NetCDF 
<http://dirac.cnrs-orleans.fr/plone/software/scientificpython/>}, and should be 
familiar to users of that module.

Most new features of netCDF 4 are implemented, such as multiple
unlimited dimensions, groups and zlib data compression.  All the new
numeric data types (such as 64 bit and unsigned integer types) are
implemented. Compound and variable length (vlen) data types are supported, 
but the enum and opaque data types are not. Mixtures of compound and vlen
data types (compound types containing vlens, and vlens containing compound 
types) are not supported.

Download 
========

 - Latest bleeding-edge code from the U{github repository
   <http://github.com/Unidata/netcdf4-python>}.
 - Latest U{releases <https://pypi.python.org/pypi/netCDF4>}
   (source code and windows installers).

Requires 
======== 

 - Python 2.5 or later (python 3 works too).
 - numpy array module U{http://numpy.scipy.org}, version 1.3.0 or later (1.5.1
   or higher recommended, required if using python 3).
 - U{Cython <http://cython.org>} is optional - if it is installed setup.py will 
   use it to recompile the Cython source code into C, using conditional compilation
   to enable features in the netCDF API that have been added since version 4.1.1.  If
   Cython is not installed, these features (such as the ability to rename Group objects)
   will be disabled to preserve backward compatibility with older versions of the netCDF
   library.
 - For python < 2.7, the ordereddict module U{http://python.org/pypi/ordereddict}.
 - The HDF5 C library version 1.8.4-patch1 or higher (1.8.8 or higher
 recommended) from U{ftp://ftp.hdfgroup.org/HDF5/current/src}.
 Be sure to build with 'C{--enable-hl --enable-shared}'.
 - U{Libcurl <http://curl.haxx.se/libcurl/>}, if you want
 U{OPeNDAP<http://opendap.org/>} support.
 - U{HDF4 <http://www.hdfgroup.org/products/hdf4/>}, if you want
 to be able to read HDF4 "Scientific Dataset" (SD) files.
 - The netCDF-4 C library from U{ftp://ftp.unidata.ucar.edu/pub/netcdf}.
 Version 4.1.1 or higher is required (4.2 or higher recommended).
 Be sure to build with 'C{--enable-netcdf-4 --enable-shared}', and set
 C{CPPFLAGS="-I $HDF5_DIR/include"} and C{LDFLAGS="-L $HDF5_DIR/lib"},
 where C{$HDF5_DIR} is the directory where HDF5 was installed.
 If you want U{OPeNDAP<http://opendap.org/>} support, add 'C{--enable-dap}'.
 If you want HDF4 SD support, add 'C{--enable-hdf4}' and add
 the location of the HDF4 headers and library to C{CPPFLAGS} and C{LDFLAGS}.


Install
=======

 - install the requisite python modules and C libraries (see above). It's
 easiest if all the C libs are built as shared libraries.
 - optionally, set the C{HDF5_DIR} environment variable to point to where HDF5
 is installed (the libs in C{$HDF5_DIR/lib}, the headers in
 C{$HDF5_DIR/include}). If the headers and libs are installed in different
 places, you can use C{HDF5_INCDIR} and C{HDF5_LIBDIR} to define the locations
 of the headers and libraries independently.
 - optionally, set the C{NETCDF4_DIR} (or C{NETCDF4_INCDIR} and C{NETCDF4_LIBDIR})
 environment variable(s) to point to
 where the netCDF version 4 library and headers are installed.
 - If the locations of the HDF5 and netCDF libs and headers are not specified
 with environment variables, some standard locations will be searched.
 - if HDF5 was built as a static library  with U{szip
 <http://www.hdfgroup.org/doc_resource/SZIP/>} support,
 you may also need to set the C{SZIP_DIR} (or C{SZIP_INCDIR} and C{SZIP_LIBDIR})
 environment variable(s) to point to where szip is installed. Note that
 the netCDF library does not support creating szip compressed files, but can read szip
 compressed files if the HDF5 lib is configured to support szip.
 - if netCDF lib was built as a static library with HDF4 and/or OpenDAP
 support, you may also need to set C{HDF4_DIR}, C{JPEG_DIR} and/or
 C{CURL_DIR}.
 - Instead of using environment variables to specify the locations of the
 required libraries, you can either let setup.py try to auto-detect their
 locations, or use the file C{setup.cfg} to specify them.  To use this 
 method, copy the file C{setup.cfg.template} to C{setup.cfg},
 then open C{setup.cfg} in a text editor and follow the instructions in the  
 comments for editing.  If you use C{setup.cfg}, environment variables will be
 ignored.
 - If you are using netcdf 4.1.2 or higher, instead of setting all those
 enviroment variables defining where libs are installed, you can just set one 
 environment variable, USE_NCCONFIG, to 1.  This will tell python to run the
 netcdf nc-config utility to determine where all the dependencies live.
 - run C{python setup.py build}, then C{python setup.py install} (as root if
 necessary).
 - If using environment variables to specify build options, be sure to run
 'python setup.py build' *without* using sudo.  sudo does not pass environment
 variables. If you run 'setup.py build' first without sudo, you can run
 'setup.py install' with sudo.  
 - run the tests in the 'test' directory by running C{python run_all.py}.

Tutorial
========

1) Creating/Opening/Closing a netCDF file
-----------------------------------------

To create a netCDF file from python, you simply call the L{Dataset}
constructor. This is also the method used to open an existing netCDF
file.  If the file is open for write access (C{w, r+} or C{a}), you may
write any type of data including new dimensions, groups, variables and
attributes.  netCDF files come in several flavors (C{NETCDF3_CLASSIC,
NETCDF3_64BIT, NETCDF4_CLASSIC}, and C{NETCDF4}). The first two flavors
are supported by version 3 of the netCDF library. C{NETCDF4_CLASSIC}
files use the version 4 disk format (HDF5), but do not use any features
not found in the version 3 API. They can be read by netCDF 3 clients
only if they have been relinked against the netCDF 4 library. They can
also be read by HDF5 clients. C{NETCDF4} files use the version 4 disk
format (HDF5) and use the new features of the version 4 API.  The
C{netCDF4} module can read and write files in any of these formats. When
creating a new file, the format may be specified using the C{format}
keyword in the C{Dataset} constructor.  The default format is
C{NETCDF4}. To see how a given file is formatted, you can examine the
C{data_model} L{Dataset} attribute.  Closing the netCDF file is
accomplished via the L{close<Dataset.close>} method of the L{Dataset}
instance.

Here's an example:

>>> from netCDF4 import Dataset
>>> rootgrp = Dataset('test.nc', 'w', format='NETCDF4')
>>> print rootgrp.data_model
NETCDF4
>>>
>>> rootgrp.close()

Remote U{OPeNDAP<http://opendap.org>}-hosted datasets can be accessed for
reading over http if a URL is provided to the L{Dataset} constructor instead of a 
filename.  However, this requires that the netCDF library be built with
OPenDAP support, via the C{--enable-dap} configure option (added in
version 4.0.1).
            

2) Groups in a netCDF file
--------------------------

netCDF version 4 added support for organizing data in hierarchical
groups, which are analagous to directories in a filesystem. Groups serve
as containers for variables, dimensions and attributes, as well as other
groups.  A C{netCDF4.Dataset} defines creates a special group, called
the 'root group', which is similar to the root directory in a unix
filesystem.  To create L{Group} instances, use the
L{createGroup<Dataset.createGroup>} method of a L{Dataset} or L{Group}
instance. L{createGroup<Dataset.createGroup>} takes a single argument, a
python string containing the name of the new group. The new L{Group}
instances contained within the root group can be accessed by name using
the C{groups} dictionary attribute of the L{Dataset} instance.  Only
C{NETCDF4} formatted files support Groups, if you try to create a Group
in a netCDF 3 file you will get an error message.

>>> rootgrp = Dataset('test.nc', 'a')
>>> fcstgrp = rootgrp.createGroup('forecasts')
>>> analgrp = rootgrp.createGroup('analyses')
>>> print rootgrp.groups
OrderedDict([('forecasts', <netCDF4.Group object at 0x1b4b7b0>),
             ('analyses', <netCDF4.Group object at 0x1b4b970>)])
>>>

Groups can exist within groups in a L{Dataset}, just as directories
exist within directories in a unix filesystem. Each L{Group} instance
has a C{'groups'} attribute dictionary containing all of the group
instances contained within that group. Each L{Group} instance also has a
C{'path'} attribute that contains a simulated unix directory path to
that group. 

Here's an example that shows how to navigate all the groups in a
L{Dataset}. The function C{walktree} is a Python generator that is used
to walk the directory tree. Note that printing the L{Dataset} or L{Group}
object yields summary information about it's contents.

>>> fcstgrp1 = fcstgrp.createGroup('model1')
>>> fcstgrp2 = fcstgrp.createGroup('model2')
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
<type 'netCDF4.Dataset'>
root group (NETCDF4 file format):
    dimensions: 
    variables: 
        groups: forecasts, analyses
<type 'netCDF4.Group'>
group /forecasts:
    dimensions:
    variables:
    groups: model1, model2
<type 'netCDF4.Group'>
group /analyses:
    dimensions:
    variables:
    groups:
<type 'netCDF4.Group'>
group /forecasts/model1:
    dimensions:
    variables:
    groups:
<type 'netCDF4.Group'>
group /forecasts/model2:
    dimensions:
    variables:
    groups:
>>>

3) Dimensions in a netCDF file
------------------------------

netCDF defines the sizes of all variables in terms of dimensions, so
before any variables can be created the dimensions they use must be
created first. A special case, not often used in practice, is that of a
scalar variable, which has no dimensions. A dimension is created using
the L{createDimension<Dataset.createDimension>} method of a L{Dataset}
or L{Group} instance. A Python string is used to set the name of the
dimension, and an integer value is used to set the size. To create an
unlimited dimension (a dimension that can be appended to), the size
value is set to C{None} or 0. In this example, there both the C{time} and
C{level} dimensions are unlimited.  Having more than one unlimited
dimension is a new netCDF 4 feature, in netCDF 3 files there may be only
one, and it must be the first (leftmost) dimension of the variable.

>>> level = rootgrp.createDimension('level', None)
>>> time = rootgrp.createDimension('time', None)
>>> lat = rootgrp.createDimension('lat', 73)
>>> lon = rootgrp.createDimension('lon', 144)
            

All of the L{Dimension} instances are stored in a python dictionary.

>>> print rootgrp.dimensions
OrderedDict([('level', <netCDF4.Dimension object at 0x1b48030>),
             ('time', <netCDF4.Dimension object at 0x1b481c0>),
             ('lat', <netCDF4.Dimension object at 0x1b480f8>),
             ('lon', <netCDF4.Dimension object at 0x1b48a08>)])
>>>

Calling the python C{len} function with a L{Dimension} instance returns
the current size of that dimension.
The L{isunlimited<Dimension.isunlimited>} method of a L{Dimension} instance
can be used to determine if the dimensions is unlimited, or appendable.

>>> print len(lon)
144
>>> print len.is_unlimited()
False
>>> print time.is_unlimited()
True
>>>

Printing the L{Dimension} object
provides useful summary info, including the name and length of the dimension,
and whether it is unlimited.

>>> for dimobj in rootgrp.dimensions.values():
>>>    print dimobj
<type 'netCDF4.Dimension'> (unlimited): name = 'level', size = 0
<type 'netCDF4.Dimension'> (unlimited): name = 'time', size = 0
<type 'netCDF4.Dimension'>: name = 'lat', size = 73
<type 'netCDF4.Dimension'>: name = 'lon', size = 144
<type 'netCDF4.Dimension'> (unlimited): name = 'time', size = 0
>>>

L{Dimension} names can be changed using the
L{renameDimension<Dataset.renameDimension>} method of a L{Dataset} or
L{Group} instance.
            
4) Variables in a netCDF file
-----------------------------

netCDF variables behave much like python multidimensional array objects
supplied by the U{numpy module <http://numpy.scipy.org>}. However,
unlike numpy arrays, netCDF4 variables can be appended to along one or
more 'unlimited' dimensions. To create a netCDF variable, use the
L{createVariable<Dataset.createVariable>} method of a L{Dataset} or
L{Group} instance. The L{createVariable<Dataset.createVariable>} method
has two mandatory arguments, the variable name (a Python string), and
the variable datatype. The variable's dimensions are given by a tuple
containing the dimension names (defined previously with
L{createDimension<Dataset.createDimension>}). To create a scalar
variable, simply leave out the dimensions keyword. The variable
primitive datatypes correspond to the dtype attribute of a numpy array. 
You can specify the datatype as a numpy dtype object, or anything that
can be converted to a numpy dtype object.  Valid datatype specifiers
include: C{'f4'} (32-bit floating point), C{'f8'} (64-bit floating
point), C{'i4'} (32-bit signed integer), C{'i2'} (16-bit signed
integer), C{'i8'} (64-bit singed integer), C{'i1'} (8-bit signed
integer), C{'u1'} (8-bit unsigned integer), C{'u2'} (16-bit unsigned
integer), C{'u4'} (32-bit unsigned integer), C{'u8'} (64-bit unsigned
integer), or C{'S1'} (single-character string).  The old Numeric
single-character typecodes (C{'f'},C{'d'},C{'h'},
C{'s'},C{'b'},C{'B'},C{'c'},C{'i'},C{'l'}), corresponding to
(C{'f4'},C{'f8'},C{'i2'},C{'i2'},C{'i1'},C{'i1'},C{'S1'},C{'i4'},C{'i4'}),
will also work. The unsigned integer types and the 64-bit integer type
can only be used if the file format is C{NETCDF4}.

The dimensions themselves are usually also defined as variables, called
coordinate variables. The L{createVariable<Dataset.createVariable>}
method returns an instance of the L{Variable} class whose methods can be
used later to access and set variable data and attributes.

>>> times = rootgrp.createVariable('time','f8',('time',))
>>> levels = rootgrp.createVariable('level','i4',('level',))
>>> latitudes = rootgrp.createVariable('latitude','f4',('lat',))
>>> longitudes = rootgrp.createVariable('longitude','f4',('lon',))
>>> # two dimensions unlimited.
>>> temp = rootgrp.createVariable('temp','f4',('time','level','lat','lon',))

All of the variables in the L{Dataset} or L{Group} are stored in a
Python dictionary, in the same way as the dimensions:

>>> print rootgrp.variables
OrderedDict([('time', <netCDF4.Variable object at 0x1b4ba70>),
             ('level', <netCDF4.Variable object at 0x1b4bab0>), 
             ('latitude', <netCDF4.Variable object at 0x1b4baf0>),
             ('longitude', <netCDF4.Variable object at 0x1b4bb30>),
             ('temp', <netCDF4.Variable object at 0x1b4bb70>)])
>>>

To get summary info on a L{Variable} instance in an interactive session, just print it.

>>> print rootgrp.variables['temp']
<type 'netCDF4.Variable'>
float32 temp(time, level, lat, lon)
    least_significant_digit: 3
    units: K
unlimited dimensions: time, level
current shape = (0, 0, 73, 144)
>>>

L{Variable} names can be changed using the
L{renameVariable<Dataset.renameVariable>} method of a L{Dataset}
instance.
            

5) Attributes in a netCDF file
------------------------------

There are two types of attributes in a netCDF file, global and variable. 
Global attributes provide information about a group, or the entire
dataset, as a whole. L{Variable} attributes provide information about
one of the variables in a group. Global attributes are set by assigning
values to L{Dataset} or L{Group} instance variables. L{Variable}
attributes are set by assigning values to L{Variable} instances
variables. Attributes can be strings, numbers or sequences. Returning to
our example,

>>> import time
>>> rootgrp.description = 'bogus example script'
>>> rootgrp.history = 'Created ' + time.ctime(time.time())
>>> rootgrp.source = 'netCDF4 python module tutorial'
>>> latitudes.units = 'degrees north'
>>> longitudes.units = 'degrees east'
>>> levels.units = 'hPa'
>>> temp.units = 'K'
>>> times.units = 'hours since 0001-01-01 00:00:00.0'
>>> times.calendar = 'gregorian'

The L{ncattrs<Dataset.ncattrs>} method of a L{Dataset}, L{Group} or
L{Variable} instance can be used to retrieve the names of all the netCDF
attributes. This method is provided as a convenience, since using the
built-in C{dir} Python function will return a bunch of private methods
and attributes that cannot (or should not) be modified by the user.

>>> for name in rootgrp.ncattrs():
>>>     print 'Global attr', name, '=', getattr(rootgrp,name)
Global attr description = bogus example script
Global attr history = Created Mon Nov  7 10.30:56 2005
Global attr source = netCDF4 python module tutorial

The C{__dict__} attribute of a L{Dataset}, L{Group} or L{Variable} 
instance provides all the netCDF attribute name/value pairs in a python 
dictionary:

>>> print rootgrp.__dict__
OrderedDict([(u'description', u'bogus example script'),
             (u'history', u'Created Thu Mar  3 19:30:33 2011'), 
             (u'source', u'netCDF4 python module tutorial')])

Attributes can be deleted from a netCDF L{Dataset}, L{Group} or
L{Variable} using the python C{del} statement (i.e. C{del grp.foo}
removes the attribute C{foo} the the group C{grp}).

6) Writing data to and retrieving data from a netCDF variable
-------------------------------------------------------------

Now that you have a netCDF L{Variable} instance, how do you put data
into it? You can just treat it like an array and assign data to a slice.

>>> import numpy 
>>> lats =  numpy.arange(-90,91,2.5)
>>> lons =  numpy.arange(-180,180,2.5)
>>> latitudes[:] = lats
>>> longitudes[:] = lons
>>> print 'latitudes =\\n',latitudes[:]
latitudes =
[-90.  -87.5 -85.  -82.5 -80.  -77.5 -75.  -72.5 -70.  -67.5 -65.  -62.5
 -60.  -57.5 -55.  -52.5 -50.  -47.5 -45.  -42.5 -40.  -37.5 -35.  -32.5
 -30.  -27.5 -25.  -22.5 -20.  -17.5 -15.  -12.5 -10.   -7.5  -5.   -2.5
   0.    2.5   5.    7.5  10.   12.5  15.   17.5  20.   22.5  25.   27.5
  30.   32.5  35.   37.5  40.   42.5  45.   47.5  50.   52.5  55.   57.5
  60.   62.5  65.   67.5  70.   72.5  75.   77.5  80.   82.5  85.   87.5
  90. ]
>>>

Unlike NumPy's array objects, netCDF L{Variable} 
objects with unlimited dimensions will grow along those dimensions if you 
assign data outside the currently defined range of indices.

>>> # append along two unlimited dimensions by assigning to slice.
>>> nlats = len(rootgrp.dimensions['lat'])
>>> nlons = len(rootgrp.dimensions['lon'])
>>> print 'temp shape before adding data = ',temp.shape
temp shape before adding data =  (0, 0, 73, 144)
>>>
>>> from numpy.random import uniform
>>> temp[0:5,0:10,:,:] = uniform(size=(5,10,nlats,nlons))
>>> print 'temp shape after adding data = ',temp.shape
temp shape after adding data =  (6, 10, 73, 144)
>>>
>>> # levels have grown, but no values yet assigned.
>>> print 'levels shape after adding pressure data = ',levels.shape
levels shape after adding pressure data =  (10,)
>>>

Note that the size of the levels variable grows when data is appended
along the C{level} dimension of the variable C{temp}, even though no
data has yet been assigned to levels.

>>> # now, assign data to levels dimension variable.
>>> levels[:] =  [1000.,850.,700.,500.,300.,250.,200.,150.,100.,50.]

However, that there are some differences between NumPy and netCDF 
variable slicing rules. Slices behave as usual, being specified as a 
C{start:stop:step} triplet. Using a scalar integer index C{i} takes the ith 
element and reduces the rank of the output array by one. Boolean array and
integer sequence indexing behaves differently for netCDF variables
than for numpy arrays.  Only 1-d boolean arrays and integer sequences are
allowed, and these indices work independently along each dimension (similar
to the way vector subscripts work in fortran).  This means that

>>> temp[0, 0, [0,1,2,3], [0,1,2,3]]

returns an array of shape (4,4) when slicing a netCDF variable, but for a
numpy array it returns an array of shape (4,).  
Similarly, a netCDF variable of shape C{(2,3,4,5)} indexed
with C{[0, array([True, False, True]), array([False, True, True, True]), :]}
would return a C{(2, 3, 5)} array. In NumPy, this would raise an error since
it would be equivalent to C{[0, [0,1], [1,2,3], :]}. While this behaviour can
cause some confusion for those used to NumPy's 'fancy indexing' rules, it
provides a very powerful way to extract data from multidimensional netCDF
variables by using logical operations on the dimension arrays to create slices.

For example, 

>>> tempdat = temp[::2, [1,3,6], lats>0, lons>0]

will extract time indices 0,2 and 4, pressure levels
850, 500 and 200 hPa, all Northern Hemisphere latitudes and Eastern
Hemisphere longitudes, resulting in a numpy array of shape  (3, 3, 36, 71).

>>> print 'shape of fancy temp slice = ',tempdat.shape
shape of fancy temp slice =  (3, 3, 36, 71)
>>>

Time coordinate values pose a special challenge to netCDF users.  Most
metadata standards (such as CF and COARDS) specify that time should be
measure relative to a fixed date using a certain calendar, with units
specified like C{hours since YY:MM:DD hh-mm-ss}.  These units can be
awkward to deal with, without a utility to convert the values to and
from calendar dates.  The functione called L{num2date} and L{date2num} are
provided with this package to do just that.  Here's an example of how they
can be used:

>>> # fill in times.
>>> from datetime import datetime, timedelta
>>> from netCDF4 import num2date, date2num
>>> dates = [datetime(2001,3,1)+n*timedelta(hours=12) for n in range(temp.shape[0])]
>>> times[:] = date2num(dates,units=times.units,calendar=times.calendar)
>>> print 'time values (in units %s): ' % times.units+'\\n',times[:]
time values (in units hours since January 1, 0001): 
[ 17533056.  17533068.  17533080.  17533092.  17533104.]
>>>
>>> dates = num2date(times[:],units=times.units,calendar=times.calendar)
>>> print 'dates corresponding to time values:\\n',dates
dates corresponding to time values:
[2001-03-01 00:00:00 2001-03-01 12:00:00 2001-03-02 00:00:00
 2001-03-02 12:00:00 2001-03-03 00:00:00]
>>>

L{num2date} converts numeric values of time in the specified C{units}
and C{calendar} to datetime objects, and L{date2num} does the reverse.
All the calendars currently defined in the U{CF metadata convention 
<http://cf-pcmdi.llnl.gov/documents/cf-conventions/>} are supported.
A function called L{date2index} is also provided which returns the indices
of a netCDF time variable corresponding to a sequence of datetime instances.


7) Reading data from a multi-file netCDF dataset.
-------------------------------------------------

If you want to read data from a variable that spans multiple netCDF files,
you can use the L{MFDataset} class to read the data as if it were 
contained in a single file. Instead of using a single filename to create
a L{Dataset} instance, create a L{MFDataset} instance with either a list
of filenames, or a string with a wildcard (which is then converted to
a sorted list of files using the python glob module).
Variables in the list of files that share the same unlimited 
dimension are aggregated together, and can be sliced across multiple
files.  To illustrate this, let's first create a bunch of netCDF files with
the same variable (with the same unlimited dimension).  The files
must in be in C{NETCDF3_64BIT}, C{NETCDF3_CLASSIC} or 
C{NETCDF4_CLASSIC format} (C{NETCDF4} formatted multi-file
datasets are not supported).

>>> for nfile in range(10):
>>>     f = Dataset('mftest'+repr(nfile)+'.nc','w',format='NETCDF4_CLASSIC')
>>>     f.createDimension('x',None)
>>>     x = f.createVariable('x','i',('x',))
>>>     x[0:10] = numpy.arange(nfile*10,10*(nfile+1))
>>>     f.close()

Now read all the files back in at once with L{MFDataset}

>>> from netCDF4 import MFDataset
>>> f = MFDataset('mftest*nc')
>>> print f.variables['x'][:]
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74
 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99]
>>>

Note that MFDataset can only be used to read, not write, multi-file
datasets. 

8) Efficient compression of netCDF variables
--------------------------------------------

Data stored in netCDF 4 L{Variable} objects can be compressed and
decompressed on the fly. The parameters for the compression are
determined by the C{zlib}, C{complevel} and C{shuffle} keyword arguments
to the L{createVariable<Dataset.createVariable>} method. To turn on
compression, set C{zlib=True}.  The C{complevel} keyword regulates the
speed and efficiency of the compression (1 being fastest, but lowest
compression ratio, 9 being slowest but best compression ratio). The
default value of C{complevel} is 4. Setting C{shuffle=False} will turn
off the HDF5 shuffle filter, which de-interlaces a block of data before
compression by reordering the bytes.  The shuffle filter can
significantly improve compression ratios, and is on by default.  Setting
C{fletcher32} keyword argument to
L{createVariable<Dataset.createVariable>} to C{True} (it's C{False} by
default) enables the Fletcher32 checksum algorithm for error detection.
It's also possible to set the HDF5 chunking parameters and endian-ness
of the binary data stored in the HDF5 file with the C{chunksizes}
and C{endian} keyword arguments to
L{createVariable<Dataset.createVariable>}.  These keyword arguments only
are relevant for C{NETCDF4} and C{NETCDF4_CLASSIC} files (where the
underlying file format is HDF5) and are silently ignored if the file
format is C{NETCDF3_CLASSIC} or C{NETCDF3_64BIT},

If your data only has a certain number of digits of precision (say for
example, it is temperature data that was measured with a precision of
0.1 degrees), you can dramatically improve zlib compression by
quantizing (or truncating) the data using the C{least_significant_digit}
keyword argument to L{createVariable<Dataset.createVariable>}. The least
significant digit is the power of ten of the smallest decimal place in
the data that is a reliable value. For example if the data has a
precision of 0.1, then setting C{least_significant_digit=1} will cause
data the data to be quantized using C{numpy.around(scale*data)/scale}, where
scale = 2**bits, and bits is determined so that a precision of 0.1 is
retained (in this case bits=4).  Effectively, this makes the compression
'lossy' instead of 'lossless', that is some precision in the data is
sacrificed for the sake of disk space.

In our example, try replacing the line

>>> temp = rootgrp.createVariable('temp','f4',('time','level','lat','lon',))

with

>>> temp = dataset.createVariable('temp','f4',('time','level','lat','lon',),zlib=True)

and then

>>> temp = dataset.createVariable('temp','f4',('time','level','lat','lon',),zlib=True,least_significant_digit=3)

and see how much smaller the resulting files are.

9) Beyond homogenous arrays of a fixed type - compound data types
-----------------------------------------------------------------

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
L{createCompoundType<Dataset.createCompoundType>} method of a L{Dataset} or L{Group} instance.
Since there is no native complex data type in netcdf, compound types are handy
for storing numpy complex arrays.  Here's an example:

>>> f = Dataset('complex.nc','w')
>>> size = 3 # length of 1-d complex array
>>> # create sample complex data.
>>> datac = numpy.exp(1j*(1.+numpy.linspace(0, numpy.pi, size)))
>>> # create complex128 compound data type.
>>> complex128 = numpy.dtype([('real',numpy.float64),('imag',numpy.float64)])
>>> complex128_t = f.createCompoundType(complex128,'complex128')
>>> # create a variable with this data type, write some data to it.
>>> f.createDimension('x_dim',None)
>>> v = f.createVariable('cmplx_var',complex128_t,'x_dim')
>>> data = numpy.empty(size,complex128) # numpy structured array
>>> data['real'] = datac.real; data['imag'] = datac.imag
>>> v[:] = data # write numpy structured array to netcdf compound var
>>> # close and reopen the file, check the contents.
>>> f.close(); f = Dataset('complex.nc')
>>> v = f.variables['cmplx_var']
>>> datain = v[:] # read in all the data into a numpy structured array
>>> # create an empty numpy complex array
>>> datac2 = numpy.empty(datain.shape,numpy.complex128)
>>> # .. fill it with contents of structured array.
>>> datac2.real = datain['real']; datac2.imag = datain['imag']
>>> print datac.dtype,datac # original data
complex128 [ 0.54030231+0.84147098j -0.84147098+0.54030231j  -0.54030231-0.84147098j]
>>>
>>> print datac2.dtype,datac2 # data from file
complex128 [ 0.54030231+0.84147098j -0.84147098+0.54030231j  -0.54030231-0.84147098j]
>>>

Compound types can be nested, but you must create the 'inner'
ones first. All of the compound types defined for a L{Dataset} or L{Group} are stored in a
Python dictionary, just like variables and dimensions. As always, printing
objects gives useful summary information in an interactive session:

>>> print f
<type 'netCDF4.Dataset'>
root group (NETCDF4 file format):
    dimensions: x_dim
    variables: cmplx_var
    groups:
<type 'netCDF4.Variable'>
>>> print f.variables['cmplx_var']
compound cmplx_var(x_dim)
compound data type: [('real', '<f8'), ('imag', '<f8')]
unlimited dimensions: x_dim
current shape = (3,)
>>> print f.cmptypes
OrderedDict([('complex128', <netCDF4.CompoundType object at 0x1029eb7e8>)])
>>> print f.cmptypes['complex128']
<type 'netCDF4.CompoundType'>: name = 'complex128', numpy dtype = [(u'real','<f8'), (u'imag', '<f8')]
>>>

10) Variable-length (vlen) data types.
--------------------------------------

NetCDF 4 has support for variable-length or "ragged" arrays.  These are arrays
of variable length sequences having the same type. To create a variable-length 
data type, use the L{createVLType<Dataset.createVLType>} method
method of a L{Dataset} or L{Group} instance.

>>> f = Dataset('tst_vlen.nc','w')
>>> vlen_t = f.createVLType(numpy.int32, 'phony_vlen')

The numpy datatype of the variable-length sequences and the name of the 
new datatype must be specified. Any of the primitive datatypes can be 
used (signed and unsigned integers, 32 and 64 bit floats, and characters),
but compound data types cannot.
A new variable can then be created using this datatype.

>>> x = f.createDimension('x',3)
>>> y = f.createDimension('y',4)
>>> vlvar = f.createVariable('phony_vlen_var', vlen_t, ('y','x'))

Since there is no native vlen datatype in numpy, vlen arrays are represented
in python as object arrays (arrays of dtype C{object}). These are arrays whose 
elements are Python object pointers, and can contain any type of python object.
For this application, they must contain 1-D numpy arrays all of the same type
but of varying length.
In this case, they contain 1-D numpy C{int32} arrays of random length betwee
1 and 10.

>>> import random
>>> data = numpy.empty(len(y)*len(x),object)
>>> for n in range(len(y)*len(x)):
>>>    data[n] = numpy.arange(random.randint(1,10),dtype='int32')+1
>>> data = numpy.reshape(data,(len(y),len(x)))
>>> vlvar[:] = data
>>> print 'vlen variable =\\n',vlvar[:]
vlen variable =
[[[ 1  2  3  4  5  6  7  8  9 10] [1 2 3 4 5] [1 2 3 4 5 6 7 8]]
 [[1 2 3 4 5 6 7] [1 2 3 4 5 6] [1 2 3 4 5]]
 [[1 2 3 4 5] [1 2 3 4] [1]]
 [[ 1  2  3  4  5  6  7  8  9 10] [ 1  2  3  4  5  6  7  8  9 10]
  [1 2 3 4 5 6 7 8]]]
>>> print f
<type 'netCDF4.Dataset'>
root group (NETCDF4 file format):
    dimensions: x, y
    variables: phony_vlen_var
    groups:
>>> print f.variables['phony_vlen_var']
<type 'netCDF4.Variable'>
vlen phony_vlen_var(y, x)
vlen data type: int32
unlimited dimensions:
current shape = (4, 3)
>>> print f.VLtypes['phony_vlen']
<type 'netCDF4.VLType'>: name = 'phony_vlen', numpy dtype = int32
>>>

Numpy object arrays containing python strings can also be written as vlen
variables,  For vlen strings, you don't need to create a vlen data type. 
Instead, simply use the python C{str} builtin (or a numpy string datatype
with fixed length greater than 1) when calling the
L{createVariable<Dataset.createVariable>} method.  

>>> z = f.createDimension('z',10)
>>> strvar = rootgrp.createVariable('strvar', str, 'z')

In this example, an object array is filled with random python strings with
random lengths between 2 and 12 characters, and the data in the object 
array is assigned to the vlen string variable.

>>> chars = '1234567890aabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
>>> data = numpy.empty(10,'O')
>>> for n in range(10):
>>>     stringlen = random.randint(2,12)
>>>     data[n] = ''.join([random.choice(chars) for i in range(stringlen)])
>>> strvar[:] = data
>>> print 'variable-length string variable:\\n',strvar[:]
variable-length string variable:
[aDy29jPt jd7aplD b8t4RM jHh8hq KtaPWF9cQj Q1hHN5WoXSiT MMxsVeq td LUzvVTzj
 5DS9X8S]
>>> print f
<type 'netCDF4.Dataset'>
root group (NETCDF4 file format):
    dimensions: x, y, z
    variables: phony_vlen_var, strvar
    groups:
>>> print f.variables['strvar']
<type 'netCDF4.Variable'>
vlen strvar(z)
vlen data type: <type 'str'>
unlimited dimensions:
current size = (10,)
>>>

It is also possible to set contents of vlen string variables with numpy arrays
of any string or unicode data type. Note, however, that accessing the contents
of such variables will always return numpy arrays with dtype C{object}.

All of the code in this tutorial is available in C{examples/tutorial.py},
Unit tests are in the C{test} directory.

@contact: Jeffrey Whitaker <jeffrey.s.whitaker@noaa.gov>

@copyright: 2008 by Jeffrey Whitaker.

@license: Permission to use, copy, modify, and distribute this software and
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
PERFORMANCE OF THIS SOFTWARE."""
__test__ = None
del __test__ # hack so epydoc doesn't show __test__

# Make changes to this file, not the c-wrappers that Pyrex generates.

# pure python utilities
from netCDF4_utils import _StartCountStride, _quantize, _find_dim, _walk_grps, \
                          _out_array_shape, _sortbylist, _tostr
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

__version__ = "1.1.1"

# Initialize numpy
import posixpath
import netcdftime
import numpy
import weakref
import sys
import warnings
from glob import glob
from numpy import ma
from numpy import __version__ as _npversion
if _npversion.split('.')[0] < '1':
    raise ImportError('requires numpy version 1.0rc1 or later')
import_array()
include "netCDF4.pxi"
# include pure python utility functions and MFDataset class.
# (use include instead of importing them so docstrings
#  get included in C extension code).
include "utils.pyx"
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

__netcdf4libversion__ = getlibversion().split()[0]
__hdf5libversion__ = _gethdf5libversion()
__has_rename_grp__ = HAS_RENAME_GRP
__has_nc_inq_path__ = HAS_NC_INQ_PATH
__has_nc_inq_format_extended__ = HAS_NC_INQ_FORMAT_EXTENDED


# numpy data type <--> netCDF 4 data type mapping.

_nptonctype  = {'U1' : NC_CHAR,
                'S1' : NC_CHAR,
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

default_fillvals = {#'S1':NC_FILL_CHAR, 
                     'U1':'\0',
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
        ierr = nc_inq_natts(grpid, &numatts)
    else:
        ierr = nc_inq_varnatts(grpid, varid, &numatts)
    if ierr != NC_NOERR:
        raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
    attslist = []
    for n from 0 <= n < numatts:
        ierr = nc_inq_attname(grpid, varid, n, namstring)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        attslist.append(namstring.decode(default_encoding,unicode_error))
    return attslist

cdef _get_att(grp, int varid, name):
    # Private function to get an attribute value given its name
    cdef int ierr, n
    cdef size_t att_len
    cdef char *attname
    cdef char *stratt
    cdef nc_type att_type
    cdef ndarray value_arr
    bytestr = _strencode(name)
    attname = bytestr
    ierr = nc_inq_att(grp._grpid, varid, attname, &att_type, &att_len)
    if ierr != NC_NOERR:
        raise AttributeError((<char *>nc_strerror(ierr)).decode('ascii'))
    # attribute is a character or string ...
    if att_type == NC_CHAR:
        value_arr = numpy.empty(att_len,'S1')
        ierr = nc_get_att_text(grp._grpid, varid, attname, <char *>value_arr.data)
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
            ierr = nc_get_att_string(grp._grpid, varid, attname, &stratt)
            pstring = stratt.decode(default_encoding,unicode_error).replace('\x00','')
            return pstring
        else:
            raise KeyError('vlen string array attributes not supported')
    else:
    # a regular numeric or compound type.
        if att_type == NC_LONG:
            att_type = NC_INT
        try:
            type_att = _nctonptype[att_type] # see if it is a primitive type
        except KeyError:
            # check if it's a compound
            try:
                type_att = _read_compound(grp, att_type)
            except:
                raise KeyError('attribute %s has unsupported datatype' % attname)
        value_arr = numpy.empty(att_len,type_att)
        ierr = nc_get_att(grp._grpid, varid, attname, value_arr.data)
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
    if format == 'NETCDF4':
        nc_set_default_format(NC_FORMAT_NETCDF4, NULL)
    elif format == 'NETCDF4_CLASSIC':
        nc_set_default_format(NC_FORMAT_NETCDF4_CLASSIC, NULL)
    elif format == 'NETCDF3_64BIT':
        nc_set_default_format(NC_FORMAT_64BIT, NULL)
    elif format == 'NETCDF3_CLASSIC':
        nc_set_default_format(NC_FORMAT_CLASSIC, NULL)
    else:
        raise ValueError("format must be 'NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_64BIT', or 'NETCDF3_CLASSIC', got '%s'" % format)

cdef _get_format(int grpid):
    # Private function to get the netCDF file format
    cdef int ierr, formatp
    ierr = nc_inq_format(grpid, &formatp)
    if ierr != NC_NOERR:
        raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
    if formatp == NC_FORMAT_NETCDF4:
        return 'NETCDF4'
    elif formatp == NC_FORMAT_NETCDF4_CLASSIC:
        return 'NETCDF4_CLASSIC'
    elif formatp == NC_FORMAT_64BIT:
        return 'NETCDF3_64BIT'
    elif formatp == NC_FORMAT_CLASSIC:
        return 'NETCDF3_CLASSIC'

cdef _get_full_format(int grpid):
    # Private function to get the underlying disk format
    cdef int ierr, formatp, modep
    IF HAS_NC_INQ_FORMAT_EXTENDED:
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

cdef _set_att(grp, int varid, name, value):
    # Private function to set an attribute name/value pair
    cdef int i, ierr, lenarr, n
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
    # if array contains strings, write a text attribute.
    if value_arr.dtype.char in ['S','U']:
        if not value_arr.shape:
            dats = _strencode(value_arr.item())
        else:
            value_arr1 = value_arr.ravel()
            dats = _strencode(''.join(value_arr1.tolist()))
        lenarr = len(dats)
        datstring = dats
        ierr = nc_put_att_text(grp._grpid, varid, attname, lenarr, datstring)
        if ierr != NC_NOERR:
            raise AttributeError((<char *>nc_strerror(ierr)).decode('ascii'))
    # a 'regular' array type ('f4','i4','f8' etc)
    else:
        if value_arr.dtype.kind == 'V': # compound attribute.
            xtype = _find_cmptype(grp,value_arr.dtype)
        elif value_arr.dtype.str[1:] not in _supportedtypes:
            raise TypeError, 'illegal data type for attribute, must be one of %s, got %s' % (_supportedtypes, value_arr.dtype.str[1:])
        else:
            xtype = _nptonctype[value_arr.dtype.str[1:]]
        lenarr = PyArray_SIZE(value_arr)
        ierr = nc_put_att(grp._grpid, varid, attname, xtype, lenarr, value_arr.data)
        if ierr != NC_NOERR:
            raise AttributeError((<char *>nc_strerror(ierr)).decode('ascii'))

cdef _get_types(group):
    # Private function to create L{CompoundType} or L{VLType} instances for all the
    # compound or VLEN types in a L{Group} or L{Dataset}.
    cdef int ierr, ntypes, classp, n
    cdef nc_type xtype
    cdef nc_type typeids[NC_MAX_VARS]
    cdef char namstring[NC_MAX_NAME+1]
    # get the number of user defined types in this group.
    ierr = nc_inq_typeids(group._grpid, &ntypes, typeids)
    if ierr != NC_NOERR:
        raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
    # create empty dictionary for CompoundType instances.
    cmptypes = OrderedDict()
    vltypes = OrderedDict()
    if ntypes > 0:
        for n from 0 <= n < ntypes:
            xtype = typeids[n]
            ierr = nc_inq_user_type(group._grpid, xtype, namstring,
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
                    #print 'WARNING: unsupported compound type, skipping...'
                    continue
                cmptypes[name] = cmptype
            elif classp == NC_VLEN: # a vlen
                name = namstring.decode(default_encoding,unicode_error)
                # read the VLEN type info from the file,
                # create a VLType instance from it.
                try:
                    vltype = _read_vlen(group, xtype)
                except KeyError:
                    #print 'WARNING: unsupported VLEN type, skipping...'
                    continue
                vltypes[name] = vltype
                pass
    return cmptypes, vltypes

cdef _get_dims(group):
    # Private function to create L{Dimension} instances for all the
    # dimensions in a L{Group} or Dataset
    cdef int ierr, numdims, n
    cdef int dimids[NC_MAX_DIMS]
    cdef char namstring[NC_MAX_NAME+1]
    # get number of dimensions in this Group.
    ierr = nc_inq_ndims(group._grpid, &numdims)
    if ierr != NC_NOERR:
        raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
    # create empty dictionary for dimensions.
    dimensions = OrderedDict()
    if numdims > 0:
        if group.data_model == 'NETCDF4':
            ierr = nc_inq_dimids(group._grpid, &numdims, dimids, 0)
            if ierr != NC_NOERR:
                raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        else:
            for n from 0 <= n < numdims:
                dimids[n] = n
        for n from 0 <= n < numdims:
            ierr = nc_inq_dimname(group._grpid, dimids[n], namstring)
            if ierr != NC_NOERR:
                raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
            name = namstring.decode(default_encoding,unicode_error)
            dimensions[name] = Dimension(group, name, id=dimids[n])
    return dimensions

cdef _get_grps(group):
    # Private function to create L{Group} instances for all the
    # groups in a L{Group} or Dataset
    cdef int ierr, numgrps, n
    cdef int *grpids
    cdef char namstring[NC_MAX_NAME+1]
    # get number of groups in this Group.
    ierr = nc_inq_grps(group._grpid, &numgrps, NULL)
    if ierr != NC_NOERR:
        raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
    # create dictionary containing L{Group} instances for groups in this group
    groups = OrderedDict()
    if numgrps > 0:
        grpids = <int *>malloc(sizeof(int) * numgrps)
        ierr = nc_inq_grps(group._grpid, NULL, grpids)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        for n from 0 <= n < numgrps:
             ierr = nc_inq_grpname(grpids[n], namstring)
             if ierr != NC_NOERR:
                 raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
             name = namstring.decode(default_encoding,unicode_error)
             groups[name] = Group(group, name, id=grpids[n])
        free(grpids)
    return groups

cdef _get_vars(group):
    # Private function to create L{Variable} instances for all the
    # variables in a L{Group} or Dataset
    cdef int ierr, numvars, n, nn, numdims, varid, classp
    cdef int *varids
    cdef int dim_sizes[NC_MAX_DIMS]
    cdef int dimids[NC_MAX_DIMS]
    cdef nc_type xtype
    cdef char namstring[NC_MAX_NAME+1]
    cdef char namstring_cmp[NC_MAX_NAME+1]
    # get number of variables in this Group.
    ierr = nc_inq_nvars(group._grpid, &numvars)
    if ierr != NC_NOERR:
        raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
    # create empty dictionary for variables.
    variables = OrderedDict()
    if numvars > 0:
        # get variable ids.
        varids = <int *>malloc(sizeof(int) * numvars)
        if group.data_model == 'NETCDF4':
            ierr = nc_inq_varids(group._grpid, &numvars, varids)
            if ierr != NC_NOERR:
                raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        else:
            for n from 0 <= n < numvars:
                varids[n] = n
        # loop over variables. 
        for n from 0 <= n < numvars:
             varid = varids[n]
             # get variable name.
             ierr = nc_inq_varname(group._grpid, varid, namstring)
             if ierr != NC_NOERR:
                 raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
             name = namstring.decode(default_encoding,unicode_error)
             if ierr != NC_NOERR:
                 raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
             # get variable type.
             ierr = nc_inq_vartype(group._grpid, varid, &xtype)
             if ierr != NC_NOERR:
                 raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
             # check to see if it is a supported user-defined type.
             try:
                 datatype = _nctonptype[xtype]
             except KeyError:
                 if xtype == NC_STRING:
                     datatype = str
                 else:
                     ierr = nc_inq_user_type(group._grpid, xtype, namstring_cmp,
                                             NULL, NULL, NULL, &classp)
                     if classp == NC_COMPOUND: # a compound type
                         # create CompoundType instance describing this compound type.
                         try:
                             datatype = _read_compound(group, xtype)
                         except KeyError:
                             #print "WARNING: variable '%s' has unsupported compound datatype, skipping .." % name
                             continue
                     elif classp == NC_VLEN: # a compound type
                         # create VLType instance describing this compound type.
                         try:
                             datatype = _read_vlen(group, xtype)
                         except KeyError:
                             #print "WARNING: variable '%s' has unsupported VLEN datatype, skipping .." % name
                             continue
                     else:
                         #print "WARNING: variable '%s' has unsupported datatype, skipping .." % name
                         continue
             # get number of dimensions.
             ierr = nc_inq_varndims(group._grpid, varid, &numdims)
             if ierr != NC_NOERR:
                 raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
             # get dimension ids.
             ierr = nc_inq_vardimid(group._grpid, varid, dimids)
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
             # create new variable instance.
             variables[name] = Variable(group, name, datatype, dimensions, id=varid)
        free(varids) # free pointer holding variable ids.
    return variables

# these are class attributes that 
# only exist at the python level (not in the netCDF file).

_private_atts =\
['_grpid','_grp','_varid','groups','dimensions','variables','dtype','data_model','disk_format',
 '_nunlimdim','path','parent','ndim','mask','scale','cmptypes','vltypes','_isprimitive',
 'file_format','_isvlen','_iscompound','_cmptype','_vltype','keepweakref']


cdef class Dataset:
    """
Dataset(self, filename, mode="r", clobber=True, diskless=False, persist=False, keepweakref=False, format='NETCDF4')

A netCDF L{Dataset} is a collection of dimensions, groups, variables and 
attributes. Together they describe the meaning of data and relations among 
data fields stored in a netCDF file.

B{Parameters:}

B{C{filename}} - Name of netCDF file to hold dataset.

B{Keywords}:

B{C{mode}} - access mode. C{r} means read-only; no data can be
modified. C{w} means write; a new file is created, an existing file with
the same name is deleted. C{a} and C{r+} mean append (in analogy with
serial files); an existing file is opened for reading and writing.
Appending C{s} to modes C{w}, C{r+} or C{a} will enable unbuffered shared
access to C{NETCDF3_CLASSIC} or C{NETCDF3_64BIT} formatted files.
Unbuffered acesss may be useful even if you don't need shared 
access, since it may be faster for programs that don't access data
sequentially. This option is ignored for C{NETCDF4} and C{NETCDF4_CLASSIC}
formatted files.

B{C{clobber}} - if C{True} (default), opening a file with C{mode='w'}
will clobber an existing file with the same name.  if C{False}, an
exception will be raised if a file with the same name already exists.

B{C{format}} - underlying file format (one of C{'NETCDF4', 
'NETCDF4_CLASSIC', 'NETCDF3_CLASSIC'} or C{'NETCDF3_64BIT'}.  Only 
relevant if C{mode = 'w'} (if C{mode = 'r','a'} or C{'r+'} the file format 
is automatically detected). Default C{'NETCDF4'}, which means the data is 
stored in an HDF5 file, using netCDF 4 API features.  Setting 
C{format='NETCDF4_CLASSIC'} will create an HDF5 file, using only netCDF 3 
compatibile API features. netCDF 3 clients must be recompiled and linked 
against the netCDF 4 library to read files in C{NETCDF4_CLASSIC} format. 
C{'NETCDF3_CLASSIC'} is the classic netCDF 3 file format that does not 
handle 2+ Gb files very well. C{'NETCDF3_64BIT'} is the 64-bit offset 
version of the netCDF 3 file format, which fully supports 2+ GB files, but 
is only compatible with clients linked against netCDF version 3.6.0 or 
later.

C{diskless} - create diskless (in memory) file.  This is an experimental 
feature added to the C library after the netcdf-4.2 release.

C{persist} - if diskless=True, persist file to disk when closed (default False).

C{keepweakref} - if keepweakref=True, child Dimension and Variable instances will keep weak
references to the parent Dataset or Group object.  Default is False, which
means strong references will be kept.  Having Dimension and Variable instances
keep a strong reference to the parent Dataset instance, which in turn keeps a
reference to child Dimension and Variable instances, creates circular references.
Circular references complicate garbage collection, which may mean increased
memory usage for programs that create may Dataset instances with lots of
Variables.  Setting keepweakref to True allows Dataset instances to be 
garbage collected as soon as they go out of scope, potential reducing memory
usage.  However, in most cases this is not desirable, since the associated
Variable instances may still be needed, but are rendered unusable when the
parent Dataset instance is garbage collected.

B{Returns:}

a L{Dataset} instance.  All further operations on the netCDF
Dataset are accomplised via L{Dataset} instance methods.

A list of attribute names corresponding to global netCDF attributes 
defined for the L{Dataset} can be obtained with the L{ncattrs()} method. 
These attributes can be created by assigning to an attribute of the 
L{Dataset} instance. A dictionary containing all the netCDF attribute
name/value pairs is provided by the C{__dict__} attribute of a
L{Dataset} instance.

The instance variables C{dimensions, variables, groups, 
cmptypes, data_model, disk_format} and C{path} are read-only (and should not be modified by the 
user).

@ivar dimensions: The C{dimensions} dictionary maps the names of 
dimensions defined for the L{Group} or L{Dataset} to instances of the 
L{Dimension} class.

@ivar variables: The C{variables} dictionary maps the names of variables 
defined for this L{Dataset} or L{Group} to instances of the L{Variable} 
class.

@ivar groups: The groups dictionary maps the names of groups created for 
this L{Dataset} or L{Group} to instances of the L{Group} class (the 
L{Dataset} class is simply a special case of the L{Group} class which 
describes the root group in the netCDF file).

@ivar cmptypes: The C{cmptypes} dictionary maps the names of 
compound types defined for the L{Group} or L{Dataset} to instances of the 
L{CompoundType} class.

@ivar vltypes: The C{vltypes} dictionary maps the names of 
variable-length types defined for the L{Group} or L{Dataset} to instances of the 
L{VLType} class.

@ivar data_model: The C{data_model} attribute describes the netCDF
data model version, one of C{NETCDF3_CLASSIC}, C{NETCDF4},
C{NETCDF4_CLASSIC} or C{NETCDF3_64BIT}. 

@ivar file_format: same as C{data_model}, retained for backwards
compatibility.

@ivar disk_format: The C{disk_format} attribute describes the underlying
file format, one of C{NETCDF3}, C{HDF5}, C{HDF4},
C{PNETCDF}, C{DAP2}, C{DAP4} or C{UNDEFINED}. Only available if using
netcdf C library version >= 4.3.1, otherwise will always return C{UNDEFINED}.

@ivar path: The C{path} attribute shows the location of the L{Group} in
the L{Dataset} in a unix directory format (the names of groups in the
hierarchy separated by backslashes). A L{Dataset} instance is the root
group, so the path is simply C{'/'}.

@ivar parent:  The C{parent} attribute is a reference to the parent
L{Group} instance. C{None} for a the root group or L{Dataset} instance"""
    cdef object __weakref__
    cdef public int _grpid
    cdef public int _isopen
    cdef public groups, dimensions, variables, disk_format, path, parent,\
    file_format, data_model, cmptypes, vltypes, keepweakref

    def __init__(self, filename, mode='r', clobber=True, format='NETCDF4',
                 diskless=False, persist=False, keepweakref=False, **kwargs):
        cdef int grpid, ierr, numgrps, numdims, numvars
        cdef char *path
        cdef char namstring[NC_MAX_NAME+1]
        if diskless and __netcdf4libversion__ < '4.2.1':
            #diskless = False # don't raise error, instead silently ignore
            raise ValueError('diskless mode requires netcdf lib >= 4.2.1, you have %s' % __netcdf4libversion__)
        bytestr = _strencode(filename)
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
            _set_default_format(format='NETCDF3_64BIT')
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
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
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
        # get compound and vlen types in the root Group.
        self.cmptypes, self.vltypes = _get_types(self)
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

    def filepath(self):
        """
filepath(self)

Get the file system path (or the opendap URL) which was used to
open/create the Dataset. Requires netcdf >= 4.1.2"""
        cdef int ierr
        cdef size_t pathlen
        cdef char path[NC_MAX_NAME + 1]
        IF HAS_NC_INQ_PATH:
            ierr = nc_inq_path(self._grpid, &pathlen, path)
            return path.decode('ascii')
        ELSE:
            msg = """
filepath method not enabled.  To enable, install Cython, make sure you have 
version 4.1.2 or higher of the netcdf C lib, and rebuild netcdf4-python."""
            raise ValueError(msg)

    def __str__(self):
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
close(self)

Close the Dataset."""
        cdef int ierr 
        ierr = nc_close(self._grpid)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        self._isopen = 0 # indicates file already closed, checked by __dealloc__

    def __dealloc__(self):
        # close file when there are no references to object left
        cdef int ierr
        if self._isopen:
            ierr = nc_close(self._grpid)

    def sync(self):
        """
sync(self)

Writes all buffered data in the L{Dataset} to the disk file."""
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
set_fill_on(self)

Sets the fill mode for a L{Dataset} open for writing to C{on}.

This causes data to be pre-filled with fill values. The fill values can be 
controlled by the variable's C{_Fill_Value} attribute, but is usually 
sufficient to the use the netCDF default C{_Fill_Value} (defined 
separately for each variable type). The default behavior of the netCDF 
library correspongs to C{set_fill_on}.  Data which are equal to the 
C{_Fill_Value} indicate that the variable was created, but never written 
to."""
        cdef int ierr, oldmode
        ierr = nc_set_fill (self._grpid, NC_FILL, &oldmode)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))

    def set_fill_off(self):
        """
set_fill_off(self)

Sets the fill mode for a L{Dataset} open for writing to C{off}. 

This will prevent the data from being pre-filled with fill values, which 
may result in some performance improvements. However, you must then make 
sure the data is actually written before being read."""
        cdef int ierr, oldmode
        ierr = nc_set_fill (self._grpid, NC_NOFILL, &oldmode)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))

    def createDimension(self, dimname, size=None):
        """
createDimension(self, dimname, size=None)

Creates a new dimension with the given C{dimname} and C{size}. 

C{size} must be a positive integer or C{None}, which stands for 
"unlimited" (default is C{None}). Specifying a size of 0 also
results in an unlimited dimension. The return value is the L{Dimension} 
class instance describing the new dimension.  To determine the current 
maximum size of the dimension, use the C{len} function on the L{Dimension} 
instance. To determine if a dimension is 'unlimited', use the 
C{isunlimited()} method of the L{Dimension} instance."""
        self.dimensions[dimname] = Dimension(self, dimname, size=size)
        return self.dimensions[dimname]

    def renameDimension(self, oldname, newname):
        """
renameDimension(self, oldname, newname)

rename a L{Dimension} named C{oldname} to C{newname}."""
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
createCompoundType(self, datatype, datatype_name)

Creates a new compound data type named C{datatype_name} from the numpy
dtype object C{datatype}.

@attention: If the new compound data type contains other compound data types
(i.e. it is a 'nested' compound type, where not all of the elements
are homogenous numeric data types), then the 'inner' compound types B{must} be
created first.

The return value is the L{CompoundType} class instance describing the new
datatype."""
        self.cmptypes[datatype_name] = CompoundType(self, datatype,\
                datatype_name)
        return self.cmptypes[datatype_name]

    def createVLType(self, datatype, datatype_name):
        """
createVLType(self, datatype, datatype_name)

Creates a new VLEN data type named C{datatype_name} from a numpy
dtype object C{datatype}.

The return value is the L{VLType} class instance describing the new
datatype."""
        self.vltypes[datatype_name] = VLType(self, datatype, datatype_name)
        return self.vltypes[datatype_name]

    def createVariable(self, varname, datatype, dimensions=(), zlib=False,
            complevel=4, shuffle=True, fletcher32=False, contiguous=False,
            chunksizes=None, endian='native', least_significant_digit=None,
            fill_value=None, chunk_cache=None):
        """
createVariable(self, varname, datatype, dimensions=(), zlib=False, complevel=4, shuffle=True, fletcher32=False, contiguous=False, chunksizes=None, endian='native', least_significant_digit=None, fill_value=None)

Creates a new variable with the given C{varname}, C{datatype}, and 
C{dimensions}. If dimensions are not given, the variable is assumed to be 
a scalar.

The C{datatype} can be a numpy datatype object, or a string that describes 
a numpy dtype object (like the C{dtype.str} attribue of a numpy array). 
Supported specifiers include: C{'S1' or 'c' (NC_CHAR), 'i1' or 'b' or 'B' 
(NC_BYTE), 'u1' (NC_UBYTE), 'i2' or 'h' or 's' (NC_SHORT), 'u2' 
(NC_USHORT), 'i4' or 'i' or 'l' (NC_INT), 'u4' (NC_UINT), 'i8' (NC_INT64), 
'u8' (NC_UINT64), 'f4' or 'f' (NC_FLOAT), 'f8' or 'd' (NC_DOUBLE)}.
C{datatype} can also be a L{CompoundType} instance
(for a structured, or compound array), a L{VLType} instance
(for a variable-length array), or the python C{str} builtin 
(for a variable-length string array). Numpy string and unicode datatypes with
length greater than one are aliases for C{str}.

Data from netCDF variables is presented to python as numpy arrays with
the corresponding data type. 

C{dimensions} must be a tuple containing dimension names (strings) that 
have been defined previously using C{createDimension}. The default value 
is an empty tuple, which means the variable is a scalar.

If the optional keyword C{zlib} is C{True}, the data will be compressed in 
the netCDF file using gzip compression (default C{False}).

The optional keyword C{complevel} is an integer between 1 and 9 describing 
the level of compression desired (default 4). Ignored if C{zlib=False}.

If the optional keyword C{shuffle} is C{True}, the HDF5 shuffle filter 
will be applied before compressing the data (default C{True}).  This 
significantly improves compression. Default is C{True}. Ignored if
C{zlib=False}.

If the optional keyword C{fletcher32} is C{True}, the Fletcher32 HDF5 
checksum algorithm is activated to detect errors. Default C{False}.

If the optional keyword C{contiguous} is C{True}, the variable data is 
stored contiguously on disk.  Default C{False}. Setting to C{True} for
a variable with an unlimited dimension will trigger an error.

The optional keyword C{chunksizes} can be used to manually specify the
HDF5 chunksizes for each dimension of the variable. A detailed
discussion of HDF chunking and I/O performance is available U{here
<http://www.hdfgroup.org/HDF5/doc/H5.user/Chunking.html>}. 
Basically, you want the chunk size for each dimension to match as
closely as possible the size of the data block that users will read
from the file.  C{chunksizes} cannot be set if C{contiguous=True}.

The optional keyword C{endian} can be used to control whether the
data is stored in little or big endian format on disk. Possible
values are C{little, big} or C{native} (default). The library
will automatically handle endian conversions when the data is read,
but if the data is always going to be read on a computer with the
opposite format as the one used to create the file, there may be
some performance advantage to be gained by setting the endian-ness.

The C{zlib, complevel, shuffle, fletcher32, contiguous, chunksizes} and C{endian}
keywords are silently ignored for netCDF 3 files that do not use HDF5.

The optional keyword C{fill_value} can be used to override the default 
netCDF C{_FillValue} (the value that the variable gets filled with before 
any data is written to it, defaults given in netCDF4.default_fillvals).
If fill_value is set to C{False}, then the variable is not pre-filled.

If the optional keyword parameter C{least_significant_digit} is
specified, variable data will be truncated (quantized). In conjunction
with C{zlib=True} this produces 'lossy', but significantly more
efficient compression. For example, if C{least_significant_digit=1},
data will be quantized using C{numpy.around(scale*data)/scale}, where
scale = 2**bits, and bits is determined so that a precision of 0.1 is
retained (in this case bits=4). From
U{http://www.cdc.noaa.gov/cdc/conventions/cdc_netcdf_standard.shtml}:
"least_significant_digit -- power of ten of the smallest decimal place
in unpacked data that is a reliable value." Default is C{None}, or no
quantization, or 'lossless' compression.

When creating variables in a C{NETCDF4} or C{NETCDF4_CLASSIC} formatted file, 
HDF5 creates something called a 'chunk cache' for each variable.  The
default size of the chunk cache may be large enough to completely fill 
available memory when creating thousands of variables.  The optional
keyword C{chunk_cache} allows you to reduce (or increase) the size of
the default chunk cache when creating a variable.  The setting only
persists as long as the Dataset is open - you can use the set_var_chunk_cache
method to change it the next time the Dataset is opened.
Warning - messing with this parameter can seriously degrade performance.

The return value is the L{Variable} class instance describing the new 
variable.

A list of names corresponding to netCDF variable attributes can be 
obtained with the L{Variable} method C{ncattrs()}. A dictionary
containing all the netCDF attribute name/value pairs is provided by
the C{__dict__} attribute of a L{Variable} instance.

L{Variable} instances behave much like array objects. Data can be
assigned to or retrieved from a variable with indexing and slicing
operations on the L{Variable} instance. A L{Variable} instance has five
Dataset standard attributes: C{dimensions, dtype, shape, ndim} and
C{least_significant_digit}. Application programs should never modify
these attributes. The C{dimensions} attribute is a tuple containing the
names of the dimensions associated with this variable. The C{dtype}
attribute is a string describing the variable's data type (C{i4, f8,
S1,} etc). The C{shape} attribute is a tuple describing the current
sizes of all the variable's dimensions. The C{least_significant_digit}
attributes describes the power of ten of the smallest decimal place in
the data the contains a reliable value.  assigned to the L{Variable}
instance. If C{None}, the data is not truncated. The C{ndim} attribute
is the number of variable dimensions."""
        self.variables[varname] = Variable(self, varname, datatype,
        dimensions=dimensions, zlib=zlib, complevel=complevel, shuffle=shuffle,
        fletcher32=fletcher32, contiguous=contiguous, chunksizes=chunksizes,
        endian=endian, least_significant_digit=least_significant_digit,
        fill_value=fill_value, chunk_cache=chunk_cache)
        return self.variables[varname]

    def renameVariable(self, oldname, newname):
        """
renameVariable(self, oldname, newname)

rename a L{Variable} named C{oldname} to C{newname}"""
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
createGroup(self, groupname)

Creates a new L{Group} with the given C{groupname}.

The return value is a L{Group} class instance describing the new group."""
        self.groups[groupname] = Group(self, groupname)
        return self.groups[groupname]
     
    def ncattrs(self):
        """
ncattrs(self)

return netCDF global attribute names for this L{Dataset} or L{Group} in a list."""
        return _get_att_names(self._grpid, NC_GLOBAL)

    def setncattr(self,name,value):
        """
setncattr(self,name,value)

set a netCDF dataset or group attribute using name,value pair.  Only use if you need to set a
netCDF attribute with the same name as one of the reserved python
attributes."""
        if self.data_model != 'NETCDF4': self._redef()
        _set_att(self, NC_GLOBAL, name, value)
        if self.data_model !=  'NETCDF4': self._enddef()

    def setncatts(self,attdict):
        """
setncatts(self,attdict)

set a bunch of netCDF dataset or group attributes at once using a python dictionary. 
This may be faster when setting a lot of attributes for a NETCDF3 
formatted file, since nc_redef/nc_enddef is not called in between setting
each attribute"""
        if self.data_model != 'NETCDF4': self._redef()
        for name, value in attdict.items():
            _set_att(self, NC_GLOBAL, name, value)
        if self.data_model != 'NETCDF4': self._enddef()

    def getncattr(self,name):
        """
getncattr(self,name)

retrievel a netCDF dataset or group attribute.  Only use if you need to set a
netCDF attribute with the same name as one of the reserved python
attributes."""
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
delncattr(self,name,value)

delete a netCDF dataset or group attribute.  Only use if you need to delete a
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
renameAttribute(self, oldname, newname)

rename a L{Dataset} or L{Group} attribute named C{oldname} to C{newname}."""
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
renameGroup(self, oldname, newname)

rename a L{Group} named C{oldname} to C{newname} (requires netcdf >= 4.3.1)."""
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
set_auto_maskandscale(self, True_or_False)

Call L{set_auto_maskandscale} for all variables contained in this L{Dataset} or
L{Group}, as well as for all variables in all its subgroups.

B{Parameters}:

B{C{True_or_False}} - Boolean determining if automatic conversion to masked arrays
and variable scaling shall be applied for all variables.

B{Notes}:

Calling this function only affects existing variables. Variables created
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
set_auto_mask(self, True_or_False)

Call L{set_auto_mask} for all variables contained in this L{Dataset} or
L{Group}, as well as for all variables in all its subgroups.

B{Parameters}:

B{C{True_or_False}} - Boolean determining if automatic conversion to masked arrays
shall be applied for all variables.

B{Notes}:

Calling this function only affects existing variables. Variables created
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
set_auto_scale(self, True_or_False)

Call L{set_auto_scale} for all variables contained in this L{Dataset} or
L{Group}, as well as for all variables in all its subgroups.

B{Parameters}:

B{C{True_or_False}} - Boolean determining if automatic variable scaling
shall be applied for all variables.

B{Notes}:

Calling this function only affects existing variables. Variables created
after calling this function will follow the default behaviour.
        """

        for var in self.variables.values():
            var.set_auto_scale(value)

        for groups in _walk_grps(self):
            for group in groups:
                for var in group.variables.values():
                    var.set_auto_scale(value)


cdef class Group(Dataset):
    """
Group(self, parent, name) 

Groups define a hierarchical namespace within a netCDF file. They are 
analagous to directories in a unix filesystem. Each L{Group} behaves like 
a L{Dataset} within a Dataset, and can contain it's own variables, 
dimensions and attributes (and other Groups).

L{Group} instances should be created using the
L{createGroup<Dataset.createGroup>} method of a L{Dataset} instance, or
another L{Group} instance, not using this class directly.

B{Parameters:}

B{C{parent}} - L{Group} instance for the parent group.  If being created
in the root group, use a L{Dataset} instance.

B{C{name}} - Name of the group.

B{Returns:}

a L{Group} instance.  All further operations on the netCDF
Group are accomplished via L{Group} instance methods.

L{Group} inherits from L{Dataset}, so all the L{Dataset} class methods and 
variables are available to a L{Group} instance (except the C{close} 
method)."""
    def __init__(self, parent, name, **kwargs):
        cdef int ierr
        cdef char *groupname
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
            # get compound and vlen types in this Group.
            self.cmptypes, self.vltypes = _get_types(self)
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
            self.dimensions = OrderedDict()
            self.variables = OrderedDict()
            self.groups = OrderedDict()

    def close(self):
        """
close(self)

overrides L{Dataset} close method which does not apply to L{Group} 
instances, raises IOError."""
        raise IOError('cannot close a L{Group} (only applies to Dataset)')


cdef class Dimension:
    """
Dimension(self, group, name, size=None)

A netCDF L{Dimension} is used to describe the coordinates of a L{Variable}.

L{Dimension} instances should be created using the
L{createDimension<Dataset.createDimension>} method of a L{Group} or
L{Dataset} instance, not using this class directly.

B{Parameters:}

B{C{group}} - L{Group} instance to associate with dimension.

B{C{name}}  - Name of the dimension.

B{Keywords:}

B{C{size}}  - Size of the dimension. C{None} or 0 means unlimited. (Default C{None}).

B{Returns:}

a L{Dimension} instance.  All further operations on the netCDF Dimension 
are accomplised via L{Dimension} instance methods.

The current maximum size of a L{Dimension} instance can be obtained by
calling the python C{len} function on the L{Dimension} instance. The
C{isunlimited()} method of a L{Dimension} instance can be used to
determine if the dimension is unlimited"""
    cdef public int _dimid, _grpid
    cdef public _data_model, _name, _grp

    def __init__(self, grp, name, size=None, **kwargs):
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

    def __str__(self):
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
        # len(L{Dimension} instance) returns current size of dimension
        cdef int ierr
        cdef size_t lengthp
        ierr = nc_inq_dimlen(self._grpid, self._dimid, &lengthp)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        return lengthp

    def group(self):
        """
group(self)

return the group that this L{Dimension} is a member of."""
        return self._grp

    def isunlimited(self):
        """
isunlimited(self)

returns C{True} if the L{Dimension} instance is unlimited, C{False} otherwise."""
        cdef int ierr, n, numunlimdims, ndims, nvars, ngatts, xdimid
        cdef int unlimdimids[NC_MAX_DIMS]
        if self._data_model == 'NETCDF4':
            ierr = nc_inq_unlimdims(self._grpid, &numunlimdims, NULL)
            if ierr != NC_NOERR:
                raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
            if numunlimdims == 0:
                return False
            else:
                dimid = self._dimid
                ierr = nc_inq_unlimdims(self._grpid, &numunlimdims, unlimdimids)
                if ierr != NC_NOERR:
                    raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
                unlimdim_ids = []
                for n from 0 <= n < numunlimdims:
                    unlimdim_ids.append(unlimdimids[n])
                if dimid in unlimdim_ids: 
                    return True
                else:
                    return False
        else: # if not NETCDF4, there is only one unlimited dimension.
            # nc_inq_unlimdims only works for NETCDF4.
            ierr = nc_inq(self._grpid, &ndims, &nvars, &ngatts, &xdimid)
            if self._dimid == xdimid:
                return True
            else:
                return False

cdef class Variable:
    """
Variable(self, group, name, datatype, dimensions=(), zlib=False, complevel=4, shuffle=True, fletcher32=False, contiguous=False, chunksizes=None, endian='native', least_significant_digit=None,fill_value=None)

A netCDF L{Variable} is used to read and write netCDF data.  They are 
analagous to numpy array objects.

L{Variable} instances should be created using the
L{createVariable<Dataset.createVariable>} method of a L{Dataset} or
L{Group} instance, not using this class directly.

B{Parameters:}

B{C{group}} - L{Group} or L{Dataset} instance to associate with variable.

B{C{name}}  - Name of the variable.

B{C{datatype}} - L{Variable} data type. Can be specified by providing a 
numpy dtype object, or a string that describes a numpy dtype object. 
Supported values, corresponding to C{str} attribute of numpy dtype 
objects, include C{'f4'} (32-bit floating point), C{'f8'} (64-bit floating 
point), C{'i4'} (32-bit signed integer), C{'i2'} (16-bit signed integer), 
C{'i8'} (64-bit singed integer), C{'i4'} (8-bit singed integer), C{'i1'} 
(8-bit signed integer), C{'u1'} (8-bit unsigned integer), C{'u2'} (16-bit 
unsigned integer), C{'u4'} (32-bit unsigned integer), C{'u8'} (64-bit 
unsigned integer), or C{'S1'} (single-character string).  From 
compatibility with Scientific.IO.NetCDF, the old Numeric single character 
typecodes can also be used (C{'f'} instead of C{'f4'}, C{'d'} instead of 
C{'f8'}, C{'h'} or C{'s'} instead of C{'i2'}, C{'b'} or C{'B'} instead of 
C{'i1'}, C{'c'} instead of C{'S1'}, and C{'i'} or C{'l'} instead of 
C{'i4'}). C{datatype} can also be a L{CompoundType} instance
(for a structured, or compound array), a L{VLType} instance
(for a variable-length array), or the python C{str} builtin 
(for a variable-length string array). Numpy string and unicode datatypes with
length greater than one are aliases for C{str}.

B{Keywords:}

B{C{dimensions}} - a tuple containing the variable's dimension names 
(defined previously with C{createDimension}). Default is an empty tuple 
which means the variable is a scalar (and therefore has no dimensions).

B{C{zlib}} - if C{True}, data assigned to the L{Variable}  
instance is compressed on disk. Default C{False}.

B{C{complevel}} - the level of zlib compression to use (1 is the fastest, 
but poorest compression, 9 is the slowest but best compression). Default 4.
Ignored if C{zlib=False}. 

B{C{shuffle}} - if C{True}, the HDF5 shuffle filter is applied 
to improve compression. Default C{True}. Ignored if C{zlib=False}.

B{C{fletcher32}} - if C{True} (default C{False}), the Fletcher32 checksum 
algorithm is used for error detection.

B{C{contiguous}} - if C{True} (default C{False}), the variable data is
stored contiguously on disk.  Default C{False}. Setting to C{True} for
a variable with an unlimited dimension will trigger an error.

B{C{chunksizes}} - Can be used to specify the HDF5 chunksizes for each
dimension of the variable. A detailed discussion of HDF chunking and I/O
performance is available U{here
<http://www.hdfgroup.org/HDF5/doc/H5.user/Chunking.html>}. 
Basically, you want the chunk size for each dimension to match as
closely as possible the size of the data block that users will read
from the file. C{chunksizes} cannot be set if C{contiguous=True}.

B{C{endian}} - Can be used to control whether the
data is stored in little or big endian format on disk. Possible
values are C{little, big} or C{native} (default). The library
will automatically handle endian conversions when the data is read,
but if the data is always going to be read on a computer with the
opposite format as the one used to create the file, there may be
some performance advantage to be gained by setting the endian-ness.
For netCDF 3 files (that don't use HDF5), only C{endian='native'} is allowed.

The C{zlib, complevel, shuffle, fletcher32, contiguous} and {chunksizes}
keywords are silently ignored for netCDF 3 files that do not use HDF5.

B{C{least_significant_digit}} - If specified, variable data will be
truncated (quantized). In conjunction with C{zlib=True} this produces
'lossy', but significantly more efficient compression. For example, if
C{least_significant_digit=1}, data will be quantized using
around(scale*data)/scale, where scale = 2**bits, and bits is determined
so that a precision of 0.1 is retained (in this case bits=4). Default is
C{None}, or no quantization.

B{C{fill_value}} - If specified, the default netCDF C{_FillValue} (the 
value that the variable gets filled with before any data is written to it) 
is replaced with this value.  If fill_value is set to C{False}, then
the variable is not pre-filled. The default netCDF fill values can be found
in netCDF4.default_fillvals.
 
B{Returns:}

a L{Variable} instance.  All further operations on the netCDF Variable are 
accomplised via L{Variable} instance methods.

A list of attribute names corresponding to netCDF attributes defined for
the variable can be obtained with the C{ncattrs()} method. These
attributes can be created by assigning to an attribute of the
L{Variable} instance. A dictionary containing all the netCDF attribute
name/value pairs is provided by the C{__dict__} attribute of a
L{Variable} instance.

The instance variables C{dimensions, dtype, ndim, shape}
and C{least_significant_digit} are read-only (and 
should not be modified by the user).

@ivar dimensions: A tuple containing the names of the dimensions 
associated with this variable.

@ivar dtype: A numpy dtype object describing the variable's data type.

@ivar ndim: The number of variable dimensions.

@ivar shape: a tuple describing the current size of all the variable's 
dimensions.

@ivar scale:  if True, C{scale_factor} and C{add_offset} are automatically
applied. Default is C{True}, can be reset using L{set_auto_scale} and
L{set_auto_maskandscale} methods.

@ivar mask:  if True, data is automatically converted to/from masked arrays
when missing values or fill values are present. Default is C{True}, can be
reset using L{set_auto_mask} and L{set_auto_maskandscale} methods.

@ivar least_significant_digit: Describes the power of ten of the smallest
decimal place in the data the contains a reliable value.  Data is 
truncated to this decimal place when it is assigned to the L{Variable} 
instance. If C{None}, the data is not truncated. """
    cdef public int _varid, _grpid, _nunlimdim
    cdef public _name, ndim, dtype, mask, scale, _isprimitive, _iscompound,\
    _isvlen, _grp,_cmptype,_vltype

    def __init__(self, grp, name, datatype, dimensions=(), zlib=False,
            complevel=4, shuffle=True, fletcher32=False, contiguous=False,
            chunksizes=None, endian='native', least_significant_digit=None,
            fill_value=None, chunk_cache=None, **kwargs):
        cdef int ierr, ndims, icontiguous, ideflate_level, numdims
        cdef char *varname
        cdef nc_type xtype
        cdef int dimids[NC_MAX_DIMS]
        cdef size_t sizep, nelemsp
        cdef size_t *chunksizesp
        cdef float preemptionp
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
        # convert to a real numpy datatype object if necessary.
        if (not isinstance(datatype, CompoundType) and \
            not isinstance(datatype, VLType)) and \
            datatype != str and \
            type(datatype) != numpy.dtype:
            datatype = numpy.dtype(datatype)
        # convert numpy string dtype with length > 1 
        # or any numpy unicode dtype into str
        if (isinstance(datatype, numpy.dtype) and
            ((datatype.kind == 'S' and datatype.itemsize > 1) or
              datatype.kind == 'U')):
            datatype = str
	# check if endian keyword consistent with datatype specification.
        dtype_endian = getattr(datatype,'byteorder',None)
        if dtype_endian == '=': dtype_endian='native'
        if dtype_endian == '>': dtype_endian='big'
        if dtype_endian == '<': dtype_endian='little'
        if dtype_endian == '|': dtype_endian=None
        if dtype_endian is not None and dtype_endian != endian:
            # endian keyword prevails, issue warning
            msg = 'endian-ness of dtype and endian kwarg do not match, using endian kwarg'
            #msg = 'endian-ness of dtype and endian kwarg do not match, dtype over-riding endian kwarg'
            warnings.warn(msg)
            #endian = dtype_endian # dtype prevails
        # check validity of datatype.
        self._isprimitive = False
        self._iscompound = False
        self._isvlen = False
        if isinstance(datatype, CompoundType) or isinstance(datatype, VLType)\
                      or datatype == str:
            if isinstance(datatype, CompoundType):
               self._iscompound = True
               self._cmptype = datatype
            if isinstance(datatype, VLType) or datatype==str:
               self._isvlen = True
               self._vltype = datatype
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
                for n from 0 <= n < ndims:
                    dimname = dimensions[n]
                    # look for dimension in this group, and if not
                    # found there, look in parent (and it's parent, etc, back to root).
                    dim = _find_dim(grp, dimname)
                    if dim is None:
                        raise KeyError("dimension %s not defined in group %s or any group in it's family tree" % (dimname, grp.path))
                    dimids[n] = dim._dimid
            # go into define mode if it's a netCDF 3 compatible
            # file format.  Be careful to exit define mode before
            # any exceptions are raised.
            if grp.data_model != 'NETCDF4': grp._redef()
            # define variable.
            if ndims:
                ierr = nc_def_var(self._grpid, varname, xtype, ndims,
                                  dimids, &self._varid)
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
                    ierr = nc_def_var_fill(self._grpid, self._varid, 1, NULL)
                    if ierr != NC_NOERR:
                        if grp.data_model != 'NETCDF4': grp._enddef()
                        raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
                else:
                    # cast fill_value to type of variable.
                    if self._isprimitive:
                        fillval = numpy.array(fill_value, self.dtype)
                        _set_att(self._grp, self._varid, '_FillValue', fillval)
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
        ierr = nc_inq_varndims(self._grpid, self._varid, &numdims)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        self.ndim = numdims
        self._name = name
        # default for automatically applying scale_factor and
        # add_offset, and converting to/from masked arrays is True.
        self.scale = True
        self.mask = True

    def __array__(self):
        # numpy special method that returns a numpy array.
	# allows numpy ufuncs to work faster on Variable objects 
	# (issue 216).
        return self[...]

    def __str__(self):
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
        else:
            ncdump_var.append('%s %s(%s)\n' %\
            (self.dtype,self._name,', '.join(dimnames)))
        ncdump_var = ncdump_var + attrs
        if self._iscompound:
            ncdump_var.append('compound data type: %s\n' % self.dtype)
        elif self._isvlen:
            ncdump_var.append('vlen data type: %s\n' % self.dtype)
        unlimdims = []
        for dimname in self.dimensions:
            dim = _find_dim(self._grp, dimname)
            if dim.isunlimited():
                unlimdims.append(dimname)
        if (self._grp.path != '/'): ncdump_var.append('path = %s\n' % self._grp.path)
        ncdump_var.append('unlimited dimensions: %s\n' % ', '.join(unlimdims))
        ncdump_var.append('current shape = %s\n' % repr(self.shape))
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
        cdef int dimids[NC_MAX_DIMS]
        # get number of dimensions for this variable.
        ierr = nc_inq_varndims(self._grpid, self._varid, &numdims)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        # get dimension ids.
        ierr = nc_inq_vardimid(self._grpid, self._varid, dimids)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        # loop over dimensions, retrieve names.
        dimensions = ()
        for nn from 0 <= nn < numdims:
            ierr = nc_inq_dimname(self._grpid, dimids[nn], namstring)
            if ierr != NC_NOERR:
                raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
            name = namstring.decode(default_encoding,unicode_error)
            dimensions = dimensions + (name,)
        return dimensions

    property datatype:
        """numpy data type (for primitive data types) or VLType/CompoundType instance (for compound or vlen data types)"""
        def __get__(self):
            if self._iscompound:
                return self._cmptype
            elif self._isvlen:
                return self._vltype
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
group(self)

return the group that this L{Variable} is a member of."""
        return self._grp

    def ncattrs(self):
        """
ncattrs(self)

return netCDF attribute names for this L{Variable} in a list."""
        return _get_att_names(self._grpid, self._varid)

    def setncattr(self,name,value):
        """
setncattr(self,name,value)

set a netCDF variable attribute using name,value pair.  Only use if you need to set a
netCDF attribute with the same name as one of the reserved python
attributes."""
        if self._grp.data_model != 'NETCDF4': self._grp._redef()
        _set_att(self._grp, self._varid, name, value)
        if self._grp.data_model != 'NETCDF4': self._grp._enddef()

    def setncatts(self,attdict):
        """
setncatts(self,attdict)

set a bunch of netCDF variable attributes at once using a python dictionary. 
This may be faster when setting a lot of attributes for a NETCDF3 
formatted file, since nc_redef/nc_enddef is not called in between setting
each attribute"""
        if self._grp.data_model != 'NETCDF4': self._grp._redef()
        for name, value in attdict.items():
            _set_att(self._grp, self._varid, name, value)
        if self._grp.data_model != 'NETCDF4': self._grp._enddef()

    def getncattr(self,name):
        """
getncattr(self,name)

retrievel a netCDF variable attribute.  Only use if you need to set a
netCDF attribute with the same name as one of the reserved python
attributes."""
        return _get_att(self._grp, self._varid, name)

    def delncattr(self, name):
        """
delncattr(self,name,value)

delete a netCDF variable attribute.  Only use if you need to delete a
netCDF attribute with the same name as one of the reserved python
attributes."""
        cdef char *attname
        bytestr = _strencode(name)
        attname = bytestr
        if self._grp.data_model != 'NETCDF4': self._grp._redef()
        ierr = nc_del_att(self._grpid, self._varid, attname)
        if self._grp.data_model != 'NETCDF4': self._grp._enddef()
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))

    def filters(self):
        """
filters(self)

return dictionary containing HDF5 filter parameters."""
        cdef int ierr,ideflate,ishuffle,ideflate_level,ifletcher32
        filtdict = {'zlib':False,'shuffle':False,'complevel':0,'fletcher32':False}
        if self._grp.data_model not in ['NETCDF4_CLASSIC','NETCDF4']: return
        ierr = nc_inq_var_deflate(self._grpid, self._varid, &ishuffle, &ideflate, &ideflate_level)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
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
endian(self)

return endian-ness (little,big,native) of variable (as stored in HDF5 file)."""
        cdef int ierr, iendian
        if self._grp.data_model not in ['NETCDF4_CLASSIC','NETCDF4']: 
            return 'native'
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
chunking(self)

return variable chunking information.  If the dataset is 
defined to be contiguous (and hence there is no chunking) the word 'contiguous'
is returned.  Otherwise, a sequence with the chunksize for
each dimension is returned."""
        cdef int ierr, icontiguous, ndims
        cdef size_t *chunksizesp
        if self._grp.data_model not in ['NETCDF4_CLASSIC','NETCDF4']: return None
        ndims = self.ndim
        chunksizesp = <size_t *>malloc(sizeof(size_t) * ndims)
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
get_var_chunk_cache(self)

return variable chunk cache information in a tuple (size,nelems,preemption).
See netcdf C library documentation for C{nc_get_var_chunk_cache} for
details."""
        cdef int ierr
        cdef size_t sizep, nelemsp
        cdef float preemptionp
        ierr = nc_get_var_chunk_cache(self._grpid, self._varid, &sizep,
               &nelemsp, &preemptionp)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        size = sizep; nelems = nelemsp; preemption = preemptionp
        return (size,nelems,preemption)

    def set_var_chunk_cache(self,size=None,nelems=None,preemption=None):
        """
set_var_chunk_cache(self,size=None,nelems=None,preemption=None)

change variable chunk cache settings.
See netcdf C library documentation for C{nc_set_var_chunk_cache} for
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
renameAttribute(self, oldname, newname)

rename a L{Variable} attribute named C{oldname} to C{newname}."""
        cdef int ierr
        cdef char *oldnamec
        cdef char *newnamec
        bytestr = _strencode(oldname)
        oldnamec = bytestr
        bytestr = _strencode(newname)
        newnamec = bytestr
        ierr = nc_rename_att(self._grpid, self._varid, oldnamec, newnamec)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))

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
                data[tuple(i)] = datout.reshape(shape)

        # Remove extra singleton dimensions. 
        if hasattr(data,'shape'):
            data = data[tuple(squeeze)]
        if self.ndim == 0:
            # Make sure a numpy scalar is returned instead of a 1-d array of
            # length 1.
            data = data[0]

        # if auto_scale mode set to True, (through
        # a call to set_auto_scale or set_auto_maskandscale),
        # perform automatic unpacking using scale_factor/add_offset.
        # if auto_mask mode is set to True (through a call to
        # set_auto_mask or set_auto_maskandscale), perform
        # automatic conversion to masked array using
        # missing_value/_Fill_Value.
        # ignore for compound and vlen datatypes.
        try: # check to see if scale_factor and add_offset is valid (issue 176).
            if hasattr(self,'scale_factor'): float(self.scale_factor)
            if hasattr(self,'add_offset'): float(self.add_offset)
            valid_scaleoffset = True
        except:
            valid_scaleoffset = False
            if self.scale:
                msg = 'invalid scale_factor or add_offset attribute, no unpacking done...'
                warnings.warn(msg)
        if self.mask and self._isprimitive:
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
                data += self.add_offset
        return data

    def _toma(self,data):
        cdef int ierr, no_fill
        # private function for creating a masked array, masking missing_values
        # and/or _FillValues.
        totalmask = numpy.zeros(data.shape, numpy.bool)
        fill_value = None
        if hasattr(self, 'missing_value'):
            mval = numpy.array(self.missing_value, self.dtype)
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
            # is _FillValue a NaN?
            try:
                fvalisnan = numpy.isnan(fval)
            except TypeError: # isnan fails on some dtypes (issue 202)
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
                 fillval = default_fillvals[self.dtype.str[1:]]
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
        return data

    def _assign_vlen(self, elem, data):
        """private method to assign data to a single item in a VLEN variable"""
        cdef size_t startp[NC_MAX_DIMS]
        cdef size_t countp[NC_MAX_DIMS]
        cdef int ndims, n
        cdef nc_vlen_t *vldata
        cdef char **strdata
        cdef ndarray data2
        if not self._isvlen:
            raise TypeError('_assign_vlen method only for use with VLEN variables')
        ndims = self.ndim
        msg="single element VLEN slices must be specified by integers only"
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
                    self._assign_vlen(elem, data)
                    return
                elif data.dtype.kind in ['S', 'U']:
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
                    self._assign_vlen(elem, data)
                    return

        # A numpy array is needed. Convert if necessary.
        # assume it's a numpy or masked array if it has an 'ndim' attribute.
        if not hasattr(data,'ndim'): 
            # if auto scaling is to be done, don't cast to an integer yet. 
            if self.scale and self.dtype.kind == 'i' and \
               hasattr(self, 'scale_factor') or hasattr(self, 'add_offset'):
                data = numpy.array(data,numpy.float)
            else:
                data = numpy.array(data,self.dtype)

        start, count, stride, put_ind =\
        _StartCountStride(elem,self.shape,self.dimensions,self._grp,datashape=data.shape)
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

        if 'least_significant_digit' in self.ncattrs():
            data = _quantize(data,self.least_significant_digit)
        # if auto_scale mode set to True, (through
        # a call to set_auto_scale or set_auto_maskandscale),
        # perform automatic unpacking using scale_factor/add_offset.
        # if auto_mask mode is set to True (through a call to
        # set_auto_mask or set_auto_maskandscale), perform
        # automatic conversion to masked array using
        # missing_value/_Fill_Value.
        # ignore if not a primitive (not compound or vlen) datatype.
        if self.mask and self._isprimitive:
            # use missing_value as fill value.
            # if no missing value set, use _FillValue.
            if hasattr(self, 'scale_factor') or hasattr(self, 'add_offset'):
                # if not masked, create a masked array.
                if not ma.isMA(data): data = self._toma(data)
        if self.scale and self._isprimitive:
            # pack non-masked values using scale_factor and add_offset
            if hasattr(self, 'scale_factor') and hasattr(self, 'add_offset'):
                data = (data - self.add_offset)/self.scale_factor
                if self.dtype.kind == 'i': data = numpy.around(data)
            elif hasattr(self, 'scale_factor'):
                data = data/self.scale_factor
                if self.dtype.kind == 'i': data = numpy.around(data)
            elif hasattr(self, 'add_offset'):
                data = data - self.add_offset
                if self.dtype.kind == 'i': data = numpy.around(data)
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
        return self.shape[0]


    def assignValue(self,val):
        """
assignValue(self, val)

assign a value to a scalar variable.  Provided for compatibility with 
Scientific.IO.NetCDF, can also be done by assigning to a slice ([:])."""
        if len(self.dimensions):
            raise IndexError('to assign values to a non-scalar variable, use a slice')
        self[:]=val

    def getValue(self):
        """
getValue(self)

get the value of a scalar variable.  Provided for compatibility with 
Scientific.IO.NetCDF, can also be done by slicing ([:])."""
        if len(self.dimensions):
            raise IndexError('to retrieve values from a non-scalar variable, use slicing')
        return self[slice(None)]

    def set_auto_maskandscale(self,maskandscale):
        """
set_auto_maskandscale(self,maskandscale)

turn on or off automatic conversion of variable data to and
from masked arrays and automatic packing/unpacking of variable
data using C{scale_factor} and C{add_offset} attributes.

If C{maskandscale} is set to C{True}, when data is read from a variable
it is converted to a masked array if any of the values are exactly
equal to the either the netCDF _FillValue or the value specified by the 
missing_value variable attribute. The fill_value of the masked array
is set to the missing_value attribute (if it exists), otherwise
the netCDF _FillValue attribute (which has a default value
for each data type).  When data is written to a variable, the masked
array is converted back to a regular numpy array by replacing all the
masked values by the fill_value of the masked array.

If C{maskandscale} is set to C{True}, and the variable has a
C{scale_factor} or an C{add_offset} attribute, then data read
from that variable is unpacked using::

    data = self.scale_factor*data + self.add_offset
            
When data is written to a variable it is packed using::

    data = (data - self.add_offset)/self.scale_factor

If either scale_factor is present, but add_offset is missing, add_offset
is assumed zero.  If add_offset is present, but scale_factor is missing,
scale_factor is assumed to be one.
For more information on how C{scale_factor} and C{add_offset} can be 
used to provide simple compression, see
U{http://www.cdc.noaa.gov/cdc/conventions/cdc_netcdf_standard.shtml
<http://www.cdc.noaa.gov/cdc/conventions/cdc_netcdf_standard.shtml>}.

The default value of C{maskandscale} is C{True}
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
set_auto_scale(self,scale)

turn on or off automatic packing/unpacking of variable
data using C{scale_factor} and C{add_offset} attributes.

If C{scale} is set to C{True}, and the variable has a
C{scale_factor} or an C{add_offset} attribute, then data read
from that variable is unpacked using::

    data = self.scale_factor*data + self.add_offset

When data is written to a variable it is packed using::

    data = (data - self.add_offset)/self.scale_factor

If either scale_factor is present, but add_offset is missing, add_offset
is assumed zero.  If add_offset is present, but scale_factor is missing,
scale_factor is assumed to be one.
For more information on how C{scale_factor} and C{add_offset} can be
used to provide simple compression, see
U{http://www.cdc.noaa.gov/cdc/conventions/cdc_netcdf_standard.shtml
<http://www.cdc.noaa.gov/cdc/conventions/cdc_netcdf_standard.shtml>}.

The default value of C{scale} is C{True}
(automatic conversions are performed).
        """
        if scale:
            self.scale = True
        else:
            self.scale = False

    def set_auto_mask(self,mask):
        """
set_auto_mask(self,mask)

turn on or off automatic conversion of variable data to and
from masked arrays .

If C{mask} is set to C{True}, when data is read from a variable
it is converted to a masked array if any of the values are exactly
equal to the either the netCDF _FillValue or the value specified by the
missing_value variable attribute. The fill_value of the masked array
is set to the missing_value attribute (if it exists), otherwise
the netCDF _FillValue attribute (which has a default value
for each data type).  When data is written to a variable, the masked
array is converted back to a regular numpy array by replacing all the
masked values by the fill_value of the masked array.

The default value of C{mask} is C{True}
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
        cdef size_t startp[NC_MAX_DIMS]
        cdef size_t countp[NC_MAX_DIMS]
        cdef ptrdiff_t stridep[NC_MAX_DIMS]
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
        if self._isprimitive or self._iscompound:
            # primitive or compound data type.
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

    def _get(self,start,count,stride):
        """Private method to retrieve data from a netCDF variable"""
        cdef int ierr, ndims
        cdef size_t startp[NC_MAX_DIMS]
        cdef size_t countp[NC_MAX_DIMS]
        cdef ptrdiff_t stridep[NC_MAX_DIMS]
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
        if self._isprimitive or self._iscompound:
            data = numpy.empty(shapeout, self.dtype)
            # strides all 1 or scalar variable, use get_vara (faster)
            if sum(stride) == ndims or ndims == 0: 
                ierr = nc_get_vara(self._grpid, self._varid,
                                   startp, countp, data.data)
            else:
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
                free(strdata)
            else:
                # regular vlen
                # allocate struct array to hold vlen data.
                vldata = <nc_vlen_t *>malloc(totelem*sizeof(nc_vlen_t))
                # strides all 1 or scalar variable, use get_vara (faster)
                if sum(stride) == ndims or ndims == 0: 
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
                    dataarr.data = <char *>vldata[i].p
                    data[i] = dataarr
                # reshape the output array
                data = numpy.reshape(data, shapeout)
                # free the pointer array.
                free(vldata)
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
A L{CompoundType} instance is used to describe a compound data type.

Constructor: C{CompoundType(group, datatype, datatype_name)}

@attention: When creating nested compound data types,
the inner compound data types must already be associated with CompoundType
instances (so create CompoundType instances for the innermost structures
first).

L{CompoundType} instances should be created using the
L{createCompoundType<Dataset.createCompoundType>}
method of a Dataset or L{Group} instance, not using this class directly.

B{Parameters:}

B{C{group}} - L{Group} instance to associate with the compound datatype.

B{C{datatype}} - A numpy dtype object describing a structured (a.k.a record)
array.  Can be composed of homogeneous numeric or character data types, or 
other structured array data types. 

B{C{datatype_name}} - a Python string containing a description of the 
compound data type.

B{Returns:}

a L{CompoundType} instance, which can be passed to the C{createVariable} 
method of a L{Dataset} or L{Group} instance.

The instance variables C{dtype} and C{name} should not be modified by
the user.

@ivar dtype: A numpy dtype object describing the compound data type.

@ivar name: A python string describing the compound type.
"""
    cdef public nc_type _nc_type
    cdef public dtype, name
    def __init__(self, grp, object dt, object dtype_name, **kwargs):
        cdef nc_type xtype
        dt = numpy.dtype(dt,align=True)
        if 'typeid' in kwargs:
            xtype = kwargs['typeid']
        else:
            xtype = _def_compound(grp, dt, dtype_name)
        self._nc_type = xtype
        self.dtype = dt
        self.name = dtype_name

    def __str__(self):
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
    cdef int dim_sizes[NC_MAX_DIMS]
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

cdef _read_compound(group, nc_type xtype):
    # read a compound data type id from an existing file,
    # construct a corresponding numpy dtype instance, 
    # then use that to create a CompoundType instance.
    # called by _get_vars, _get_types and _get_att.
    # Calls itself recursively for nested compound types.
    cdef int ierr, nf, numdims, ndim, classp
    cdef size_t nfields, offset
    cdef nc_type field_typeid
    cdef int dim_sizes[NC_MAX_DIMS]
    cdef char field_namstring[NC_MAX_NAME+1]
    cdef char cmp_namstring[NC_MAX_NAME+1]
    # get name and number of fields.
    ierr = nc_inq_compound(group._grpid, xtype, cmp_namstring, NULL, &nfields)
    if ierr != NC_NOERR:
        raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
    name = cmp_namstring.decode(default_encoding,unicode_error)
    # loop over fields.
    names = []
    formats = []
    offsets = []
    for nf from 0 <= nf < nfields:
        ierr = nc_inq_compound_field(group._grpid,
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
        # check to see if this field is a nested compound type.
        try: 
            field_type =  _nctonptype[field_typeid]
        except KeyError:
            ierr = nc_inq_user_type(group._grpid,
                   field_typeid,NULL,NULL,NULL,NULL,&classp)
            if classp == NC_COMPOUND: # a compound type
                # recursively call this function?
                field_type = _read_compound(group, field_typeid)
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
A L{VLType} instance is used to describe a variable length (VLEN) data type.

Constructor: C{VLType(group, datatype, datatype_name)}

L{VLType} instances should be created using the 
L{createVLType<Dataset.createVLType>}
method of a Dataset or L{Group} instance, not using this class directly.

B{Parameters:}

B{C{group}} - L{Group} instance to associate with the VLEN datatype.

B{C{datatype}} - An numpy dtype object describing a the component type for the
variable length array.  

B{C{datatype_name}} - a Python string containing a description of the 
VLEN data type.

B{Returns:}

a L{VLType} instance, which can be passed to the C{createVariable} 
method of a L{Dataset} or L{Group} instance.

The instance variables C{dtype} and C{name} should not be modified by
the user.

@ivar dtype: An object describing the VLEN type.

@ivar name: A python string describing the VLEN type.
"""
    cdef public nc_type _nc_type
    cdef public dtype, name
    def __init__(self, grp, object dt, object dtype_name, **kwargs):
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

    def __str__(self):
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
    cdef int dim_sizes[NC_MAX_DIMS]
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

cdef _read_vlen(group, nc_type xtype):
    # read a VLEN data type id from an existing file,
    # construct a corresponding numpy dtype instance, 
    # then use that to create a VLType instance.
    # called by _get_types, _get_vars.
    cdef int ierr
    cdef size_t vlsize
    cdef nc_type base_xtype
    cdef char vl_namstring[NC_MAX_NAME+1]
    if xtype == NC_STRING:
        dt = str
        name = None
    else:
        ierr = nc_inq_vlen(group._grpid, xtype, vl_namstring, &vlsize, &base_xtype)
        if ierr != NC_NOERR:
            raise RuntimeError((<char *>nc_strerror(ierr)).decode('ascii'))
        name = vl_namstring.decode(default_encoding,unicode_error)
        try:
            dt = numpy.dtype(_nctonptype[base_xtype]) # see if it is a primitive type
        except KeyError:
            raise KeyError("unsupported component type for VLEN")
    return VLType(group, dt, name, typeid=xtype)

cdef _strencode(pystr,encoding=None):
    # encode a string into bytes.  If already bytes, do nothing.
    # uses default_encoding module variable for default encoding.
    if encoding is None:
        encoding = default_encoding
    try:
        return pystr.encode(encoding)
    except (AttributeError, UnicodeDecodeError):
        return pystr # already bytes or unicode?
