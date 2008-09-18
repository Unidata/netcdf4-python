"""
Introduction
============

Python interface to the netCDF version 4 library.  U{netCDF version 4 
<http://www.unidata.ucar.edu/software/netcdf/netcdf-4>} has many features 
not found in earlier versions of the library and is implemented on top of 
U{HDF5 <http://hdf.ncsa.uiuc.edu/HDF5>}. This module can read and write 
files in both the new netCDF 4 and the old netCDF 3 format, and can create 
files that are readable by HDF5 clients. The API modelled after 
U{Scientific.IO.NetCDF 
<http://starship.python.net/~hinsen/ScientificPython>}, and should be 
familiar to users of that module.

Many new features of netCDF 4 are implemented, such as multiple
unlimited dimensions, groups and zlib data compression.  All the new
primitive data types (such as 64 bit and unsigned integer types) are
implemented, except variable-length strings (C{NC_STRING}). User
defined data types (compound, vlen, enum etc.) are not supported.

Download 
========

 - U{Project page <http://code.google.com/p/netcdf4-python/>}.
 - U{Subversion repository <http://code.google.com/p/netcdf4-python/source>}.
 - U{Source tar.gz <http://code.google.com/p/netcdf4-python/downloads/list>}.

Requires 
======== 

 - numpy array module U{http://numpy.scipy.org}, version 1.0 or later.
 - The HDF5 C library version 1.8.1 or higher from U{ftp://ftp.hdfgroup.org/HDF5/current/src}.
 Be sure to build with 'C{--enable-hl --enable-shared}'.
 - The netCDF-4 C library from U{ftp://ftp.unidata.ucar.edu/pub/netcdf}.
 Version 4.0 or higher is required.
 Be sure to build with 'C{--enable-netcdf-4 --with-hdf5=$HDF5_DIR --enable-shared}',
 where C{$HDF5_DIR} is the directory where HDF5 was installed.


Install
=======

 - install the requisite python modules and C libraries (see above).
 - optionally, set the C{HDF5_DIR} environment variable to point to where HDF5
 is installed. (the libs in C{$HDF5_DIR/lib}, the headers in
 C{$HDF5_DIR/include}).
 - optionally, set the C{NETCDF4_DIR} environment variable to point to
 where the netCDF version 4 library and headers are installed.
 If C{HDF5_DIR} and C{NETCDF4_DIR} are not set, some standard
 locations will be searched.
 - if HDF5 was build with U{szip <http://hdf.ncsa.uiuc.edu/doc_resource/SZIP/>},
 you may also need to set the C{SZIP_DIR} 
 environment variable to point to where szip is installed. Note that
 netCDF 4.0 does not yet support szip compression.
 - run 'python setup.py install'
 - run some of the tests in the 'test' directory.

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
C{file_format} L{Dataset} attribute.  Closing the netCDF file is
accomplished via the L{close<Dataset.close>} method of the L{Dataset}
instance.

Here's an example:

>>> import netCDF4
>>> rootgrp = netCDF4.Dataset('test.nc', 'w', format='NETCDF4')
>>> print rootgrp.file_format
NETCDF4
>>>
>>> rootgrp.close()
            

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

>>> rootgrp = netCDF4.Dataset('test.nc', 'a')
>>> fcstgrp = rootgrp.createGroup('forecasts')
>>> analgrp = rootgrp.createGroup('analyses')
>>> print rootgrp.groups
{'analyses': <netCDF4._Group object at 0x24a54c30>, 
 'forecasts': <netCDF4._Group object at 0x24a54bd0>}
>>>

Groups can exist within groups in a L{Dataset}, just as directories
exist within directories in a unix filesystem. Each L{Group} instance
has a C{'groups'} attribute dictionary containing all of the group
instances contained within that group. Each L{Group} instance also has a
C{'path'} attribute that contains a simulated unix directory path to
that group. 

Here's an example that shows how to navigate all the groups in a
L{Dataset}. The function C{walktree} is a Python generator that is used
to walk the directory tree.

>>> fcstgrp1 = fcstgrp.createGroup('model1')
>>> fcstgrp2 = fcstgrp.createGroup('model2')
>>> def walktree(top):
>>>     values = top.groups.values()
>>>     yield values
>>>     for value in top.groups.values():
>>>         for children in walktree(value):
>>>             yield children
>>> print rootgrp.path, rootgrp
>>> for children in walktree(rootgrp):
>>>      for child in children:
>>>          print child.path, child
/ <netCDF4.Dataset object at 0x24a54c00>
/analyses <netCDF4.Group object at 0x24a54c30>
/forecasts <netCDF4.Group object at 0x24a54bd0>
/forecasts/model2 <netCDF4.Group object at 0x24a54cc0>
/forecasts/model1 <netCDF4.Group object at 0x24a54c60>
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
value is set to C{None}. In this example, there both the C{time} and
C{level} dimensions are unlimited.  Having more than one unlimited
dimension is a new netCDF 4 feature, in netCDF 3 files there may be only
one, and it must be the first (leftmost) dimension of the variable.

>>> rootgrp.createDimension('level', None)
>>> rootgrp.createDimension('time', None)
>>> rootgrp.createDimension('lat', 73)
>>> rootgrp.createDimension('lon', 144)
            

All of the L{Dimension} instances are stored in a python dictionary.

>>> print rootgrp.dimensions
{'lat': <netCDF4.Dimension object at 0x24a5f7b0>, 
 'time': <netCDF4.Dimension object at 0x24a5f788>, 
 'lon': <netCDF4.Dimension object at 0x24a5f7d8>, 
 'level': <netCDF4.Dimension object at 0x24a5f760>}
>>>

Calling the python C{len} function with a L{Dimension} instance returns
the current size of that dimension. The
L{isunlimited<Dimension.isunlimited>} method of a L{Dimension} instance
can be used to determine if the dimensions is unlimited, or appendable.

>>> for dimname, dimobj in rootgrp.dimensions.iteritems():
>>>    print dimname, len(dimobj), dimobj.isunlimited()
lat 73 False
time 0 True
lon 144 False
level 0 True
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
{'temp': <netCDF4.Variable object at 0x24a61068>,
 'level': <netCDF4.Variable object at 0.35f0f80>, 
 'longitude': <netCDF4.Variable object at 0x24a61030>,
 'pressure': <netCDF4.Variable object at 0x24a610a0>, 
 'time': <netCDF4.Variable object at 02x45f0.4.58>, 
 'latitude': <netCDF4.Variable object at 0.3f0fb8>}
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
>>> pressure.units = 'hPa'
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
{'source': 'netCDF4 python module tutorial',
'description': 'bogus example script',
'history': 'Created Mon Nov  7 10.30:56 2005'}

Attributes can be deleted from a netCDF L{Dataset}, L{Group} or
L{Variable} using the python C{del} statement (i.e. C{del grp.foo}
removes the attribute C{foo} the the group C{grp}).

6) Writing data to and retrieving data from a netCDF variable
-------------------------------------------------------------

Now that you have a netCDF L{Variable} instance, how do you put data
into it? You can just treat it like an array and assign data to a slice.

>>> import numpy 
>>> latitudes[:] = numpy.arange(-90,91,2.5)
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

Unlike numpy array objects, netCDF L{Variable} objects with unlimited
dimensions will grow along those dimensions if you assign data outside
the currently defined range of indices.

>>> # append along two unlimited dimensions by assigning to slice.
>>> nlats = len(rootgrp.dimensions['lat'])
>>> nlons = len(rootgrp.dimensions['lon'])
>>> print 'temp shape before adding data = ',temp.shape
temp shape before adding data =  (0, 0, 73, 144)
>>>
>>> from numpy.random.mtrand import uniform
>>> temp[0:5,0:10,:,:] = uniform(size=(5,10,nlats,nlons))
>>> print 'temp shape after adding data = ',temp.shape
temp shape after adding data =  (5, 10, 73, 144)
>>>
>>> # levels have grown, but no values yet assigned.
>>> print 'levels shape after adding pressure data = ',levels.shape
levels shape after adding pressure data =  (10,)
>>>

Note that the size of the levels variable grows when data is appended
along the C{level} dimension of the variable C{temp}, even though no
data has yet been assigned to levels.

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
C{NETCDF4_CLASSIC format} (C{NETCDF4} formatted files multi-file
datasets are not supported).

>>> for nfile in range(10):
>>>     f = Dataset('mftest'+repr(nfile)+'.nc','w',format='NETCDF4_CLASSIC')
>>>     f.createDimension('x',None)
>>>     x = f.createVariable('x','i',('x',))
>>>     x[0:10] = numpy.arange(nfile*10,10*(nfile+1))
>>>     f.close()

Now read all the files back in at once with L{MFDataset}

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
default value of C{complevel} is 6. Setting C{shuffle=False} will turn
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
data the data to be quantized using {numpy.around(scale*data)/scale}, where
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

# Make changes to this file, not the c-wrappers that Pyrex generates.

# pure python utilities
from netCDF4_utils import _buildStartCountStride, _quantize, _find_dim

__version__ = "0.7.6"

# Initialize numpy
import os
import netcdftime
import numpy
from glob import glob
from numpy import ma
from numpy import __version__ as _npversion
if _npversion.split('.')[0] < '1':
    raise ImportError('requires numpy version 1.0rc1 or later')
import_array()
include "netCDF4.pxi"

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

_default_fillvals = {#'S1':NC_FILL_CHAR, 
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

_nctonptype = {}
for _key,_value in _nptonctype.iteritems():
    _nctonptype[_value] = _key
_supportedtypes = _nptonctype.keys()

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
        raise RuntimeError(nc_strerror(ierr))
    attslist = []
    for n from 0 <= n < numatts:
        ierr = nc_inq_attname(grpid, varid, n, namstring)
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))
        attslist.append(namstring)
    return attslist

cdef _get_att(int grpid, int varid, name):
    # Private function to get an attribute value given its name
    cdef int ierr, n
    cdef size_t att_len
    cdef char *attname
    cdef nc_type att_type
    cdef ndarray value_arr
    cdef char *strdata 
    attname = PyString_AsString(name)
    ierr = nc_inq_att(grpid, varid, attname, &att_type, &att_len)
    if ierr != NC_NOERR:
        raise AttributeError(nc_strerror(ierr))
    # attribute is a character or string ...
    if att_type == NC_CHAR or att_type == NC_STRING:
        value_arr = numpy.empty(att_len,'S1')
        ierr = nc_get_att_text(grpid, varid, attname, <char *>value_arr.data)
        if ierr != NC_NOERR:
            raise AttributeError(nc_strerror(ierr))
        pstring = value_arr.tostring()
        # remove NULL characters from python string
        return pstring.replace('\x00','')
    # a regular numeric type.
    else:
        if att_type == NC_LONG:
            att_type = NC_INT
        if att_type not in _nctonptype.keys():
            raise ValueError, 'unsupported attribute type'
        try:
            type_att = _nctonptype[att_type]
        except:
            raise KeyError('attribute %s has unsupported datatype' % attname)
        value_arr = numpy.empty(att_len,type_att)
        ierr = nc_get_att(grpid, varid, attname, value_arr.data)
        if ierr != NC_NOERR:
            raise AttributeError(nc_strerror(ierr))
        if value_arr.shape == ():
            # return a scalar for a scalar array
            return value_arr.item()
        elif att_len == 1:
            # return a scalar for a single element array
            return value_arr[0]
        else:
            return value_arr

def _set_default_format(object format='NETCDF4',object verbose=False):
    # Private function to set the netCDF file format
    if format == 'NETCDF4':
        if verbose:
            print "Switching to netCDF-4 format"
        nc_set_default_format(NC_FORMAT_NETCDF4, NULL)
    elif format == 'NETCDF4_CLASSIC':
        if verbose:
            print "Switching to netCDF-4 format (with NC_CLASSIC_MODEL)"
        nc_set_default_format(NC_FORMAT_NETCDF4_CLASSIC, NULL)
    elif format == 'NETCDF3_64BIT':
        if verbose:
            print "Switching to 64-bit offset format"
        nc_set_default_format(NC_FORMAT_64BIT, NULL)
    elif format == 'NETCDF3_CLASSIC':
        if verbose:
            print "Switching to netCDF classic format"
        nc_set_default_format(NC_FORMAT_CLASSIC, NULL)
    else:
        raise ValueError, "format must be 'NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_64BIT', or 'NETCDF3_CLASSIC', got '%s'" % format

cdef _get_format(int grpid):
    # Private function to get the netCDF file format
    cdef int ierr, formatp
    ierr = nc_inq_format(grpid, &formatp)
    if ierr != NC_NOERR:
        raise RuntimeError(nc_strerror(ierr))
    if formatp == NC_FORMAT_NETCDF4:
        return 'NETCDF4'
    elif formatp == NC_FORMAT_NETCDF4_CLASSIC:
        return 'NETCDF4_CLASSIC'
    elif formatp == NC_FORMAT_64BIT:
        return 'NETCDF3_64BIT'
    elif formatp == NC_FORMAT_CLASSIC:
        return 'NETCDF3_CLASSIC'

cdef _set_att(int grpid, int varid, name, value):
    # Private function to set an attribute name/value pair
    cdef int i, ierr, lenarr, n
    cdef char *attname, *datstring, *strdata
    cdef ndarray value_arr 
    attname = PyString_AsString(name)
    # put attribute value into a numpy array.
    value_arr = numpy.array(value)
    # if array is 64 bit integers, cast to 32 bit integers
    # if 64-bit datatype not supported.
    if value_arr.dtype.str[1:] == 'i8' and 'i8' not in _supportedtypes:
        value_arr = value_arr.astype('i4')
    # if array contains strings, write a text attribute.
    if value_arr.dtype.char == 'S':
        dats = value_arr.tostring()
        datstring = dats
        lenarr = len(dats)
        ierr = nc_put_att_text(grpid, varid, attname, lenarr, datstring)
        if ierr != NC_NOERR:
            raise AttributeError(nc_strerror(ierr))
    # a 'regular' array type ('f4','i4','f8' etc)
    else:
        if value_arr.dtype.str[1:] not in _supportedtypes:
            raise TypeError, 'illegal data type for attribute, must be one of %s, got %s' % (_supportedtypes, value_arr.dtype.str[1:])
        xtype = _nptonctype[value_arr.dtype.str[1:]]
        lenarr = PyArray_SIZE(value_arr)
        ierr = nc_put_att(grpid, varid, attname, xtype, lenarr, value_arr.data)
        if ierr != NC_NOERR:
            raise AttributeError(nc_strerror(ierr))

cdef _get_dims(group):
    # Private function to create L{Dimension} instances for all the
    # dimensions in a L{Group} or Dataset
    cdef int ierr, numdims, n
    cdef int dimids[NC_MAX_DIMS]
    cdef char namstring[NC_MAX_NAME+1]
    # get number of dimensions in this Group.
    ierr = nc_inq_ndims(group._grpid, &numdims)
    if ierr != NC_NOERR:
        raise RuntimeError(nc_strerror(ierr))
    # create empty dictionary for dimensions.
    dimensions = {}
    if numdims > 0:
        if group.file_format == 'NETCDF4':
            ierr = nc_inq_dimids(group._grpid, &numdims, dimids, 0)
            if ierr != NC_NOERR:
                raise RuntimeError(nc_strerror(ierr))
        else:
            for n from 0 <= n < numdims:
                dimids[n] = n
        for n from 0 <= n < numdims:
            ierr = nc_inq_dimname(group._grpid, dimids[n], namstring)
            if ierr != NC_NOERR:
                raise RuntimeError(nc_strerror(ierr))
            name = namstring
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
        raise RuntimeError(nc_strerror(ierr))
    # create dictionary containing L{Group} instances for groups in this group
    groups = {}
    if numgrps > 0:
        grpids = <int *>malloc(sizeof(int) * numgrps)
        ierr = nc_inq_grps(group._grpid, NULL, grpids)
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))
        for n from 0 <= n < numgrps:
             ierr = nc_inq_grpname(grpids[n], namstring)
             if ierr != NC_NOERR:
                 raise RuntimeError(nc_strerror(ierr))
             name = namstring
             groups[name] = Group(group, name, id=grpids[n])
        free(grpids)
    return groups

cdef _get_vars(group):
    # Private function to create L{Variable} instances for all the
    # variables in a L{Group} or Dataset
    cdef int ierr, numvars, n, nn, numdims, varid
    cdef size_t sizein
    cdef int *varids
    cdef int dim_sizes[NC_MAX_DIMS], dimids[NC_MAX_DIMS]
    cdef nc_type xtype
    cdef char namstring[NC_MAX_NAME+1]
    # get number of variables in this Group.
    ierr = nc_inq_nvars(group._grpid, &numvars)
    if ierr != NC_NOERR:
        raise RuntimeError(nc_strerror(ierr))
    # create empty dictionary for variables.
    variables = {}
    if numvars > 0:
        # get variable ids.
        varids = <int *>malloc(sizeof(int) * numvars)
        if group.file_format == 'NETCDF4':
            ierr = nc_inq_varids(group._grpid, &numvars, varids)
            if ierr != NC_NOERR:
                raise RuntimeError(nc_strerror(ierr))
        else:
            for n from 0 <= n < numvars:
                varids[n] = n
        # loop over variables. 
        for n from 0 <= n < numvars:
             varid = varids[n]
             # get variable name.
             ierr = nc_inq_varname(group._grpid, varid, namstring)
             if ierr != NC_NOERR:
                 raise RuntimeError(nc_strerror(ierr))
             name = namstring
             # get variable type.
             ierr = nc_inq_vartype(group._grpid, varid, &xtype)
             if ierr != NC_NOERR:
                 raise RuntimeError(nc_strerror(ierr))
             try:
                 datatype = _nctonptype[xtype]
             except:
                 #raise KeyError('variable %s has unsupported data type' % name)
                 print "WARNING: variable '%s' has unsupported datatype, skipping .." % name
                 continue
             # get number of dimensions.
             ierr = nc_inq_varndims(group._grpid, varid, &numdims)
             if ierr != NC_NOERR:
                 raise RuntimeError(nc_strerror(ierr))
             # get dimension ids.
             ierr = nc_inq_vardimid(group._grpid, varid, dimids)
             if ierr != NC_NOERR:
                 raise RuntimeError(nc_strerror(ierr))
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
                     for key, value in grp.dimensions.iteritems():
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

_private_atts = ['_grpid','_grp','_varid','groups','dimensions','variables','dtype','file_format', '_nunlimdim','path','parent','ndim','maskandscale']


cdef class Dataset:
    """
Dataset(self, filename, mode="r", clobber=True, format='NETCDF4')

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
file_format} and C{path} are read-only (and should not be modified by the 
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

@ivar file_format: The C{file_format} attribute describes the netCDF
file format version, one of C{NETCDF3_CLASSIC}, C{NETCDF4},
C{NETCDF4_CLASSIC} or C{NETCDF3_64BIT}.  This module can read and 
write all formats.

@ivar path: The C{path} attribute shows the location of the L{Group} in
the L{Dataset} in a unix directory format (the names of groups in the
hierarchy separated by backslashes). A L{Dataset}, instance is the root
group, so the path is simply C{'/'}."""
    cdef public int _grpid
    cdef public groups, dimensions, variables, file_format, path, parent, maskanscale

    def __init__(self, filename, mode='r', clobber=True, format='NETCDF4', **kwargs):
        cdef int grpid, ierr, numgrps, numdims, numvars
        cdef char *path
        cdef int *grpids, *dimids
        cdef char namstring[NC_MAX_NAME+1]
        path = filename
        if mode == 'w':
            _set_default_format(format=format)
            if clobber:
                ierr = nc_create(path, NC_CLOBBER, &grpid)
            else:
                ierr = nc_create(path, NC_NOCLOBBER, &grpid)
            # initialize group dict.
        elif mode == 'r':
            ierr = nc_open(path, NC_NOWRITE, &grpid)
        elif mode == 'r+' or mode == 'a':
            ierr = nc_open(path, NC_WRITE, &grpid)
        else:
            raise ValueError("mode must be 'w', 'r', 'a' or 'r+', got '%s'" % mode)
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))
        # file format attribute.
        self.file_format = _get_format(grpid)
        self._grpid = grpid
        self.path = '/'
        self.parent = None
        # get dimensions in the root group.
        self.dimensions = _get_dims(self)
        # get variables in the root Group.
        self.variables = _get_vars(self)
        # get groups in the root Group.
        if self.file_format == 'NETCDF4':
            self.groups = _get_grps(self)
        else:
            self.groups = {}

    def close(self):
        """
close(self)

Close the Dataset."""
        cdef int ierr 
        ierr = nc_close(self._grpid)
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))

    def sync(self):
        """
sync(self)

Writes all buffered data in the L{Dataset} to the disk file."""
        cdef int ierr
        ierr = nc_sync(self._grpid)
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))

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
            raise RuntimeError(nc_strerror(ierr))

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
            raise RuntimeError(nc_strerror(ierr))

    def createDimension(self, dimname, size=None):
        """
createDimension(self, dimname, size=None)

Creates a new dimension with the given C{dimname} and C{size}. 

C{size} must be a positive integer or C{None}, which stands for 
"unlimited" (default is C{None}). The return value is the L{Dimension} 
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
        dim = self.dimensions[oldname]
        namstring = PyString_AsString(newname)
        if self.file_format != 'NETCDF4': self._redef()
        ierr = nc_rename_dim(self._grpid, dim._dimid, namstring)
        if self.file_format != 'NETCDF4': self._enddef()
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))
        # remove old key from dimensions dict.
        self.dimensions.pop(oldname)
        # add new key.
        self.dimensions[newname] = dim
        # Variable.dimensions is determined by a method that
        # looks in the file, so no need to manually update.


    def createVariable(self, varname, datatype, dimensions=(), zlib=False, complevel=6, shuffle=True, fletcher32=False, contiguous=False, chunksizes=None, endian='native', least_significant_digit=None, fill_value=None):
        """
createVariable(self, varname, datatype, dimensions=(), zlib=False, complevel=6, shuffle=True, fletcher32=False, contiguous=False, chunksizes=None, endian='native', least_significant_digit=None, fill_value=None)

Creates a new variable with the given C{varname}, C{datatype}, and 
C{dimensions}. If dimensions are not given, the variable is assumed to be 
a scalar.

The C{datatype} can be a numpy datatype object, or a string that describes 
a numpy dtype object (like the C{dtype.str} attribue of a numpy array). 
Supported specifiers include: C{'S1' or 'c' (NC_CHAR), 'i1' or 'b' or 'B' 
(NC_BYTE), 'u1' (NC_UBYTE), 'i2' or 'h' or 's' (NC_SHORT), 'u2' 
(NC_USHORT), 'i4' or 'i' or 'l' (NC_INT), 'u4' (NC_UINT), 'i8' (NC_INT64), 
'u8' (NC_UINT64), 'f4' or 'f' (NC_FLOAT), 'f8' or 'd' (NC_DOUBLE)}.

Data from netCDF variables is presented to python as numpy arrays with
the corresponding data type. 

C{dimensions} must be a tuple containing dimension names (strings) that 
have been defined previously using C{createDimension}. The default value 
is an empty tuple, which means the variable is a scalar.

If the optional keyword C{zlib} is C{True}, the data will be compressed in 
the netCDF file using gzip compression (default C{False}).

The optional keyword C{complevel} is an integer between 1 and 9 describing 
the level of compression desired (default 6). Ignored if C{zlib=False}.

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
any data is written to it).  If fill_value is set to C{False}, then
the variable is not pre-filled.

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

The return value is the L{Variable} class instance describing the new 
variable.

A list of names corresponding to netCDF variable attributes can be 
obtained with the L{Variable} method C{ncattrs()}. A dictionary
containing all the netCDF attribute name/value pairs is provided by
the C{__dict__} attribute of a L{Variable} instance.

L{Variable} instances behave much like array objects. Data can be
assigned to or retrieved from a variable with indexing and slicing
operations on the L{Variable} instance. A L{Variable} instance has five
standard attributes: C{dimensions, dtype, shape, ndim} and
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
        self.variables[varname] = Variable(self, varname, datatype, dimensions=dimensions, zlib=zlib, complevel=complevel, shuffle=shuffle, fletcher32=fletcher32, contiguous=contiguous, chunksizes=chunksizes, endian=endian, least_significant_digit=least_significant_digit, fill_value=fill_value)
        return self.variables[varname]

    def renameVariable(self, oldname, newname):
        """
renameVariable(self, oldname, newname)

rename a L{Variable} named C{oldname} to C{newname}"""
        cdef char *namstring
        try:
            var = self.variables[oldname]
        except:
            raise KeyError('%s not a valid variable name' % oldname)
        namstring = PyString_AsString(newname)
        ierr = nc_rename_var(self._grpid, var._varid, namstring)
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))
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

    def __delattr__(self,name):
        cdef char *attname
        # if it's a netCDF attribute, remove it
        if name not in _private_atts:
            attname = PyString_AsString(name)
            if self.file_format != 'NETCDF4': self._redef()
            ierr = nc_del_att(self._grpid, NC_GLOBAL, attname)
            if self.file_format != 'NETCDF4': self._enddef()
            if ierr != NC_NOERR:
                raise RuntimeError(nc_strerror(ierr))
        else:
            raise AttributeError, "'%s' is one of the reserved attributes %s, cannot delete" % (name, tuple(_private_atts))

    def __setattr__(self,name,value):
        # if name in _private_atts, it is stored at the python
        # level and not in the netCDF file.
        if name not in _private_atts:
            if self.file_format != 'NETCDF4': self._redef()
            _set_att(self._grpid, NC_GLOBAL, name, value)
            if self.file_format != 'NETCDF4': self._enddef()
        elif not name.endswith('__'):
            if hasattr(self,name):
                raise AttributeError("'%s' is one of the reserved attributes %s, cannot rebind" % (name, tuple(_private_atts)))
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
                    values.append(_get_att(self._grpid, NC_GLOBAL, name))
                return dict(zip(names,values))
            else:
                raise AttributeError
        elif name in _private_atts:
            return self.__dict__[name]
        else:
            return _get_att(self._grpid, NC_GLOBAL, name)

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
        cdef int ierr, n, numgrps, numdims, numvars
        cdef int *grpids, *dimids
        cdef char *groupname
        cdef char namstring[NC_MAX_NAME+1]
        groupname = name
        if kwargs.has_key('id'):
            self._grpid = kwargs['id']
        else:
            ierr = nc_def_grp(parent._grpid, groupname, &self._grpid)
            if ierr != NC_NOERR:
                raise RuntimeError(nc_strerror(ierr))
        # set file_format attribute.
        self.file_format = _get_format(self._grpid)
        # get number of groups in this group.
        ierr = nc_inq_grps(self._grpid, &numgrps, NULL)
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))
        # full path to Group.
        self.path = os.path.join(parent.path, name)
        # parent group.
        self.parent = parent
        # get dimensions in this Group.
        self.dimensions = _get_dims(self)
        # get variables in this Group.
        self.variables = _get_vars(self)
        # get groups in this Group.
        self.groups = _get_grps(self)

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

B{C{size}}  - Size of the dimension (Default C{None} means unlimited).

B{Returns:}

a L{Dimension} instance.  All further operations on the netCDF Dimension 
are accomplised via L{Dimension} instance methods.

The current maximum size of a L{Dimension} instance can be obtained by
calling the python C{len} function on the L{Dimension} instance. The
C{isunlimited()} method of a L{Dimension} instance can be used to
determine if the dimension is unlimited"""
    cdef public int _dimid, _grpid
    cdef public _file_format

    def __init__(self, grp, name, size=None, **kwargs):
        cdef int ierr
        cdef char *dimname
        cdef size_t lendim
        self._grpid = grp._grpid
        self._file_format = grp.file_format
        if kwargs.has_key('id'):
            self._dimid = kwargs['id']
        else:
            dimname = name
            if size is not None:
                lendim = size
            else:
                lendim = NC_UNLIMITED
            if grp.file_format != 'NETCDF4': grp._redef()
            ierr = nc_def_dim(self._grpid, dimname, lendim, &self._dimid)
            if grp.file_format != 'NETCDF4': grp._enddef()
            if ierr != NC_NOERR:
                raise RuntimeError(nc_strerror(ierr))

    def __len__(self):
        # len(L{Dimension} instance) returns current size of dimension
        cdef int ierr
        cdef size_t lengthp
        ierr = nc_inq_dimlen(self._grpid, self._dimid, &lengthp)
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))
        return lengthp

    def isunlimited(self):
        """
isunlimited(self)

returns C{True} if the L{Dimension} instance is unlimited, C{False} otherwise."""
        cdef int ierr, n, numunlimdims, ndims, nvars, ngatts, xdimid
        cdef int unlimdimids[NC_MAX_DIMS]
        if self._file_format == 'NETCDF4':
            ierr = nc_inq_unlimdims(self._grpid, &numunlimdims, NULL)
            if ierr != NC_NOERR:
                raise RuntimeError(nc_strerror(ierr))
            if numunlimdims == 0:
                return False
            else:
                dimid = self._dimid
                ierr = nc_inq_unlimdims(self._grpid, &numunlimdims, unlimdimids)
                if ierr != NC_NOERR:
                    raise RuntimeError(nc_strerror(ierr))
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
Variable(self, group, name, datatype, dimensions=(), zlib=False, complevel=6, shuffle=True, fletcher32=False, contiguous=False, chunksizes=None, endian='native', least_significant_digit=None,fill_value=None)

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
C{'i4'}).

B{Keywords:}

B{C{dimensions}} - a tuple containing the variable's dimension names 
(defined previously with C{createDimension}). Default is an empty tuple 
which means the variable is a scalar (and therefore has no dimensions).

B{C{zlib}} - if C{True}, data assigned to the L{Variable}  
instance is compressed on disk. Default C{False}.

B{C{complevel}} - the level of zlib compression to use (1 is the fastest, 
but poorest compression, 9 is the slowest but best compression). Default 6.
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

The C{zlib, complevel, shuffle, fletcher32, contiguous, chunksizes} and C{endian}
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
the variable is not pre-filled.
 
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

@ivar least_significant_digit: Describes the power of ten of the smallest 
decimal place in the data the contains a reliable value.  Data is 
truncated to this decimal place when it is assigned to the L{Variable} 
instance. If C{None}, the data is not truncated. """
    cdef public int _varid, _grpid, _nunlimdim
    cdef object _grp
    cdef public ndim, dtype, maskandscale

    def __init__(self, grp, name, datatype, dimensions=(), zlib=False, complevel=6, shuffle=True, fletcher32=False, contiguous=False, chunksizes=None, endian='native', least_significant_digit=None, fill_value=None,  **kwargs):
        cdef int ierr, ndims, icontiguous, ideflate_level, numdims
        cdef char *varname
        cdef nc_type xtype, vltypeid
        cdef int dimids[NC_MAX_DIMS]
        cdef int *chunksizesp
        # if dimensions is a string, convert to a tuple
        # this prevents a common error that occurs when
        # dimensions = ('lat') instead of ('lat',)
        if type(dimensions) == type(''):
            dimensions = dimensions,
        self._grpid = grp._grpid
        self._grp = grp
        # convert to a real numpy datatype object if necessary.
        if type(datatype) != numpy.dtype:
            datatype = numpy.dtype(datatype)
        # check validity of datatype.
        if datatype.str[1:] not in _supportedtypes:
            raise TypeError('illegal data type, must be one of %s, got %s' % (_supportedtypes,datatype))
        # dtype variable attribute is a numpy datatype object.
        self.dtype = datatype
        if kwargs.has_key('id'):
            self._varid = kwargs['id']
        else:
            varname = name
            ndims = len(dimensions)
            # find netCDF primitive data type corresponding to 
            # specified numpy data type.
            xtype = _nptonctype[datatype.str[1:]]
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
            if grp.file_format != 'NETCDF4': grp._redef()
            # define variable.
            if ndims:
                ierr = nc_def_var(self._grpid, varname, xtype, ndims,
                                  dimids, &self._varid)
            else: # a scalar variable.
                ierr = nc_def_var(self._grpid, varname, xtype, ndims,
                                  NULL, &self._varid)
            if ierr != NC_NOERR:
                if grp.file_format != 'NETCDF4': grp._enddef()
                raise RuntimeError(nc_strerror(ierr))
            # set zlib, shuffle, chunking, fletcher32 and endian
            # variable settings.
            # don't bother for NETCDF3* formats.
            # for NETCDF3* formats, the zlib,shuffle,chunking,endian
            # and fletcher32 are silently ignored.
            if grp.file_format in ['NETCDF4','NETCDF4_CLASSIC']:
                # set zlib and shuffle parameters.
                if zlib and ndims: # don't bother for scalar variable
                    ideflate_level = complevel
                    if shuffle:
                        ierr = nc_def_var_deflate(self._grpid, self._varid, 1, 1, ideflate_level)
                    else:
                        ierr = nc_def_var_deflate(self._grpid, self._varid, 0, 1, ideflate_level)
                    if ierr != NC_NOERR:
                        if grp.file_format != 'NETCDF4': grp._enddef()
                        raise RuntimeError(nc_strerror(ierr))
                # set checksum.
                if fletcher32:
                    ierr = nc_def_var_fletcher32(self._grpid, self._varid, 1)
                    if ierr != NC_NOERR:
                        if grp.file_format != 'NETCDF4': grp._enddef()
                        raise RuntimeError(nc_strerror(ierr))
                # set chunking stuff.
                if ndims: # don't bother for scalar variable.
                    if contiguous:
                        icontiguous = 1
                        if chunksizes is not None:
                            raise ValueError('cannot specify chunksizes for a contiguous dataset')
                    else:
                        icontiguous = 0
                    if chunksizes is None:
                        chunksizesp = NULL
                    else:
                        if len(chunksizes) != len(dimensions):
                            if grp.file_format != 'NETCDF4': grp._enddef()
                            raise ValueError('chunksizes must be a sequence with the same length as dimensions')
                        chunksizesp = <int *>malloc(sizeof(int) * ndims)
                        for n from 0 <= n < ndims:
                            chunksizesp[n] = chunksizes[n]
                    if chunksizes is not None or contiguous:
                        ierr = nc_def_var_chunking(self._grpid, self._varid, icontiguous, chunksizesp)
                        free(chunksizesp)
                        if ierr != NC_NOERR:
                            if grp.file_format != 'NETCDF4': grp._enddef()
                            raise RuntimeError(nc_strerror(ierr))
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
                    if grp.file_format != 'NETCDF4': grp._enddef()
                    raise RuntimeError(nc_strerror(ierr))
            # set a fill value for this variable if fill_value keyword
            # given.  This avoids the HDF5 overhead of deleting and 
            # recreating the dataset if it is set later (after the enddef).
            if fill_value is not None:
                if not fill_value and isinstance(fill_value,bool):
                    # no filling for this variable if fill_value==False.
                    ierr = nc_def_var_fill(self._grpid, self._varid, 1, NULL)
                    if ierr != NC_NOERR:
                        if grp.file_format != 'NETCDF4': grp._enddef()
                        raise RuntimeError(nc_strerror(ierr))
                else:
                    # cast fill_value to type of variable.
                    fillval = numpy.array(fill_value, self.dtype)
                    _set_att(self._grpid, self._varid, '_FillValue', fillval)
            if least_significant_digit is not None:
                self.least_significant_digit = least_significant_digit
            # leave define mode if not a NETCDF4 format file.
            if grp.file_format != 'NETCDF4': grp._enddef()
        # count how many unlimited dimensions there are.
        self._nunlimdim = 0
        for dimname in self.dimensions:
            # look in current group, and parents for dim.
            dim = _find_dim(self._grp, dimname)
            if dim.isunlimited(): self._nunlimdim = self._nunlimdim + 1
        # set ndim attribute (number of dimensions).
        ierr = nc_inq_varndims(self._grpid, self._varid, &numdims)
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))
        self.ndim = numdims
        # default for automatically applying scale_factor and
        # add_offset, and converting to/from masked arrays is False.
        self.maskandscale = False

    property shape:
        """find current sizes of all variable dimensions"""
        def __get__(self):
            shape = ()
            for dimname in self.dimensions:
                # look in current group, and parents for dim.
                dim = _find_dim(self._grp,dimname)
                shape = shape + (len(dim),)
            return shape
        def __set__(self,value):
            raise AttributeError("shape cannot be altered")

    property dimensions:
        """get variables's dimension names"""
        def __get__(self):
            # Private method to get variables's dimension names
            cdef int ierr, numdims, n, nn
            cdef char namstring[NC_MAX_NAME+1]
            cdef int dimids[NC_MAX_DIMS]
            # get number of dimensions for this variable.
            ierr = nc_inq_varndims(self._grpid, self._varid, &numdims)
            if ierr != NC_NOERR:
                raise RuntimeError(nc_strerror(ierr))
            # get dimension ids.
            ierr = nc_inq_vardimid(self._grpid, self._varid, dimids)
            if ierr != NC_NOERR:
                raise RuntimeError(nc_strerror(ierr))
            # loop over dimensions, retrieve names.
            dimensions = ()
            for nn from 0 <= nn < numdims:
                ierr = nc_inq_dimname(self._grpid, dimids[nn], namstring)
                if ierr != NC_NOERR:
                    raise RuntimeError(nc_strerror(ierr))
                name = namstring
                dimensions = dimensions + (name,)
            return dimensions
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

    def filters(self):
        """
filters(self)

return dictionary containing HDF5 filter parameters."""
        cdef int ierr,ideflate,ishuffle,ideflate_level,ifletcher32
        filtdict = {'zlib':False,'shuffle':False,'complevel':0,'fletcher32':False}
        if self._grp.file_format not in ['NETCDF4_CLASSIC','NETCDF4']: return
        ierr = nc_inq_var_deflate(self._grpid, self._varid, &ishuffle, &ideflate, &ideflate_level)
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))
        ierr = nc_inq_var_fletcher32(self._grpid, self._varid, &ifletcher32)
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))
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
        if self._grp.file_format not in ['NETCDF4_CLASSIC','NETCDF4']: return None
        ierr = nc_inq_var_endian(self._grpid, self._varid, &iendian)
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))
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
        cdef int *chunksizesp
        if self._grp.file_format not in ['NETCDF4_CLASSIC','NETCDF4']: return None
        chunksizesp = <int *>malloc(sizeof(int) * ndims)
        ierr = nc_inq_var_chunking(self._grpid, self._varid, &icontiguous, chunksizesp)
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))
        ndims = self.ndim
        chunksizes=[]
        for n from 0 <= n < ndims:
            chunksizes.append(chunksizesp[n])
        free(chunksizesp)
        if icontiguous:
            return 'contiguous'
        else:
            return chunksizes

    def __delattr__(self,name):
        cdef char *attname
        # if it's a netCDF attribute, remove it
        if name not in _private_atts:
            attname = PyString_AsString(name)
            if self._grp.file_format != 'NETCDF4': self._grp._redef()
            ierr = nc_del_att(self._grpid, self._varid, attname)
            if self._grp.file_format != 'NETCDF4': self._grp._enddef()
            if ierr != NC_NOERR:
                raise RuntimeError(nc_strerror(ierr))
        else:
            raise AttributeError("'%s' is one of the reserved attributes %s, cannot delete" % (name, tuple(_private_atts)))

    def __setattr__(self,name,value):
        # if name in _private_atts, it is stored at the python
        # level and not in the netCDF file.
        if name not in _private_atts:
            # if setting _FillValue, make sure value
            # has same type as variable.
            if name == '_FillValue':
                value = numpy.array(value, self.dtype)
            if self._grp.file_format != 'NETCDF4': self._grp._redef()
            _set_att(self._grpid, self._varid, name, value)
            if self._grp.file_format != 'NETCDF4': self._grp._enddef()
        elif not name.endswith('__'):
            if hasattr(self,name):
                raise AttributeError("'%s' is one of the reserved attributes %s, cannot rebind" % (name, tuple(_private_atts)))
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
                    values.append(_get_att(self._grpid, self._varid, name))
                return dict(zip(names,values))
            else:
                raise AttributeError
        elif name in _private_atts:
            return self.__dict__[name]
        else:
            return _get_att(self._grpid, self._varid, name)

    def __getitem__(self, elem):
        # This special method is used to index the netCDF variable
        # using the "extended slice syntax". The extended slice syntax
        # is a perfect match for the "start", "count" and "stride"
        # arguments to the nc_get_var() function, and is much more easy
        # to use.
        start, count, stride, sliceout = _buildStartCountStride(elem,self.shape,self.dimensions,self._grp)
        data =  self._get(start, count, stride)
        if sliceout is not None:
            data = data[sliceout].squeeze() # slice resulting array with 'fancy' indices
        # if auto_maskandscale mode set to True, (through
        # a call to set_auto_maskandscale), perform
        # automatic unpacking using scale_factor/add_offset
        # and automatic conversion to masked array using
        # missing_value/_Fill_Value.
        if self.maskandscale:
            totalmask = numpy.zeros(data.shape, numpy.bool)
            fill_value = None
            if hasattr(self, 'missing_value') and (data == self.missing_value).any():
                mask=data==self.missing_value
                fill_value = self.missing_value
                totalmask += mask
            if hasattr(self, '_FillValue') and (data == self._FillValue).any():
                mask=data==self._FillValue
                if fill_value is None:
                    fill_value = self._FillValue
                totalmask += mask
            else:
                fillval = _default_fillvals[self.dtype.str[1:]]
                if (data == fillval).any():
                    mask=data==fillval
                    if fill_value is None:
                        fill_value = fillval
                    totalmask += mask
            # all values where data == missing_value or _FillValue are
            # masked.  fill_value set to missing_value if it exists,
            # otherwise _FillValue.
            if fill_value is not None:
                data = ma.masked_array(data,mask=totalmask,fill_value=fill_value)
            # if variable has scale_factor and add_offset attributes, rescale.
            if hasattr(self, 'scale_factor') and hasattr(self, 'add_offset'):
                data = self.scale_factor*data + self.add_offset
        return data
 
    def __setitem__(self, elem, data):
        # This special method is used to assign to the netCDF variable
        # using "extended slice syntax". The extended slice syntax
        # is a perfect match for the "start", "count" and "stride"
        # arguments to the nc_put_var() function, and is much more easy
        # to use.
        if hasattr(data,"shape"):
            start, count, stride, sliceout = _buildStartCountStride(elem,self.shape,self.dimensions,self._grp,datashape=data.shape)
        else:
            start, count, stride, sliceout = _buildStartCountStride(elem,self.shape,self.dimensions,self._grp)
        # quantize data if least_significant_digit attribute set.
        if 'least_significant_digit' in self.ncattrs():
            data = _quantize(data,self.least_significant_digit)
        # if auto_maskandscale mode set to True, (through
        # a call to set_auto_maskandscale), perform
        # automatic packing using scale_factor/add_offset
        # and automatic filling of masked arrays using
        # missing_value/_Fill_Value.
        if self.maskandscale:
            # use missing_value as fill value.
            # if no missing value set, use _FillValue.
            if hasattr(data,'mask'):
                if hasattr(self, 'missing_value'):
                    fillval = self.missing_value
                elif hasattr(self, '_FillValue'):
                    fillval = self._FillValue
                else:
                    fillval = _default_fillvals[self.dtype.str[1:]]
                data = data.filled(fill_value=fillval)
            # pack using scale_factor and add_offset.
            if hasattr(self, 'scale_factor') and hasattr(self, 'add_offset'):
                data = (data - self.add_offset)/self.scale_factor
        # A numpy array is needed. Convert if necessary.
        if not type(data) == numpy.ndarray:
            data = numpy.array(data,self.dtype)
        self._put(data, start, count, stride)

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
        return self[:]

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
C{scale_factor} and an C{add_offset} attribute, then data read
from that variable is unpacked using::

    data = self.scale_factor*data + self.add_offset
            
When data is written to a variable it is packed using::

    data = (data - self.add_offset)/self.scale_factor

For more information on how C{scale_factor} and C{add_offset} can be 
used to provide simple compression, see
U{http://www.cdc.noaa.gov/cdc/conventions/cdc_netcdf_standard.shtml
<http://www.cdc.noaa.gov/cdc/conventions/cdc_netcdf_standard.shtml>}.

The default value of C{maskandscale} is C{False}
(no automatic conversions are performed).
        """
        if maskandscale:
            self.maskandscale = True
        else:
            self.maskandscale = False

    def _put(self,ndarray data,start,count,stride):
        """Private method to put data into a netCDF variable"""
        cdef int ierr, ndims, totelem
        cdef size_t startp[NC_MAX_DIMS], countp[NC_MAX_DIMS]
        cdef ptrdiff_t stridep[NC_MAX_DIMS]
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
            # If just one element given, make a new array of desired
            # size and fill it with that data.
            if dataelem == 1:
                #datanew = numpy.empty(totelem,self.dtype)
                #datanew[:] = data
                #data = datanew
                data = data*numpy.ones(totelem,self.dtype)
            else:
                raise IndexError('size of data array does not conform to slice')
        # if data type of array doesn't match variable, 
        # try to cast the data.
        if self.dtype != data.dtype:
            data = data.astype(self.dtype) # cast data, if necessary.
        # if there is a negative stride, reverse the data, then use put_vars.
        if negstride:
            # reverse data along axes with negative strides.
            data = data[sl].copy() # make sure a copy is made.
        # strides all 1 or scalar variable, use put_vara (faster)
        if sum(stride) == ndims or ndims == 0:
            ierr = nc_put_vara(self._grpid, self._varid,
                               startp, countp, data.data)
        else:  
            ierr = nc_put_vars(self._grpid, self._varid,
                                  startp, countp, stridep, data.data)
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))

    def _get(self,start,count,stride):
        """Private method to retrieve data from a netCDF variable"""
        cdef int ierr, ndims
        cdef size_t startp[NC_MAX_DIMS], countp[NC_MAX_DIMS]
        cdef ptrdiff_t stridep[NC_MAX_DIMS]
        cdef ndarray data
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
        data = numpy.empty(shapeout, self.dtype)
        # strides all 1 or scalar variable, use get_vara (faster)
        if sum(stride) == ndims or ndims == 0: 
            ierr = nc_get_vara(self._grpid, self._varid,
                               startp, countp, data.data)
        else:
            ierr = nc_get_vars(self._grpid, self._varid,
                               startp, countp, stridep, data.data)
        if negstride:
            # reverse data along axes with negative strides.
            data = data[sl].copy() # make a copy so data is contiguous.
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))
        if not self.dimensions: 
            return data[0] # a scalar 
        #elif data.shape == (1,):
        #    # if a single item, just return a python scalar
        #    # (instead of a scalar array).
        #    return data.item()
        elif squeeze_out:
            return data.squeeze()
        else:
            return data

# include pure python utility functions and MFDataset class.
# (use include instead of importing them so docstrings
#  get included in C extension code).
include "utils.pyx"
