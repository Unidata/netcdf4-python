"""
Introduction
============

Python interface to the netCDF version 4 library.  U{netCDF version 4 
<http://www.unidata.ucar.edu/software/netcdf/netcdf-4>} has many features 
not found in earlier versions of the library and is implemented on top of 
U{HDF5 <http://hdf.ncsa.uiuc.edu/HDF5>}. This module can read files 
created with netCDF versions 2 and 3, but writes files which are only 
compatible with netCDF version 4. To read and write netCDF 3 files, and 
netCDF 4 files that are compatible with netCDF 3 clients, use the 
companion L{netCDF4_classic} module. The API is modelled after 
U{Scientific.IO.NetCDF 
<http://starship.python.net/~hinsen/ScientificPython>}, and should be 
familiar to users of that module.

Many new features of netCDF 4 are implemented, such as multiple
unlimited dimensions, groups and zlib data compression.  All the new
primitive data types (such as 64 bit and unsigned integer types) are
implemented, including variable-length strings (C{NC_STRING}). User
defined datatypes are not supported.

Download 
========

 - U{Project page <http://code.google.com/p/netcdf4-python/>}.
 - U{Subversion repository <http://code.google.com/p/netcdf4-python/source>}.
 - U{Source tar.gz <http://cheeseshop.python.org/pypi/netCDF4/>}.

Requires 
======== 

 - Pyrex module (U{http://www.cosc.canterbury.ac.nz/greg.ewing/python/Pyrex/}).
 If you're using python 2.5, you'll need the unofficial version of Pyrex from
 U{http://codespeak.net/svn/lxml/pyrex}, since the official version does
 not yet support python 2.5.
 - numpy array module U{http://numpy.scipy.org}, version 1.0rc1 or later.
 - A pre-release version of HDF5 1.8 is required.  Version
 1.7.52 is currently required. It is 
 available at U{ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4}.
 Be sure to build with 'C{--enable-hl}'.
 - netCDF version 4.  netCDF4 is now in alpha,
 and is a bit of a moving target.  This release is has only
 been tested with netcdf-4.0-alpha17, available from
 U{ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4}.
 Be sure to build with 'C{--enable-netcdf-4}' and 'C{--with-hdf5=$HD5_DIR}',
 where C{$HDF5_DIR} is the directory where HDF5 was installed.


Install
=======

 - set the C{HDF5_DIR} environment variable to point to where HDF5 is installed.
 (the libs in C{$HDF5_DIR/lib}, the headers in C{$HDF5_DIR/include}).
 - set the C{NETCDF4_DIR} environment variable to point to where netCDF version
 4 is installed.
 - run 'python setup.py install'
 - run some of the tests in the 'test' directory.

Tutorial
========

1) Creating/Opening/Closing a netCDF file
-----------------------------------------

To create a netCDF file from python, you simply call the L{Dataset} 
constructor. This is also the method used to open an existing netCDF file.  
If the file is open for write access (C{w, r+} or C{a}), you may write any 
type of data including new dimensions, groups, variables and attributes.  
netCDF files come in several flavors (C{NETCDF3_CLASSIC, NETCDF3_64BIT, 
NETCDF4_CLASSIC}, and C{NETCDF4}). The first two flavors are supported by 
version 3 of the netCDF library. C{NETCDF4_CLASSIC} files use the version 
4 disk format (HDF5), but do not use any features not found in the version 
3 API. They can be read by netCDF 3 clients only if they have been 
relinked against the netCDF 4 library. They can also be read by HDF5 
clients. C{NETCDF4} files use the version 4 disk format (HDF5) and use the 
new features of the version 4 API.  The C{netCDF4} module can read files 
with any of these formats, but only writes C{NETCDF4} formatted files. To 
write C{NETCDF4_CLASSIC}, C{NETCDF3_CLASSIC} or C{NETCDF3_64BIT} formatted 
files, use the L{netCDF4_classic} module. To see what how a given file is 
formatted, you can examine the C{file_format} L{Dataset} attribute. 
Closing the netCDF file is accomplished via the C{close} method of the 
L{Dataset} instance.

Here's an example:

>>> import netCDF4_simple as netCDF4
>>> rootgrp = netCDF4.Dataset('test.nc', 'w')
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
filesystem.  To create L{Group} instances, use the C{createGroup} method
of a L{Dataset} or L{Group} instance. C{createGroup} takes a single
argument, a python string containing the name of the new group. The new
L{Group} instances contained within the root group can be accessed by
name using the C{groups} dictionary attribute of the L{Dataset}
instance.

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
the C{createDimension} method of a L{Dataset} or L{Group} instance. A
Python string is used to set the name of the dimension, and an integer
value is used to set the size. To create an unlimited dimension (a
dimension that can be appended to), the size value is set to C{None}. In
this example, there both the C{time} and C{level} dimensions are
unlimited.

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
the current size of that dimension. The C{isunlimited()} method of a
L{Dimension} instance can be used to determine if the dimensions is
unlimited, or appendable.

>>> for dimname, dimobj in rootgrp.dimensions.iteritems():
>>>    print dimname, len(dimobj), dimobj.isunlimited()
lat 73 False
time 0 True
lon 144 False
level 0 True
>>>

L{Dimension} names can be changed using the C{renameDimension} method
of a L{Dataset} or L{Group} instance.
            
4) Variables in a netCDF file
-----------------------------

netCDF variables behave much like python multidimensional array objects
supplied by the U{numpy module <http://numpy.scipy.org>}. However,
unlike numpy arrays, netCDF4 variables can be appended to along one or
more 'unlimited' dimensions. To create a netCDF variable, use the
C{createVariable} method of a L{Dataset} or L{Group} instance. The
C{createVariable} method has two mandatory arguments, the variable name
(a Python string), and the variable datatype. The variable's dimensions
are given by a tuple containing the dimension names (defined previously
with C{createDimension}). To create a scalar variable, simply leave out
the dimensions keyword. The variable primitive datatypes correspond to
the dtype.str attribute of a numpy array, and can be one of C{'f4'}
(32-bit floating point), C{'f8'} (64-bit floating point), C{'i4'}
(32-bit signed integer), C{'i2'} (16-bit signed integer), C{'i8'}
(64-bit singed integer), C{'i1'} (8-bit signed integer), C{'u1'} (8-bit
unsigned integer), C{'u2'} (16-bit unsigned integer), C{'u4'} (32-bit
unsigned integer), C{'u8'} (64-bit unsigned integer), or C{'S1'}
(single-character string). There is also a C{'S'} datatype for variable
length strings, which have no corresponding numpy data type (they are
stored in numpy object arrays). Variables of datatype C{'S'} can be used
to store arbitrary python objects, since each element will be pickled
into a string (if it is not already a string) before being saved in the
netCDF file (see section 9 for more on storing arrays of python
objects). Pickle strings will be automatically un-pickled back into
python objects when they are read back in. The dimensions themselves 
are usually also defined as variables, called coordinate variables. The
C{createVariable} method returns an instance of the L{Variable} class
whose methods can be used later to access and set variable data and
attributes.

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

L{Variable} names can be changed using the C{renameVariable} method of a
L{Dataset} instance.
            

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
>>> times.units = 'days since January 1, 0001'
>>> times.calendar = 'proleptic_gregorian'

The C{ncattrs()} method of a L{Dataset}, L{Group} or L{Variable}
instance can be used to retrieve the names of all the netCDF attributes.
This method is provided as a convenience, since using the built-in
C{dir} Python function will return a bunch of private methods and
attributes that cannot (or should not) be modified by the user.

>>> for name in rootgrp.ncattrs():
>>>     print 'Global attr', name, '=', getattr(rootgrp,name)
Global attr description = bogus example script
Global attr history = Created Mon Nov  7 10.30:56 2005
Global attr source = netCDF4 python module tutorial

The C{__dict__} attribute of a L{Dataset}, L{Group} or L{Variable} instance provides
all the netCDF attribute name/value pairs in a python dictionary:

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

>>> import numpy as NP
>>> latitudes[:] = NP.arange(-90,91,2.5)
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
from calendar dates.  A module called L{netcdftime.netcdftime} is
provided with this package to do just that.  Here's an example of how it
can be used:

>>> # fill in times.
>>> from datetime import datetime, timedelta
>>> from netcdftime import utime
>>> cdftime = utime(times.units,calendar=times.calendar,format='%B %d, %Y') 
>>> dates = [datetime(2001,3,1)+n*timedelta(hours=12) for n in range(temp.shape[0])]
>>> times[:] = cdftime.date2num(dates)
>>> print 'time values (in units %s): ' % times.units+'\\n',times[:]
time values (in units hours since January 1, 0001): 
[ 17533056.  17533068.  17533080.  17533092.  17533104.]
>>>
>>> dates = cdftime.num2date(times[:])
>>> print 'dates corresponding to time values:\\n',dates
dates corresponding to time values:
[2001-03-01 00:00:00 2001-03-01 12:00:00 2001-03-02 00:00:00
 2001-03-02 12:00:00 2001-03-03 00:00:00]
>>>

Values of time in the specified units and calendar are converted to and
from python C{datetime} instances using the C{num2date} and C{date2num}
methods of the C{utime} class. See the L{netcdftime.netcdftime}
documentation for more details.
            
7) Efficient compression of netCDF variables
--------------------------------------------

Data stored in netCDF L{Variable} objects is compressed on disk by
default. The parameters for the compression are determined by the
C{zlib} and C{complevel} and C{shuffle} keyword arguments to the
C{createVariable} method.  The default values are C{zlib=True},
C{complevel=6} and C{shuffle=True}.  To turn off compression, set
C{zlib=False}.  C{complevel} regulates the speed and efficiency of the
compression (1 being fastest, but lowest compression ratio, 9 being
slowest but best compression ratio).  C{shuffle=False} will turn off the
HDF5 shuffle filter, which de-interlaces a block of data by reordering
the bytes.  The shuffle filter can significantly improve compression
ratios.  Setting C{fletcher32} keyword argument to C{createVariable} to
C{True} (it's C{False} by default) enables the Fletcher32 checksum
algorithm for error detection.

If your data only has a certain number of digits of precision (say for
example, it is temperature data that was measured with a precision of
0.1 degrees), you can dramatically improve compression by quantizing (or
truncating) the data using the C{least_significant_digit} keyword
argument to C{createVariable}. The least significant digit is the power
of ten of the smallest decimal place in the data that is a reliable
value. For example if the data has a precision of 0.1, then setting
C{least_significant_digit=1} will cause data the data to be quantized
using {NP.around(scale*data)/scale}, where scale = 2**bits, and bits is
determined so that a precision of 0.1 is retained (in this case bits=4). 
Effectively, this makes the compression 'lossy' instead of 'lossless',
that is some precision in the data is sacrificed for the sake of disk
space.

In our example, try replacing the line

>>> temp = rootgrp.createVariable('temp','f4',('time','level','lat','lon',))

with

>>> temp = rootgrp.createVariable('temp','f4',('time','level','lat','lon',),
                                  least_significant_digit=3)
            

and see how much smaller the resulting file is.

8) Converting netCDF 3 files to netCDF 4 files (with compression)
-----------------------------------------------------------------

A command line utility (C{nc3tonc4}) is provided which can convert a
netCDF 3 file (in C{NETCDF3_CLASSIC} or C{NETCDF3_64BIT} format) to a
C{NETCDF4_CLASSIC} file, optionally unpacking variables packed as short
integers (with scale_factor and add_offset) to floats, and adding zlib
compression (with the HDF5 shuffle filter and fletcher32 checksum). Data
may also be quantized (truncated) to a specified precision to improve
compression.

>>> os.system('nc3tonc4 -h')
nc3tonc4 [-h] [-o] [--zlib=(0|1)] [--complevel=(1-9)] [--shuffle=(0|1)]
         [--fletcher32=(0|1)] [--unpackshort=(0|1)]
         [--quantize=var1=n1,var2=n2,..] netcdf3filename netcdf4filename
-h -- Print usage message.
-o -- Overwite destination file
      (default is to raise an error if output file already exists).
--zlib=(0|1) -- Activate (or disable) zlib compression (default is activate).
--complevel=(1-9) -- Set zlib compression level (6 is default).
--shuffle=(0|1) -- Activate (or disable) the shuffle filter
                   (active by default).
--fletcher32=(0|1) -- Activate (or disable) the fletcher32 checksum
                      (not active by default).
--unpackshort=(0|1) -- Unpack short integer variables to float variables
                       using scale_factor and add_offset netCDF 
                       variable attributes (active by default).
--quantize=(comma separated list of "variable name=integer" pairs) --
  Truncate the data in the specified variables to a given decimal precision.
  For example, 'speed=2, height=-2, temp=0' will cause the variable
  'speed' to be truncated to a precision of 0.01, 
  'height' to a precision of 100 and 'temp' to 1.
  This can significantly improve compression. The default
  is not to quantize any of the variables.

If C{--zlib=1}, the resulting C{NETCDF4_CLASSIC} file will take up less
disk space than the original netCDF 3 file (especially if the
C{--quantize} option is used), and will be readable by netCDF 3 clients
as long as they have been linked against the netCDF 4 library.

9) Storing arrays of arbitrary python objects using the 'S' datatype
---------------------------------------------------------------------

Variables with datatype C{'S'} can be used to store variable-length
strings, or python objects.  Here's an example.

>>> strvar = rootgrp.createVariable('strvar','S',('level'))

Typically, a string variable is used to hold variable-length strings.
They are represented in python as numpy object arrays containing python
strings. Below an object array is filled with random python strings with
random lengths between 2 and 12 characters.

>>> import random
>>> chars = '1234567890aabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
>>> data = NP.empty(10,'O')
>>> for n in range(10):
>>>     stringlen = random.randint(2,12)
>>>     data[n] = ''.join([random.choice(chars) for i in range(stringlen)])

Now, we replace the first element of the object array with a python dictionary.

>>> data[0] = {'spam':1,'eggs':2,'ham':False}

When the data is assigned to the string variable, elements which are not
python strings are converted to strings using the python C{cPickle}
module.

>>> strvar[:] = data

When the data is read back in from the netCDF file, strings which are
determined to be pickled python objects are unpickled back into objects.

>>> print 'string variable with embedded python objects:\\n',strvar[:]
string variable with embedded python objects:
[{'eggs': 2, 'ham': False, 'spam': 1} QnXTY8B nbt4zisk pMHIn1F wl3suHW0OquZ
 wn5kxEzgE nk AGBL pe kay81]

Attributes can also be python objects, although the rules for whetherr
they are saved as pickled strings are different.  Attributes are
converted to numpy arrays before being saved to the netCDF file.  If the
attribute is cast to an object array by numpy, it is pickled and saved
as a text attribute (and then automatically unpickled when the attribute
is accessed).  So, an attribute which is a list of integers will be
saved as an array of integers, while an attribute that is a python
dictionary will be saved as a pickled string, then unpickled
automatically when it is retrieved. For example,

>>> from datetime import datetime
>>> strvar.timestamp = datetime.now()
>>> print strvar.timestamp
2006-02-11 13:26:27.238042

Note that data saved as pickled strings will not be very useful if the
data is to be read by a non-python client (the data will appear to the
client as an ugly looking binary string).

All of the code in this tutorial is available in examples/tutorial.py,
along with several other examples. Unit tests are in the test directory.

@contact: Jeffrey Whitaker <jeffrey.s.whitaker@noaa.gov>

@copyright: 2006 by Jeffrey Whitaker.

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

# numpy data type <--> netCDF 4 data type mapping.

_nptonctype  = {'S1' : NC_CHAR,
                'S'  : NC_STRING,
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
_supportedtypes = _nptonctype.keys()
_nctonptype = {}
for _key,_value in _nptonctype.iteritems():
    _nctonptype[_value] = _key

# utility functions (internal)

# pull in code from netCDF4_common.pyx.
include 'netCDF4_common.pyx'

# pure python utilities
from netCDF4_utils import _buildStartCountStride, _quantize, _find_dim

def _get_dims(group):
    """Private function to create L{Dimension} instances for all the
    dimensions in a L{Group} or Dataset"""
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

def _get_grps(group):
    """Private function to create L{Group} instances for all the
    groups in a L{Group} or Dataset"""
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

def _get_vars(group):
    """Private function to create L{Variable} instances for all the
    variables in a L{Group} or Dataset"""
    cdef int ierr, numvars, n, nn, numdims, varid, classp, nf, ndim
    cdef size_t sizein, nfields, offset
    cdef int *varids
    cdef int dim_sizes[NC_MAX_DIMS], dimids[NC_MAX_DIMS]
    cdef nc_type xtype, field_typeid
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
             # NC_LONG is the same as NC_INT anyway.
             if xtype == NC_LONG:
                 xtype == NC_INT
             try:
                 datatype = _nctonptype[xtype]
             except:
                 raise KeyError('unsupported data type')
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
             # use the 'foo' dimensoin from the parent?  
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

_private_atts = ['_grpid','_grp','_varid','groups','dimensions','variables','dtype','file_format', '_nunlimdim','path','parent']


cdef class Dataset:
    """
A netCDF L{Dataset} is a collection of dimensions, groups, variables and 
attributes. Together they describe the meaning of data and relations among 
data fields stored in a netCDF file.

Constructor: C{Dataset(filename, mode="r", clobber=True)}

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
C{NETCDF4_CLASSIC} or C{NETCDF3_64BIT}.  This module can read all
formats, but only writes C{NETCDF4}. To write files in the other
formats, use the L{netCDF4_classic} module.

@ivar path: The C{path} attribute shows the location of the L{Group} in
the L{Dataset} in a unix directory format (the names of groups in the
hierarchy separated by backslashes). A L{Dataset}, instance is the root
group, so the path is simply C{'/'}.

"""
    cdef public int _grpid
    cdef public groups, dimensions, variables, file_format, path, parent

    def __init__(self, filename, mode='r', clobber=True, **kwargs):
        cdef int grpid, ierr, numgrps, numdims, numvars
        cdef char *path
        cdef int *grpids, *dimids
        cdef char namstring[NC_MAX_NAME+1]
        path = filename
        if mode == 'w':
            _set_default_format(format='NETCDF4') # set default format to NETCDF4
            if clobber:
                ierr = nc__create(path, NC_CLOBBER, <size_t>0,
                                 <size_t *>NC_SIZEHINT_DEFAULT, &grpid)
            else:
                ierr = nc__create(path, NC_NOCLOBBER, <size_t>0,
                                  <size_t *>NC_SIZEHINT_DEFAULT, &grpid)
            # initialize group dict.
        elif mode == 'r':
            ierr = nc__open(path, NC_NOWRITE, <size_t *>NC_SIZEHINT_DEFAULT, &grpid)
        elif mode == 'r+' or mode == 'a':
            ierr = nc__open(path, NC_WRITE, <size_t *>NC_SIZEHINT_DEFAULT, &grpid)
        else:
            raise ValueError("mode must be 'w', 'r', 'a' or 'r+', got '%s'" % mode)
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))
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
Close the Dataset.

C{close()}"""
        cdef int ierr 
        ierr = nc_close(self._grpid)
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))

    def sync(self):
        """
Writes all buffered data in the L{Dataset} to the disk file.

C{sync()}""" 
        cdef int ierr
        ierr = nc_sync(self._grpid)
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))

    def set_fill_on(self):
        """
Sets the fill mode for a L{Dataset} open for writing to C{on}.

C{set_fill_on()}

This causes data to be pre-filled with fill values. The fill values can be 
controlled by the variable's C{_Fill_Value} attribute, but is usually 
sufficient to the use the netCDF default C{_Fill_Value} (defined 
separately for each variable type). The default behavior of the netCDF 
library correspongs to C{set_fill_on}.  Data which are equal to the 
C{_Fill_Value} indicate that the variable was created, but never written 
to.

"""
        cdef int ierr, oldmode
        ierr = nc_set_fill (self._grpid, NC_FILL, &oldmode)
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))

    def set_fill_off(self):
        """
Sets the fill mode for a L{Dataset} open for writing to C{off}. 

C{set_fill_off()}

This will prevent the data from being pre-filled with fill values, which 
may result in some performance improvements. However, you must then make 
sure the data is actually written before being read.

"""
        cdef int ierr, oldmode
        ierr = nc_set_fill (self._grpid, NC_NOFILL, &oldmode)
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))

    def createDimension(self, dimname, size=None):
        """
Creates a new dimension with the given C{dimname} and C{size}. 

C{createDimension(dimname, size=None)}

C{size} must be a positive integer or C{None}, which stands for 
"unlimited" (default is C{None}). The return value is the L{Dimension} 
class instance describing the new dimension.  To determine the current 
maximum size of the dimension, use the C{len} function on the L{Dimension} 
instance. To determine if a dimension is 'unlimited', use the 
C{isunlimited()} method of the L{Dimension} instance.

"""
        self.dimensions[dimname] = Dimension(self, dimname, size=size)

    def renameDimension(self, oldname, newname):
        """
rename a L{Dimension} named C{oldname} to C{newname}.

C{renameDimension(oldname, newname)}"""
        cdef char *namstring
        dim = self.dimensions[oldname]
        namstring = PyString_AsString(newname)
        ierr = nc_rename_dim(self._grpid, dim._dimid, namstring)
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))
        # remove old key from dimensions dict.
        self.dimensions.pop(oldname)
        # add new key.
        self.dimensions[newname] = dim
        # Variable.dimensions is determined by a method that
        # looks in the file, so no need to manually update.


    def createVariable(self, varname, datatype, dimensions=(), zlib=True, complevel=6, shuffle=True, fletcher32=False, chunking='seq', least_significant_digit=None, fill_value=None):
        """
Creates a new variable with the given C{varname}, C{datatype}, and 
C{dimensions}. If dimensions are not given, the variable is assumed to be 
a scalar.

C{createVariable(varname, datatype, dimensions=(), zlib=True, complevel=6, shuffle=True, fletcher32=False, chunking='seq', least_significant_digit=None, fill_value=None)}

The C{datatype} must be a string with the
same meaning as the C{dtype.str} attribute of arrays in module numpy.
Supported data types are: C{'S1' (NC_CHAR), 'i1' (NC_BYTE), 'u1'
(NC_UBYTE), 'i2' (NC_SHORT), 'u2' (NC_USHORT), 'i4' (NC_INT), 'u4'
(NC_UINT), 'i8' (NC_INT64), 'u8' (NC_UINT64), 'f4' (NC_FLOAT), 'f8'
(NC_DOUBLE)} and C{'S' (NC_STRING)}.

Data from netCDF variables of a primitive data type are presented to
python as numpy arrays with the corresponding data type, except for
variables with C{datatype=S} (variable-length strings).  Variables
containing variable-length strings are presented to python as numpy
object arrays. Numpy arrays of arbitrary python objects can be stored in
variables with C{datatype=S}, if the object is not a python string it is
converted to one using the python cPickle module. Pickle strings are
automatically converted back into python objects when they are read back
in from the netCDF file.

C{dimensions} must be a tuple containing dimension names (strings) that 
have been defined previously using C{createDimension}. The default value 
is an empty tuple, which means the variable is a scalar.

If the optional keyword C{zlib} is C{True}, the data will be compressed in 
the netCDF file using gzip compression (default C{True}).

The optional keyword C{complevel} is an integer between 1 and 9 describing 
the level of compression desired (default 6).

If the optional keyword C{shuffle} is C{True}, the HDF5 shuffle filter 
will be applied before compressing the data (default C{True}).  This 
significantly improves compression.

If the optional keyword C{fletcher32} is C{True}, the Fletcher32 HDF5 
checksum algorithm is activated to detect errors. Default C{False}.

If the optional keyword C{chunking} is C{'seq'} (Default) HDF5 chunk sizes 
are set to favor sequential access.  If C{chunking='sub'}, chunk sizes are 
set to favor subsetting equally in all dimensions.

The optional keyword C{fill_value} can be used to override the default 
netCDF C{_FillValue} (the value that the variable gets filled with before 
any data is written to it).  If fill_value is set to C{False}, then
the variable is not pre-filled.

If the optional keyword parameter C{least_significant_digit} is
specified, variable data will be truncated (quantized). This produces
'lossy', but significantly more efficient compression. For example, if
C{least_significant_digit=1}, data will be quantized using
C{numpy.around(scale*data)/scale}, where scale = 2**bits, and bits is
determined so that a precision of 0.1 is retained (in this case bits=4). 
From
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
operations on the L{Variable} instance. A L{Variable} instance has seven
standard attributes: C{dimensions, dtype, shape} and C{least_significant_digit},
Application programs should never modify these attributes. The C{dimensions}
attribute is a tuple containing the names of the dimensions associated
with this variable. The C{dtype} attribute is a string describing the
variable's data type (C{i4, f8, S1,} etc).
The C{shape} attribute
is a tuple describing the current sizes of all the variable's
dimensions. The C{least_significant_digit} attributes describes the
power of ten of the smallest decimal place in the data the contains a
reliable value.  Data is truncated to this decimal place when it is
assigned to the L{Variable} instance. If C{None}, the data is not
truncated."""

        self.variables[varname] = Variable(self, varname, datatype, dimensions=dimensions, zlib=zlib, complevel=complevel, shuffle=shuffle, fletcher32=fletcher32, chunking=chunking, least_significant_digit=least_significant_digit, fill_value=fill_value)
        return self.variables[varname]

    def renameVariable(self, oldname, newname):
        """
rename a L{Variable} named C{oldname} to C{newname}

C{renameVariable(oldname, newname)}"""
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
Creates a new L{Group} with the given C{groupname}.

C{createGroup(groupname)}

The return value is a L{Group} class instance describing the new group."""

        self.groups[groupname] = Group(self, groupname)
        return self.groups[groupname]
     
    def ncattrs(self):
        """
return netCDF global attribute names for this L{Dataset} or L{Group} in a list.

C{ncattrs()}""" 
        return _get_att_names(self._grpid, NC_GLOBAL)

    def __delattr__(self,name):
        cdef char *attname
        # if it's a netCDF attribute, remove it
        if name not in _private_atts:
            attname = PyString_AsString(name)
            ierr = nc_del_att(self._grpid, NC_GLOBAL, attname)
            if ierr != NC_NOERR:
                raise RuntimeError(nc_strerror(ierr))
        else:
            raise AttributeError, "'%s' is one of the reserved attributes %s, cannot delete" % (name, tuple(_private_atts))

    def __setattr__(self,name,value):
        # if name in _private_atts, it is stored at the python
        # level and not in the netCDF file.
        if name not in _private_atts:
            _set_att(self._grpid, NC_GLOBAL, name, value)
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
Groups define a hierarchical namespace within a netCDF file. They are 
analagous to directories in a unix filesystem. Each L{Group} behaves like 
a L{Dataset} within a Dataset, and can contain it's own variables, 
dimensions and attributes (and other Groups).

Constructor: C{Group(parent, name)} 

L{Group} instances should be created using the C{createGroup} method of a 
L{Dataset} instance, or another L{Group} instance, not using this class 
directly.

B{Parameters:}

B{C{parent}} - L{Group} instance for the parent group.  If being created
in the root group, use a L{Dataset} instance.

B{C{name}} - Name of the group.

B{Returns:}

a L{Group} instance.  All further operations on the netCDF
Group are accomplised via L{Group} instance methods.

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
overrides L{Dataset} close method which does not apply to L{Group} 
instances, raises IOError.

C{close()}"""
        raise IOError('cannot close a L{Group} (only applies to Dataset)')


cdef class Dimension:
    """
A netCDF L{Dimension} is used to describe the coordinates of a L{Variable}.

Constructor: C{Dimension(group, name, size=None)}

L{Dimension} instances should be created using the C{createDimension} 
method of a L{Group} or L{Dataset} instance, not using this class 
directly.

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
            ierr = nc_def_dim(self._grpid, dimname, lendim, &self._dimid)
            if ierr != NC_NOERR:
                raise RuntimeError(nc_strerror(ierr))

    def __len__(self):
        """
len(L{Dimension} instance) returns current size of dimension"""
        cdef int ierr
        cdef size_t lengthp
        ierr = nc_inq_dimlen(self._grpid, self._dimid, &lengthp)
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))
        return lengthp

    def isunlimited(self):
        """
returns C{True} if the L{Dimension} instance is unlimited, C{False} otherwise.

C{isunlimited()}"""
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
A netCDF L{Variable} is used to read and write netCDF data.  They are 
analagous to numpy array objects.

C{Variable(group, name, datatype, dimensions=(), zlib=True, complevel=6, 
shuffle=True, fletcher32=False, chunking='seq', 
least_significant_digit=None,fill_value=None)}
   
L{Variable} instances should be created using the C{createVariable} method 
of a L{Dataset} or L{Group} instance, not using this class directly.

B{Parameters:}

B{C{group}} - L{Group} or L{Dataset} instance to associate with variable.

B{C{name}}  - Name of the variable.

B{C{datatype}} - L{Variable} data type, one of
C{'f4'} (32-bit floating point),
C{'f8'} (64-bit floating point), C{'i4'} (32-bit signed
integer), C{'i2'} (16-bit signed integer), C{'i8'} (64-bit singed
integer), C{'i4'} (8-bit singed integer), C{'i1'} (8-bit signed
integer), C{'u1'} (8-bit unsigned integer), C{'u2'} (16-bit unsigned
integer), C{'u4'} (32-bit unsigned integer), C{'u8'} (64-bit unsigned
integer), C{'S1'} (single-character string), or C{'S'} (variable-length
string). 

B{Keywords:}

B{C{dimensions}} - a tuple containing the variable's dimension names 
(defined previously with C{createDimension}). Default is an empty tuple 
which means the variable is a scalar (and therefore has no dimensions).

B{C{zlib}} - if C{True} (default), data assigned to the L{Variable} 
instance is compressed on disk.

B{C{complevel}} - the level of zlib compression to use (1 is the fastest, 
but poorest compression, 9 is the slowest but best compression). Default 6.

B{C{shuffle}} - if C{True} (default), the HDF5 shuffle filter is applied 
to improve compression.

B{C{fletcher32}} - if C{True} (default C{False}), the Fletcher32 checksum 
algorithm is used for error detection.

B{C{chunking}} - Chunking is required in any dataset with one or more 
unlimited dimension in HDF5. NetCDF-4 supports setting the chunking 
algorithm at variable creation.  If C{chunking = 'seq'} (default) chunk 
sizes are set to favor sequential access. Setting C{chunking = 'sub'} will 
cause chunk sizes to be set to favor subsetting equally in any dimension.

B{C{least_significant_digit}} - If specified, variable data will be 
truncated (quantized). This produces 'lossy', but significantly more 
efficient compression. For example, if C{least_significant_digit=1}, data 
will be quantized using around(scale*data)/scale, where scale = 2**bits, 
and bits is determined so that a precision of 0.1 is retained (in this 
case bits=4). Default is C{None}, or no quantization.

B{C{fill_value}} - If specified, the default netCDF C{_FillValue} (the 
value that the variable gets filled with before any data is written to it) 
is replaced with this value.  If fill_value is set to C{False}, then
the variable is not pre-filled.
 
B{Returns:}

a L{Variable} instance.  All further operations on the netCDF Variable are 
accomplised via L{Variable} instance methods.

A list of attribute names corresponding to netCDF attributes defined for 
the variable can be obtained with the C{ncattrs()} method. These 
attributes can be created by assigning to an attribute of the L{Variable} 
instance. A dictionary containing all the netCDF attribute
name/value pairs is provided by the C{__dict__} attribute of a
L{Variable} instance.

The instance variables C{dimensions, dtype, shape}
and C{least_significant_digits} are read-only (and 
should not be modified by the user).

@ivar dimensions: A tuple containing the names of the dimensions 
associated with this variable.

@ivar dtype: A string describing the variable's data type, 
(C{'i4','f8','S1'}, etc).

@ivar shape: a tuple describing the current size of all the variable's 
dimensions.

@ivar least_significant_digit: Describes the power of ten of the smallest 
decimal place in the data the contains a reliable value.  Data is 
truncated to this decimal place when it is assigned to the L{Variable} 
instance. If C{None}, the data is not truncated. """

    cdef public int _varid, _grpid, _nunlimdim
    cdef object _grp
    cdef public dtype

    def __init__(self, grp, name, datatype, dimensions=(), zlib=True, complevel=6, shuffle=True, fletcher32=False, chunking='seq', least_significant_digit=None, fill_value=None,  **kwargs):
        cdef int ierr, ndims
        cdef char *varname
        cdef int *ichunkalgp
        cdef nc_type xtype, vltypeid
        cdef int dimids[NC_MAX_DIMS]
        self._grpid = grp._grpid
        self._grp = grp
        if datatype not in _supportedtypes:
            raise TypeError('illegal data type, must be one of %s, got %s' % (_supportedtypes,datatype))
        self.dtype = datatype
        if kwargs.has_key('id'):
            self._varid = kwargs['id']
        else:
            varname = name
            ndims = len(dimensions)
            # find netCDF primitive data type corresponding to 
            # specified numpy data type.
            xtype = _nptonctype[datatype]
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
            # define variable.
            if ndims:
                ierr = nc_def_var(self._grpid, varname, xtype, ndims,
                                  dimids, &self._varid)
            else: # a scalar variable.
                ierr = nc_def_var(self._grpid, varname, xtype, ndims,
                                  NULL, &self._varid)
            if ierr != NC_NOERR:
                raise RuntimeError(nc_strerror(ierr))
            # set zlib, shuffle, chunking and fletcher32 variable settings.
            # don't bother for scalar variables.
            if ndims:
                if zlib:
                    ideflate_level = complevel
                    if shuffle:
                        ierr = nc_def_var_deflate(self._grpid, self._varid, 1, 1, ideflate_level)
                    else:
                        ierr = nc_def_var_deflate(self._grpid, self._varid, 1, 0, ideflate_level)
                    if ierr != NC_NOERR:
                        raise RuntimeError(nc_strerror(ierr))
                if fletcher32:
                    ierr = nc_def_var_fletcher32(self._grpid, self._varid, ifletcher32)
                    if ierr != NC_NOERR:
                        raise RuntimeError(nc_strerror(ierr))
                if chunking == 'sub':
                    ichunkalgp = <int *>NC_CHUNK_SUB
                elif chunking == 'seq':
                    ichunkalgp = <int *>NC_CHUNK_SEQ
                else:
                    raise ValueError("chunking keyword must be 'seq' or 'sub', got %s" % chunking)
                ierr = nc_def_var_chunking(self._grpid, self._varid, ichunkalgp, NULL, NULL)
                if ierr != NC_NOERR:
                    raise RuntimeError(nc_strerror(ierr))
            # set a fill value for this variable if fill_value keyword
            # given.  This avoids the HDF5 overhead of deleting and 
            # recreating the dataset if it is set later (after the enddef).
            if fill_value is not None:
                if not fill_value and isinstance(fill_value,bool):
                    # no filling for this variable if fill_value==False.
                    ierr = nc_def_var_fill(self._grpid, self._varid, 1, NULL)
                    if ierr != NC_NOERR:
                        raise RuntimeError(nc_strerror(ierr))
                else:
                    # cast fill_value to type of variable.
                    fillval = NP.array(fill_value, self.dtype)
                    _set_att(self._grpid, self._varid, '_FillValue', fillval)
                    #ierr = nc_def_var_fill(self._grpid, self._varid, 0, fillval.data)
                    #if ierr != NC_NOERR:
                    #    raise RuntimeError(nc_strerror(ierr))
            if least_significant_digit is not None:
                self.least_significant_digit = least_significant_digit
        # count how many unlimited dimensions there are.
        self._nunlimdim = 0
        for dimname in self.dimensions:
            # look in current group, and parents for dim.
            dim = _find_dim(self._grp, dimname)
            if dim.isunlimited(): self._nunlimdim = self._nunlimdim + 1

    def _getDimensions(self):
        """Private method to get variables's dimension names"""
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

    def _shape(self):
        """Private method to find current sizes of all variable dimensions"""
        varshape = ()
        for dimname in self.dimensions:
            # look in current group, and parents for dim.
            dim = _find_dim(self._grp,dimname)
            varshape = varshape + (len(dim),)
        return varshape

    def group(self):
        """
return the group that this L{Variable} is a member of.

C{group()}"""
        return self._grp

    def ncattrs(self):
        """
return netCDF attribute names for this L{Variable} in a list

C{ncattrs()}"""

        return _get_att_names(self._grpid, self._varid)

    def __delattr__(self,name):
        cdef char *attname
        # if it's a netCDF attribute, remove it
        if name not in _private_atts:
            attname = PyString_AsString(name)
            ierr = nc_del_att(self._grpid, self._varid, attname)
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
            if self.dtype != 'S' and name == '_FillValue':
                value = NP.array(value, self.dtype)
            _set_att(self._grpid, self._varid, name, value)
        elif not name.endswith('__'):
            if hasattr(self,name):
                raise AttributeError("'%s' is one of the reserved attributes %s, cannot rebind" % (name, tuple(_private_atts)))
            else:
                self.__dict__[name]=value

    def __getattr__(self,name):
        # special treatment for 'shape' - pass to _shape method.
        if name == 'shape': return self._shape()
        # special treatment for 'dimensions' - pass to _getDimensions method.
        if name == 'dimensions': return self._getDimensions()
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
        start, count, stride = _buildStartCountStride(elem,self.shape,self.dimensions,self._grp)
        # Get elements.
        if self.dtype != 'S':
            return self._get(start, count, stride)
        else:
            return self._get_string(start, count, stride)
 
    def __setitem__(self, elem, data):
        # This special method is used to assign to the netCDF variable
        # using "extended slice syntax". The extended slice syntax
        # is a perfect match for the "start", "count" and "stride"
        # arguments to the nc_put_var() function, and is much more easy
        # to use.
        start, count, stride = _buildStartCountStride(elem,self.shape,self.dimensions,self._grp)
        # quantize data if least_significant_digit attribute set.
        if 'least_significant_digit' in self.ncattrs():
            data = _quantize(data,self.least_significant_digit)
        # A numpy array is needed. Convert if necessary.
        if not type(data) == NP.ndarray:
            if self.dtype == 'S': # if a string, convert to a scalar object arr
                data = NP.array(data, 'O')
            # otherwise, let numpy to the casting
            else:
                data = NP.array(data,self.dtype)
        # append the data to the variable object.
        if self.dtype != 'S':
            self._put(data, start, count, stride)
        else:
            self._put_string(data, start, count, stride)

    def assignValue(self,val):
        """
assign a value to a scalar variable.  Provided for compatibility with 
Scientific.IO.NetCDF, can also be done by assigning to a slice ([:]).

C{assignValue(val)}"""
        if len(self.dimensions):
            raise IndexError('to assign values to a non-scalar variable, use a slice')
        self[:]=val

    def getValue(self):
        """
get the value of a scalar variable.  Provided for compatibility with 
Scientific.IO.NetCDF, can also be done by slicing ([:]).

C{getValue()}"""
        if len(self.dimensions):
            raise IndexError('to retrieve values from a non-scalar variable, use slicing')
        return self[:]

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
                datanew = NP.empty(totelem,self.dtype)
                datanew[:] = data
                data = datanew
            else:
                raise IndexError('size of data array does not conform to slice')
        # if data type of array doesn't match variable, try to 
        # cast the data.
        if self.dtype != data.dtype.str[1:]:
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

    def _put_string(self,ndarray data,start,count,stride):
        """Private method to put data into a netCDF variable with dtype='S'"""
        cdef int ierr, ndims, totelem, n, buflen
        cdef size_t startp[NC_MAX_DIMS], countp[NC_MAX_DIMS]
        cdef ptrdiff_t stridep[NC_MAX_DIMS]
        cdef char **strdata
        cdef void *bufdat
        cdef char *strbuf
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
                datanew = NP.empty(totelem,'O')
                datanew[:] = data
                data = datanew
            else:
                raise IndexError('size of data array does not conform to slice')
        if data.dtype.char !='O':
            raise TypeError('data to put in string variable must be an object array containing Python strings')
        if negstride:
            # reverse data along axes with negative strides.
            data = data[sl].copy() # make sure a copy is made.
        # flatten data array.
        data = data.flatten()
        # loop over elements of object array, put data buffer for
        # each element in struct.
        strdata = <char **>malloc(sizeof(char *)*totelem)
        for i from 0<=i<totelem:
            pystring = data[i]
            if PyString_Check(pystring) != 1:
                # if not a python string, pickle it into a string
                # (use protocol 2)
                pystring = cPickle.dumps(pystring,2)
            strdata[i] = PyString_AsString(pystring)
        # strides all 1 or scalar variable, use put_vara (faster)
        if sum(stride) == ndims or ndims == 0: 
            ierr = nc_put_vara(self._grpid, self._varid,
                               startp, countp, strdata)
        else:  
            raise IndexError('strides must all be 1 for string variables')
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))
        free(strdata)

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
        data = NP.empty(shapeout, self.dtype)
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
        elif squeeze_out:
            return NP.squeeze(data)
        else:
            return data

    def _get_string(self,start,count,stride):
        """Private method to retrieve data from a netCDF variable with dtype='S'"""
        cdef int i,ierr, ndims, totelem, arrlen
        cdef size_t startp[NC_MAX_DIMS], countp[NC_MAX_DIMS]
        cdef ptrdiff_t stridep[NC_MAX_DIMS]
        cdef ndarray data
        cdef void *elptr
        cdef char *strbuf
        cdef char **strdata
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
        # allocate array of correct primitive type.
        data = NP.empty(shapeout, 'O')
        # flatten data array.
        data = data.flatten()
        totelem = PyArray_SIZE(data)
        # allocate pointer array to hold string data.
        strdata = <char **>malloc(sizeof(char *) * totelem)
        # strides all 1 or scalar variable, use get_vara (faster)
        if sum(stride) == ndims or ndims == 0: 
            ierr = nc_get_vara(self._grpid, self._varid,
                               startp, countp, strdata)
        else:
            raise IndexError('strides must all be 1 for string variables')
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))
        # loop over elements of object array, fill array with
        # contents of strdata.
        for i from 0<=i<totelem:
            data[i] = PyString_FromString(strdata[i])
            # if it's a pickle string, unpickle it.
            # (see if first element is the pickle protocol 2
            # identifier - '\x80')
            if data[i][0] == '\x80': # use pickle.PROTO instead?
                data[i] = cPickle.loads(data[i])
        # reshape the output array
        data = NP.reshape(data, shapeout)
        if negstride:
            # reverse data along axes with negative strides.
            data = data[sl].copy() # make a copy so data is contiguous.
        free(strdata)
        if not self.dimensions: 
            return data[0] # a scalar 
        elif squeeze_out:
            return NP.squeeze(data)
        else:
            return data
