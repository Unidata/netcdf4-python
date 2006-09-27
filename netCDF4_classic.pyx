"""
Introduction
============

Python interface to the netCDF version 4 library that maintains backward
compatibility with netCDF version 3 clients.  U{netCDF version 4
<http://www.unidata.ucar.edu/software/netcdf/netcdf-4>} has many
features not found in earlier versions of the library and is implemented
on top of U{HDF5 <http://hdf.ncsa.uiuc.edu/HDF5>}. This module does not
implement any of the new features of netCDF 4, except zlib compression.
To use the other new features of netCDF 4, use the companion L{netCDF4}
module (which produces netCDF 4 files that can only be read by netCDF 4
clients).  The API modelled after U{Scientific.IO.NetCDF
<http://starship.python.net/~hinsen/ScientificPython>}, and should be
familiar to users of that module.

Download 
========

 - U{Project page <http://code.google.com/p/netcdf4-python/>}.
 - U{Subversion repository <http://code.google.com/p/netcdf4-python/source>}.
 - U{Source tar.gz <http://cheeseshop.python.org/pypi/netCDF4/>}.

Requires 
======== 

 - numpy array module U{http://numpy.scipy.org}.
 - netCDF version 4.  netCDF4 is now in alpha,
 and is a bit of a moving target.  This release is has only
 been tested with netcdf-4.0-alpha16, available from
 U{ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4}.
 - A pre-release version of HDF5 1.8 is required.  Version
 1.7.52 is required for netcdf-4.0-alpha16. It is also 
 available at U{ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4}.


Install
=======

 - set the HDF5_DIR environment variable to point to where HDF5 is installed.
 (the libs in $HDF5_DIR/lib, the headers in $HDF5_DIR/include).
 - set the NETCDF4_DIR environment variable to point to where netCDF version
 4 is installed.
 - run 'python setup.py install'
 - run some of the tests in the 'test_classic' directory.

Tutorial
========

1) Creating/Opening/Closing a netCDF file
-----------------------------------------

To create a netCDF file from python, you simply call the L{Dataset} 
constructor. This is also the method used to open an existing netCDF file.  
If the file is open for write access (C{w, r+} or C{a}), you may write any 
type of data including new dimensions, variables and attributes.  netCDF 
files come in several flavors (C{NETCDF3_CLASSIC,NETCDF3_64BIT, 
NETCDF4_CLASSIC}, and C{NETCDF4}). The first two flavors are supported by 
version 3 of the netCDF library. C{NETCDF4_CLASSIC} files use the version 
4 disk format (HDF5), but do not use any features not found in the version 
3 API, except zlib compression. They can be read by netCDF 3 clients only 
if they have been relinked against the netCDF 4 library. They can also be 
read by HDF5 clients, using the HDF5 API. C{NETCDF4} files use the HDF5 
file format and use the new features of the netCDF4 version 4 API, and 
thus cannot be read by netCDF 3 clients.  The netCDF4_classic module can 
read and write C{NETCDF3_CLASSIC}, C{NETCDF3_64BIT} and C{NETCDF4_CLASSIC} 
files. To write C{NETCDF4} files, use the L{netCDF4} module. To see what 
how a given file is formatted, you can examine the C{file_format} 
L{Dataset} attribute. Closing the netCDF file is accomplished via the 
C{close} method of the L{Dataset} instance.


Here's an example:

>>> import netCDF4_classic as netCDF
>>> dataset = netCDF.Dataset('test.nc', 'w')
>>> print dataset.file_format
NETCDF4_CLASSIC
>>>
>>> dataset.close()
	    

2) Dimensions in a netCDF file
------------------------------

netCDF defines the sizes of all variables in terms of dimensions, so
before any variables can be created the dimensions they use must be
created first. A special case, not often used in practice, is that of a
scalar variable, which has no dimensions. A dimension is created using
the C{createDimension} method of a L{Dataset} instance. A Python string
is used to set the name of the dimension, and an integer value is used
to set the size. To create an unlimited dimension (a dimension that can
be appended to), the size value is set to C{None}. In this example, the
C{time} is unlimited.  Only one unlimited dimension per file is allowed
in netCDF 3, and it must be the first (or leftmost) dimension. 
C{NETCDF4} formatted files may have multiple unlimited dimensions (see
the L{netCDF4} documentation).

>>> dataset = netCDF.Dataset('test.nc', 'a')
>>> dataset.createDimension('time', None)
>>> dataset.createDimension('level', 10)
>>> dataset.createDimension('lat', 73)
>>> dataset.createDimension('lon', 144)
	    

All of the L{Dimension} instances are stored in a python dictionary.

>>> print dataset.dimensions
{'lat': <netCDF4_classic.Dimension object at 0x24a5f7b0>, 
 'time': <netCDF4_classic.Dimension object at 0x24a5f788>, 
 'lon': <netCDF4_classic.Dimension object at 0x24a5f7d8>, 
 'level': <netCDF4_classic.Dimension object at 0x24a5f760>}
>>>

Calling the python C{len} function with a L{Dimension} instance returns
the current size of that dimension. The C{isunlimited()} method of a
L{Dimension} instance can be used to determine if the dimensions is
unlimited, or appendable.

>>> for dimname, dimobj in dataset.dimensions.iteritems():
>>>    print dimname, len(dimobj), dimobj.isunlimited()
lat 73 False
time 0 True
lon 144 False
level 0 False
>>>
	    
L{Dimension} names can be changed using the C{renameDimension} method of
a L{Dataset} instance.

3) Variables in a netCDF file
-----------------------------

netCDF variables behave much like python multidimensional array objects supplied by 
the U{numpy module <http://numpy.scipy.org>}. However, unlike numpy arrays, netCDF 
variables can be appended to along the 'unlimited' dimension. To create a netCDF 
variable, use the C{createVariable} method of a L{Dataset} instance. The 
C{createVariable} method has two mandatory arguments, the variable name (a Python 
string), and the variable datatype. The variable's dimensions are given by a tuple 
containing the dimension names (defined previously with C{createDimension}). To 
create a scalar variable, simply leave out the dimensions keyword. The variable 
primitive datatypes correspond to the dtype.str attribute of a numpy array, and can 
be one of C{'f4'} (32-bit floating point), C{'f8'} (64-bit floating point), C{'i4'} 
(32-bit signed integer), C{'i2'} (16-bit signed integer), C{'i1'} (8-bit signed 
integer), integer), C{'S1'} (single-character string).  The old single character 
Numeric typecodes (C{'f','d','i','h','b','c'}) are also accepted for compatibility 
with Scientific.IO.NetCDF. The dimensions themselves are usually also defined as 
variables, called coordinate variables. The C{createVariable} method returns an 
instance of the L{Variable} class whose methods can be used later to access and set 
variable data and attributes.

>>> times = dataset.createVariable('time','f8',('time',))
>>> levels = dataset.createVariable('level','i4',('level',))
>>> latitudes = dataset.createVariable('latitude','f4',('lat',))
>>> longitudes = dataset.createVariable('longitude','f4',('lon',))
>>> temp = dataset.createVariable('temp','f4',('time','level','lat','lon',))

All of the variables in the file are stored in a Python dictionary, in
the same way as the dimensions:

>>> print dataset.variables
{'temp': <netCDF4_classic.Variable object at 0x24a61068>,
 'level': <netCDF4_classic.Variable object at 0.3f0f80>, 
 'longitude': <netCDF4_classic.Variable object at 0x24a61030>,
 'pressure': <netCDF4_classic.Variable object at 0x24a610a0>, 
 'time': <netCDF4_classic.Variable object at 0.3f0.4.58>, 
 'latitude': <netCDF4_classic.Variable object at 0.3f0fb8>}
>>>

L{Variable} names can be changed using the C{renameVariable} method of a
L{Dataset} instance.


4) Attributes in a netCDF file
------------------------------

There are two types of attributes in a netCDF file, global and variable. 
Global attributes provide information about an entire dataset as a
whole. L{Variable} attributes provide information about one of the
variables in a group. Global attributes are set by assigning values to
L{Dataset} instance variables. L{Variable} attributes are set by
assigning values to L{Variable} instances variables. Attributes can be
strings, numbers or sequences. Returning to our example,


>>> import time
>>> dataset.description = 'bogus example script'
>>> dataset.history = 'Created ' + time.ctime(time.time())
>>> dataset.source = 'netCDF4 python module tutorial'
>>> latitudes.units = 'degrees north'
>>> longitudes.units = 'degrees east'
>>> pressure.units = 'hPa'
>>> temp.units = 'K'
>>> times.units = 'days since January 1, 0001'
>>> times.calendar = 'proleptic_gregorian'

The C{ncattrs()} method of a L{Dataset} or L{Variable} instance can be
used to retrieve the names of all the netCDF attributes. This method is
provided as a convenience, since using the built-in C{dir} Python
function will return a bunch of private methods and attributes that
cannot (or should not) be modified by the user.

>>> for name in dataset.ncattrs():
>>>     print 'Global attr', name, '=', getattr(dataset,name)
Global attr description = bogus example script
Global attr history = Created Mon Nov  7 10.30:56 2005
Global attr source = netCDF4_classic python module tutorial

The C{__dict__} attribute of a L{Dataset} or L{Variable} instance provides 
all the netCDF attribute name/value pairs in a python dictionary:

>>> print dataset.__dict__
{'source': 'netCDF4_classic python module tutorial',
'description': 'bogus example script', 
'history': 'Created Mon Nov  7 10.30:56 2005'}

Attributes can also be python objects. netCDF4_classic tries to convert
attributes to numpy arrays before saving them to the netCDF file.  If
the attribute is cast to an object array by numpy, it is pickled and
saved as a text attribute (and then automatically unpickled when the
attribute is accessed).  So, an attribute which is a list of integers
will be saved as an array of integers, while an attribute that is a
python dictionary will be saved as a pickled string, then unpickled
automatically when it is retrieved. For example,

>>> from datetime import datetime
>>> dataset.timestamp = datetime.now()
>>> print 'Global attr timestamp =',dataset.timestamp
Global attr timestamp = 2006-03-06 09:20:21.520926

Note that data saved as pickled strings will not be very useful if the
data is to be read by a non-python client (the data will appear to the
client as an ugly looking binary string).

Attributes can be deleted from a netCDF L{Dataset} or L{Variable} using
the python C{del} statement (i.e. C{del dset.foo} removes the attribute
C{foo} the the dataset C{dset}).

5) Writing data to and retrieving data from a netCDF variable
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

Unlike numpy array objects, netCDF L{Variable} objects with an unlimited
dimension will grow along that dimension if you assign data outside the
currently defined range of indices.

>>> # append along two unlimited dimensions by assigning to slice.
>>> nlats = len(dataset.dimensions['lat'])
>>> nlons = len(dataset.dimensions['lon'])
>>> nlevs = len(dataset.dimensions['level'])
>>> print 'temp shape before adding data = ',temp.shape
temp shape before adding data =  (0, 10, 73, 144)
>>>
>>> from numpy.random.mtrand import uniform
>>> temp[0:5,:,:,:] = uniform(size=(5,10,nlats,nlons))
>>> print 'temp shape after adding data = ',temp.shape
temp shape after adding data =  (5, 10, 73, 144)
>>>
>>> # times have grown, but no values yet assigned.
>>> print 'times shape after adding pressure data = ',times.shape
times shape after adding pressure data =  (5,)
>>>

Note that the size of the times variable grows when data is appended
along the C{time} dimension of the variable C{temp}, even though no data
has yet been assigned to the variable C{times}.

Time coordinate values pose a special challenge to netCDF users.  Most
metadata standards (such as CF and COARDS) specify that time should be
measure relative to a fixed date using a certain calendar, with units
specified like C{hours since YY:MM:DD hh-mm-ss}.  These units can be
awkward to deal with, without a utility to convert the values to and
from calendar dates.  A module called L{netcdftime.netcdftime} is
provided with this package to do just that.  Here's an example of how it
can be used:

>>> # fill in times.
>>> from datetime import timedelta
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
	    
6) Efficient compression of netCDF variables
--------------------------------------------

Data stored in netCDF L{Variable} objects is compressed on disk by
default, if the file format is C{NETCDF4_CLASSIC}. This a new feature of
netCDF 4, but the resulting files can still be read by netCDF 3 clients
that have been linked against the netCDF 4 library. The parameters for
the compression are determined by the C{zlib} and C{complevel} and
C{shuffle} keyword arguments to the C{createVariable} method.  The
default values are C{zlib=True}, C{complevel=6} and C{shuffle=True}.  To
turn off compression, set C{zlib=False}.  C{complevel} regulates the
speed and efficiency of the compression (1 being fastest, but lowest
compression ratio, 9 being slowest but best compression ratio). 
C{shuffle=False} will turn off the HDF5 shuffle filter, which
de-interlaces a block of data by reordering the bytes.  The shuffle
filter can significantly improve compression ratios.  Setting
C{fletcher32} keyword argument to C{createVariable} to C{True} (it's
C{False} by default) enables the Fletcher32 checksum algorithm for error
detection.

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

>>> temp = dataset.createVariable('temp','f4',('time','level','lat','lon',))

with

>>> temp = dataset.createVariable('temp','f4',('time','level','lat','lon',),
                                  least_significant_digit=3)
	    

and see how much smaller the resulting file is.  If the file format is
not C{NETCDF4_CLASSIC}, using the least_significant_digit keyword will
not result in a smaller file, since on-the-fly zlib compression will not
be done.  However, the resulting file will still be smaller when
gzipped.

7) Converting netCDF 3 files to netCDF 4 files (with compression)
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

All of the code in this tutorial is available in
examples_classic/tutorial.py, along with several other examples. Unit
tests are in the test_classic directory.

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

# numpy data type <--> netCDF 3 data type mapping.

_nptonctype  = {'S1' : NC_CHAR,
                'i1' : NC_BYTE,
                'i2' : NC_SHORT,
                'i4' : NC_INT,   
                'f4' : NC_FLOAT,
                'f8' : NC_DOUBLE}
_supportedtypes = _nptonctype.keys()
_nctonptype = {}
for _key,_value in _nptonctype.iteritems():
    _nctonptype[_value] = _key

# also allow old Numeric single character typecodes.
_nptonctype['d'] = NC_DOUBLE
_nptonctype['f'] = NC_FLOAT
_nptonctype['i'] = NC_INT
_nptonctype['l'] = NC_INT
_nptonctype['s'] = NC_SHORT
_nptonctype['h'] = NC_SHORT
_nptonctype['c'] = NC_CHAR
_nptonctype['S'] = NC_CHAR
_nptonctype['b'] = NC_BYTE
_nptonctype['B'] = NC_BYTE

# Python wrappers.

# utility functions (internal)

# pull in code from netCDF4_common.pyx.
include 'netCDF4_common.pyx'

def _get_dims(dataset):
    """Private function to create Dimension instances for all the
    dimensions in a Dataset"""
    cdef int ierr, numdims, n
    cdef char namstring[NC_MAX_NAME+1]
    # get number of dimensions in this Group.
    ierr = nc_inq_ndims(dataset._dsetid, &numdims)
    if ierr != NC_NOERR:
        raise RuntimeError(nc_strerror(ierr))
    # create empty dictionary for dimensions.
    dimensions = {}
    if numdims > 0:
        for n from 0 <= n < numdims:
             ierr = nc_inq_dimname(dataset._dsetid, n, namstring)
             if ierr != NC_NOERR:
                 raise RuntimeError(nc_strerror(ierr))
             name = namstring
             dimensions[name] = Dimension(dataset, name, id=n)
    return dimensions

def _get_vars(dataset):
    """Private function to create Variable instances for all the
    variables in a Dataset"""
    cdef int ierr, numvars, n, nn, numdims
    cdef int dimids[NC_MAX_DIMS]
    cdef nc_type xtype
    cdef char namstring[NC_MAX_NAME+1]
    # get number of variables in this Group.
    ierr = nc_inq_nvars(dataset._dsetid, &numvars)
    if ierr != NC_NOERR:
        raise RuntimeError(nc_strerror(ierr))
    # create empty dictionary for variables.
    variables = {}
    if numvars > 0:
        # loop over variables. 
        for n from 0 <= n < numvars:
             # get variable name.
             ierr = nc_inq_varname(dataset._dsetid, n, namstring)
             if ierr != NC_NOERR:
                 raise RuntimeError(nc_strerror(ierr))
             name = namstring
             # get variable type.
             ierr = nc_inq_vartype(dataset._dsetid, n, &xtype)
             if ierr != NC_NOERR:
                 raise RuntimeError(nc_strerror(ierr))
             # NC_LONG is the same as NC_INT anyway.
             if xtype == NC_LONG:
                 xtype == NC_INT
             try:
                 type = _nctonptype[xtype]
             except:
                 raise KeyError('unsupported data type')
             # get number of dimensions.
             ierr = nc_inq_varndims(dataset._dsetid, n, &numdims)
             if ierr != NC_NOERR:
                 raise RuntimeError(nc_strerror(ierr))
             # get dimension ids.
             ierr = nc_inq_vardimid(dataset._dsetid, n, dimids)
             if ierr != NC_NOERR:
                 raise RuntimeError(nc_strerror(ierr))
             # loop over dimensions, retrieve names.
             dimensions = []
             for nn from 0 <= nn < numdims:
                 for key, value in dataset.dimensions.iteritems():
                     if value._dimid == dimids[nn]:
                         dimensions.append(key)
             # create new variable instance.
             variables[name] = Variable(dataset, name, type, dimensions, id=n)
    return variables

# these are class attributes that 
# only exist at the python level (not in the netCDF file).

_private_atts = ['_dsetid','_dset','_varid','dimensions','variables','dtype','file_format']

cdef class Dataset:
    """
A netCDF L{Dataset} is a collection of dimensions, variables and
attributes. Together they describe the meaning of data and relations
among data fields stored in a netCDF file.

Constructor: C{Dataset(filename, mode="r", clobber=True, format='NETCDF4_CLASSIC')}

B{Parameters:}

B{C{filename}} - Name of netCDF file to hold dataset.

B{Keywords}:

B{C{mode}} - access mode. C{r} means read-only; no data can be modified.
C{w} means write; a new file is created, an existing file with the same
name is deleted. C{a} and C{r+} mean append (in analogy with serial
files); an existing file is opened for reading and writing.

B{C{clobber}} - if C{True} (default), opening a file with C{mode='w'}
will clobber an existing file with the same name.  if C{False}, an
exception will be raised if a file with the same name already exists.

B{C{format}} - underlying file format (one of C{'NETCDF4_CLASSIC',
'NETCDF3_CLASSIC'} or C{'NETCDF3_64BIT'}.  Only relevant if C{mode =
'w'} (if C{mode = 'r','a'} or C{'r+'} the file format is automatically
detected). Default C{'NETCDF4_CLASSIC'}, which means the data is stored
in an HDF5 file, but using only netCDF 3 compatibile API features.
netCDF 3 clients must be recompiled and linked against the netCDF 4
library to read files in C{NETCDF4_CLASSIC} format.  The advantage is
that the files are also readable by HDF5 clients, and you get to use
on-the-fly zlib compression (which makes for much smaller files). 
C{'NETCDF3_CLASSIC'} is the classic netCDF 3 file format that does not
handle 2+ Gb files very well. C{'NETCDF3_64BIT'} is the 64-bit offset
version of the netCDF 3 file format, which fully supports 2+ GB files,
but is only compatible with clients linked against netCDF version 3.6.0
or later.

B{Returns:}

a L{Dataset} instance.  All further operations on the netCDF Dataset are
accomplised via L{Dataset} instance methods.

A list of attribute names corresponding to global netCDF attributes
defined for the L{Dataset} can be obtained with the L{ncattrs()} method. 
These attributes can be created by assigning to an attribute of the
L{Dataset} instance. A dictionary containing all the netCDF attribute
name/value pairs is provided by the C{__dict__} attribute of a
L{Dataset} instance.

The instance variables C{dimensions, variables} amd C{file_format} are
read-only (and should not be modified by the user).

@ivar dimensions: The C{dimensions} dictionary maps the names of
dimensions defined for this L{Dataset} to instances of the L{Dimension}
class.

@ivar variables: The C{variables} dictionary maps the names of variables
defined for this L{Dataset} to instances of the L{Variable} class.

@ivar file_format: The C{file_format} attribute describes the netCDF
file format version, one of C{NETCDF3_CLASSIC}, C{NETCDF4_CLASSIC} or
C{NETCDF3_64BIT}.  To read or write files in C{NETCDFr} format, use the
L{netCDF4} module.

"""
    cdef public int _dsetid
    cdef public dimensions, variables, file_format

    def __init__(self, filename, mode='r', clobber=True, format='NETCDF4_CLASSIC', **kwargs):
        cdef int dsetid, ierr, numdsets, numdims, numvars
        cdef char *path
        cdef int dimids[NC_MAX_DIMS]
        cdef char namstring[NC_MAX_NAME+1]
        path = filename
        if mode == 'w':
            _set_default_format(format=format)
            if clobber:
                ierr = nc__create(path, NC_CLOBBER, <size_t>0,
                                 <size_t *>NC_SIZEHINT_DEFAULT, &dsetid)
            else:
                ierr = nc__create(path, NC_NOCLOBBER, <size_t>0,
                                  <size_t *>NC_SIZEHINT_DEFAULT, &dsetid)
        elif mode == 'r':
            ierr = nc__open(path, NC_NOWRITE, <size_t *>NC_SIZEHINT_DEFAULT, &dsetid)
        elif mode == 'r+' or mode == 'a':
            ierr = nc__open(path, NC_WRITE, <size_t *>NC_SIZEHINT_DEFAULT, &dsetid)
        else:
            raise ValueError("mode must be 'w', 'r', 'a' or 'r+', got '%s'" % mode)
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))
        # file format attribute.
        self.file_format = _get_format(dsetid)
        if self.file_format == 'NETCDF4':
            raise IOError('use the netCDF4 module to read/write NETCDF4 format data files (files that use new features of the netCDF 4 API)')
        self._dsetid = dsetid
        # get dimensions in the Dataset.
        self.dimensions = _get_dims(self)
        # get variables in the Dataset.
        self.variables = _get_vars(self)

    def close(self):
        """
Close the Dataset.

C{close()}"""
        cdef int ierr 
        ierr = nc_close(self._dsetid)
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))

    def sync(self):
        """
Writes all buffered data in the L{Dataset} to the disk file.

C{sync()}""" 
        cdef int ierr
        ierr = nc_sync(self._dsetid)
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))

    def _redef(self):
        cdef int ierr
        ierr = nc_redef(self._dsetid)

    def _enddef(self):
        cdef int ierr
        ierr = nc_enddef(self._dsetid)

    def set_fill_on(self):
        """
Sets the fill mode for a L{Dataset} open for writing to C{on}.

C{set_fill_on()}

This causes data to be pre-filled with fill values. The fill values can
be controlled by the variable's C{_Fill_Value} attribute, but is usually
sufficient to the use the netCDF default C{_Fill_Value} (defined
separately for each variable type). The default behavior of the netCDF
library correspongs to C{set_fill_on}.  Data which are equal to the
C{_Fill_Value} indicate that the variable was created, but never written
to.

"""

        cdef int ierr, oldmode
        ierr = nc_set_fill (self._dsetid, NC_FILL, &oldmode)
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
        ierr = nc_set_fill (self._dsetid, NC_NOFILL, &oldmode)
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))

    def createDimension(self, dimname, size=None):
        """
Creates a new dimension with the given C{dimname} and C{size}. 

C{createDimension(dimname, size=None)}

C{size} must be a positive integer or C{None}, which stands for
"unlimited" (default is C{None}). The return value is the L{Dimension}
class instance describing the new dimension.  To determine the current
maximum size of the dimension, use the C{len} function on the
L{Dimension} instance. To determine if a dimension is 'unlimited', use
the C{isunlimited()} method of the L{Dimension} instance.

"""
        self.dimensions[dimname] = Dimension(self, dimname, size=size)

    def renameDimension(self, oldname, newname):
        """
rename a L{Dimension} named C{oldname} to C{newname}.

C{renameDimension(oldname, newname)}"""
        cdef char *namstring
        dim = self.dimensions[oldname]
        namstring = PyString_AsString(newname)
        ierr = nc_rename_dim(self._dsetid, dim._dimid, namstring)
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))
        # remove old key from dimensions dict.
        self.dimensions.pop(oldname)
        # add new key.
        self.dimensions[newname] = dim
        # Variable.dimensions is determined by a method that
        # looks in the file, so no need to manually update.

    def createVariable(self, varname, datatype, dimensions=(), zlib=True, complevel=6, shuffle=True, fletcher32=False, least_significant_digit=None, fill_value=None, chunking='seq'):
        """
Creates a new variable with the given C{varname}, C{datatype}, and
C{dimensions}. If dimensions are not given, the variable is assumed to
be a scalar.

C{createVariable(varname, datatype, dimensions=(), zlib=True, complevel=6, shuffle=True, fletcher32=False, chunking='seq', least_significant_digit=None, fill_value=None)}

The C{datatype} is a string with the same meaning as the C{dtype.str}
attribute of arrays in module nump. Supported data types are: C{'S1'
(NC_CHAR), 'i1' (NC_BYTE), 'i2' (NC_SHORT), 'i4' (NC_INT), 'f4'
(NC_FLOAT)} and C{'f8' (NC_DOUBLE)}. The old single character 
Numeric typecodes (C{'f','d','c','i','h','b'}) are also accepted.

Data from netCDF variables are presented to python as numpy arrays with
the corresponding data type. 

C{dimensions} must be a tuple containing dimension names (strings) that
have been defined previously using C{createDimension}. The default value
is an empty tuple, which means the variable is a scalar.

If the optional keyword C{zlib} is C{True}, the data will be compressed
in the netCDF file using gzip compression (default C{True}).

The optional keyword C{complevel} is an integer between 1 and 9
describing the level of compression desired (default 6).

If the optional keyword C{shuffle} is C{True}, the HDF5 shuffle filter
will be applied before compressing the data (default C{True}).  This
significantly improves compression.

If the optional keyword C{fletcher32} is C{True}, the Fletcher32 HDF5
checksum algorithm is activated to detect errors. Default C{False}.

If the optional keyword C{chunking} is C{'seq'} (Default) HDF5 chunk
sizes are set to favor sequential access.  If C{chunking='sub'}, chunk
sizes are set to favor subsetting equally in all dimensions.

The optional keyword C{fill_value} can be used to override the default
netCDF C{_FillValue} (the value that the variable gets filled with
before any data is written to it).

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
standard attributes: C{dimensions, dtype, shape} and
C{least_significant_digit} Application programs should never modify
these attributes. The C{dimensions} attribute is a tuple containing the
names of the dimensions associated with this variable. The C{dtype}
attribute is a string describing the variable's data type (C{i4, f8,
S1,} etc). The C{shape} attribute is a tuple describing the current
sizes of all the variable's dimensions. The C{least_significant_digit}
attributes describes the power of ten of the smallest decimal place in
the data the contains a reliable value.  Data is truncated to this
decimal place when it is assigned to the L{Variable} instance. If
C{None}, the data is not truncated.

"""
        self.variables[varname] = Variable(self, varname, datatype, dimensions, zlib=zlib, complevel=complevel, shuffle=shuffle, fletcher32=fletcher32, least_significant_digit=least_significant_digit, chunking=chunking, fill_value=fill_value)
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
        ierr = nc_rename_var(self._dsetid, var._varid, namstring)
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))
        # remove old key from dimensions dict.
        self.variables.pop(oldname)
        # add new key.
        self.variables[newname] = var

    def __delattr__(self,name):
        cdef char *attname
        # if it's a netCDF attribute, remove it
        if name not in _private_atts:
            self._redef()
            attname = PyString_AsString(name)
            ierr = nc_del_att(self._dsetid, NC_GLOBAL, attname)
            if ierr != NC_NOERR:
                raise RuntimeError(nc_strerror(ierr))
            self._enddef()
        else:
            raise AttributeError("'%s' is one of the reserved attributes %s, cannot delete" % (name, tuple(_private_atts)))

    def __setattr__(self,name,value):
        # if name in _private_atts, it is stored at the python
        # level and not in the netCDF file.
        if name not in _private_atts:
            self._redef()
            _set_att(self._dsetid, NC_GLOBAL, name, value)
            self._enddef()
        elif not name.endswith('__'):
            if hasattr(self,name):
                raise AttributeError("'%s' is one of the reserved attributes %s, cannot rebind" % (name, tuple(_private_atts)))
            else:
                self.__dict__[name]=value

    def ncattrs(self):
        """
return names of netCDF attribute for this L{Dataset} in a list

C{ncattrs()}""" 
        return _get_att_names(self._dsetid, NC_GLOBAL)

    def __getattr__(self,name):
        # if name in _private_atts, it is stored at the python
        # level and not in the netCDF file.
        if name.startswith('__') and name.endswith('__'):
            # if __dict__ requested, return a dict with netCDF attributes.
            if name == '__dict__': 
                names = self.ncattrs()
                values = []
                for name in names:
                    values.append(_get_att(self._dsetid, NC_GLOBAL, name))
                return dict(zip(names,values))
            else:
                raise AttributeError
        elif name in _private_atts:
            return self.__dict__[name]
        else:
            return _get_att(self._dsetid, NC_GLOBAL, name)

cdef class Dimension:
    """
A netCDF L{Dimension} is used to describe the coordinates of a L{Variable}.

Constructor: C{Dimension(dataset, name, size=None)}

L{Dimension} instances should be created using the C{createDimension}
method of a L{Dataset} instance, not using this class directly.

B{Parameters:}

B{C{dataset}} - L{Dataset} instance to associate with dimension.

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


    cdef public int _dimid, _dsetid

    def __init__(self, dset, name, size=None, **kwargs):
        cdef int ierr
        cdef char *dimname
        cdef size_t lendim
        self._dsetid = dset._dsetid
        if kwargs.has_key('id'):
            self._dimid = kwargs['id']
        else:
            dimname = name
            if size is not None:
                lendim = size
            else:
                lendim = NC_UNLIMITED
            dset._redef()
            ierr = nc_def_dim(self._dsetid, dimname, lendim, &self._dimid)
            dset._enddef()
            if ierr != NC_NOERR:
                raise RuntimeError(nc_strerror(ierr))

    def __len__(self):
        """
len(L{Dimension} instance) returns current size of dimension"""
        cdef int ierr
        cdef size_t lengthp
        ierr = nc_inq_dimlen(self._dsetid, self._dimid, &lengthp)
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))
        return lengthp

    def isunlimited(self):
        """
returns C{True} if the L{Dimension} instance is unlimited, C{False} otherwise.

C{isunlimited()}"""
        cdef int ierr, n, numunlimdims, ndims, nvars, ngatts, xdimid
        cdef int *unlimdimids
        ierr = nc_inq(self._dsetid, &ndims, &nvars, &ngatts, &xdimid)
        if self._dimid == xdimid:
            return True
        else:
            return False

cdef class Variable:
    """
A netCDF L{Variable} is used to read and write netCDF data.  They are 
analagous to numpy array objects.

C{Variable(dataset, name, datatype, dimensions=(), zlib=True, complevel=6, 
shuffle=True, fletcher32=False, chunking='seq', 
least_significant_digit=None)}
   
L{Variable} instances should be created using the C{createVariable} method 
of a L{Dataset} instance, not using this class directly.

B{Parameters:}

B{C{dataset}} - L{Dataset} instance to associate with variable.

B{C{name}}  - Name of the variable.

B{C{datatype}} - L{Variable} data type, one of C{'f4'} (32-bit floating
point), C{'f8'} (64-bit floating point), C{'i4'} (32-bit signed
integer), C{'i2'} (16-bit signed integer), C{'i4'} (8-bit singed
integer), C{'i1'} (8-bit signed integer) or C{'S1'} (single-character
string). The old single character Numeric typecodes 
(C{'f','d','c','i','h','b'}) are also accepted

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
sizes are set to favor sequential access. Setting C{chunking = 'sub'}
will cause chunk sizes to be set to favor subsetting equally in any
dimension.

B{C{least_significant_digit}} - If specified, variable data will be
truncated (quantized). This produces 'lossy', but significantly more
efficient compression. For example, if C{least_significant_digit=1},
data will be quantized using around(scale*data)/scale, where scale =
2**bits, and bits is determined so that a precision of 0.1 is retained
(in this case bits=4). Default is C{None}, or no quantization.

B{C{fill_value}} - If specified, the default netCDF C{_FillValue} (the 
value that the variable gets filled with before any data is written to it) 
is replaced with this value.
 
B{Returns:}

a L{Variable} instance.  All further operations on the netCDF Variable are 
accomplised via L{Variable} instance methods.

A list of attribute names corresponding to netCDF attributes defined for
the variable can be obtained with the C{ncattrs()} method. These
attributes can be created by assigning to an attribute of the
L{Variable} instance. A dictionary containing all the netCDF attribute
name/value pairs is provided by the C{__dict__} attribute of a
L{Variable} instance.

The instance variables C{dimensions, dtype, shape} and
C{least_significant_digits} are read-only (and should not be modified by
the user).

@ivar dimensions: A tuple containing the names of the dimensions 
associated with this variable.

@ivar dtype: A description of the variable's data type (C{'i4','f8','S1'}, 
etc).

@ivar shape: a tuple describing the current size of all the variable's 
dimensions.

@ivar least_significant_digit: Describes the power of ten of the
smallest decimal place in the data the contains a reliable value.  Data
is truncated to this decimal place when it is assigned to the
L{Variable} instance. If C{None}, the data is not truncated. """

    cdef public int _varid, _dsetid
    cdef object _dset
    cdef public dtype

    def __init__(self, dset, name, datatype, dimensions=(), zlib=True, complevel=6, shuffle=True, fletcher32=False, least_significant_digit=None, chunking='seq', fill_value=None, **kwargs):
        cdef int ierr, ndims
        cdef char *varname
        cdef nc_type xtype
        cdef int dimids[NC_MAX_DIMS]
        cdef nc_var_options ncvaropt
        self._dsetid = dset._dsetid
        self._dset = dset
        if datatype not in _nptonctype.keys():
            raise TypeError('illegal data type, must be one of %s, got %s' % (_supportedtypes,datatype))
        self.dtype = _nctonptype[_nptonctype[datatype]]
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
                    try:
                        dimids[n] = dset.dimensions[dimname]._dimid
                    except:
                        raise KeyError('dimension %s not defined' % dimname)
            # set HDF5 filter parameters (in nc_var_options struct).
            if zlib:
                ncvaropt.deflate = 1
                ncvaropt.deflate_level = complevel
            else:
                ncvaropt.deflate = 0
                ncvaropt.deflate_level = 0
            # set chunking algorithm.
            if chunking == 'seq':
                ncvaropt.chunkalg = NC_CHUNK_SEQ
            elif chunking == 'sub':
                ncvaropt.chunkalg = NC_CHUNK_SUB
            else:
                raise ValueError("chunking keyword must be 'seq' or 'sub', got %s" % chunking)
            ncvaropt.chunksizes = NULL
            ncvaropt.extend_increments = NULL
            if fletcher32:
                ncvaropt.fletcher32 = 1
            else:
                ncvaropt.fletcher32 = 0
            # only enable shuffle if zlib is True and complevel != 0:
            if shuffle and zlib and complevel:
                ncvaropt.shuffle = 1
            else:
                ncvaropt.shuffle = 0
            dset._redef()
            if ndims:
                if dset.file_format=='NETCDF4_CLASSIC':
                    ierr = nc_def_var_full(self._dsetid, varname, xtype, ndims,
                                           dimids, &self._varid, &ncvaropt)
                else: # compression stuff ignored for NETCDF3 files.
                    ierr = nc_def_var(self._dsetid, varname, xtype, ndims,
                                      dimids, &self._varid)
            else: # a scalar variable.
                ierr = nc_def_var(self._dsetid, varname, xtype, ndims,
                                  NULL, &self._varid)
            if ierr != NC_NOERR:
                raise RuntimeError(nc_strerror(ierr))
            # set a fill value for this variable if fill_value keyword
            # given.  This avoids the HDF5 overhead of deleting and 
            # recreating the dataset if it is set later (after the enddef).
            if fill_value is not None:
                # cast fill_value to type of variable.
                fill_value = NP.array(fill_value, self.dtype)
                _set_att(self._dsetid, self._varid, '_FillValue', fill_value)
            dset._enddef()
            if least_significant_digit is not None:
                self.least_significant_digit = least_significant_digit

    def typecode(self):
        """return dtype attribute, provided for compatibility with Scientific.IO.NetCDF"""
        return self.dtype

    def _getDimensions(self):
        """Private method to get variables's dimension names"""
        cdef int ierr, numdims, n, nn
        cdef char namstring[NC_MAX_NAME+1]
        cdef int dimids[NC_MAX_DIMS]
        # get number of dimensions for this variable.
        ierr = nc_inq_varndims(self._dsetid, self._varid, &numdims)
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))
        # get dimension ids.
        ierr = nc_inq_vardimid(self._dsetid, self._varid, dimids)
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))
        # loop over dimensions, retrieve names.
        dimensions = ()
        for nn from 0 <= nn < numdims:
            ierr = nc_inq_dimname(self._dsetid, dimids[nn], namstring)
            if ierr != NC_NOERR:
                raise RuntimeError(nc_strerror(ierr))
            name = namstring
            dimensions = dimensions + (name,)
        return dimensions

    def _shape(self):
        """Private method to find current sizes of all variable dimensions"""
        varshape = ()
        for dimname in self.dimensions:
            dim = self._dset.dimensions[dimname]
            varshape = varshape + (len(dim),)
        return varshape

    def ncattrs(self):
        """
return names of netCDF attribute for this L{Variable} in a list

C{ncattrs()}"""

        return _get_att_names(self._dsetid, self._varid)

    def __delattr__(self,name):
        cdef char *attname
        # if it's a netCDF attribute, remove it
        if name not in _private_atts:
            self._dset._redef()
            attname = PyString_AsString(name)
            ierr = nc_del_att(self._dsetid, self._varid, attname)
            if ierr != NC_NOERR:
                raise RuntimeError(nc_strerror(ierr))
            self._dset._enddef()
        else:
            raise AttributeError("'%s' is one of the reserved attributes %s, cannot delete" % (name, tuple(_private_atts)))

    def __setattr__(self,name,value):
        # if name in _private_atts, it is stored at the python
        # level and not in the netCDF file.
        if name not in _private_atts:
            self._dset._redef()
            # if setting _FillValue, make sure value
            # has same type as variable.
            if name == '_FillValue':
                value = NP.array(value, self.dtype)
            _set_att(self._dsetid, self._varid, name, value)
            self._dset._enddef()
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
                    values.append(_get_att(self._dsetid, self._varid, name))
                return dict(zip(names,values))
            else:
                raise AttributeError
        elif name in _private_atts:
            return self.__dict__[name]
        else:
            return _get_att(self._dsetid, self._varid, name)

    def __getitem__(self, elem):
        # This special method is used to index the netCDF variable
        # using the "extended slice syntax". The extended slice syntax
        # is a perfect match for the "start", "count" and "stride"
        # arguments to the nc_get_var() function, and is much more easy
        # to use.
        start, count, stride = _buildStartCountStride(elem,self.shape,self.dimensions,self._dset.dimensions)
        # Get elements.
        return self._get(start, count, stride)

    def __setitem__(self, elem, data):
        # This special method is used to assign to the netCDF variable
        # using "extended slice syntax". The extended slice syntax
        # is a perfect match for the "start", "count" and "stride"
        # arguments to the nc_put_var() function, and is much more easy
        # to use.
        start, count, stride = _buildStartCountStride(elem,self.shape,self.dimensions,self._dset.dimensions)
        # quantize data if least_significant_digit attribute set.
        if 'least_significant_digit' in self.ncattrs():
            data = _quantize(data,self.least_significant_digit)
        # A numpy array is needed. Convert if necessary.
        if not type(data) == NP.ndarray:
            data = NP.array(data, self.dtype)
        # append the data to the variable object.
        self._put(data, start, count, stride)

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

    def _put(self,ndarray data, start, count, stride):
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
            else: # raise IndexError
                raise IndexError('size of data array does not conform to slice')
        if self.dtype != data.dtype.str[1:]:
            data = data.astype(self.dtype) # cast data, if necessary.

        if negstride:
            # reverse data along axes with negative strides.
            data = data[sl].copy() # make sure a copy is made.
        # strides all 1 or scalar variable, use put_vara (faster)
        if sum(stride) == ndims or ndims == 0:
            ierr = nc_put_vara(self._dsetid, self._varid,
                               startp, countp, data.data)
        else:  
            ierr = nc_put_vars(self._dsetid, self._varid,
                               startp, countp, stridep, data.data)
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))

    def _get(self, start, count, stride):
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
        # fill up startp,count,stridep.
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
                sl.append(slice(None, None, -1)) # this slice will reverse the data
            else:
                startp[n] = start[n]
                stridep[n] = stride[n]
                sl.append(slice(None,None, 1))
        # allocate array of correct primitive type.
        data = NP.empty(shapeout, self.dtype)
        # strides all 1 or scalar variable, use get_vara (faster)
        if sum(stride) == ndims or ndims ==  0: 
            ierr = nc_get_vara(self._dsetid, self._varid,
                               startp, countp, data.data)
        else:
            ierr = nc_get_vars(self._dsetid, self._varid,
                               startp, countp, stridep, data.data)
        if negstride:
            # reverse data along axes with negative strides.
            data = data[sl].copy() # make copy so data is contiguous.
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))
        if not self.dimensions: 
            return data[0] # a scalar 
        elif squeeze_out:
            return NP.squeeze(data)
        else:
            return data
