"""
Introduction
============

Python interface to the netCDF version 3 library. The API modelled after 
U{Scientific.IO.NetCDF 
<http://starship.python.net/~hinsen/ScientificPython>}, and should be 
familiar to users of that module. Some new features not found in
Scientific.IO.NetCDF:

 - ability to read multi-file netCDF Datasets, making variables
 spanning multiple files appear as if they were in one file
 (see L{MFDataset} for more details).
 - support for masked arrays, automatic packing and unpacking of
 packed integer data (see L{Variable.set_auto_maskandscale} for details).
 - supports more complicated slicing (including numpy 'fancy indexing').
 - includes convenience functions for converting to and from datetime
 objects to numeric time values (see L{num2date} and L{date2num}), using
 all the calendars in the CF standard.
 - convenience functions for converting arrays of characters to arrays
 of strings, and vice-versa (L{stringtochar} and L{chartostring}).
 - can use numpy dtype objects to specify netCDF variable datatype.

Download 
========

 - U{Project page <http://code.google.com/p/netcdf4-python/>}.
 - U{Subversion repository <http://code.google.com/p/netcdf4-python/source>}.
 - U{Source tar.gz <http://code.google.com/p/netcdf4-python/downloads/list>}.

Requires 
======== 

 - numpy array module U{http://numpy.scipy.org}, version 1.0 or later.
 - The netCDF-3 C library (version 3.6 or later), available at
 U{ftp://ftp.unidata.ucar.edu/pub/netcdf}.


Install
=======

 - install the requisite python modules and C libraries (see above).
 Optionally, set the C{NETCDF3_DIR} environment variable to point to where the
 netCDF version 3 library and headers are installed. If C{NETCDF3_DIR} is
 not set, some standard locations will be searched.
 - run 'python setup-nc3.py install'
 - run some of the tests in the 'test3' directory.

Tutorial
========

1) Creating/Opening/Closing a netCDF file
-----------------------------------------

To create a netCDF file from python, you simply call the L{Dataset}
constructor. This is also the method used to open an existing netCDF
file.  If the file is open for write access (C{w, r+} or C{a}), you may
write any type of data including new dimensions, variables and
attributes.  netCDF files come in several flavors (C{NETCDF3_CLASSIC,
NETCDF3_64BIT, NETCDF4_CLASSIC}, and C{NETCDF4}). The first two flavors
are supported by version 3 of the netCDF library, and are supported
in this module. To read or write C{NETCDF4} and C{NETCDF4_CLASSIC}
files use the companion L{netCDF4} python module. The default format
C{NETCDF3_64BIT}. To see how a given file is formatted, you can examine the
C{file_format} L{Dataset} attribute.  Closing the netCDF file is
accomplished via the L{close<Dataset.close>} method of the L{Dataset}
instance.

Here's an example:

>>> import netCDF3
>>> ncfile = netCDF3.Dataset('test.nc', 'w')
>>> print ncfile.file_format
NETCDF3_64BIT
>>>
>>> ncfile.close()
            
2) Dimensions in a netCDF file
------------------------------

netCDF defines the sizes of all variables in terms of dimensions, so
before any variables can be created the dimensions they use must be
created first. A special case, not often used in practice, is that of a
scalar variable, which has no dimensions. A dimension is created using
the L{createDimension<Dataset.createDimension>} method of a L{Dataset}
instance. A Python string is used to set the name of the
dimension, and an integer value is used to set the size. To create an
unlimited dimension (a dimension that can be appended to), the size
value is set to C{None}. netCDF 3 files can only have one unlimited
dimension, and it must be the first (leftmost) dimension of the variable.

>>> ncfile.createDimension('press', 10)
>>> ncfile.createDimension('time', None)
>>> ncfile.createDimension('lat', 73)
>>> ncfile.createDimension('lon', 144)
            

All of the L{Dimension} instances are stored in a python dictionary.

>>> print ncfile.dimensions
{'lat': <netCDF3.Dimension object at 0x24a5f7b0>, 
 'time': <netCDF3.Dimension object at 0x24a5f788>, 
 'lon': <netCDF3.Dimension object at 0x24a5f7d8>, 
 'press': <netCDF3.Dimension object at 0x24a5f760>}
>>>

Calling the python C{len} function with a L{Dimension} instance returns
the current size of that dimension. The
L{isunlimited<Dimension.isunlimited>} method of a L{Dimension} instance
can be used to determine if the dimensions is unlimited, or appendable.

>>> for dimname, dimobj in ncfile.dimensions.iteritems():
>>>    print dimname, len(dimobj), dimobj.isunlimited()
lat 73 False
time 0 True
lon 144 False
press 10 False
>>>

L{Dimension} names can be changed using the
L{renameDimension<Dataset.renameDimension>} method of a L{Dataset} instance.
            
3) Variables in a netCDF file
-----------------------------

netCDF variables behave much like python multidimensional array objects
supplied by the U{numpy module <http://numpy.scipy.org>}. However,
unlike numpy arrays, netCDF3 variables can be appended to along one 
'unlimited' dimension. To create a netCDF variable, use the
L{createVariable<Dataset.createVariable>} method of a L{Dataset}
instance. The L{createVariable<Dataset.createVariable>} method
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
integer), C{'i1'} (8-bit signed integer), or C{'S1'} (single-character string)
The old Numeric single-character typecodes (C{'f'},C{'d'},C{'h'},
C{'s'},C{'b'},C{'B'},C{'c'},C{'i'},C{'l'}), corresponding to
(C{'f4'},C{'f8'},C{'i2'},C{'i2'},C{'i1'},C{'i1'},C{'S1'},C{'i4'},C{'i4'}),
will also work. 

The dimensions themselves are usually also defined as variables, called
coordinate variables. The L{createVariable<Dataset.createVariable>}
method returns an instance of the L{Variable} class whose methods can be
used later to access and set variable data and attributes.

>>> times = ncfile.createVariable('time','f8',('time',))
>>> pressure = ncfile.createVariable('press','i4',('press',))
>>> latitudes = ncfile.createVariable('latitude','f4',('lat',))
>>> longitudes = ncfile.createVariable('longitude','f4',('lon',))
>>> # two dimensions unlimited.
>>> temp = ncfile.createVariable('temp','f4',('time','press','lat','lon',))

All of the variables in the L{Dataset} are stored in a
Python dictionary, in the same way as the dimensions:

>>> print ncfile.variables
{'temp': <netCDF3.Variable object at 0x24a61068>,
 'pressure': <netCDF3.Variable object at 0.35f0f80>, 
 'longitude': <netCDF3.Variable object at 0x24a61030>,
 'pressure': <netCDF3.Variable object at 0x24a610a0>, 
 'time': <netCDF3.Variable object at 02x45f0.4.58>, 
 'latitude': <netCDF3.Variable object at 0.3f0fb8>}
>>>

L{Variable} names can be changed using the
L{renameVariable<Dataset.renameVariable>} method of a L{Dataset}
instance.
            

4) Attributes in a netCDF file
------------------------------

There are two types of attributes in a netCDF file, global and variable. 
Global attributes provide information about a dataset as a whole.
L{Variable} attributes provide information about
one of the variables in a dataset. Global attributes are set by assigning
values to L{Dataset} instance variables. L{Variable}
attributes are set by assigning values to L{Variable} instance
variables. Attributes can be strings, numbers or sequences. Returning to
our example,

>>> import time
>>> ncfile.description = 'bogus example script'
>>> ncfile.history = 'Created ' + time.ctime(time.time())
>>> ncfile.source = 'netCDF3 python module tutorial'
>>> latitudes.units = 'degrees north'
>>> longitudes.units = 'degrees east'
>>> pressure.units = 'hPa'
>>> temp.units = 'K'
>>> times.units = 'hours since 0001-01-01 00:00:00.0'
>>> times.calendar = 'gregorian'

The L{ncattrs<Dataset.ncattrs>} method of a L{Dataset} or
L{Variable} instance can be used to retrieve the names of all the netCDF
attributes. This method is provided as a convenience, since using the
built-in C{dir} Python function will return a bunch of private methods
and attributes that cannot (or should not) be modified by the user.

>>> for name in ncfile.ncattrs():
>>>     print 'Global attr', name, '=', getattr(ncfile,name)
Global attr description = bogus example script
Global attr history = Created Mon Nov  7 10.30:56 2005
Global attr source = netCDF3 python module tutorial

The C{__dict__} attribute of a L{Dataset} or L{Variable} 
instance provides all the netCDF attribute name/value pairs in a python 
dictionary:

>>> print ncfile.__dict__
{'source': 'netCDF3 python module tutorial',
'description': 'bogus example script',
'history': 'Created Mon Nov  7 10.30:56 2005'}

Attributes can be deleted from a netCDF L{Dataset} or
L{Variable} using the python C{del} statement (i.e. C{del var.foo}
removes the attribute C{foo} the the variable C{var}).

6) Writing data to and retrieving data from a netCDF variable
-------------------------------------------------------------

Now that you have a netCDF L{Variable} instance, how do you put data
into it? You can just treat it like an array and assign data to a slice.

>>> import numpy 
>>> latitudes[:] = numpy.arange(-90,91,2.5)
>>> pressure[:] = numpy.arange(1000,90,-100)
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
>>> print 'pressure levels =\\n',pressure[:]
[1000  900  800  700  600  500  400  300  200  100]
>>>

Unlike numpy array objects, netCDF L{Variable} objects with unlimited
dimensions will grow along those dimensions if you assign data outside
the currently defined range of indices.

>>> # append along two unlimited dimensions by assigning to slice.
>>> nlats = len(ncfile.dimensions['lat'])
>>> nlons = len(ncfile.dimensions['lon'])
>>> nlevs = len(ncfile.dimensions['press'])
>>> print 'temp shape before adding data = ',temp.shape
temp shape before adding data =  (0, 10, 73, 144)
>>>
>>> from numpy.random.mtrand import uniform
>>> temp[0:5,:,:,:] = uniform(size=(5,nlevs,nlats,nlons))
>>> print 'temp shape after adding data = ',temp.shape
temp shape after adding data =  (5, 16, 73, 144)
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
>>> from netCDF3 import num2date, date2num
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
            
All of the code in this tutorial is available in C{examples/tutorial-nc3.py},
Unit tests are in the C{test3} directory.

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
the same variable (with the same unlimited dimension).

>>> for nfile in range(10):
>>>     f = Dataset('mftest'+repr(nfile)+'.nc','w')
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
from netCDF4_utils import _buildStartCountStride

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
include "netCDF3.pxi"

# numpy data type <--> netCDF 3 data type mapping.

_nptonctype  = {'S1' : NC_CHAR,
                'i1' : NC_BYTE,
                'i2' : NC_SHORT,
                'i4' : NC_INT,   
                'f8' : NC_DOUBLE,   
                'f4' : NC_FLOAT}

_default_fillvals = {#'S1':NC_FILL_CHAR, 
                     'S1':'\0',
                     'i1':NC_FILL_BYTE,
                     'i2':NC_FILL_SHORT,
                     'i4':NC_FILL_INT,
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
    if att_type == NC_CHAR:
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

def _set_default_format(object format='NETCDF3_64BIT',object verbose=False):
    # Private function to set the netCDF file format
    if format == 'NETCDF3_64BIT':
        if verbose:
            print "Switching to 64-bit offset format"
        nc_set_default_format(NC_FORMAT_64BIT, NULL)
    elif format == 'NETCDF3_CLASSIC':
        if verbose:
            print "Switching to netCDF classic format"
        nc_set_default_format(NC_FORMAT_CLASSIC, NULL)
    else:
        raise ValueError, "format must be 'NETCDF3_64BIT' or 'NETCDF3_CLASSIC', got '%s'" % format

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
    # dimensions in a L{Dataset}.
    cdef int ierr, numdims, n
    cdef char namstring[NC_MAX_NAME+1]
    # get number of dimensions in this Dataset.
    ierr = nc_inq_ndims(group._grpid, &numdims)
    if ierr != NC_NOERR:
        raise RuntimeError(nc_strerror(ierr))
    # create empty dictionary for dimensions.
    dimensions = {}
    if numdims > 0:
        for n from 0 <= n < numdims:
            ierr = nc_inq_dimname(group._grpid, n, namstring)
            if ierr != NC_NOERR:
                raise RuntimeError(nc_strerror(ierr))
            name = namstring
            dimensions[name] = Dimension(group, name, id=n)
    return dimensions

cdef _get_vars(group):
    # Private function to create L{Variable} instances for all the
    # variables in a L{Dataset}.
    cdef int ierr, numvars, n, nn, numdims, varid
    cdef size_t sizein
    cdef int dim_sizes[NC_MAX_DIMS], dimids[NC_MAX_DIMS]
    cdef nc_type xtype
    cdef char namstring[NC_MAX_NAME+1]
    # get number of variables in this Dataset.
    ierr = nc_inq_nvars(group._grpid, &numvars)
    if ierr != NC_NOERR:
        raise RuntimeError(nc_strerror(ierr))
    # create empty dictionary for variables.
    variables = {}
    if numvars > 0:
        # loop over variables. 
        for n from 0 <= n < numvars:
             varid = n
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
                 raise KeyError('variable %s has unsupported data type' % name)
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
             dimensions = []
             for nn from 0 <= nn < numdims:
                 for key, value in group.dimensions.iteritems():
                     if value._dimid == dimids[nn]:
                         dimensions.append(key)
                         found = True
                         break
             # create new variable instance.
             variables[name] = Variable(group, name, datatype, dimensions, id=varid)
    return variables

# these are class attributes that 
# only exist at the python level (not in the netCDF file).

_private_atts = ['_grpid','_grp','_varid','dimensions','variables','dtype','file_format', 'ndim','maskandscale']


cdef class Dataset:
    """
Dataset(self, filename, mode="r", clobber=True, format='NETCDF3_64BIT')

A netCDF L{Dataset} is a collection of dimensions, variables and 
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

B{C{format}} - underlying file format (either C{'NETCDF3_64BIT'} 
or C{'NETCDF3_CLASSIC'}.  Only 
relevant if C{mode = 'w'} (if C{mode = 'r','a'} or C{'r+'} the file format 
is automatically detected). Default C{'NETCDF3_64BIT'} (the 64-bit offset
version of the netCDF 3 file format, which fully supports 2+ GB files)).
C{'NETCDF3_CLASSIC'} is the classic netCDF 3 file format that does not 
handle 2+ Gb files very well.

B{Returns:}

a L{Dataset} instance.  All further operations on the netCDF
Dataset are accomplised via L{Dataset} instance methods.

A list of attribute names corresponding to global netCDF attributes 
defined for the L{Dataset} can be obtained with the L{ncattrs()} method. 
These attributes can be created by assigning to an attribute of the 
L{Dataset} instance. A dictionary containing all the netCDF attribute
name/value pairs is provided by the C{__dict__} attribute of a
L{Dataset} instance.

The instance variables C{dimensions, variables,
file_format} and C{path} are read-only (and should not be modified by the 
user).

@ivar dimensions: The C{dimensions} dictionary maps the names of 
dimensions defined for the L{Dataset} to instances of the 
L{Dimension} class.

@ivar variables: The C{variables} dictionary maps the names of variables 
defined for this L{Dataset} to instances of the L{Variable} 
class.

@ivar file_format: The C{file_format} attribute describes the netCDF
file format version, either C{NETCDF3_CLASSIC} or
or C{NETCDF3_64BIT}."""
    cdef public int _grpid
    cdef public dimensions, variables, file_format, maskanscale

    def __init__(self, filename, mode='r', clobber=True, format='NETCDF3_64BIT', **kwargs):
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
        # get dimensions in the root group.
        self.dimensions = _get_dims(self)
        # get variables in the Dataset.
        self.variables = _get_vars(self)

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
        self._redef()
        ierr = nc_rename_dim(self._grpid, dim._dimid, namstring)
        self._enddef()
        if ierr != NC_NOERR:
            raise RuntimeError(nc_strerror(ierr))
        # remove old key from dimensions dict.
        self.dimensions.pop(oldname)
        # add new key.
        self.dimensions[newname] = dim
        # Variable.dimensions is determined by a method that
        # looks in the file, so no need to manually update.


    def createVariable(self, varname, datatype, dimensions=(), fill_value=None):
        """
createVariable(self, varname, datatype, dimensions=(), fill_value=None)

Creates a new variable with the given C{varname}, C{datatype}, and 
C{dimensions}. If dimensions are not given, the variable is assumed to be 
a scalar.

The C{datatype} can be a numpy datatype object, or a string that describes 
a numpy dtype object (like the C{dtype.str} attribue of a numpy array). 
Supported specifiers include: C{'S1' or 'c' (NC_CHAR), 'i1' or 'b' or 'B' 
(NC_BYTE), 'i2' or 'h' or 's' (NC_SHORT), 'u2' 
(NC_USHORT), 'i4' or 'i' or 'l' (NC_INT), 
'f4' or 'f' (NC_FLOAT), 'f8' or 'd' (NC_DOUBLE)}.

Data from netCDF variables is presented to python as numpy arrays with
the corresponding data type. 

C{dimensions} must be a tuple containing dimension names (strings) that 
have been defined previously using C{createDimension}. The default value 
is an empty tuple, which means the variable is a scalar.

The optional keyword C{fill_value} can be used to override the default 
netCDF C{_FillValue} (the value that the variable gets filled with before 
any data is written to it).

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
sizes of all the variable's dimensions.  The C{ndim} attribute
is the number of variable dimensions."""
        self.variables[varname] = Variable(self, varname, datatype, dimensions=dimensions, fill_value=fill_value)
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

    def ncattrs(self):
        """
ncattrs(self)

return netCDF global attribute names for this L{Dataset} in a list."""
        return _get_att_names(self._grpid, NC_GLOBAL)

    def __delattr__(self,name):
        cdef char *attname
        # if it's a netCDF attribute, remove it
        if name not in _private_atts:
            attname = PyString_AsString(name)
            self._redef()
            ierr = nc_del_att(self._grpid, NC_GLOBAL, attname)
            self._enddef()
            if ierr != NC_NOERR:
                raise RuntimeError(nc_strerror(ierr))
        else:
            raise AttributeError, "'%s' is one of the reserved attributes %s, cannot delete" % (name, tuple(_private_atts))

    def __setattr__(self,name,value):
        # if name in _private_atts, it is stored at the python
        # level and not in the netCDF file.
        if name not in _private_atts:
            self._redef()
            _set_att(self._grpid, NC_GLOBAL, name, value)
            self._enddef()
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

cdef class Dimension:
    """
Dimension(self, dset, name, size=None)

A netCDF L{Dimension} is used to describe the coordinates of a L{Variable}.

L{Dimension} instances should be created using the
L{createDimension<Dataset.createDimension>} method of a 
L{Dataset} instance, not using this class directly.

B{Parameters:}

B{C{dset}}  - Dataset instance.

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
            grp._redef()
            ierr = nc_def_dim(self._grpid, dimname, lendim, &self._dimid)
            grp._enddef()
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
        ierr = nc_inq(self._grpid, &ndims, &nvars, &ngatts, &xdimid)
        if self._dimid == xdimid:
            return True
        else:
            return False

cdef class Variable:
    """
Variable(self, dset, name, datatype, dimensions=(), fill_value=None)

A netCDF L{Variable} is used to read and write netCDF data.  They are 
analagous to numpy array objects.

L{Variable} instances should be created using the
L{createVariable<Dataset.createVariable>} method of a L{Dataset}
instance, not using this class directly.

B{Parameters:}

B{C{dset}}  - Dataset instance.

B{C{name}}  - Name of the variable.

B{C{datatype}} - L{Variable} data type. Can be specified by providing a 
numpy dtype object, or a string that describes a numpy dtype object. 
Supported values, corresponding to C{str} attribute of numpy dtype 
objects, include C{'f4'} (32-bit floating point), C{'f8'} (64-bit floating 
point), C{'i4'} (32-bit signed integer), C{'i2'} (16-bit signed integer), 
C{'i4'} (8-bit singed integer), C{'i1'} (8-bit signed integer),
or C{'S1'} (single-character string).  From 
compatibility with Scientific.IO.NetCDF, the old Numeric single character 
typecodes can also be used (C{'f'} instead of C{'f4'}, C{'d'} instead of 
C{'f8'}, C{'h'} or C{'s'} instead of C{'i2'}, C{'b'} or C{'B'} instead of 
C{'i1'}, C{'c'} instead of C{'S1'}, and C{'i'} or C{'l'} instead of 
C{'i4'}).

B{Keywords:}

B{C{dimensions}} - a tuple containing the variable's dimension names 
(defined previously with C{createDimension}). Default is an empty tuple 
which means the variable is a scalar (and therefore has no dimensions).

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

The instance variables C{dimensions, dtype, ndim, shape} are read-only
(and should not be modified by the user).

@ivar dimensions: A tuple containing the names of the dimensions 
associated with this variable.

@ivar dtype: A numpy dtype object describing the variable's data type.

@ivar ndim: The number of variable dimensions.

@ivar shape: a tuple describing the current size of all the variable's 
dimensions."""
    cdef public int _varid, _grpid, _nunlimdim
    cdef object _grp
    cdef public ndim, dtype, maskandscale

    def __init__(self, grp, name, datatype, dimensions=(), fill_value=None, **kwargs):
        cdef int ierr, ndims, ichunkalg, ideflate_level, numdims
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
                    dim = grp.dimensions[dimname]
                    if dim is None:
                        raise KeyError("dimension %s not defined in Dataset" % dimname)
                    dimids[n] = dim._dimid
            # go into define mode if it's a netCDF 3 compatible
            # file format.  Be careful to exit define mode before
            # any exceptions are raised.
            grp._redef()
            # define variable.
            if ndims:
                ierr = nc_def_var(self._grpid, varname, xtype, ndims,
                                  dimids, &self._varid)
            else: # a scalar variable.
                ierr = nc_def_var(self._grpid, varname, xtype, ndims,
                                  NULL, &self._varid)
            if ierr != NC_NOERR:
                grp._enddef()
                raise RuntimeError(nc_strerror(ierr))
            # set a fill value for this variable if fill_value keyword
            # given.
            if fill_value is not None:
                # cast fill_value to type of variable.
                fillval = numpy.array(fill_value, self.dtype)
                _set_att(self._grpid, self._varid, '_FillValue', fillval)
            # leave define mode.
            grp._enddef()
        # count how many unlimited dimensions there are.
        self._nunlimdim = 0
        for dimname in self.dimensions:
            dim = grp.dimensions[dimname]
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
                dim = self._grp.dimensions[dimname]
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


    def ncattrs(self):
        """
ncattrs(self)

return netCDF attribute names for this L{Variable} in a list."""
        return _get_att_names(self._grpid, self._varid)

    def __delattr__(self,name):
        cdef char *attname
        # if it's a netCDF attribute, remove it
        if name not in _private_atts:
            attname = PyString_AsString(name)
            self._grp._redef()
            ierr = nc_del_att(self._grpid, self._varid, attname)
            self._grp._enddef()
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
            self._grp._redef()
            _set_att(self._grpid, self._varid, name, value)
            self._grp._enddef()
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

# include pure python utility functions.
# (use include instead of importing them so docstrings
#  get included in C extension code).
include "utils.pyx"
