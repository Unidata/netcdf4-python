# utility functions (visible from python).

def stringtochar(a):
    """
stringtochar(a)

convert a string array to a character array with one extra dimension

@param a:  Input numpy string array with numpy datatype 'SN', where N
is the number of characters in each string.  Will be converted to
an array of characters (datatype 'S1') of shape a.shape + (N,).

@return: A numpy character array with datatype 'S1' and shape 
a.shape + (N,), where N is the length of each string in a."""
    b = NP.array(tuple(a.tostring()),'S1')
    b.shape = a.shape + (a.itemsize,)
    return b

def chartostring(b):
    """
chartostring(b)

convert a character array to a string array with one less dimension.

@param b:  Input character array (numpy datatype 'S1').
Will be converted to a array of strings, where each string has a fixed
length of b.shape[-1] characters.

@return: A numpy string array with datatype 'SN' and shape b.shape[:-1],
where N=b.shape[-1]."""
    bs = b.tostring()
    slen = b.shape[-1]
    a = NP.array([bs[n1:n1+slen] for n1 in range(0,len(bs),slen)],'S'+repr(slen))
    a.shape = b.shape[:-1]
    return a

def date2num(dates,units,calendar='standard'):
    """
date2num(dates,units,calendar='standard')

Return numeric time values given datetime objects. The units
of the numeric time values are described by the L{units} argument
and the L{calendar} keyword. The datetime objects must
be in UTC with no time-zone offset.  If there is a 
time-zone offset in C{units}, it will be applied to the
returned numeric values.

Like the matplotlib C{date2num} function, except that it allows
for different units and calendars.  Behaves the same if
C{units = 'days since 0001-01-01 00:00:00'} and 
C{calendar = 'proleptic_gregorian'}.

@param dates: A datetime object or a sequence of datetime objects.
 The datetime objects should not include a time-zone offset.

@param units: a string of the form C{'B{time units} since B{reference time}}'
 describing the time units. B{C{time units}} can be days, hours, minutes
 or seconds.  B{C{reference time}} is the time origin. A valid choice
 would be units=C{'hours since 1800-01-01 00:00:00 -6:00'}.

@param calendar: describes the calendar used in the time calculations. 
 All the values currently defined in the U{CF metadata convention 
 <http://cf-pcmdi.llnl.gov/documents/cf-conventions/>} are supported.
 Valid calendars C{'standard', 'gregorian', 'proleptic_gregorian'
 'noleap', '365_day', '360_day', 'julian', 'all_leap', '366_day'}.
 Default is C{'standard'}, which is a mixed Julian/Gregorian calendar.

@return: a numeric time value, or an array of numeric time values.

The maximum resolution of the numeric time values is 1 second.
    """
    cdftime = netcdftime.utime(units,calendar=calendar)
    return cdftime.date2num(dates)

def num2date(times,units,calendar='standard'):
    """
num2date(times,units,calendar='standard')

Return datetime objects given numeric time values. The units
of the numeric time values are described by the C{units} argument
and the C{calendar} keyword. The returned datetime objects represent 
UTC with no time-zone offset, even if the specified 
C{units} contain a time-zone offset.

Like the matplotlib C{num2date} function, except that it allows
for different units and calendars.  Behaves the same if
C{units = 'days since 001-01-01 00:00:00'} and 
C{calendar = 'proleptic_gregorian'}.

@param times: numeric time values. Maximum resolution is 1 second.

@param units: a string of the form C{'B{time units} since B{reference time}}'
describing the time units. B{C{time units}} can be days, hours, minutes
or seconds.  B{C{reference time}} is the time origin. A valid choice
would be units=C{'hours since 1800-01-01 00:00:00 -6:00'}.

@param calendar: describes the calendar used in the time calculations. 
All the values currently defined in the U{CF metadata convention 
<http://cf-pcmdi.llnl.gov/documents/cf-conventions/>} are supported.
Valid calendars C{'standard', 'gregorian', 'proleptic_gregorian'
'noleap', '365_day', '360_day', 'julian', 'all_leap', '366_day'}.
Default is C{'standard'}, which is a mixed Julian/Gregorian calendar.

@return: a datetime instance, or an array of datetime instances.

The datetime instances returned are 'real' python datetime 
objects if the date falls in the Gregorian calendar (i.e. 
C{calendar='proleptic_gregorian'}, or C{calendar = 'standard'} or C{'gregorian'}
and the date is after 1582-10-15). Otherwise, they are 'phony' datetime 
objects which support some but not all the methods of 'real' python
datetime objects.  This is because the python datetime module cannot
the uses the C{'proleptic_gregorian'} calendar, even before the switch
occured from the Julian calendar in 1582. The datetime instances
do not contain a time-zone offset, even if the specified C{units}
contains one.
    """
    cdftime = netcdftime.utime(units,calendar=calendar)
    return cdftime.num2date(times)

def getlibversion():
    """
getlibversion()

returns a string describing the version of the netcdf-4 library
used to build the module, and when it was built.
    """
    cdef char *libstring
    libstring = nc_inq_libvers()
    return PyString_FromString(libstring)
