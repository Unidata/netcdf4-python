"""
Performs conversions of netCDF time coordinate data to/from datetime objects.
"""

from cpython.object cimport PyObject_RichCompare

import numpy as np
import math
import numpy
import re

from datetime import datetime as real_datetime
from datetime import timedelta
import time                     # strftime

try:
    from itertools import izip as zip
except ImportError:  # python 3.x
    pass

microsec_units = ['microseconds','microsecond', 'microsec', 'microsecs']
millisec_units = ['milliseconds', 'millisecond', 'millisec', 'millisecs']
sec_units =      ['second', 'seconds', 'sec', 'secs', 's']
min_units =      ['minute', 'minutes', 'min', 'mins']
hr_units =       ['hour', 'hours', 'hr', 'hrs', 'h']
day_units =      ['day', 'days', 'd']
_units = microsec_units+millisec_units+sec_units+min_units+hr_units+day_units
_calendars = ['standard', 'gregorian', 'proleptic_gregorian',
              'noleap', 'julian', 'all_leap', '365_day', '366_day', '360_day']

__version__ = '1.4.1'

# Adapted from http://delete.me.uk/2005/03/iso8601.html
ISO8601_REGEX = re.compile(r"(?P<year>[+-]?[0-9]{1,4})(-(?P<month>[0-9]{1,2})(-(?P<day>[0-9]{1,2})"
                           r"(((?P<separator1>.)(?P<hour>[0-9]{1,2}):(?P<minute>[0-9]{1,2})(:(?P<second>[0-9]{1,2})(\.(?P<fraction>[0-9]+))?)?)?"
                           r"((?P<separator2>.?)(?P<timezone>Z|(([-+])([0-9]{1,2}):([0-9]{1,2}))))?)?)?)?"
                           )
TIMEZONE_REGEX = re.compile(
    "(?P<prefix>[+-])(?P<hours>[0-9]{1,2}):(?P<minutes>[0-9]{1,2})")

def JulianDayFromDate(date, calendar='standard'):
    """

    creates a Julian Day from a 'datetime-like' object.  Returns the fractional
    Julian Day (approximately millisecond accuracy).

    if calendar='standard' or 'gregorian' (default), Julian day follows Julian
    Calendar on and before 1582-10-5, Gregorian calendar after 1582-10-15.

    if calendar='proleptic_gregorian', Julian Day follows gregorian calendar.

    if calendar='julian', Julian Day follows julian calendar.

    Algorithm:

    Meeus, Jean (1998) Astronomical Algorithms (2nd Edition). Willmann-Bell,
    Virginia. p. 63

    """

    # based on redate.py by David Finlayson.

    # check if input was scalar and change return accordingly
    isscalar = False
    try:
        date[0]
    except:
        isscalar = True

    date = np.atleast_1d(np.array(date))
    year = np.empty(len(date), dtype=np.int32)
    month = year.copy()
    day = year.copy()
    hour = year.copy()
    minute = year.copy()
    second = year.copy()
    microsecond = year.copy()
    for i, d in enumerate(date):
        year[i] = d.year
        month[i] = d.month
        day[i] = d.day
        hour[i] = d.hour
        minute[i] = d.minute
        second[i] = d.second
        microsecond[i] = d.microsecond
    # convert years in BC era to astronomical years (so that 1 BC is year zero)
    # (fixes issue #596)
    year[year < 0] = year[year < 0] + 1
    # Convert time to fractions of a day
    day = day + hour / 24.0 + minute / 1440.0 + (second + microsecond/1.e6) / 86400.0

    # Start Meeus algorithm (variables are in his notation)
    month_lt_3 = month < 3
    month[month_lt_3] = month[month_lt_3] + 12
    year[month_lt_3] = year[month_lt_3] - 1

    # MC - assure array
    # A = np.int64(year / 100)
    A = (year / 100).astype(np.int64)

    # MC
    # jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + \
    #      day - 1524.5
    jd = 365. * year + np.int32(0.25 * year + 2000.) + np.int32(30.6001 * (month + 1)) + \
        day + 1718994.5

    # optionally adjust the jd for the switch from
    # the Julian to Gregorian Calendar
    # here assumed to have occurred the day after 1582 October 4
    if calendar in ['standard', 'gregorian']:
        # MC - do not have to be contiguous dates
        # if np.min(jd) >= 2299170.5:
        #     # 1582 October 15 (Gregorian Calendar)
        #     B = 2 - A + np.int32(A / 4)
        # elif np.max(jd) < 2299160.5:
        #     # 1582 October 5 (Julian Calendar)
        #     B = np.zeros(len(jd))
        # else:
        #     print(date, calendar, jd)
        #     raise ValueError(
        #         'impossible date (falls in gap between end of Julian calendar and beginning of Gregorian calendar')
        if np.any((jd >= 2299160.5) & (jd < 2299170.5)): # missing days in Gregorian calendar
            raise ValueError(
                'impossible date (falls in gap between end of Julian calendar and beginning of Gregorian calendar')
        B = np.zeros(len(jd))             # 1582 October 5 (Julian Calendar)
        ii = np.where(jd >= 2299170.5)[0] # 1582 October 15 (Gregorian Calendar)
        if ii.size>0:
            B[ii] = 2 - A[ii] + np.int32(A[ii] / 4)
    elif calendar == 'proleptic_gregorian':
        B = 2 - A + np.int32(A / 4)
    elif calendar == 'julian':
        B = np.zeros(len(jd))
    else:
        raise ValueError(
            'unknown calendar, must be one of julian,standard,gregorian,proleptic_gregorian, got %s' % calendar)

    # adjust for Julian calendar if necessary
    jd = jd + B

    # Add a small offset (proportional to Julian date) for correct re-conversion.
    # This is about 45 microseconds in 2000 for Julian date starting -4712.
    # (pull request #433).
    eps = np.finfo(float).eps
    eps = np.maximum(eps*jd, eps)
    jd += eps

    if isscalar:
        return jd[0]
    else:
        return jd


cdef _NoLeapDayFromDate(date):
    """

creates a Julian Day for a calendar with no leap years from a datetime
instance.  Returns the fractional Julian Day (approximately millisecond accuracy).

    """

    year = date.year
    month = date.month
    day = date.day
    hour = date.hour
    minute = date.minute
    second = date.second
    microsecond = date.microsecond
    # Convert time to fractions of a day
    day = day + hour / 24.0 + minute / 1440.0 + (second + microsecond/1.e6) / 86400.0

    # Start Meeus algorithm (variables are in his notation)
    if (month < 3):
        month = month + 12
        year = year - 1

    jd = int(365. * (year + 4716)) + int(30.6001 * (month + 1)) + \
        day - 1524.5

    return jd


cdef _AllLeapFromDate(date):
    """

creates a Julian Day for a calendar where all years have 366 days from
a 'datetime-like' object.
Returns the fractional Julian Day (approximately millisecond accuracy).

    """

    year = date.year
    month = date.month
    day = date.day
    hour = date.hour
    minute = date.minute
    second = date.second
    microsecond = date.microsecond
    # Convert time to fractions of a day
    day = day + hour / 24.0 + minute / 1440.0 + (second + microsecond/1.e6) / 86400.0

    # Start Meeus algorithm (variables are in his notation)
    if (month < 3):
        month = month + 12
        year = year - 1

    jd = int(366. * (year + 4716)) + int(30.6001 * (month + 1)) + \
        day - 1524.5

    return jd


cdef _360DayFromDate(date):
    """

creates a Julian Day for a calendar where all months have 30 days from
a 'datetime-like' object.
Returns the fractional Julian Day (approximately millisecond accuracy).

    """

    year = date.year
    month = date.month
    day = date.day
    hour = date.hour
    minute = date.minute
    second = date.second
    microsecond = date.microsecond
    # Convert time to fractions of a day
    day = day + hour / 24.0 + minute / 1440.0 + (second + microsecond/1.e6) / 86400.0

    jd = int(360. * (year + 4716)) + int(30. * (month - 1)) + day

    return jd


def DateFromJulianDay(JD, calendar='standard'):
    """

    returns a 'datetime-like' object given Julian Day. Julian Day is a
    fractional day with approximately millisecond accuracy.

    if calendar='standard' or 'gregorian' (default), Julian day follows Julian
    Calendar on and before 1582-10-5, Gregorian calendar after  1582-10-15.

    if calendar='proleptic_gregorian', Julian Day follows gregorian calendar.

    if calendar='julian', Julian Day follows julian calendar.

    The datetime object is a 'real' datetime object if the date falls in
    the Gregorian calendar (i.e. calendar='proleptic_gregorian', or
    calendar = 'standard'/'gregorian' and the date is after 1582-10-15).
    Otherwise, it's a 'phony' datetime object which is actually an instance
    of netcdftime.datetime.


    Algorithm:

    Meeus, Jean (1998) Astronomical Algorithms (2nd Edition). Willmann-Bell,
    Virginia. p. 63
    """

    # based on redate.py by David Finlayson.

    julian = np.array(JD, dtype=float)

    if np.min(julian) < 0:
        raise ValueError('Julian Day must be positive')

    dayofwk = np.atleast_1d(np.int32(np.fmod(np.int32(julian + 1.5), 7)))
    # get the day (Z) and the fraction of the day (F)
    # add 0.000005 which is 452 ms in case of jd being after
    # second 23:59:59 of a day we want to round to the next day see issue #75
    Z = np.atleast_1d(np.int32(np.round(julian)))
    F = np.atleast_1d(julian + 0.5 - Z).astype(np.float64)
    if calendar in ['standard', 'gregorian']:
        # MC
        # alpha = int((Z - 1867216.25)/36524.25)
        # A = Z + 1 + alpha - int(alpha/4)
        alpha = np.int32(((Z - 1867216.) - 0.25) / 36524.25)
        A = Z + 1 + alpha - np.int32(0.25 * alpha)
        # check if dates before oct 5th 1582 are in the array
        ind_before = np.where(julian < 2299160.5)[0]
        if len(ind_before) > 0:
            A[ind_before] = Z[ind_before]

    elif calendar == 'proleptic_gregorian':
        # MC
        # alpha = int((Z - 1867216.25)/36524.25)
        # A = Z + 1 + alpha - int(alpha/4)
        alpha = np.int32(((Z - 1867216.) - 0.25) / 36524.25)
        A = Z + 1 + alpha - np.int32(0.25 * alpha)
    elif calendar == 'julian':
        A = Z
    else:
        raise ValueError(
            'unknown calendar, must be one of julian,standard,gregorian,proleptic_gregorian, got %s' % calendar)

    B = A + 1524
    # MC
    # C = int((B - 122.1)/365.25)
    # D = int(365.25 * C)
    C = np.atleast_1d(np.int32(6680. + ((B - 2439870.) - 122.1) / 365.25))
    D = np.atleast_1d(np.int32(365 * C + np.int32(0.25 * C)))
    E = np.atleast_1d(np.int32((B - D) / 30.6001))

    # Convert to date
    day = np.clip(B - D - np.int64(30.6001 * E) + F, 1, None)
    nday = B - D - 123
    dayofyr = nday - 305
    ind_nday_before = np.where(nday <= 305)[0]
    if len(ind_nday_before) > 0:
        dayofyr[ind_nday_before] = nday[ind_nday_before] + 60
    # MC
    # if E < 14:
    #     month = E - 1
    # else:
    #     month = E - 13

    # if month > 2:
    #     year = C - 4716
    # else:
    #     year = C - 4715
    month = E - 1
    month[month > 12] = month[month > 12] - 12
    year = C - 4715
    year[month > 2] = year[month > 2] - 1
    year[year <= 0] = year[year <= 0] - 1

    # a leap year?
    leap = np.zeros(len(year),dtype=dayofyr.dtype)
    leap[year % 4 == 0] = 1
    if calendar == 'proleptic_gregorian':
        leap[(year % 100 == 0) & (year % 400 != 0)] = 0
    elif calendar in ['standard', 'gregorian']:
        leap[(year % 100 == 0) & (year % 400 != 0) & (julian < 2299160.5)] = 0

    inc_idx = np.where((leap == 1) & (month > 2))[0]
    dayofyr[inc_idx] = dayofyr[inc_idx] + leap[inc_idx]

    # Subtract the offset from JulianDayFromDate from the microseconds (pull
    # request #433).
    eps = np.finfo(float).eps
    eps = np.maximum(eps*julian, eps)
    hour = np.clip((F * 24.).astype(np.int64), 0, 23)
    F   -= hour / 24.
    minute = np.clip((F * 1440.).astype(np.int64), 0, 59)
    # this is an overestimation due to added offset in JulianDayFromDate
    second = np.clip((F - minute / 1440.) * 86400., 0, None)
    microsecond = (second % 1)*1.e6
    # remove the offset from the microsecond calculation.
    microsecond = np.clip(microsecond - eps*86400.*1e6, 0, 999999)

    # convert year, month, day, hour, minute, second to int32
    year = year.astype(np.int32)
    month = month.astype(np.int32)
    day = day.astype(np.int32)
    hour = hour.astype(np.int32)
    minute = minute.astype(np.int32)
    second = second.astype(np.int32)
    microsecond = microsecond.astype(np.int32)

    # check if input was scalar and change return accordingly
    isscalar = False
    try:
        JD[0]
    except:
        isscalar = True

    if calendar in 'proleptic_gregorian':
        # FIXME: datetime.datetime does not support years < 1
        if (year < 0).any():
            datetime_type = DatetimeProlepticGregorian
        else:
            datetime_type = real_datetime
    elif calendar in ('standard', 'gregorian'):
        # return a 'real' datetime instance if calendar is proleptic
        # Gregorian or Gregorian and all dates are after the
        # Julian/Gregorian transition
        if len(ind_before) == 0:
            datetime_type = real_datetime
        else:
            datetime_type = DatetimeGregorian
    elif calendar == "julian":
        datetime_type = DatetimeJulian
    else:
        raise ValueError("unsupported calendar: {0}".format(calendar))

    if not isscalar:
        return np.array([datetime_type(*args)
                         for args in
                         zip(year, month, day, hour, minute, second,
                             microsecond)])

    else:
        return datetime_type(year[0], month[0], day[0], hour[0],
                             minute[0], second[0], microsecond[0])


cdef _DateFromNoLeapDay(JD):
    """

returns a 'datetime-like' object given Julian Day for a calendar with no leap
days. Julian Day is a fractional day with approximately millisecond accuracy.

    """

    # based on redate.py by David Finlayson.

    if JD < 0:
        year_offset = int(-JD) // 365 + 1
        JD += year_offset * 365
    else:
        year_offset = 0

    dayofwk = int(math.fmod(int(JD + 1.5), 7))
    (F, Z) = math.modf(JD + 0.5)
    Z = int(Z)
    A = Z
    B = A + 1524
    C = int((B - 122.1) / 365.)
    D = int(365. * C)
    E = int((B - D) / 30.6001)

    # Convert to date
    day = B - D - int(30.6001 * E) + F
    nday = B - D - 123
    if nday <= 305:
        dayofyr = nday + 60
    else:
        dayofyr = nday - 305
    if E < 14:
        month = E - 1
    else:
        month = E - 13

    if month > 2:
        year = C - 4716
    else:
        year = C - 4715

    # Convert fractions of a day to time
    (dfrac, days) = math.modf(day / 1.0)
    (hfrac, hours) = math.modf(dfrac * 24.0)
    (mfrac, minutes) = math.modf(hfrac * 60.0)
    (sfrac, seconds) = math.modf(mfrac * 60.0)
    microseconds = sfrac*1.e6

    if year_offset > 0:
        # correct dayofwk

        # 365 mod 7 = 1, so the day of the week changes by one day for
        # every year in year_offset
        dayofwk -= int(math.fmod(year_offset, 7))

        if dayofwk < 0:
            dayofwk += 7

    return DatetimeNoLeap(year - year_offset, month, int(days), int(hours), int(minutes),
                          int(seconds), int(microseconds),dayofwk, dayofyr)


cdef _DateFromAllLeap(JD):
    """

returns a 'datetime-like' object given Julian Day for a calendar where all
years have 366 days.
Julian Day is a fractional day with approximately millisecond accuracy.

    """

    # based on redate.py by David Finlayson.

    if JD < 0:
        raise ValueError('Julian Day must be positive')

    dayofwk = int(math.fmod(int(JD + 1.5), 7))
    (F, Z) = math.modf(JD + 0.5)
    Z = int(Z)
    A = Z
    B = A + 1524
    C = int((B - 122.1) / 366.)
    D = int(366. * C)
    E = int((B - D) / 30.6001)

    # Convert to date
    day = B - D - int(30.6001 * E) + F
    nday = B - D - 123
    if nday <= 305:
        dayofyr = nday + 60
    else:
        dayofyr = nday - 305
    if E < 14:
        month = E - 1
    else:
        month = E - 13
    if month > 2:
        dayofyr = dayofyr + 1

    if month > 2:
        year = C - 4716
    else:
        year = C - 4715

    # Convert fractions of a day to time
    (dfrac, days) = math.modf(day / 1.0)
    (hfrac, hours) = math.modf(dfrac * 24.0)
    (mfrac, minutes) = math.modf(hfrac * 60.0)
    (sfrac, seconds) = math.modf(mfrac * 60.0)
    microseconds = sfrac*1.e6

    return DatetimeAllLeap(year, month, int(days), int(hours), int(minutes),
                           int(seconds), int(microseconds),dayofwk, dayofyr)


cdef _DateFrom360Day(JD):
    """

returns a 'datetime-like' object given Julian Day for a calendar where all
months have 30 days.
Julian Day is a fractional day with approximately millisecond accuracy.

    """

    if JD < 0:
        year_offset = int(-JD) // 360 + 1
        JD += year_offset * 360
    else:
        year_offset = 0

    #jd = int(360. * (year + 4716)) + int(30. * (month - 1)) + day
    (F, Z) = math.modf(JD)
    year = int((Z - 0.5) / 360.) - 4716
    dayofyr = Z - (year + 4716) * 360
    month = int((dayofyr - 0.5) / 30) + 1
    day = dayofyr - (month - 1) * 30 + F

    # Convert fractions of a day to time
    (dfrac, days) = math.modf(day / 1.0)
    (hfrac, hours) = math.modf(dfrac * 24.0)
    (mfrac, minutes) = math.modf(hfrac * 60.0)
    (sfrac, seconds) = math.modf(mfrac * 60.0)
    microseconds = sfrac*1.e6

    return Datetime360Day(year - year_offset, month, int(days), int(hours), int(minutes),
                          int(seconds), int(microseconds), -1, dayofyr)


cdef _dateparse(timestr):
    """parse a string of the form time-units since yyyy-mm-dd hh:mm:ss
    return a tuple (units,utc_offset, datetimeinstance)"""
    timestr_split = timestr.split()
    units = timestr_split[0].lower()
    if units not in _units:
        raise ValueError(
            "units must be one of 'seconds', 'minutes', 'hours' or 'days' (or singular version of these), got '%s'" % units)
    if timestr_split[1].lower() != 'since':
        raise ValueError("no 'since' in unit_string")
    # parse the date string.
    n = timestr.find('since') + 6
    year, month, day, hour, minute, second, utc_offset = _parse_date(
        timestr[n:].strip())
    return units, utc_offset, datetime(year, month, day, hour, minute, second)


class utime:

    """
Performs conversions of netCDF time coordinate
data to/from datetime objects.

To initialize: C{t = utime(unit_string,calendar='standard')}

where

B{C{unit_string}} is a string of the form
C{'time-units since <time-origin>'} defining the time units.

Valid time-units are days, hours, minutes and seconds (the singular forms
are also accepted). An example unit_string would be C{'hours
since 0001-01-01 00:00:00'}.

The B{C{calendar}} keyword describes the calendar used in the time calculations.
All the values currently defined in the U{CF metadata convention
<http://cf-pcmdi.llnl.gov/documents/cf-conventions/1.1/cf-conventions.html#time-coordinate>}
are accepted. The default is C{'standard'}, which corresponds to the mixed
Gregorian/Julian calendar used by the C{udunits library}. Valid calendars
are:

C{'gregorian'} or C{'standard'} (default):

Mixed Gregorian/Julian calendar as defined by udunits.

C{'proleptic_gregorian'}:

A Gregorian calendar extended to dates before 1582-10-15. That is, a year
is a leap year if either (i) it is divisible by 4 but not by 100 or (ii)
it is divisible by 400.

C{'noleap'} or C{'365_day'}:

Gregorian calendar without leap years, i.e., all years are 365 days long.
all_leap or 366_day Gregorian calendar with every year being a leap year,
i.e., all years are 366 days long.

C{'360_day'}:

All years are 360 days divided into 30 day months.

C{'julian'}:

Proleptic Julian calendar, extended to dates after 1582-10-5. A year is a
leap year if it is divisible by 4.

The C{L{num2date}} and C{L{date2num}} class methods can used to convert datetime
instances to/from the specified time units using the specified calendar.

The datetime instances returned by C{num2date} are 'real' python datetime
objects if the date falls in the Gregorian calendar (i.e.
C{calendar='proleptic_gregorian', 'standard'} or C{'gregorian'} and
the date is after 1582-10-15). Otherwise, they are 'phony' datetime
objects which are actually instances of C{L{netcdftime.datetime}}.  This is
because the python datetime module cannot handle the weird dates in some
calendars (such as C{'360_day'} and C{'all_leap'}) which don't exist in any real
world calendar.


Example usage:

>>> from netcdftime import utime
>>> from datetime import  datetime
>>> cdftime = utime('hours since 0001-01-01 00:00:00')
>>> date = datetime.now()
>>> print date
2016-10-05 08:46:27.245015
>>>
>>> t = cdftime.date2num(date)
>>> print t
17669840.7742
>>>
>>> date = cdftime.num2date(t)
>>> print date
2016-10-05 08:46:27.244996
>>>

The resolution of the transformation operation is approximately a millisecond.

Warning:  Dates between 1582-10-5 and 1582-10-15 do not exist in the
C{'standard'} or C{'gregorian'} calendars.  An exception will be raised if you pass
a 'datetime-like' object in that range to the C{L{date2num}} class method.

Words of Wisdom from the British MetOffice concerning reference dates:

"udunits implements the mixed Gregorian/Julian calendar system, as
followed in England, in which dates prior to 1582-10-15 are assumed to use
the Julian calendar. Other software cannot be relied upon to handle the
change of calendar in the same way, so for robustness it is recommended
that the reference date be later than 1582. If earlier dates must be used,
it should be noted that udunits treats 0 AD as identical to 1 AD."

@ivar origin: datetime instance defining the origin of the netCDF time variable.
@ivar calendar:  the calendar used (as specified by the C{calendar} keyword).
@ivar unit_string:  a string defining the the netCDF time variable.
@ivar units:  the units part of C{unit_string} (i.e. 'days', 'hours', 'seconds').
    """

    def __init__(self, unit_string, calendar='standard'):
        """
@param unit_string: a string of the form
C{'time-units since <time-origin>'} defining the time units.

Valid time-units are days, hours, minutes and seconds (the singular forms
are also accepted). An example unit_string would be C{'hours
since 0001-01-01 00:00:00'}.

@keyword calendar: describes the calendar used in the time calculations.
All the values currently defined in the U{CF metadata convention
<http://cf-pcmdi.llnl.gov/documents/cf-conventions/1.1/cf-conventions.html#time-coordinate>}
are accepted. The default is C{'standard'}, which corresponds to the mixed
Gregorian/Julian calendar used by the C{udunits library}. Valid calendars
are:
 - C{'gregorian'} or C{'standard'} (default):
 Mixed Gregorian/Julian calendar as defined by udunits.
 - C{'proleptic_gregorian'}:
 A Gregorian calendar extended to dates before 1582-10-15. That is, a year
 is a leap year if either (i) it is divisible by 4 but not by 100 or (ii)
 it is divisible by 400.
 - C{'noleap'} or C{'365_day'}:
 Gregorian calendar without leap years, i.e., all years are 365 days long.
 - C{'all_leap'} or C{'366_day'}:
 Gregorian calendar with every year being a leap year, i.e.,
 all years are 366 days long.
 -C{'360_day'}:
 All years are 360 days divided into 30 day months.
 -C{'julian'}:
 Proleptic Julian calendar, extended to dates after 1582-10-5. A year is a
 leap year if it is divisible by 4.

@returns: A class instance which may be used for converting times from netCDF
units to datetime objects.
        """
        calendar = calendar.lower()
        if calendar in _calendars:
            self.calendar = calendar
        else:
            raise ValueError(
                "calendar must be one of %s, got '%s'" % (str(_calendars), calendar))
        units, tzoffset, self.origin = _dateparse(unit_string)
        # real-world calendars limited to positive reference years.
        if self.calendar in ['julian', 'standard', 'gregorian', 'proleptic_gregorian']:
            if self.origin.year == 0:
                msg='zero not allowed as a reference year, does not exist in Julian or Gregorian calendars'
                raise ValueError(msg)
            elif self.origin.year < 0:
                msg='negative reference year in time units, must be >= 1'
                raise ValueError(msg)
        self.tzoffset = tzoffset  # time zone offset in minutes
        self.units = units
        self.unit_string = unit_string
        if self.calendar in ['noleap', '365_day'] and self.origin.month == 2 and self.origin.day == 29:
            raise ValueError(
                'cannot specify a leap day as the reference time with the noleap calendar')
        if self.calendar == '360_day' and self.origin.day > 30:
            raise ValueError(
                'there are only 30 days in every month with the 360_day calendar')
        if self.calendar in ['noleap', '365_day']:
            self._jd0 = _NoLeapDayFromDate(self.origin)
        elif self.calendar in ['all_leap', '366_day']:
            self._jd0 = _AllLeapFromDate(self.origin)
        elif self.calendar == '360_day':
            self._jd0 = _360DayFromDate(self.origin)
        else:
            self._jd0 = JulianDayFromDate(self.origin, calendar=self.calendar)

    def date2num(self, date):
        """
        Returns C{time_value} in units described by L{unit_string}, using
        the specified L{calendar}, given a 'datetime-like' object.

        The datetime object must represent UTC with no time-zone offset.
        If there is a time-zone offset implied by L{unit_string}, it will
        be applied to the returned numeric values.

        Resolution is approximately a millisecond.

        If C{calendar = 'standard'} or C{'gregorian'} (indicating
        that the mixed Julian/Gregorian calendar is to be used), an
        exception will be raised if the 'datetime-like' object describes
        a date between 1582-10-5 and 1582-10-15.

        Works for scalars, sequences and numpy arrays.
        Returns a scalar if input is a scalar, else returns a numpy array.
        """
        isscalar = False
        try:
            date[0]
        except:
            isscalar = True
        if not isscalar:
            date = numpy.array(date)
            shape = date.shape
        if self.calendar in ['julian', 'standard', 'gregorian', 'proleptic_gregorian']:
            if isscalar:
                jdelta = JulianDayFromDate(date, self.calendar) - self._jd0
            else:
                jdelta = JulianDayFromDate(
                    date.flat, self.calendar) - self._jd0
        elif self.calendar in ['noleap', '365_day']:
            if isscalar:
                if date.month == 2 and date.day == 29:
                    raise ValueError(
                        'there is no leap day in the noleap calendar')
                jdelta = _NoLeapDayFromDate(date) - self._jd0
            else:
                jdelta = []
                for d in date.flat:
                    if d.month == 2 and d.day == 29:
                        raise ValueError(
                            'there is no leap day in the noleap calendar')
                    jdelta.append(_NoLeapDayFromDate(d) - self._jd0)
        elif self.calendar in ['all_leap', '366_day']:
            if isscalar:
                jdelta = _AllLeapFromDate(date) - self._jd0
            else:
                jdelta = [_AllLeapFromDate(d) - self._jd0 for d in date.flat]
        elif self.calendar == '360_day':
            if isscalar:
                if date.day > 30:
                    raise ValueError(
                        'there are only 30 days in every month with the 360_day calendar')
                jdelta = _360DayFromDate(date) - self._jd0
            else:
                jdelta = []
                for d in date.flat:
                    if d.day > 30:
                        raise ValueError(
                            'there are only 30 days in every month with the 360_day calendar')
                    jdelta.append(_360DayFromDate(d) - self._jd0)
        if not isscalar:
            jdelta = numpy.array(jdelta)
        # convert to desired units, subtract time zone offset.
        if self.units in microsec_units:
            jdelta = jdelta * 86400. * 1.e6  - self.tzoffset * 60. * 1.e6
        elif self.units in millisec_units:
            jdelta = jdelta * 86400. * 1.e3  - self.tzoffset * 60. * 1.e3
        elif self.units in sec_units:
            jdelta = jdelta * 86400. - self.tzoffset * 60.
        elif self.units in min_units:
            jdelta = jdelta * 1440. - self.tzoffset
        elif self.units in hr_units:
            jdelta = jdelta * 24. - self.tzoffset / 60.
        elif self.units in day_units:
            jdelta = jdelta - self.tzoffset / 1440.
        else:
            raise ValueError('unsupported time units')
        if isscalar:
            return jdelta
        else:
            return numpy.reshape(jdelta, shape)

    def num2date(self, time_value):
        """
        Return a 'datetime-like' object given a C{time_value} in units
        described by L{unit_string}, using L{calendar}.

        dates are in UTC with no offset, even if L{unit_string} contains
        a time zone offset from UTC.

        Resolution is approximately a millisecond.

        Works for scalars, sequences and numpy arrays.
        Returns a scalar if input is a scalar, else returns a numpy array.

        The datetime instances returned by C{num2date} are 'real' python datetime
        objects if the date falls in the Gregorian calendar (i.e.
        C{calendar='proleptic_gregorian'}, or C{calendar = 'standard'/'gregorian'} and
        the date is after 1582-10-15). Otherwise, they are 'phony' datetime
        objects which are actually instances of netcdftime.datetime.  This is
        because the python datetime module cannot handle the weird dates in some
        calendars (such as C{'360_day'} and C{'all_leap'}) which
        do not exist in any real world calendar.
        """
        isscalar = False
        try:
            time_value[0]
        except:
            isscalar = True
        ismasked = False
        if hasattr(time_value, 'mask'):
            mask = time_value.mask
            ismasked = True
        if not isscalar:
            time_value = numpy.array(time_value, dtype='d')
            shape = time_value.shape
        # convert to desired units, add time zone offset.
        if self.units in microsec_units:
            jdelta = time_value / 86400000000. + self.tzoffset / 1440.
        elif self.units in millisec_units:
            jdelta = time_value / 86400000. + self.tzoffset / 1440.
        elif self.units in sec_units:
            jdelta = time_value / 86400. + self.tzoffset / 1440.
        elif self.units in min_units:
            jdelta = time_value / 1440. + self.tzoffset / 1440.
        elif self.units in hr_units:
            jdelta = time_value / 24. + self.tzoffset / 1440.
        elif self.units in day_units:
            jdelta = time_value + self.tzoffset / 1440.
        else:
            raise ValueError('unsupported time units')
        jd = self._jd0 + jdelta
        if self.calendar in ['julian', 'standard', 'gregorian', 'proleptic_gregorian']:
            if not isscalar:
                if ismasked:
                    date = []
                    for j, m in zip(jd.flat, mask.flat):
                        if not m:
                            date.append(DateFromJulianDay(j, self.calendar))
                        else:
                            date.append(None)
                else:
                    date = DateFromJulianDay(jd.flat, self.calendar)
            else:
                if ismasked and mask.item():
                    date = None
                else:
                    date = DateFromJulianDay(jd, self.calendar)
        elif self.calendar in ['noleap', '365_day']:
            if not isscalar:
                date = [_DateFromNoLeapDay(j) for j in jd.flat]
            else:
                date = _DateFromNoLeapDay(jd)
        elif self.calendar in ['all_leap', '366_day']:
            if not isscalar:
                date = [_DateFromAllLeap(j) for j in jd.flat]
            else:
                date = _DateFromAllLeap(jd)
        elif self.calendar == '360_day':
            if not isscalar:
                date = [_DateFrom360Day(j) for j in jd.flat]
            else:
                date = _DateFrom360Day(jd)
        if isscalar:
            return date
        else:
            return numpy.reshape(numpy.array(date), shape)


cdef _parse_timezone(tzstring):
    """Parses ISO 8601 time zone specs into tzinfo offsets

    Adapted from pyiso8601 (http://code.google.com/p/pyiso8601/)
    """
    if tzstring == "Z":
        return 0
    # This isn't strictly correct, but it's common to encounter dates without
    # time zones so I'll assume the default (which defaults to UTC).
    if tzstring is None:
        return 0
    m = TIMEZONE_REGEX.match(tzstring)
    prefix, hours, minutes = m.groups()
    hours, minutes = int(hours), int(minutes)
    if prefix == "-":
        hours = -hours
        minutes = -minutes
    return minutes + hours * 60.


cpdef _parse_date(datestring):
    """Parses ISO 8601 dates into datetime objects

    The timezone is parsed from the date string, assuming UTC
    by default.

    Adapted from pyiso8601 (http://code.google.com/p/pyiso8601/)
    """
    if not isinstance(datestring, str) and not isinstance(datestring, unicode):
        raise ValueError("Expecting a string %r" % datestring)
    m = ISO8601_REGEX.match(datestring.strip())
    if not m:
        raise ValueError("Unable to parse date string %r" % datestring)
    groups = m.groupdict()
    tzoffset_mins = _parse_timezone(groups["timezone"])
    if groups["hour"] is None:
        groups["hour"] = 0
    if groups["minute"] is None:
        groups["minute"] = 0
    if groups["second"] is None:
        groups["second"] = 0
    # if groups["fraction"] is None:
    #    groups["fraction"] = 0
    # else:
    #    groups["fraction"] = int(float("0.%s" % groups["fraction"]) * 1e6)
    iyear = int(groups["year"])
    return iyear, int(groups["month"]), int(groups["day"]),\
        int(groups["hour"]), int(groups["minute"]), int(groups["second"]),\
        tzoffset_mins

cdef _check_index(indices, times, nctime, calendar, select):
    """Return True if the time indices given correspond to the given times,
    False otherwise.

    Parameters:

    indices : sequence of integers
    Positive integers indexing the time variable.

    times : sequence of times.
    Reference times.

    nctime : netCDF Variable object
    NetCDF time object.

    calendar : string
    Calendar of nctime.

    select : string
    Index selection method.
    """
    N = nctime.shape[0]
    if (indices < 0).any():
        return False

    if (indices >= N).any():
        return False

    try:
        t = nctime[indices]
        nctime = nctime
    # WORKAROUND TO CHANGES IN SLICING BEHAVIOUR in 1.1.2
    # this may be unacceptably slow...
    # if indices are unsorted, or there are duplicate
    # values in indices, read entire time variable into numpy
    # array so numpy slicing rules can be used.
    except IndexError:
        nctime = nctime[:]
        t = nctime[indices]
# if fancy indexing not available, fall back on this.
#   t=[]
#   for ind in indices:
#       t.append(nctime[ind])

    if select == 'exact':
        return numpy.all(t == times)

    elif select == 'before':
        ta = nctime[numpy.clip(indices + 1, 0, N - 1)]
        return numpy.all(t <= times) and numpy.all(ta > times)

    elif select == 'after':
        tb = nctime[numpy.clip(indices - 1, 0, N - 1)]
        return numpy.all(t >= times) and numpy.all(tb < times)

    elif select == 'nearest':
        ta = nctime[numpy.clip(indices + 1, 0, N - 1)]
        tb = nctime[numpy.clip(indices - 1, 0, N - 1)]
        delta_after = ta - t
        delta_before = t - tb
        delta_check = numpy.abs(times - t)
        return numpy.all(delta_check <= delta_after) and numpy.all(delta_check <= delta_before)


def date2index(dates, nctime, calendar=None, select='exact'):
    """
    date2index(dates, nctime, calendar=None, select='exact')

    Return indices of a netCDF time variable corresponding to the given dates.

    @param dates: A datetime object or a sequence of datetime objects.
    The datetime objects should not include a time-zone offset.

    @param nctime: A netCDF time variable object. The nctime object must have a
    C{units} attribute. The entries are assumed to be stored in increasing
    order.

    @param calendar: Describes the calendar used in the time calculation.
    Valid calendars C{'standard', 'gregorian', 'proleptic_gregorian'
    'noleap', '365_day', '360_day', 'julian', 'all_leap', '366_day'}.
    Default is C{'standard'}, which is a mixed Julian/Gregorian calendar
    If C{calendar} is None, its value is given by C{nctime.calendar} or
    C{standard} if no such attribute exists.

    @param select: C{'exact', 'before', 'after', 'nearest'}
    The index selection method. C{exact} will return the indices perfectly
    matching the dates given. C{before} and C{after} will return the indices
    corresponding to the dates just before or just after the given dates if
    an exact match cannot be found. C{nearest} will return the indices that
    correspond to the closest dates.
    """
    try:
        nctime.units
    except AttributeError:
        raise AttributeError("netcdf time variable is missing a 'units' attribute")
    # Setting the calendar.
    if calendar == None:
        calendar = getattr(nctime, 'calendar', 'standard')
    cdftime = utime(nctime.units,calendar=calendar)
    times = cdftime.date2num(dates)
    return time2index(times, nctime, calendar=calendar, select=select)


def time2index(times, nctime, calendar=None, select='exact'):
    """
    time2index(times, nctime, calendar=None, select='exact')

    Return indices of a netCDF time variable corresponding to the given times.

    @param times: A numeric time or a sequence of numeric times.

    @param nctime: A netCDF time variable object. The nctime object must have a
    C{units} attribute. The entries are assumed to be stored in increasing
    order.

    @param calendar: Describes the calendar used in the time calculation.
    Valid calendars C{'standard', 'gregorian', 'proleptic_gregorian'
    'noleap', '365_day', '360_day', 'julian', 'all_leap', '366_day'}.
    Default is C{'standard'}, which is a mixed Julian/Gregorian calendar
    If C{calendar} is None, its value is given by C{nctime.calendar} or
    C{standard} if no such attribute exists.

    @param select: C{'exact', 'before', 'after', 'nearest'}
    The index selection method. C{exact} will return the indices perfectly
    matching the times given. C{before} and C{after} will return the indices
    corresponding to the times just before or just after the given times if
    an exact match cannot be found. C{nearest} will return the indices that
    correspond to the closest times.
    """
    try:
        nctime.units
    except AttributeError:
        raise AttributeError("netcdf time variable is missing a 'units' attribute")
    # Setting the calendar.
    if calendar == None:
        calendar = getattr(nctime, 'calendar', 'standard')

    num = numpy.atleast_1d(times)
    N = len(nctime)

    # Trying to infer the correct index from the starting time and the stride.
    # This assumes that the times are increasing uniformly.
    if len(nctime) >= 2:
        t0, t1 = nctime[:2]
        dt = t1 - t0
    else:
        t0 = nctime[0]
        dt = 1.
    if select in ['exact', 'before']:
        index = numpy.array((num - t0) / dt, int)
    elif select == 'after':
        index = numpy.array(numpy.ceil((num - t0) / dt), int)
    else:
        index = numpy.array(numpy.around((num - t0) / dt), int)

    # Checking that the index really corresponds to the given time.
    # If the times do not correspond, then it means that the times
    # are not increasing uniformly and we try the bisection method.
    if not _check_index(index, times, nctime, calendar, select):

        # Use the bisection method. Assumes nctime is ordered.
        import bisect
        index = numpy.array([bisect.bisect_right(nctime, n) for n in num], int)
        before = index == 0

        index = numpy.array([bisect.bisect_left(nctime, n) for n in num], int)
        after = index == N

        if select in ['before', 'exact'] and numpy.any(before):
            raise ValueError(
                'Some of the times given are before the first time in `nctime`.')

        if select in ['after', 'exact'] and numpy.any(after):
            raise ValueError(
                'Some of the times given are after the last time in `nctime`.')

        # Find the times for which the match is not perfect.
        # Use list comprehension instead of the simpler `nctime[index]` since
        # not all time objects support numpy integer indexing (eg dap).
        index[after] = N - 1
        ncnum = numpy.squeeze([nctime[i] for i in index])
        mismatch = numpy.nonzero(ncnum != num)[0]

        if select == 'exact':
            if len(mismatch) > 0:
                raise ValueError(
                    'Some of the times specified were not found in the `nctime` variable.')

        elif select == 'before':
            index[after] = N
            index[mismatch] -= 1

        elif select == 'after':
            pass

        elif select == 'nearest':
            nearest_to_left = num[mismatch] < numpy.array(
                [float(nctime[i - 1]) + float(nctime[i]) for i in index[mismatch]]) / 2.
            index[mismatch] = index[mismatch] - 1 * nearest_to_left

        else:
            raise ValueError(
                "%s is not an option for the `select` argument." % select)

        # Correct for indices equal to -1
        index[before] = 0

    # convert numpy scalars or single element arrays to python ints.
    return _toscalar(index)


cdef _toscalar(a):
    if a.shape in [(), (1,)]:
        return a.item()
    else:
        return a

cdef to_tuple(dt):
    """Turn a datetime.datetime instance into a tuple of integers. Elements go
    in the order of decreasing significance, making it easy to compare
    datetime instances. Parts of the state that don't affect ordering
    are omitted. Compare to datetime.timetuple()."""
    return (dt.year, dt.month, dt.day, dt.hour, dt.minute,
            dt.second, dt.microsecond)

# a cache of converters (utime instances) for different calendars
cdef dict _converters
_converters = {}
for calendar in _calendars:
    _converters[calendar] = utime("seconds since 1-1-1", calendar)

cdef class datetime(object):
    """
The base class implementing most methods of datetime classes that
mimic datetime.datetime but support calendars other than the proleptic
Gregorial calendar.
    """
    cdef readonly int year, month, day, hour, minute, dayofwk, dayofyr
    cdef readonly int second, microsecond
    cdef readonly str calendar

    # Python's datetime.datetime uses the proleptic Gregorian
    # calendar. This boolean is used to decide whether a
    # netcdftime.datetime instance can be converted to
    # datetime.datetime.
    cdef readonly bint datetime_compatible

    def __init__(self, int year, int month, int day, int hour=0, int minute=0, int second=0,
                 int microsecond=0, int dayofwk=-1, int dayofyr=1):
        """dayofyr set to 1 by default - otherwise time.strftime will complain"""

        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        self.minute = minute
        self.dayofwk = dayofwk
        self.dayofyr = dayofyr
        self.second = second
        self.microsecond = microsecond
        self.calendar = ""

        self.datetime_compatible = True

    @property
    def format(self):
        return '%Y-%m-%d %H:%M:%S'

    def strftime(self, format=None):
        if format is None:
            format = self.format
        return _strftime(self, format)

    def replace(self, **kwargs):
        "Return datetime with new specified fields."
        args = {"year": self.year,
                "month": self.month,
                "day": self.day,
                "hour": self.hour,
                "minute": self.minute,
                "second": self.second,
                "microsecond": self.microsecond,
                "dayofwk": self.dayofwk,
                "dayofyr": self.dayofyr}

        for name, value in kwargs.items():
            args[name] = value

        return self.__class__(**args)

    def timetuple(self):
        return (self.year, self.month, self.day, self.hour,
                self.minute, self.second, self.dayofwk, self.dayofyr, -1)

    cpdef _to_real_datetime(self):
        return real_datetime(self.year, self.month, self.day,
                             self.hour, self.minute, self.second,
                             self.microsecond)

    def __repr__(self):
        return "{0}.{1}{2}".format(self.__class__.__module__,
                                   self.__class__.__name__,
                                   self._getstate())

    def __str__(self):
        return self.strftime(self.format)

    def __hash__(self):
        try:
            d = self._to_real_datetime()
        except ValueError:
            return hash(self.timetuple())
        return hash(d)

    cdef to_tuple(self):
        return (self.year, self.month, self.day, self.hour, self.minute,
                self.second, self.microsecond)

    def __richcmp__(self, other, int op):
        cdef datetime dt, dt_other
        dt = self
        if isinstance(other, datetime):
            dt_other = other
            # comparing two datetime instances
            if dt.calendar == dt_other.calendar:
                return PyObject_RichCompare(dt.to_tuple(), dt_other.to_tuple(), op)
            else:
                # Note: it *is* possible to compare datetime
                # instances that use difference calendars by using
                # utime.date2num(), but this implementation does
                # not attempt it.
                raise TypeError("cannot compare {0!r} and {1!r} (different calendars)".format(dt, dt_other))
        elif isinstance(other, real_datetime):
            # comparing datetime and real_datetime
            if not dt.datetime_compatible:
                raise TypeError("cannot compare {0!r} and {1!r} (different calendars)".format(self, other))
            return PyObject_RichCompare(dt.to_tuple(), to_tuple(other), op)
        else:
            raise TypeError("cannot compare {0!r} and {1!r}".format(self, other))

    cdef _getstate(self):
        return (self.year, self.month, self.day, self.hour,
                self.minute, self.second, self.microsecond,
                self.dayofwk, self.dayofyr)

    def __reduce__(self):
        """special method that allows instance to be pickled"""
        return (self.__class__, self._getstate())

    cdef _add_timedelta(self, other):
        return NotImplemented

    def __add__(self, other):
        cdef datetime dt
        if isinstance(self, datetime) and isinstance(other, timedelta):
            dt = self
            delta = other
        elif isinstance(self, timedelta) and isinstance(other, datetime):
            dt = other
            delta = self
        else:
            return NotImplemented
        return dt._add_timedelta(delta)

    def __sub__(self, other):
        cdef datetime dt
        if isinstance(self, datetime): # left arg is a datetime instance
            dt = self
            if isinstance(other, datetime):
                # datetime - datetime
                if dt.calendar != other.calendar:
                    raise ValueError("cannot compute the time difference between dates with different calendars")
                if dt.calendar == "":
                    raise ValueError("cannot compute the time difference between dates that are not calendar-aware")
                converter = _converters[dt.calendar]
                return timedelta(seconds=converter.date2num(dt) - converter.date2num(other))
            elif isinstance(other, real_datetime):
                # datetime - real_datetime
                if not dt.datetime_compatible:
                    raise ValueError("cannot compute the time difference between dates with different calendars")
                return dt._to_real_datetime() - other
            elif isinstance(other, timedelta):
                # datetime - timedelta
                return dt._add_timedelta(-other)
            else:
                return NotImplemented
        else:
            if isinstance(self, real_datetime):
                # real_datetime - datetime
                if not other.datetime_compatible:
                    raise ValueError("cannot compute the time difference between dates with different calendars")
                return self - other._to_real_datetime()
            else:
                return NotImplemented

cdef class DatetimeNoLeap(datetime):
    """
Phony datetime object which mimics the python datetime object,
but uses the "noleap" ("365_day") calendar.
    """
    def __init__(self, *args, **kwargs):
        datetime.__init__(self, *args, **kwargs)
        self.calendar = "noleap"
        self.datetime_compatible = False

    cdef _add_timedelta(self, delta):
        return DatetimeNoLeap(*add_timedelta(self, delta, no_leap, False))

cdef class DatetimeAllLeap(datetime):
    """
Phony datetime object which mimics the python datetime object,
but uses the "all_leap" ("366_day") calendar.
    """
    def __init__(self, *args, **kwargs):
        datetime.__init__(self, *args, **kwargs)
        self.calendar = "all_leap"
        self.datetime_compatible = False

    cdef _add_timedelta(self, delta):
        return DatetimeAllLeap(*add_timedelta(self, delta, all_leap, False))

cdef class Datetime360Day(datetime):
    """
Phony datetime object which mimics the python datetime object,
but uses the "360_day" calendar.
    """
    def __init__(self, *args, **kwargs):
        datetime.__init__(self, *args, **kwargs)
        self.calendar = "360_day"
        self.datetime_compatible = False

    cdef _add_timedelta(self, delta):
        return Datetime360Day(*add_timedelta_360_day(self, delta))

cdef class DatetimeJulian(datetime):
    """
Phony datetime object which mimics the python datetime object,
but uses the "julian" calendar.
    """
    def __init__(self, *args, **kwargs):
        datetime.__init__(self, *args, **kwargs)
        self.calendar = "julian"
        self.datetime_compatible = False

    cdef _add_timedelta(self, delta):
        return DatetimeJulian(*add_timedelta(self, delta, is_leap_julian, False))

cdef class DatetimeGregorian(datetime):
    """
Phony datetime object which mimics the python datetime object,
but uses the mixed Julian-Gregorian ("standard", "gregorian") calendar.

The last date of the Julian calendar is 1582-10-4, which is followed
by 1582-10-15, using the Gregorian calendar.

Instances using the date after 1582-10-15 can be compared to
datetime.datetime instances and used to compute time differences
(datetime.timedelta) by subtracting a DatetimeGregorian instance from
a datetime.datetime instance or vice versa.
    """
    def __init__(self, *args, **kwargs):
        datetime.__init__(self, *args, **kwargs)
        self.calendar = "gregorian"

        # dates after 1582-10-15 can be converted to and compared to
        # proleptic Gregorian dates
        if self.to_tuple() >= (1582, 10, 15, 0, 0, 0, 0):
            self.datetime_compatible = True
        else:
            self.datetime_compatible = False

    cdef _add_timedelta(self, delta):
        return DatetimeGregorian(*add_timedelta(self, delta, is_leap_gregorian, True))

cdef class DatetimeProlepticGregorian(datetime):
    """
Phony datetime object which mimics the python datetime object,
but allows for dates that don't exist in the proleptic gregorian calendar.

Supports timedelta operations by overloading + and -.

Has strftime, timetuple, replace, __repr__, and __str__ methods. The
format of the string produced by __str__ is controlled by self.format
(default %Y-%m-%d %H:%M:%S). Supports comparisons with other phony
datetime instances using the same calendar; comparison with
datetime.datetime instances is possible for netcdftime.datetime
instances using 'gregorian' and 'proleptic_gregorian' calendars.

Instance variables are year,month,day,hour,minute,second,microsecond,dayofwk,dayofyr,
format, and calendar.
    """
    def __init__(self, *args, **kwargs):
        datetime.__init__(self, *args, **kwargs)
        self.calendar = "proleptic_gregorian"
        self.datetime_compatible = True

    cdef _add_timedelta(self, delta):
        return DatetimeProlepticGregorian(*add_timedelta(self, delta,
                                                         is_leap_proleptic_gregorian, False))

_illegal_s = re.compile(r"((^|[^%])(%%)*%s)")


cdef _findall(text, substr):
    # Also finds overlaps
    sites = []
    i = 0
    while 1:
        j = text.find(substr, i)
        if j == -1:
            break
        sites.append(j)
        i = j + 1
    return sites

# Every 28 years the calendar repeats, except through century leap
# years where it's 6 years.  But only if you're using the Gregorian
# calendar.  ;)


cdef _strftime(datetime dt, fmt):
    if _illegal_s.search(fmt):
        raise TypeError("This strftime implementation does not handle %s")
    # don't use strftime method at all.
    # if dt.year > 1900:
    #    return dt.strftime(fmt)

    year = dt.year
    # For every non-leap year century, advance by
    # 6 years to get into the 28-year repeat cycle
    delta = 2000 - year
    off = 6 * (delta // 100 + delta // 400)
    year = year + off

    # Move to around the year 2000
    year = year + ((2000 - year) // 28) * 28
    timetuple = dt.timetuple()
    s1 = time.strftime(fmt, (year,) + timetuple[1:])
    sites1 = _findall(s1, str(year))

    s2 = time.strftime(fmt, (year + 28,) + timetuple[1:])
    sites2 = _findall(s2, str(year + 28))

    sites = []
    for site in sites1:
        if site in sites2:
            sites.append(site)

    s = s1
    syear = "%4d" % (dt.year,)
    for site in sites:
        s = s[:site] + syear + s[site + 4:]
    return s

cdef bint is_leap_julian(int year):
    "Return 1 if year is a leap year in the Julian calendar, 0 otherwise."
    cdef int y
    y = year if year > 0 else year + 1
    return (y % 4) == 0

cdef bint is_leap_proleptic_gregorian(int year):
    "Return 1 if year is a leap year in the Gregorian calendar, 0 otherwise."
    cdef int y
    y = year if year > 0 else year + 1
    return (((y % 4) == 0) and ((y % 100) != 0)) or ((y % 400) == 0)

cdef bint is_leap_gregorian(int year):
    return (year > 1582 and is_leap_proleptic_gregorian(year)) or (year < 1582 and is_leap_julian(year))

cdef bint all_leap(int year):
    "Return True for all years."
    return True

cdef bint no_leap(int year):
    "Return False for all years."
    return False

# numbers of days per month for calendars supported by add_timedelta(...)
cdef int[13] month_lengths_365_day, month_lengths_366_day
#                  Dummy Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec
for j,N in enumerate([-1, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]):
    month_lengths_365_day[j] = N

#                  Dummy Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec
for j,N in enumerate([-1, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]):
    month_lengths_366_day[j] = N

cdef int* month_lengths(bint (*is_leap)(int), int year):
    if is_leap(year):
        return month_lengths_366_day
    else:
        return month_lengths_365_day

# Add a datetime.timedelta to a netcdftime.datetime instance. Uses
# integer arithmetic to avoid rounding errors and preserve
# microsecond accuracy.
#
# The argument is_leap is the pointer to a function returning 1 for leap years and 0 otherwise.
#
# This implementation supports 365_day (no_leap), 366_day (all_leap),
# julian, proleptic_gregorian, and the mixed Julian/Gregorian
# (standard, gregorian) calendars by using different is_leap and
# julian_gregorian_mixed arguments.
#
# The date of the transition from the Julian to Gregorian calendar and
# the number of invalid dates are hard-wired (1582-10-4 is the last day
# of the Julian calendar, after which follows 1582-10-15).
cdef tuple add_timedelta(datetime dt, delta, bint (*is_leap)(int), bint julian_gregorian_mixed):
    cdef int microsecond, second, minute, hour, day, month, year
    cdef int delta_microseconds, delta_seconds, delta_days
    cdef int* month_length
    cdef int extra_days, n_invalid_dates

    # extract these inputs here to avoid type conversion in the code below
    delta_microseconds = delta.microseconds
    delta_seconds = delta.seconds
    delta_days = delta.days

    # shift microseconds, seconds, days
    microsecond = dt.microsecond + delta_microseconds
    second = dt.second + delta_seconds
    minute = dt.minute
    hour = dt.hour
    day = dt.day
    month = dt.month
    year = dt.year

    # validate inputs:
    if year == 0:
        raise ValueError("invalid year in {0!r}".format(dt))

    month_length = month_lengths(is_leap, year)

    if month < 1 or month > 12:
        raise ValueError("invalid month in {0!r}".format(dt))

    if day < 1 or day > month_length[month]:
        raise ValueError("invalid day number in {0!r}".format(dt))

    if julian_gregorian_mixed and year == 1582 and month == 10 and day > 4 and day < 15:
        raise ValueError("{0!r} is not present in the mixed Julian/Gregorian calendar".format(dt))

    n_invalid_dates = 10 if julian_gregorian_mixed else 0

    # Normalize microseconds, seconds, minutes, hours.
    second += microsecond // 1000000
    microsecond = microsecond % 1000000
    minute += second // 60
    second = second % 60
    hour += minute // 60
    minute = minute % 60
    extra_days = hour // 24
    hour = hour % 24

    delta_days += extra_days

    while delta_days < 0:
        if year == 1582 and month == 10 and day > 14 and day + delta_days < 15:
            delta_days -= n_invalid_dates    # skip over invalid dates
        if day + delta_days < 1:
            delta_days += day
            # decrement month
            month -= 1
            if month < 1:
                month = 12
                year -= 1
                if year == 0:
                    year = -1
                month_length = month_lengths(is_leap, year)
            day = month_length[month]
        else:
            day += delta_days
            delta_days = 0

    while delta_days > 0:
        if year == 1582 and month == 10 and day < 5 and day + delta_days > 4:
            delta_days += n_invalid_dates    # skip over invalid dates
        if day + delta_days > month_length[month]:
            delta_days -= month_length[month] - (day - 1)
            # increment month
            month += 1
            if month > 12:
                month = 1
                year += 1
                if year == 0:
                    year = 1
                month_length = month_lengths(is_leap, year)
            day = 1
        else:
            day += delta_days
            delta_days = 0

    return (year, month, day, hour, minute, second, microsecond, -1, 1)

# Add a datetime.timedelta to a netcdftime.datetime instance with the 360_day calendar.
#
# Assumes that the 360_day calendar (unlike the rest of supported
# calendars) has the year 0. Also, there are no leap years and all
# months are 30 days long, so we can compute month and year by using
# "//" and "%".
cdef tuple add_timedelta_360_day(datetime dt, delta):
    cdef int microsecond, second, minute, hour, day, month, year
    cdef int delta_microseconds, delta_seconds, delta_days

    assert dt.month >= 1 and dt.month <= 12

    # extract these inputs here to avoid type conversion in the code below
    delta_microseconds = delta.microseconds
    delta_seconds = delta.seconds
    delta_days = delta.days

    # shift microseconds, seconds, days
    microsecond = dt.microsecond + delta_microseconds
    second = dt.second + delta_seconds
    minute = dt.minute
    hour = dt.hour
    day = dt.day + delta_days
    month = dt.month
    year = dt.year

    # Normalize microseconds, seconds, minutes, hours, days, and months.
    second += microsecond // 1000000
    microsecond = microsecond % 1000000
    minute += second // 60
    second = second % 60
    hour += minute // 60
    minute = minute % 60
    day += hour // 24
    hour = hour % 24
    # day and month are counted from 1; all months have 30 days
    month += (day - 1) // 30
    day = (day - 1) % 30 + 1
    # all years have 12 months
    year += (month - 1) // 12
    month = (month - 1) % 12 + 1

    return (year, month, day, hour, minute, second, microsecond, -1, 1)
