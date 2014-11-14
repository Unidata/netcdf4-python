"""
Performs conversions of netCDF time coordinate data to/from datetime objects.
"""
import numpy as np
import math
import numpy
import re
from datetime import datetime as real_datetime
try:
    from itertools import izip as zip
except ImportError:  # python 3.x
    pass

from ._datetime import datetime

_units = ['days', 'hours', 'minutes', 'seconds',
          'day', 'hour', 'minute', 'second']
_calendars = ['standard', 'gregorian', 'proleptic_gregorian',
              'noleap', 'julian', 'all_leap', '365_day', '366_day', '360_day']

__version__ = '1.3'

# Adapted from http://delete.me.uk/2005/03/iso8601.html
ISO8601_REGEX = re.compile(r"(?P<year>[0-9]{1,4})(-(?P<month>[0-9]{1,2})(-(?P<day>[0-9]{1,2})"
                           r"(((?P<separator1>.)(?P<hour>[0-9]{1,2}):(?P<minute>[0-9]{1,2})(:(?P<second>[0-9]{1,2})(\.(?P<fraction>[0-9]+))?)?)?"
                           r"((?P<separator2>.?)(?P<timezone>Z|(([-+])([0-9]{1,2}):([0-9]{1,2}))))?)?)?)?"
                           )
TIMEZONE_REGEX = re.compile(
    "(?P<prefix>[+-])(?P<hours>[0-9]{1,2}):(?P<minutes>[0-9]{1,2})")


def JulianDayFromDate(date, calendar='standard'):
    """

    creates a Julian Day from a 'datetime-like' object.  Returns the fractional
    Julian Day (resolution 1 second).

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
    for i, d in enumerate(date):
        year[i] = d.year
        month[i] = d.month
        day[i] = d.day
        hour[i] = d.hour
        minute[i] = d.minute
        second[i] = d.second
    # Convert time to fractions of a day
    day = day + hour / 24.0 + minute / 1440.0 + second / 86400.0

    # Start Meeus algorithm (variables are in his notation)
    month_lt_3 = month < 3
    month[month_lt_3] = month[month_lt_3] + 12
    year[month_lt_3] = year[month_lt_3] - 1

    A = np.int64(year / 100)

    # MC
    # jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + \
    #      day - 1524.5
    jd = 365. * year + np.int32(0.25 * year + 2000.) + np.int32(30.6001 * (month + 1)) + \
        day + 1718994.5

    # optionally adjust the jd for the switch from
    # the Julian to Gregorian Calendar
    # here assumed to have occurred the day after 1582 October 4
    if calendar in ['standard', 'gregorian']:
        if np.min(jd) >= 2299170.5:
            # 1582 October 15 (Gregorian Calendar)
            B = 2 - A + np.int32(A / 4)
        elif np.max(jd) < 2299160.5:
            # 1582 October 5 (Julian Calendar)
            B = np.zeros(len(jd))
        else:
            raise ValueError(
                'impossible date (falls in gap between end of Julian calendar and beginning of Gregorian calendar')
    elif calendar == 'proleptic_gregorian':
        B = 2 - A + np.int32(A / 4)
    elif calendar == 'julian':
        B = np.zeros(len(jd))
    else:
        raise ValueError(
            'unknown calendar, must be one of julian,standard,gregorian,proleptic_gregorian, got %s' % calendar)

    # adjust for Julian calendar if necessary
    jd = jd + B

    if isscalar:
        return jd[0]
    else:
        return jd


def _NoLeapDayFromDate(date):
    """

creates a Julian Day for a calendar with no leap years from a datetime
instance.  Returns the fractional Julian Day (resolution 1 second).

    """

    year = date.year
    month = date.month
    day = date.day
    hour = date.hour
    minute = date.minute
    second = date.second
    # Convert time to fractions of a day
    day = day + hour / 24.0 + minute / 1440.0 + second / 86400.0

    # Start Meeus algorithm (variables are in his notation)
    if (month < 3):
        month = month + 12
        year = year - 1

    jd = int(365. * (year + 4716)) + int(30.6001 * (month + 1)) + \
        day - 1524.5

    return jd


def _AllLeapFromDate(date):
    """

creates a Julian Day for a calendar where all years have 366 days from
a 'datetime-like' object.
Returns the fractional Julian Day (resolution 1 second).

    """

    year = date.year
    month = date.month
    day = date.day
    hour = date.hour
    minute = date.minute
    second = date.second
    # Convert time to fractions of a day
    day = day + hour / 24.0 + minute / 1440.0 + second / 86400.0

    # Start Meeus algorithm (variables are in his notation)
    if (month < 3):
        month = month + 12
        year = year - 1

    jd = int(366. * (year + 4716)) + int(30.6001 * (month + 1)) + \
        day - 1524.5

    return jd


def _360DayFromDate(date):
    """

creates a Julian Day for a calendar where all months have 30 daysfrom
a 'datetime-like' object.
Returns the fractional Julian Day (resolution 1 second).

    """

    year = date.year
    month = date.month
    day = date.day
    hour = date.hour
    minute = date.minute
    second = date.second
    # Convert time to fractions of a day
    day = day + hour / 24.0 + minute / 1440.0 + second / 86400.0

    jd = int(360. * (year + 4716)) + int(30. * (month - 1)) + day

    return jd


def DateFromJulianDay(JD, calendar='standard'):
    """

    returns a 'datetime-like' object given Julian Day. Julian Day is a
    fractional day with a resolution of 1 second.

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
    Z = np.atleast_1d(np.int32(np.round(julian + 0.00005)))
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
    C = np.int32(6680. + ((B - 2439870.) - 122.1) / 365.25)
    D = np.int32(365 * C + np.int32(0.25 * C))
    E = np.int32((B - D) / 30.6001)

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
    leap = np.zeros(len(year))
    leap[year % 4 == 0] = 1
    if calendar == 'proleptic_gregorian':
        leap[(year % 100 == 0) & (year % 400 != 0)] = 0
    elif calendar in ['standard', 'gregorian']:
        leap[(year % 100 == 0) & (year % 400 != 0) & (julian < 2299160.5)] = 0

    inc_idx = np.where((leap == 1) & (month > 2))[0]
    dayofyr[inc_idx] = dayofyr[inc_idx] + leap[inc_idx]

    eps = np.clip(
        (1e-12 * np.abs(Z)).astype(np.float64), np.float64(1e-12), None)
    hour = np.clip((F * 24. + eps).astype(np.int64), 0, 23)
    F -= hour / 24.
    minute = np.clip((F * 1440. + eps).astype(np.int64), 0, 59)
    second = np.clip((F - minute / 1440.) * 86400., 0, None)
    second = second.astype(np.int32)
    microsecond = ((second - np.int32(second)) * 1e6).astype(np.int32)

    # convert year, month and day to int
    year = year.astype(np.int32)
    month = month.astype(np.int32)
    day = day.astype(np.int32)

    # check if input was scalar and change return accordingly
    isscalar = False
    try:
        JD[0]
    except:
        isscalar = True
    # return a 'real' datetime instance if calendar is gregorian.
    if calendar in 'proleptic_gregorian' or \
            (calendar in ['standard', 'gregorian'] and len(ind_before) == 0):
        if not isscalar:
            return np.array([real_datetime(*args)
                             for args in
                             zip(year, month, day, hour, minute,
                                 second, microsecond)])

        else:
            return real_datetime(year[0], month[0], day[0], hour[0],
                                 minute[0], second[0], microsecond[0])
    else:
        # or else, return a 'datetime-like' instance.
        if not isscalar:
            return np.array([datetime(*args)
                             for args in
                             zip(year, month, day, hour, minute,
                                 second, dayofwk, dayofyr)])
        else:
            return datetime(year[0], month[0], day[0], hour[0],
                            minute[0], second[0], dayofwk[0],
                            dayofyr[0])


def _DateFromNoLeapDay(JD):
    """

returns a 'datetime-like' object given Julian Day for a calendar with no leap
days. Julian Day is a fractional day with a resolution of 1 second.

    """

    # based on redate.py by David Finlayson.

    if JD < 0:
        raise ValueError('Julian Day must be positive')

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
    seconds = round(mfrac * 60.0)  # seconds are rounded

    if seconds > 59:
        seconds = 0
        minutes = minutes + 1
    if minutes > 59:
        minutes = 0
        hours = hours + 1
    if hours > 23:
        hours = 0
        days = days + 1

    return datetime(year, month, int(days), int(hours), int(minutes), int(seconds), dayofwk, dayofyr)


def _DateFromAllLeap(JD):
    """

returns a 'datetime-like' object given Julian Day for a calendar where all
years have 366 days.
Julian Day is a fractional day with a resolution of 1 second.

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
    seconds = round(mfrac * 60.0)  # seconds are rounded

    if seconds > 59:
        seconds = 0
        minutes = minutes + 1
    if minutes > 59:
        minutes = 0
        hours = hours + 1
    if hours > 23:
        hours = 0
        days = days + 1

    return datetime(year, month, int(days), int(hours), int(minutes), int(seconds), dayofwk, dayofyr)


def _DateFrom360Day(JD):
    """

returns a 'datetime-like' object given Julian Day for a calendar where all
months have 30 days.
Julian Day is a fractional day with a resolution of 1 second.

    """

    if JD < 0:
        raise ValueError('Julian Day must be positive')

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
    seconds = round(mfrac * 60.0)  # seconds are rounded

    if seconds > 59:
        seconds = 0
        minutes = minutes + 1
    if minutes > 59:
        minutes = 0
        hours = hours + 1
    if hours > 23:
        hours = 0
        days = days + 1

    return datetime(year, month, int(days), int(hours), int(minutes), int(seconds), -1, int(dayofyr))


def _dateparse(timestr):
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
2006-03-17 16:04:02.561678
>>>
>>> t = cdftime.date2num(date)
>>> print t
17577328.0672
>>>
>>> date = cdftime.num2date(t)
>>> print date
2006-03-17 16:04:02
>>>

The resolution of the transformation operation is 1 second.

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
        if calendar in _calendars:
            self.calendar = calendar
        else:
            raise ValueError(
                "calendar must be one of %s, got '%s'" % (str(_calendars), calendar))
        units, tzoffset, self.origin = _dateparse(unit_string)
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

        Resolution is 1 second.

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
        # convert to desired units, add time zone offset.
        if self.units in ['second', 'seconds']:
            jdelta = jdelta * 86400. + self.tzoffset * 60.
        elif self.units in ['minute', 'minutes']:
            jdelta = jdelta * 1440. + self.tzoffset
        elif self.units in ['hour', 'hours']:
            jdelta = jdelta * 24. + self.tzoffset / 60.
        elif self.units in ['day', 'days']:
            jdelta = jdelta + self.tzoffset / 1440.
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

        Resolution is 1 second.

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
        # convert to desired units, remove time zone offset.
        if self.units in ['second', 'seconds']:
            jdelta = time_value / 86400. - self.tzoffset / 1440.
        elif self.units in ['minute', 'minutes']:
            jdelta = time_value / 1440. - self.tzoffset / 1440.
        elif self.units in ['hour', 'hours']:
            jdelta = time_value / 24. - self.tzoffset / 1440.
        elif self.units in ['day', 'days']:
            jdelta = time_value - self.tzoffset / 1440.
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


def _parse_timezone(tzstring):
    """Parses ISO 8601 time zone specs into tzinfo offsets

    Adapted from pyiso8601 (http://code.google.com/p/pyiso8601/)
    """
    if tzstring == "Z":
        return 0
    # This isn't strictly correct, but it's common to encounter dates without
    # timezones so I'll assume the default (which defaults to UTC).
    if tzstring is None:
        return 0
    m = TIMEZONE_REGEX.match(tzstring)
    prefix, hours, minutes = m.groups()
    hours, minutes = int(hours), int(minutes)
    if prefix == "-":
        hours = -hours
        minutes = -minutes
    return minutes + hours * 60.


def _parse_date(datestring):
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
    return int(groups["year"]), int(groups["month"]), int(groups["day"]),\
        int(groups["hour"]), int(groups["minute"]), int(groups["second"]),\
        tzoffset_mins


def date2num(dates, units, calendar='standard'):
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
    cdftime = utime(units, calendar=calendar)
    return cdftime.date2num(dates)


def num2date(times, units, calendar='standard'):
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
    cdftime = utime(units, calendar=calendar)
    return cdftime.num2date(times)


def _check_index(indices, times, nctime, calendar, select):
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
    correpond to the closest dates.
    """
    # Setting the calendar.
    if calendar == None:
        calendar = getattr(nctime, 'calendar', 'standard')
    times = date2num(dates, nctime.units, calendar=calendar)
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
    correpond to the closest times.
    """
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
                [nctime[i - 1] + nctime[i] for i in index[mismatch]]) / 2.
            index[mismatch] = index[mismatch] - 1 * nearest_to_left

        else:
            raise ValueError(
                "%s is not an option for the `select` argument." % select)

        # Correct for indices equal to -1
        index[before] = 0

    # convert numpy scalars or single element arrays to python ints.
    return _toscalar(index)


def _toscalar(a):
    if a.shape in [(), (1,)]:
        return a.item()
    else:
        return a
