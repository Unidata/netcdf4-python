from datetime import datetime as real_datetime
import operator
import re
import time

from cpython.object cimport PyObject_RichCompare


cdef class datetime(object):

    """
Phony datetime object which mimics the python datetime object,
but allows for dates that don't exist in the proleptic gregorian calendar.
Doesn't do timedelta operations, doesn't overload + and -.

Has strftime, timetuple and __repr__ methods.  The format
of the string produced by __repr__ is controlled by self.format
(default %Y-%m-%d %H:%M:%S). Does support comparisons with other
phony datetime and with datetime.datetime objects.

Instance variables are year,month,day,hour,minute,second,dayofwk,dayofyr
and format.
    """
    cdef readonly int year, month, day, hour, minute, dayofwk, dayofyr
    cdef readonly int second, microsecond

    def __init__(self, year, month, day, hour=0, minute=0, second=0,
                 microsecond=0,dayofwk=-1, dayofyr=1):
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

    property format:
        def __get__(self):
            return '%Y-%m-%d %H:%M:%S'

    def strftime(self, format=None):
        if format is None:
            format = self.format
        return _strftime(self, format)

    def timetuple(self):
        return (self.year, self.month, self.day, self.hour,
                self.minute, self.second, self.dayofwk, self.dayofyr, -1)

    def _to_real_datetime(self):
        return real_datetime(self.year, self.month, self.day,
                             self.hour, self.minute, self.second,
                             self.microsecond)

    def __repr__(self):
        return self.strftime(self.format)

    def __hash__(self):
        try:
            d = self._to_real_datetime()
        except ValueError:
            return hash(self.timetuple())
        return hash(d)

    def __richcmp__(self, other, int op):
        if hasattr(other, 'strftime'):
            self_str = self.strftime('%Y-%m-%d %H:%M:%S')
            other_str = other.strftime('%Y-%m-%d %H:%M:%S')
            return PyObject_RichCompare(self_str, other_str, op)
        return NotImplemented


_illegal_s = re.compile(r"((^|[^%])(%%)*%s)")


def _findall(text, substr):
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


def _strftime(dt, fmt):
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
