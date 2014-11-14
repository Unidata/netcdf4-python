from datetime import datetime as real_datetime
import operator
import re
import time

class datetime(object):

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

    def __init__(self, year, month, day, hour=0, minute=0, second=0,
                 dayofwk=-1, dayofyr=1):
        """dayofyr set to 1 by default - otherwise time.strftime will complain"""

        self._year = year
        self._month = month
        self._day = day
        self._hour = hour
        self._minute = minute
        self._dayofwk = dayofwk
        self._dayofyr = dayofyr
        self._second = second
        self._format = '%Y-%m-%d %H:%M:%S'

    year = property(lambda self: self._year)
    month = property(lambda self: self._month)
    day = property(lambda self: self._day)
    hour = property(lambda self: self._hour)
    minute = property(lambda self: self._minute)
    dayofwk = property(lambda self: self._dayofwk)
    dayofyr = property(lambda self: self._dayofyr)
    second = property(lambda self: self._second)
    format = property(lambda self: self._format)

    def strftime(self, format=None):
        if format is None:
            format = self.format
        return _strftime(self, format)

    def timetuple(self):
        return (self.year, self.month, self.day, self.hour,
                self.minute, self.second, self.dayofwk, self.dayofyr, -1)

    def _to_real_datetime(self):
        return real_datetime(self._year, self._month, self._day,
                             self._hour, self._minute, self._second)

    def __repr__(self):
        return self.strftime(self.format)

    def __hash__(self):
        try:
            d = self._to_real_datetime()
        except ValueError:
            return hash(tuple(sorted(self.__dict__.items())))
        return hash(d)

    def _compare(self, comparison_op, other):
        if hasattr(other, 'strftime'):
            return comparison_op(self.strftime('%Y-%m-%d %H:%M:%S'),
                                 other.strftime('%Y-%m-%d %H:%M:%S'))
        return NotImplemented

    def __eq__(self, other):
        return self._compare(operator.eq, other)

    def __ne__(self, other):
        return self._compare(operator.ne, other)

    def __lt__(self, other):
        return self._compare(operator.lt, other)

    def __le__(self, other):
        return self._compare(operator.le, other)

    def __gt__(self, other):
        return self._compare(operator.gt, other)

    def __ge__(self, other):
        return self._compare(operator.ge, other)


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
