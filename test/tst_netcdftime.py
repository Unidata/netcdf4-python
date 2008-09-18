from netcdftime import utime, JulianDayFromDate,DateFromJulianDay, date2index
from netcdftime import datetime as datetimex, date2num
import numpy
import random
import sys
import unittest
import os
from datetime import datetime
from numpy.testing import assert_almost_equal, assert_equal

# test netcdftime module for netCDF time <--> python datetime conversions.

class netcdftimeTestCase(unittest.TestCase):

    def setUp(self):
        self.cdftime_mixed = utime('hours since 0001-01-01 00:00:00')
        self.cdftime_pg = utime('seconds since 0001-01-01 00:00:00',
                          calendar='proleptic_gregorian')
        self.cdftime_noleap = utime('days since 1600-02-28 00:00:00',calendar='noleap')
        self.cdftime_leap = utime('days since 1600-02-29 00:00:00',calendar='all_leap')
        self.cdftime_360day = utime('days since 1600-02-30 00:00:00',calendar='360_day')
        self.cdftime_jul = utime('hours since 1000-01-01 00:00:00',calendar='julian')

    def runTest(self):
        """testing netcdftime"""
        # test mixed julian/gregorian calendar
        # check attributes.
        self.assert_(self.cdftime_mixed.units == 'hours')
        self.assert_(repr(self.cdftime_mixed.origin) == '   1-01-01 00:00:00')
        self.assert_(self.cdftime_mixed.unit_string == 'hours since 0001-01-01 00:00:00')
        self.assert_(self.cdftime_mixed.calendar == 'standard')
        # check date2num method. (date before switch)
        d = datetime(1582,10,4,23)
        t1 = self.cdftime_mixed.date2num(d)
        assert_almost_equal(t1,13865687.0)
        # check num2date method.
        d2 = self.cdftime_mixed.num2date(t1)
        self.assert_(str(d) == str(d2))
        # this is a non-existant date, should raise ValueError.
        d = datetime(1582,10,5,0)
        self.assertRaises(ValueError, self.cdftime_mixed.date2num, d)
        # check date2num/num2date with date after switch.
        d = datetime(1582,10,15,0)
        t2 = self.cdftime_mixed.date2num(d)
        assert_almost_equal(t2,13865688.0)
        d2 = self.cdftime_mixed.num2date(t2)
        self.assert_(str(d) == str(d2))
        # check day of year.
        ndayr = d.timetuple()[7]
        self.assert_(ndayr == 288)
        # test using numpy arrays.
        t = numpy.arange(t2,t2+240.0,12.)
        t = numpy.reshape(t,(4,5))
        d = self.cdftime_mixed.num2date(t)
        self.assert_(d.shape == t.shape)
        d_check = "1582-10-15 00:00:001582-10-15 12:00:001582-10-16 00:00:001582-10-16 12:00:001582-10-17 00:00:001582-10-17 12:00:001582-10-18 00:00:001582-10-18 12:00:001582-10-19 00:00:001582-10-19 12:00:001582-10-20 00:00:001582-10-20 12:00:001582-10-21 00:00:001582-10-21 12:00:001582-10-22 00:00:001582-10-22 12:00:001582-10-23 00:00:001582-10-23 12:00:001582-10-24 00:00:001582-10-24 12:00:00"
        d2 = [str(dd) for dd in d.flat]
        self.assert_(d_check == ''.join(d2))
        # test proleptic gregorian calendar.
        self.assert_(self.cdftime_pg.units == 'seconds')
        self.assert_(repr(self.cdftime_pg.origin) == '   1-01-01 00:00:00')
        self.assert_(self.cdftime_pg.unit_string == 'seconds since 0001-01-01 00:00:00')
        self.assert_(self.cdftime_pg.calendar == 'proleptic_gregorian')
        # check date2num method.
        d = datetime(1990,5,5,2,17)
        t1 = numpy.around(self.cdftime_pg.date2num(d))
        self.assert_(t1 == 62777470620.0)
        # check num2date method.
        d2 = self.cdftime_pg.num2date(t1)
        self.assert_(str(d) == str(d2))
        # check day of year.
        ndayr = d.timetuple()[7]
        self.assert_(ndayr == 125)
        # check noleap calendar.
        # this is a non-existant date, should raise ValueError.
        self.assertRaises(ValueError,utime,'days since 1600-02-29 00:00:00',calendar='noleap')
        self.assert_(self.cdftime_noleap.units == 'days')
        self.assert_(repr(self.cdftime_noleap.origin) == '1600-02-28 00:00:00')
        self.assert_(self.cdftime_noleap.unit_string == 'days since 1600-02-28 00:00:00')
        self.assert_(self.cdftime_noleap.calendar == 'noleap')
        assert_almost_equal(self.cdftime_noleap.date2num(self.cdftime_noleap.origin),0.0)
        # check date2num method.
        d1 = datetime(2000,2,28)
        d2 = datetime(1600,2,28)
        t1 = self.cdftime_noleap.date2num(d1)
        t2 = self.cdftime_noleap.date2num(d2)
        assert_almost_equal(t1,400*365.)
        assert_almost_equal(t2,0.)
        # check num2date method.
        d2 = self.cdftime_noleap.num2date(t1)
        self.assert_(str(d1) == str(d2))
        # check day of year.
        ndayr = d2.timetuple()[7]
        self.assert_(ndayr == 59)
        # non-existant date, should raise ValueError.
        date = datetime(2000,2,29)
        self.assertRaises(ValueError,self.cdftime_noleap.date2num,date)
        # check all_leap calendar.
        self.assert_(self.cdftime_leap.units == 'days')
        self.assert_(repr(self.cdftime_leap.origin) == '1600-02-29 00:00:00')
        self.assert_(self.cdftime_leap.unit_string == 'days since 1600-02-29 00:00:00')
        self.assert_(self.cdftime_leap.calendar == 'all_leap')
        assert_almost_equal(self.cdftime_leap.date2num(self.cdftime_leap.origin),0.0)
        # check date2num method.
        d1 = datetime(2000,2,29)
        d2 = datetime(1600,2,29)
        t1 = self.cdftime_leap.date2num(d1)
        t2 = self.cdftime_leap.date2num(d2)
        assert_almost_equal(t1,400*366.)
        assert_almost_equal(t2,0.)
        # check num2date method.
        d2 = self.cdftime_leap.num2date(t1)
        self.assert_(str(d1) == str(d2))
        # check day of year.
        ndayr = d2.timetuple()[7]
        self.assert_(ndayr == 60)
        # double check date2num,num2date methods.
        d = datetime(2000,12,31)
        t1 = self.cdftime_mixed.date2num(d)
        d2 = self.cdftime_mixed.num2date(t1)
        self.assert_(str(d) == str(d2))
        ndayr = d2.timetuple()[7]
        self.assert_(ndayr == 366)
        # check 360_day calendar.
        self.assert_(self.cdftime_360day.units == 'days')
        self.assert_(repr(self.cdftime_360day.origin) == '1600-02-30 00:00:00')
        self.assert_(self.cdftime_360day.unit_string == 'days since 1600-02-30 00:00:00')
        self.assert_(self.cdftime_360day.calendar == '360_day')
        assert_almost_equal(self.cdftime_360day.date2num(self.cdftime_360day.origin),0.0)
        # check date2num,num2date methods.
        # use datetime from netcdftime, since this date doesn't
        # exist in "normal" calendars.
        d = datetimex(2000,2,30) 
        t1 = self.cdftime_360day.date2num(d)
        assert_almost_equal(t1,360*400.)
        d2 = self.cdftime_360day.num2date(t1)
        self.assert_(str(d) == str(d2))
        # check day of year.
        d = datetime(2001,12,30)
        t = self.cdftime_360day.date2num(d)
        assert_almost_equal(t,144660.0)
        date = self.cdftime_360day.num2date(t)
        self.assert_(str(d) == str(date))
        ndayr = date.timetuple()[7]
        self.assert_(ndayr == 360)
        # test proleptic julian calendar.
        d = datetime(1858,11,17,12)
        t = self.cdftime_jul.date2num(d)
        assert_almost_equal(t,7528932.0)
        d1 = datetime(1582,10,4,23)
        d2 = datetime(1582,10,15,0)
        assert_almost_equal(self.cdftime_jul.date2num(d1)+241.0,self.cdftime_jul.date2num(d2))
        date = self.cdftime_jul.num2date(t)
        self.assert_(str(d) == str(date))
        # test julian day from date, date from julian day
        d = datetime(1858,11,17)
        mjd = JulianDayFromDate(d)
        assert_almost_equal(mjd,2400000.5)
        date = DateFromJulianDay(mjd)
        self.assert_(str(date) == str(d))


class TestDate2index(unittest.TestCase):

    class TestTime:
        """Fake a netCDF time variable."""
        def __init__(self, start, n, step, units, calendar='standard'):
            """Create an object that fakes a netCDF time variable.

            Internally, this object has a _data array attribute whose values
            corresponds to dates in the given units and calendar. `start`, `n`
            and `step` define the starting date, the length of the array and
            the distance between each date (in units).
           
            :Example:
            >>> t = TestTime(datetime(1989, 2, 18), 45, 6, 'hours since 1979-01-01')
            >>> print num2date(t[1], t.units)
            1989-02-18 06:00:00
            """
            self.units = units
            self.calendar = calendar
            t0 = date2num(start, units, calendar)
            self._data = (t0 + numpy.arange(n)*step).astype('float')

        def __getitem__(self, item):
            return self._data[item]

    def setUp(self):
        self.standardtime = self.TestTime(datetime(1950, 1, 1), 366, 24,
          'hours since 1900-01-01', 'standard')

    def test_simple(self):
        t = date2index(datetime(1950,2,1), self.standardtime)
        assert_equal(t, 31)
   
    def test_list(self):
        t = date2index([datetime(1950,2,1), datetime(1950,2,3)], self.standardtime)
        assert_equal(t, [31, 33])

    def test_nonuniform(self):
        """Check that the fallback mechanism works. """
        nutime = self.TestTime(datetime(1950, 1, 1), 366, 24,
          'hours since 1900-01-01', 'standard')
        # Let's remove the second entry, so that the computed stride is not
        # representative and the brute force method is needed.
        nutime._data = nutime._data[numpy.r_[0,slice(2,200)]]
       
        t = date2index(datetime(1950,2,1), nutime)
        assert_equal(t, 30)       

       
        t = date2index([datetime(1950,2,1), datetime(1950,2,3)], nutime)
        assert_equal(t, [30, 32])       

if __name__ == '__main__':
    unittest.main()
