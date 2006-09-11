from netcdftime import utime, datetime
import numpy

# test netcdftime module for netCDF time <--> python datetime conversions.

cdftime = utime('hours since 0001-01-01 00:00:00')
print cdftime.units,'since',cdftime.origin,'=',cdftime.unit_string
print cdftime.calendar,'calendar'
d = datetime(1582,10,4,23)
t1 = cdftime.date2num(d)
print t1,'=',13865687.0
# should raise exception
try:
    cdftime.date2num(datetime(1582,10,5,0))
except:
    print 'caught illegal date'
else:
    print 'error - did not raise an exception for illegal date.'
date = cdftime.num2date(t1)
print date,'=',d
d = datetime(1582,10,15,0)
t2 = cdftime.date2num(d)
print t2,'=',13865688.0
date = cdftime.num2date(t2)
print date,'=',d
print 'day of year = ',date.timetuple()[7],'=',288
print 'testing arrays'
t = numpy.arange(t2,t2+240.0,12.)
t = numpy.reshape(t,(4,5))
print t
print t.shape
d = cdftime.num2date(t)
print d
print d.shape
print d.dtype
print
cdftime = utime('seconds since 0001-01-01 00:00:00',
                     calendar='proleptic_gregorian')
print cdftime.units,'since',cdftime.origin,'=',cdftime.unit_string
print cdftime.calendar,'calendar'
d = datetime(1990,5,5,2,17)
print d
t = cdftime.date2num(d)
print t,'=',62777470620.0
date = cdftime.num2date(t)
print date,'=',d
print cdftime.date2num(cdftime.origin),'=',0.0
print 'day of year = ',date.timetuple()[7],'=',125
cdftime = utime('days since 1600-02-28 00:00:00',calendar='noleap')
# should raise an exception
try:
    cdftime = utime('days since 1600-02-29 00:00:00',calendar='noleap')
except:
    print 'caught illegal unit specification'
else:
    print 'error - did not raise an exception for illegal unit specification'
print cdftime.units,'since',cdftime.origin,'=',cdftime.unit_string
print cdftime.calendar,'calendar'
print cdftime.date2num(cdftime.origin),'=',0.0
date = datetime(2000,2,28)
t = cdftime.date2num(date)
# should raise an exception.
try:
    date = datetime(2000,2,29)
    t = cdftime.date2num(date)
except:
    print 'caught illegal date'
else:
    print 'error - did not raise an exception for illegal date.'
print t,'=',400.*365
print cdftime.num2date(t),'=',datetime(2000,2,28)
print cdftime.num2date(0),'=',datetime(1600,2,28)
date = cdftime.num2date(t)
print 'day of year = ',date.timetuple()[7],'=',59
date = datetime(2000,12,31)
t = cdftime.date2num(date)
date = cdftime.num2date(t)
print 'day of year = ',date.timetuple()[7],'=',366
print
cdftime = utime('days since 1600-02-29 00:00:00',calendar='all_leap')
print cdftime.units,'since',cdftime.origin,'=',cdftime.unit_string
print cdftime.calendar,'calendar'
print cdftime.date2num(cdftime.origin),'=',0.0
date = datetime(2000,2,29)
t = cdftime.date2num(date)
print t,'=',400.*366
print cdftime.num2date(t),'=',datetime(2000,2,29)
print cdftime.num2date(0),'=',datetime(1600,2,29)
date = cdftime.num2date(t)
print 'day of year = ',date.timetuple()[7],'=',60
d = datetime(2000,12,31)
t = cdftime.date2num(d)
date = cdftime.num2date(t)
print date,'=',d
print 'day of year = ',date.timetuple()[7],'=',366
print
cdftime = utime('days since 1600-02-30 00:00:00',calendar='360_day')
print cdftime.units,'since',cdftime.origin,'=',cdftime.unit_string
print cdftime.calendar,'calendar'
print cdftime.date2num(cdftime.origin),'=',0.0
date = datetime(2000,2,30)
t = cdftime.date2num(date)
print t,'=',400.*360
print cdftime.num2date(t),'=',datetime(2000,2,30)
print cdftime.num2date(0),'=',datetime(1600,2,30)
date = cdftime.num2date(t)
print 'day of year = ',date.timetuple()[7],'=',60
d = datetime(2001,12,30)
t = cdftime.date2num(d)
date = cdftime.num2date(t)
print date,'=',d
print 'day of year = ',date.timetuple()[7],'=',360
