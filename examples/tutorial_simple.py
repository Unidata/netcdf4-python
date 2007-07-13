import netCDF4_simple as netCDF4

# code from tutorial.

# create a file (Dataset object, also the root group).
rootgrp = netCDF4.Dataset('test.nc', 'w')
print rootgrp.file_format
rootgrp.close()

# create some groups.
rootgrp = netCDF4.Dataset('test.nc', 'a')
fcstgrp = rootgrp.createGroup('forecasts')
analgrp = rootgrp.createGroup('analyses')
print rootgrp.groups
fcstgrp1 = fcstgrp.createGroup('model1')
fcstgrp2 = fcstgrp.createGroup('model2')

# walk the group tree using a Python generator.
def walktree(top):
    values = top.groups.values()
    yield values
    for value in top.groups.values():
        for children in walktree(value):
            yield  children
print rootgrp.path, rootgrp
for children in walktree(rootgrp):
     for child in children:
         print child.path, child

# dimensions.
rootgrp.createDimension('level', None)
rootgrp.createDimension('time', None)
rootgrp.createDimension('lat', 73)
rootgrp.createDimension('lon', 144)

print rootgrp.dimensions

for dimname, dimobj in rootgrp.dimensions.iteritems():
    print dimname, len(dimobj), dimobj.isunlimited()

# variables.
times = rootgrp.createVariable('time','f8',('time',))
levels = rootgrp.createVariable('level','i4',('level',))
latitudes = rootgrp.createVariable('latitude','f4',('lat',))
longitudes = rootgrp.createVariable('longitude','f4',('lon',))
# 2 unlimited dimensions.
#temp = rootgrp.createVariable('temp','f4',('time','level','lat','lon',))
# this makes the compression 'lossy' (preserving a precision of 1/1000)
# try it and see how much smaller the file gets.
temp = rootgrp.createVariable('temp','f4',('time','level','lat','lon',),least_significant_digit=3)

# attributes.
import time
rootgrp.description = 'bogus example script'
rootgrp.history = 'Created ' + time.ctime(time.time())
rootgrp.source = 'netCDF4 python module tutorial'
latitudes.units = 'degrees north'
longitudes.units = 'degrees east'
temp.units = 'K'
times.units = 'hours since January 1, 0001'
times.calendar = 'proleptic_gregorian'

for name in rootgrp.ncattrs():
    print 'Global attr', name, '=', getattr(rootgrp,name)

print rootgrp.__dict__

print rootgrp.variables

import numpy as NP
# no unlimited dimension, just assign to slice.
latitudes[:] = NP.arange(-90,91,2.5)
print 'latitudes =\n',latitudes[:]

# append along two unlimited dimensions by assigning to slice.
nlats = len(rootgrp.dimensions['lat'])
nlons = len(rootgrp.dimensions['lon'])
print 'temp shape before adding data = ',temp.shape
from numpy.random.mtrand import uniform # random number generator.
temp[0:5,0:10,:,:] = uniform(size=(5,10,nlats,nlons))
print 'temp shape after adding data = ',temp.shape
# levels have grown, but no values yet assigned.
print 'levels shape after adding pressure data = ',levels.shape

# fill in times.
from datetime import datetime, timedelta
from netcdftime import utime
cdftime = utime(times.units,calendar=times.calendar,format='%B %d, %Y') 
dates = [datetime(2001,3,1)+n*timedelta(hours=12) for n in range(temp.shape[0])]
times[:] = cdftime.date2num(dates)
print 'time values (in units %s): ' % times.units+'\n',times[:]
dates = cdftime.num2date(times[:])
print 'dates corresponding to time values:\n',dates

rootgrp.close()
