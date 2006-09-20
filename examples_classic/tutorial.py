import netCDF4_classic as netCDF
# code from tutorial.

# create a file (Dataset object, also the root group).
dataset = netCDF.Dataset('test.nc', 'w')
print dataset.file_format
dataset.close()

# dimensions.
dataset = netCDF.Dataset('test.nc', 'a')
print dataset.file_format
dataset.createDimension('time', None)
dataset.createDimension('level', 10)
dataset.createDimension('lat', 73)
dataset.createDimension('lon', 144)

print dataset.dimensions

for dimname, dimobj in dataset.dimensions.iteritems():
    print dimname, len(dimobj), dimobj.isunlimited()

# variables.
times = dataset.createVariable('time','f8',('time',))
levels = dataset.createVariable('level','i4',('level',))
latitudes = dataset.createVariable('latitude','f4',('lat',))
longitudes = dataset.createVariable('longitude','f4',('lon',))
temp = dataset.createVariable('temp','f4',('time','level','lat','lon',))
# this makes the compression 'lossy' (preserving a precision of 1/1000)
# try it and see how much smaller the file gets.
#temp = dataset.createVariable('temp','f4',('time','level','lat','lon',),least_significant_digit=3)

# attributes.
import time
dataset.description = 'bogus example script'
dataset.history = 'Created ' + time.ctime(time.time())
dataset.source = 'netCDF4_classic python module tutorial'
latitudes.units = 'degrees north'
longitudes.units = 'degrees east'
temp.units = 'K'
times.units = 'hours since January 1, 0001'
times.calendar = 'proleptic_gregorian'

for name in dataset.ncattrs():
    print 'Global attr', name, '=', getattr(dataset,name)

from datetime import datetime
dataset.timestamp = datetime.now()
print 'Global attr timestamp =',dataset.timestamp

print dataset.__dict__

print dataset.variables

import numpy as NP
# no unlimited dimension, just assign to slice.
latitudes[:] = NP.arange(-90,91,2.5)
print 'latitudes =\n',latitudes[:]

# append along two unlimited dimensions by assigning to slice.
nlats = len(dataset.dimensions['lat'])
nlons = len(dataset.dimensions['lon'])
nlevs = len(dataset.dimensions['level'])
print 'temp shape before adding data = ',temp.shape
from numpy.random.mtrand import uniform # random number generator.
temp[0:5,:,:,:] = uniform(size=(5,10,nlats,nlons))
print 'temp shape after adding data = ',temp.shape
# times have grown, but no values yet assigned.
print 'times shape after adding pressure data = ',times.shape

# fill in times.
from datetime import timedelta
from netcdftime import utime
cdftime = utime(times.units,calendar=times.calendar,format='%B %d, %Y') 
dates = [datetime(2001,3,1)+n*timedelta(hours=12) for n in range(temp.shape[0])]
times[:] = cdftime.date2num(dates)
print 'time values (in units %s): ' % times.units+'\n',times[:]
dates = cdftime.num2date(times[:])
print 'dates corresponding to time values:\n',dates

dataset.close()
