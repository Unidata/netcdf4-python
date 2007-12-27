import netCDF3

# code from tutorial.

# create a file (Dataset object, also the root group).
ncfile = netCDF3.Dataset('test.nc', 'w')
print ncfile.file_format
ncfile.close()

# add dimensions.
ncfile = netCDF3.Dataset('test.nc','a')
ncfile.createDimension('press', 10)
ncfile.createDimension('time', None)
ncfile.createDimension('lat', 73)
ncfile.createDimension('lon', 144)

print ncfile.dimensions

for dimname, dimobj in ncfile.dimensions.iteritems():
    print dimname, len(dimobj), dimobj.isunlimited()

# add variables.
times = ncfile.createVariable('time','f8',('time',))
pressure = ncfile.createVariable('press','i4',('press',))
latitudes = ncfile.createVariable('latitude','f4',('lat',))
longitudes = ncfile.createVariable('longitude','f4',('lon',))
temp = ncfile.createVariable('temp','f4',('time','press','lat','lon',))

# add attributes.
import time
ncfile.description = 'bogus example script'
ncfile.history = 'Created ' + time.ctime(time.time())
ncfile.source = 'netCDF3 python module tutorial'
latitudes.units = 'degrees north'
longitudes.units = 'degrees east'
pressure.units = 'hPa'
temp.units = 'K'
times.units = 'hours since 0001-01-01 00:00:00.0'
times.calendar = 'gregorian'

for name in ncfile.ncattrs():
    print 'Global attr', name, '=', getattr(ncfile,name)

print ncfile.__dict__

print ncfile.variables

import numpy
# no unlimited dimension, just assign to slice.
latitudes[:] = numpy.arange(-90,91,2.5)
print 'latitudes =\n',latitudes[:]
pressure[:] = range(1000,99,-100)
print 'pressure levels =\n',pressure[:]

# append unlimited dimensions by assigning to slice.
nlats = len(ncfile.dimensions['lat'])
nlons = len(ncfile.dimensions['lon'])
nlevs = len(ncfile.dimensions['press'])
print 'temp shape before adding data = ',temp.shape
from numpy.random.mtrand import uniform # random number generator.
temp[0:5,:,:,:] = uniform(size=(5,10,nlats,nlons))
print 'temp shape after adding data = ',temp.shape

# fill in times.
from datetime import datetime, timedelta
from netCDF3 import num2date, date2num
dates = [datetime(2001,3,1)+n*timedelta(hours=12) for n in range(temp.shape[0])]
times[:] = date2num(dates,units=times.units,calendar=times.calendar)
print 'time values (in units %s): ' % times.units+'\\n',times[:]
dates = num2date(times[:],units=times.units,calendar=times.calendar)
print 'dates corresponding to time values:\\n',dates

ncfile.close()

# create a series of netCDF files with a variable sharing
# the same unlimited dimension.
for nfile in range(10):
    f = netCDF3.Dataset('mftest'+repr(nfile)+'.nc','w')
    f.createDimension('x',None)
    x = f.createVariable('x','i',('x',))
    x[0:10] = numpy.arange(nfile*10,10*(nfile+1))
    f.close()
# now read all those files in at once, in one Dataset.
f = netCDF3.MFDataset('mftest*nc')
print f.variables['x'][:]
