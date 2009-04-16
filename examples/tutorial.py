import netCDF4

# code from tutorial.

# create a file (Dataset object, also the root group).
rootgrp = netCDF4.Dataset('test.nc', 'w', format='NETCDF4')
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
times.units = 'hours since 0001-01-01 00:00:00.0'
times.calendar = 'gregorian'

for name in rootgrp.ncattrs():
    print 'Global attr', name, '=', getattr(rootgrp,name)

print rootgrp.__dict__

print rootgrp.variables

import numpy
# no unlimited dimension, just assign to slice.
lats =  numpy.arange(-90,91,2.5)
lons =  numpy.arange(-180,180,2.5)
latitudes[:] = lats
longitudes[:] = lons
print 'latitudes =\n',latitudes[:]
print 'longitudes =\n',longitudes[:]

# append along two unlimited dimensions by assigning to slice.
nlats = len(rootgrp.dimensions['lat'])
nlons = len(rootgrp.dimensions['lon'])
print 'temp shape before adding data = ',temp.shape
from numpy.random.mtrand import uniform # random number generator.
temp[0:20,0:10,:,:] = uniform(size=(20,10,nlats,nlons))
print 'temp shape after adding data = ',temp.shape
# levels have grown, but no values yet assigned.
print 'levels shape after adding pressure data = ',levels.shape

# assign values to levels dimension variable.
levels[:] =  [1000.,850.,700.,500.,300.,250.,200.,150.,100.,50.]
# fancy slicing
tempdat = temp[10:20:2, [1,3,6], lats>0, lons>0]
print 'shape of fancy temp slice = ',tempdat.shape
print temp[0, 0, [0,1,2,3], [0,1,2,3]].shape

# fill in times.
from datetime import datetime, timedelta
from netCDF4 import num2date, date2num, date2index
dates = [datetime(2001,3,1)+n*timedelta(hours=12) for n in range(temp.shape[0])]
times[:] = date2num(dates,units=times.units,calendar=times.calendar)
print 'time values (in units %s): ' % times.units+'\\n',times[:]
dates = num2date(times[:],units=times.units,calendar=times.calendar)
print 'dates corresponding to time values:\\n',dates

rootgrp.close()

# create a series of netCDF files with a variable sharing
# the same unlimited dimension.
for nfile in range(10):
    f = netCDF4.Dataset('mftest'+repr(nfile)+'.nc','w',format='NETCDF4_CLASSIC')
    f.createDimension('x',None)
    x = f.createVariable('x','i',('x',))
    x[0:10] = numpy.arange(nfile*10,10*(nfile+1))
    f.close()
# now read all those files in at once, in one Dataset.
f = netCDF4.MFDataset('mftest*nc')
print f.variables['x'][:]

from netCDF4 import chartostring, stringtoarr
rootgrp = netCDF4.Dataset('compound_example.nc','w')
# create an unlimited  dimension call 'station'
rootgrp.createDimension('station',None)
# define a compound data type (can contain arrays, or nested compound types).
winddtype = numpy.dtype([('speed','f4'),('direction','i4')])
statdtype = numpy.dtype([('latitude', 'f4'), ('longitude', 'f4'),\
            ('surface_wind',winddtype),\
            ('temp_sounding','f4',10),('press_sounding','i4',10),
            ('location_name','S1',80)])
print statdtype
# use this data type definition to create a compound data type
# called 'station_data_t'
wind_data_t = rootgrp.createCompoundType(winddtype,'wind_data')
station_data_t = rootgrp.createCompoundType(statdtype,'station_data')
# create nested compound data types to hold units.
winddtype_units = numpy.dtype([('speed','S1',(80,)),('direction','S1',(80,))])
statdtype_units = numpy.dtype([('latitude', 'S1',(80,)), ('longitude', 'S1',(80,)),\
            ('surface_wind',winddtype_units),\
            ('temp_sounding','S1',(80,)),('press_sounding','S1',(80,))])
wind_data_units_t = rootgrp.createCompoundType(winddtype_units,'wind_data_units')
station_data_units_t =\
rootgrp.createCompoundType(statdtype_units,'station_data_units')
# create a variable of of type 'station_data_t'
statdat = rootgrp.createVariable('station_obs', station_data_t, ('station',))
# create a numpy structured array, assign data to it.
data = numpy.empty(1,station_data_t)
data['latitude'] = 40.
data['longitude'] = -105.
data['surface_wind']['speed'] = 12.5
data['surface_wind']['direction'] = 270
data['temp_sounding'] = (280.3,272.,270.,269.,266.,258.,254.1,250.,245.5,240.)
data['press_sounding'] = range(800,300,-50)
# variable-length string datatypes are not supported, so
# to store strings in a compound data type, each string must be 
# stored as fixed-size (in this case 80) array of characters.
NUMCHARS = statdtype.fields['location_name'][0].itemsize
data['location_name'] = stringtoarr('Boulder, Colorado, USA',NUMCHARS)
# assign structured array to variable slice.
statdat[0] = data
# or just assign a tuple of values to variable slice
# (will automatically be converted to a structured array).
statdat[1] = (40.78,-73.99,(-12.5,90),\
            (290.2,282.5,279.,277.9,276.,266.,264.1,260.,255.5,243.),\
            range(900,400,-50),stringtoarr('New York, New York, USA',NUMCHARS))
windunits = numpy.empty(1,winddtype_units)
stationobs_units = numpy.empty(1,statdtype_units)
windunits['speed'] = stringtoarr('m/s',80)
windunits['direction'] = stringtoarr('degrees',80)
stationobs_units['latitude'] = stringtoarr('degrees north',80)
stationobs_units['longitude'] = stringtoarr('degrees west',80)
stationobs_units['surface_wind'] = windunits[:]
stationobs_units['temp_sounding'] = stringtoarr('Kelvin',80)
stationobs_units['press_sounding'] = stringtoarr('hPa',80)
statdat.units = stationobs_units
# close and reopen the file.
rootgrp.close(); rootgrp = netCDF4.Dataset('compound_example.nc')
statdat = rootgrp.variables['station_obs']
# print out data in variable.
print 'data in a variable of compound type:\\n----'
for data in statdat[:]:
    for name in statdat.dtype.names:
        try:
            # convert array of characters back to a string for display.
            print name,': value =',chartostring(data[name])
        except:
            print name,': value =',data[name]
    print '----'
rootgrp.close()
