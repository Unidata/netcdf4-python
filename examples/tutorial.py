import netCDF4

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

# ragged arrays (vlens).
vleni4 = rootgrp.createUserType('i4', 'vlen', 'vlen_i4')
ragged = rootgrp.createVariable('ragged',vleni4,('lat','lon'))

import random
data = NP.empty(nlats*nlons,'O')
for n in range(nlats*nlons):
    data[n] = NP.arange(random.randint(1,10))+1
data = NP.reshape(data,(nlats,nlons))
ragged[:] = data
print 'ragged array variable =\n',ragged[0:3,0:3]

# compound data types.
# create an unlimited  dimension call 'station'
rootgrp.createDimension('station',False)
# define a compound data type (a list of 3-tuples containing
# the name of each member, it's primitive data type, and it's size).
# Only fixed-size primitive data types allowed (no 'S').
# Members can be multi-dimensional arrays (in which case the third
# element is a shape tuple instead of a scalar).
datatype = [('latitude', 'f4',1), ('longitude', 'f4',1),('sfc_press','i4',1),
            ('temp_sounding','f4',10),('press_sounding','i4',10),
            ('location_name','S1',80)]
# use this data type definition to create a user-defined data type
# called 'station_data'
table = rootgrp.createUserType(datatype,'compound','station_data')
# create a variable of of type 'station_data'
statdat = rootgrp.createVariable('station_obs', table, ('station',))
# create record array, assign data to it.
ra = NP.empty(1,statdat.dtype_base)
ra['latitude'] = 40.
ra['longitude'] = -105.
ra['sfc_press'] = 818
ra['temp_sounding'] = (280.3,272.,270.,269.,266.,258.,254.1,250.,245.5,240.)
ra['press_sounding'] = range(800,300,-50)
# only fixed-size primitive data types can currenlty be used
# as compound data type members (although the library supports
# nested compound types).
# To store strings in a compound data type, each string must be
# stored as fixed-size (in this case 80) array of characters.
def stringtoarr(string,NUMCHARS):
    """function to convert a string to a array of NUMCHARS characters"""
    arr = NP.zeros(NUMCHARS,'S1')
    arr[0:len(string)] = tuple(string)
    return arr
ra['location_name'] = stringtoarr('Boulder, Colorado, USA',80)
# assign record array to variable slice.
statdat[0] = ra
# or just assign a tuple of values to variable slice
# (will automatically be converted to a record array).
statdat[1] = (40.78,-73.99,1002,
            (290.2,282.5,279.,277.9,276.,266.,264.1,260.,255.5,243.),
            range(900,400,-50),stringtoarr('New York, New York, USA',80))
# this module doesn't support attributes of compound type.
# so, to assign an attribute like 'units' to each member of 
# the compound type I do the following:
# 1) create a python dict with key/value pairs representing
#    the name of each compound type member and it's units.
# 2) convert the dict to a string using the repr function.
# 3) use that string as a variable attribute.
# When this attribute is read back in it can be converted back to
# a python dictionary using the eval function..
# This can be converted into hash-like objects in other languages
# as well (including C), since this string is also valid JSON
# (JavaScript Object Notation - http://json.org). 
# JSON is a lightweight, language-independent data serialization format.
units_dict = {'latitude': 'degrees north', 'longitude': 'degrees east',
              'sfc_press': 'Pascals', 'temp_sounding': 'Kelvin',
              'press_sounding': 'Pascals','location_name': None}
statdat.units = repr(units_dict)
# convert units string back to a python dictionary.
statdat_units = eval(statdat.units)
# print out data in variable (including units attribute)
print 'data in a variable of compound type:\n----'
for data in statdat[:]:
   for item in statdat.dtype_base:
       name = item[0]
       type = item[1]
       if type == 'S1': # if array of chars, convert value to string.
           print name,': value =',data[name].tostring(),'units =',statdat_units[name]
       else:
           print name,': value =',data[name],'units =',statdat_units[name]
   print '----'

# storing arbitrary python objects as pickled strings.
strvar = rootgrp.createVariable('strvar','S',('level',))
chars = '1234567890aabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
data = NP.empty(10,'O')
for n in range(10):
    stringlen = random.randint(2,12)
    data[n] = ''.join([random.choice(chars) for i in range(stringlen)])
data[0] = {'spam':1,'eggs':2,'ham':False}
strvar[:] = data
print 'string variable with embedded python objects: \n',strvar[:]
strvar.timestamp = datetime.now()
print strvar.timestamp

rootgrp.close()
