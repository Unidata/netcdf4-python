from typing import Literal
from netCDF4 import Dataset

# code from tutorial.

# create a file (Dataset object, also the root group).
rootgrp = Dataset('test.nc', 'w', format='NETCDF4')
print(rootgrp.file_format)
rootgrp.close()

# create some groups.
rootgrp = Dataset('test.nc', 'a')
fcstgrp = rootgrp.createGroup('forecasts')
analgrp = rootgrp.createGroup('analyses')
fcstgrp1 = rootgrp.createGroup('/forecasts/model1')
fcstgrp2 = rootgrp.createGroup('/forecasts/model2')

# walk the group tree using a Python generator.
def walktree(top):
    yield top.groups.values()
    for value in top.groups.values():
        yield from walktree(value)

print(rootgrp)
for children in walktree(rootgrp):
    for child in children:
        print(child)

# dimensions.
level_dim = rootgrp.createDimension('level', None)
time_dim = rootgrp.createDimension('time', None)
lat_dim = rootgrp.createDimension('lat', 73)
lon_dim = rootgrp.createDimension('lon', 144)

print(rootgrp.dimensions)

print(len(lon_dim))
print(lon_dim.isunlimited())
print(time_dim.isunlimited())

for dimobj in rootgrp.dimensions.values():
    print(dimobj)

print(time_dim)

# variables.
times = rootgrp.createVariable('time','f8',('time',))
levels = rootgrp.createVariable('level','i4',('level',))
latitudes = rootgrp.createVariable('lat','f4',('lat',))
longitudes = rootgrp.createVariable('lon','f4',('lon',))
# 2 unlimited dimensions.
#temp = rootgrp.createVariable('temp','f4',('time','level','lat','lon',))
# this makes the compression 'lossy' (preserving a precision of 1/1000)
# try it and see how much smaller the file gets.
temp = rootgrp.createVariable('temp','f4',('time','level','lat','lon',),least_significant_digit=3)
print(temp)
# create variable in a group using a path.
temp = rootgrp.createVariable('/forecasts/model1/temp','f4',('time','level','lat','lon',))
print(rootgrp['/forecasts/model1']) # print the Group instance
print(rootgrp['/forecasts/model1/temp']) # print the Variable instance

# attributes.
import time
rootgrp.description = 'bogus example script'
rootgrp.history = 'Created ' + time.ctime(time.time())
rootgrp.source = 'netCDF4 python module tutorial'
latitudes.units = 'degrees north'
longitudes.units = 'degrees east'
levels.units = 'hPa'
temp.units = 'K'
times.units = 'hours since 0001-01-01 00:00:00.0'
calendar: Literal['gregorian'] = 'gregorian'
times.calendar = calendar

for name in rootgrp.ncattrs():
    print('Global attr', name, '=', getattr(rootgrp,name))

print(rootgrp)

print(rootgrp.__dict__)

print(rootgrp.variables)

import numpy as np
# no unlimited dimension, just assign to slice.
lats =  np.arange(-90,91,2.5)
lons =  np.arange(-180,180,2.5)
latitudes[:] = lats
longitudes[:] = lons
print('latitudes =\n',latitudes[:])
print('longitudes =\n',longitudes[:])

# append along two unlimited dimensions by assigning to slice.
nlats = len(rootgrp.dimensions['lat'])
nlons = len(rootgrp.dimensions['lon'])
print('temp shape before adding data = ',temp.shape)
from numpy.random.mtrand import uniform # random number generator.
temp[0:5,0:10,:,:] = uniform(size=(5,10,nlats,nlons))
print('temp shape after adding data = ',temp.shape)
# levels have grown, but no values yet assigned.
print('levels shape after adding pressure data = ',levels.shape)

# assign values to levels dimension variable.
levels[:] =  [1000.,850.,700.,500.,300.,250.,200.,150.,100.,50.]
# fancy slicing
tempdat = temp[::2, [1,3,6], lats>0, lons>0]
print('shape of fancy temp slice = ',tempdat.shape)
print(temp[0, 0, [0,1,2,3], [0,1,2,3]].shape)

# fill in times.
from datetime import datetime, timedelta
from netCDF4 import num2date, date2num, date2index
dates = [datetime(2001,3,1)+n*timedelta(hours=12) for n in range(temp.shape[0])]
times[:] = date2num(dates,units=times.units,calendar=times.calendar)
print("time values (in units {}):\n{}".format(times.units, times[:]))
dates_array = num2date(times[:],units=times.units,calendar=times.calendar)
print("dates corresponding to time values:\n{}".format(dates_array))

rootgrp.close()

# create a series of netCDF files with a variable sharing
# the same unlimited dimension.
for nfile in range(10):
    nc = Dataset('mftest'+repr(nfile)+'.nc','w',format='NETCDF4_CLASSIC')
    nc.createDimension('x',None)
    x_var = nc.createVariable('x','i',('x',))
    x_var[0:10] = np.arange(nfile*10,10*(nfile+1))
    nc.close()
# now read all those files in at once, in one Dataset.
from netCDF4 import MFDataset
nc = MFDataset('mftest*nc')
print(nc.variables['x'][:])

# example showing how to save numpy complex arrays using compound types.
nc = Dataset('complex.nc','w')
size = 3 # length of 1-d complex array
# create sample complex data.
datac = np.exp(1j*(1.+np.linspace(0, np.pi, size)))
print(datac.dtype)
# create complex128 compound data type.
complex128 = np.dtype([('real',np.float64),('imag',np.float64)])
complex128_t = nc.createCompoundType(complex128,'complex128')
# create a variable with this data type, write some data to it.
nc.createDimension('x_dim',None)
var_complex = nc.createVariable('cmplx_var',complex128_t,'x_dim')
data = np.empty(size,complex128) # numpy structured array
data['real'] = datac.real; data['imag'] = datac.imag
var_complex[:] = data
# close and reopen the file, check the contents.
nc.close()
nc = Dataset('complex.nc')
print(nc)
print(nc.variables['cmplx_var'])
print(nc.cmptypes)
print(nc.cmptypes['complex128'])
var_complex = nc.variables['cmplx_var']
print(var_complex.shape)
datain = var_complex[:] # read in all the data into a numpy structured array
# create an empty numpy complex array
datac2 = np.empty(datain.shape,np.complex128)
# .. fill it with contents of structured array.
datac2.real = datain['real']
datac2.imag = datain['imag']
print(datac.dtype,datac)
print(datac2.dtype,datac2)

# more complex compound type example.
nc = Dataset('compound_example.nc','w') # create a new dataset.
# create an unlimited  dimension call 'station'
nc.createDimension('station',None)
# define a compound data type (can contain arrays, or nested compound types).
winddtype = np.dtype([('speed','f4'),('direction','i4')])
statdtype = np.dtype([('latitude', 'f4'), ('longitude', 'f4'),
                      ('surface_wind',winddtype),
                      ('temp_sounding','f4',10),('press_sounding','i4',10),
                      ('location_name','S12')])
# use this data type definitions to create a compound data types
# called using the createCompoundType Dataset method.
# create a compound type for vector wind which will be nested inside
# the station data type. This must be done first!
wind_data_t = nc.createCompoundType(winddtype,'wind_data')
# now that wind_data_t is defined, create the station data type.
station_data_t = nc.createCompoundType(statdtype,'station_data')
# create nested compound data types to hold the units variable attribute.
winddtype_units = np.dtype([('speed','S12'),('direction','S12')])
statdtype_units = np.dtype([('latitude', 'S12'), ('longitude', 'S12'),
                            ('surface_wind',winddtype_units),
                            ('temp_sounding','S12'),
                            ('location_name','S12'),
                            ('press_sounding','S12')])
# create the wind_data_units type first, since it will nested inside
# the station_data_units data type.
wind_data_units_t = nc.createCompoundType(winddtype_units,'wind_data_units')
station_data_units_t =\
nc.createCompoundType(statdtype_units,'station_data_units')
# create a variable of of type 'station_data_t'
statdat = nc.createVariable('station_obs', station_data_t, ('station',))
# create a numpy structured array, assign data to it.
data = np.empty(1,statdtype)
data['latitude'] = 40.
data['longitude'] = -105.
data['surface_wind']['speed'] = 12.5
data['surface_wind']['direction'] = 270
data['temp_sounding'] = (280.3,272.,270.,269.,266.,258.,254.1,250.,245.5,240.)
data['press_sounding'] = range(800,300,-50)
data['location_name'] = 'Boulder, CO'
# assign structured array to variable slice.
statdat[0] = data
# or just assign a tuple of values to variable slice
# (will automatically be converted to a structured array).
statdat[1] = np.array((40.78,-73.99,(-12.5,90),
             (290.2,282.5,279.,277.9,276.,266.,264.1,260.,255.5,243.),
             range(900,400,-50),'New York, NY'),data.dtype)
print(nc.cmptypes)
windunits = np.empty(1,winddtype_units)
stationobs_units = np.empty(1,statdtype_units)
windunits['speed'] = 'm/s'
windunits['direction'] = 'degrees'
stationobs_units['latitude'] = 'degrees N'
stationobs_units['longitude'] = 'degrees W'
stationobs_units['surface_wind'] = windunits
stationobs_units['location_name'] = 'None'
stationobs_units['temp_sounding'] = 'Kelvin'
stationobs_units['press_sounding'] = 'hPa'
print(stationobs_units.dtype)
statdat.units = stationobs_units
# close and reopen the file.
nc.close()
nc = Dataset('compound_example.nc')
print(nc)
statdat = nc.variables['station_obs']
print(statdat)
# print out data in variable.
print('data in a variable of compound type:')
print(statdat[:])
nc.close()

nc = Dataset('tst_vlen.nc','w')
vlen_t = nc.createVLType(np.int32, 'phony_vlen')
x = nc.createDimension('x',3)
y = nc.createDimension('y',4)
vlvar = nc.createVariable('phony_vlen_var', vlen_t, ('y','x'))
import random
data = np.empty(len(y)*len(x),object)
for n in range(len(y)*len(x)):
    data[n] = np.arange(random.randint(1,10),dtype='int32')+1
data = np.reshape(data,(len(y),len(x)))
vlvar[:] = data
print(vlvar)
print('vlen variable =\n',vlvar[:])
print(nc)
print(nc.variables['phony_vlen_var'])
print(nc.vltypes['phony_vlen'])
z = nc.createDimension('z', 10)
strvar = nc.createVariable('strvar',str,'z')
chars = '1234567890aabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
data = np.empty(10,object)
for n in range(10):
    stringlen = random.randint(2,12)
    data[n] = ''.join([random.choice(chars) for i in range(stringlen)])
strvar[:] = data
print('variable-length string variable:\n',strvar[:])
print(nc)
print(nc.variables['strvar'])
nc.close()

# Enum type example.
nc = Dataset('clouds.nc','w')
# python dict describing the allowed values and their names.
enum_dict = {'Altocumulus': 7, 'Missing': 255, 'Stratus': 2, 'Clear': 0,
'Nimbostratus': 6, 'Cumulus': 4, 'Altostratus': 5, 'Cumulonimbus': 1,
'Stratocumulus': 3}
# create the Enum type called 'cloud_t'.
cloud_type = nc.createEnumType(np.uint8,'cloud_t',enum_dict)
print(cloud_type)
time_dim = nc.createDimension('time',None)
# create a 1d variable of type 'cloud_type' called 'primary_clouds'.
# The fill_value is set to the 'Missing' named value.
cloud_var = nc.createVariable('primary_cloud',cloud_type,'time',\
fill_value=enum_dict['Missing'])
# write some data to the variable.
cloud_var[:] = [enum_dict['Clear'],enum_dict['Stratus'],enum_dict['Cumulus'],\
                enum_dict['Missing'],enum_dict['Cumulonimbus']]
# close file, reopen it.
nc.close()
nc = Dataset('clouds.nc')
cloud_var = nc.variables['primary_cloud']
print(cloud_var)
print(cloud_var.datatype.enum_dict)
print(cloud_var[:])
nc.close()

# dealing with strings
from netCDF4 import stringtochar
nc = Dataset('stringtest.nc','w',format='NETCDF4_CLASSIC')
nc.createDimension('nchars',3)
nc.createDimension('nstrings',None)
var = nc.createVariable('strings','S1',('nstrings','nchars'))
datain = np.array(['foo','bar'],dtype='S3')
var[:] = stringtochar(datain) # manual conversion to char array
print(var[:]) # data returned as char array
var._Encoding = 'ascii' # this enables automatic conversion
var[:] = datain # conversion to char array done internally
print(var[:]) # data returned in numpy string array
nc.close()
# strings in compound types
nc = Dataset('compoundstring_example.nc','w')
dtype = np.dtype([('observation', 'f4'),
                  ('station_name','S12')])
station_data_t = nc.createCompoundType(dtype,'station_data')
nc.createDimension('station',None)
statdat = nc.createVariable('station_obs', station_data_t, ('station',))
data = np.empty(2,station_data_t.dtype_view)
data['observation'][:] = (123.,3.14)
data['station_name'][:] = ('Boulder','New York')
print(statdat.dtype) # strings actually stored as character arrays
statdat[:] = data # strings converted to character arrays internally
print(statdat[:]) # character arrays converted back to strings
print(statdat[:].dtype)
statdat.set_auto_chartostring(False) # turn off auto-conversion
statdat[:] = data.view(station_data_t.dtype)
print(statdat[:]) # now structured array with char array subtype is returned
nc.close()

# create a diskless (in-memory) Dataset, and persist the file
# to disk when it is closed.
nc = Dataset('diskless_example.nc','w',diskless=True,persist=True)
d = nc.createDimension('x',None)
v = nc.createVariable('v',np.int32,'x')
v[0:5] = np.arange(5)
print(nc)
print(nc['v'][:])
nc.close() # file saved to disk
# create an in-memory dataset from an existing python memory
# buffer.
# read the newly created netcdf file into a python bytes object.
f = open('diskless_example.nc', 'rb')
nc_bytes = f.read(); f.close()
# create a netCDF in-memory dataset from the bytes object.
nc = Dataset('inmemory.nc', memory=nc_bytes)
print(nc)
print(nc['v'][:])
nc.close()
# create an in-memory Dataset and retrieve memory buffer
# estimated size is 1028 bytes - this is actually only
# used if format is NETCDF3 (ignored for NETCDF4/HDF5 files).
nc = Dataset('inmemory.nc', mode='w',memory=1028)
d = nc.createDimension('x',None)
v = nc.createVariable('v',np.int32,'x')
v[0:5] = np.arange(5)
nc_buf = nc.close() # close returns memoryview
print(type(nc_buf))
# save nc_buf to disk, read it back in and check.
f2 = open('inmemory.nc', 'wb')
f2.write(nc_buf); f2.close()
nc = Dataset('inmemory.nc')
print(nc)
print(nc['v'][:])
nc.close()

# Write complex numbers to file
complex_array = np.array([0 + 0j, 1 + 0j, 0 + 1j, 1 + 1j, 0.25 + 0.75j])
with Dataset("complex.nc", "w", auto_complex=True) as nc:
    nc.createDimension("x", size=len(complex_array))
    var = nc.createVariable("data", "c16", ("x",))
    var[:] = complex_array
    print(var)
