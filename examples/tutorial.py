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
    values = top.groups.values()
    yield values
    for value in top.groups.values():
        for children in walktree(value):
            yield  children
print(rootgrp)
for children in walktree(rootgrp):
    for child in children:
        print(child)

# dimensions.
level = rootgrp.createDimension('level', None)
time = rootgrp.createDimension('time', None)
lat = rootgrp.createDimension('lat', 73)
lon = rootgrp.createDimension('lon', 144)

print(rootgrp.dimensions)

print(len(lon))
print(lon.isunlimited())
print(time.isunlimited())

for dimobj in rootgrp.dimensions.values():
    print(dimobj)

print(time)

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
times.calendar = 'gregorian'

for name in rootgrp.ncattrs():
    print('Global attr', name, '=', getattr(rootgrp,name))

print(rootgrp)

print(rootgrp.__dict__)

print(rootgrp.variables)

import numpy
# no unlimited dimension, just assign to slice.
lats =  numpy.arange(-90,91,2.5)
lons =  numpy.arange(-180,180,2.5)
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
print('time values (in units %s): ' % times.units+'\\n',times[:])
dates = num2date(times[:],units=times.units,calendar=times.calendar)
print('dates corresponding to time values:\\n',dates)

rootgrp.close()

# create a series of netCDF files with a variable sharing
# the same unlimited dimension.
for nfile in range(10):
    f = Dataset('mftest'+repr(nfile)+'.nc','w',format='NETCDF4_CLASSIC')
    f.createDimension('x',None)
    x = f.createVariable('x','i',('x',))
    x[0:10] = numpy.arange(nfile*10,10*(nfile+1))
    f.close()
# now read all those files in at once, in one Dataset.
from netCDF4 import MFDataset
f = MFDataset('mftest*nc')
print(f.variables['x'][:])

# example showing how to save numpy complex arrays using compound types.
f = Dataset('complex.nc','w')
size = 3 # length of 1-d complex array
# create sample complex data.
datac = numpy.exp(1j*(1.+numpy.linspace(0, numpy.pi, size)))
print(datac.dtype)
# create complex128 compound data type.
complex128 = numpy.dtype([('real',numpy.float64),('imag',numpy.float64)])
complex128_t = f.createCompoundType(complex128,'complex128')
# create a variable with this data type, write some data to it.
f.createDimension('x_dim',None)
v = f.createVariable('cmplx_var',complex128_t,'x_dim')
data = numpy.empty(size,complex128) # numpy structured array
data['real'] = datac.real; data['imag'] = datac.imag
v[:] = data
# close and reopen the file, check the contents.
f.close()
f = Dataset('complex.nc')
print(f)
print(f.variables['cmplx_var'])
print(f.cmptypes)
print(f.cmptypes['complex128'])
v = f.variables['cmplx_var']
print(v.shape)
datain = v[:] # read in all the data into a numpy structured array
# create an empty numpy complex array
datac2 = numpy.empty(datain.shape,numpy.complex128)
# .. fill it with contents of structured array.
datac2.real = datain['real']
datac2.imag = datain['imag']
print(datac.dtype,datac)
print(datac2.dtype,datac2)
# more complex compound type example.
from netCDF4 import chartostring, stringtoarr
f = Dataset('compound_example.nc','w') # create a new dataset.
# create an unlimited  dimension call 'station'
f.createDimension('station',None)
# define a compound data type (can contain arrays, or nested compound types).
NUMCHARS = 80 # number of characters to use in fixed-length strings.
winddtype = numpy.dtype([('speed','f4'),('direction','i4')])
statdtype = numpy.dtype([('latitude', 'f4'), ('longitude', 'f4'),
                         ('surface_wind',winddtype),
                         ('temp_sounding','f4',10),('press_sounding','i4',10),
                         ('location_name','S1',NUMCHARS)])
# use this data type definitions to create a compound data types
# called using the createCompoundType Dataset method.
# create a compound type for vector wind which will be nested inside
# the station data type. This must be done first!
wind_data_t = f.createCompoundType(winddtype,'wind_data')
# now that wind_data_t is defined, create the station data type.
station_data_t = f.createCompoundType(statdtype,'station_data')
# create nested compound data types to hold the units variable attribute.
winddtype_units = numpy.dtype([('speed','S1',NUMCHARS),('direction','S1',NUMCHARS)])
statdtype_units = numpy.dtype([('latitude', 'S1',NUMCHARS), ('longitude', 'S1',NUMCHARS),
                               ('surface_wind',winddtype_units),
                               ('temp_sounding','S1',NUMCHARS),
                               ('location_name','S1',NUMCHARS),
                               ('press_sounding','S1',NUMCHARS)])
# create the wind_data_units type first, since it will nested inside
# the station_data_units data type.
wind_data_units_t = f.createCompoundType(winddtype_units,'wind_data_units')
station_data_units_t =\
f.createCompoundType(statdtype_units,'station_data_units')
# create a variable of of type 'station_data_t'
statdat = f.createVariable('station_obs', station_data_t, ('station',))
# create a numpy structured array, assign data to it.
data = numpy.empty(1,station_data_t)
data['latitude'] = 40.
data['longitude'] = -105.
data['surface_wind']['speed'] = 12.5
data['surface_wind']['direction'] = 270
data['temp_sounding'] = (280.3,272.,270.,269.,266.,258.,254.1,250.,245.5,240.)
data['press_sounding'] = range(800,300,-50)
# variable-length string datatypes are not supported inside compound types, so
# to store strings in a compound data type, each string must be
# stored as fixed-size (in this case 80) array of characters.
data['location_name'] = stringtoarr('Boulder, Colorado, USA',NUMCHARS)
# assign structured array to variable slice.
statdat[0] = data
# or just assign a tuple of values to variable slice
# (will automatically be converted to a structured array).
statdat[1] = (40.78,-73.99,(-12.5,90),
             (290.2,282.5,279.,277.9,276.,266.,264.1,260.,255.5,243.),
             range(900,400,-50),stringtoarr('New York, New York, USA',NUMCHARS))
print(f.cmptypes)
windunits = numpy.empty(1,winddtype_units)
stationobs_units = numpy.empty(1,statdtype_units)
windunits['speed'] = stringtoarr('m/s',NUMCHARS)
windunits['direction'] = stringtoarr('degrees',NUMCHARS)
stationobs_units['latitude'] = stringtoarr('degrees north',NUMCHARS)
stationobs_units['longitude'] = stringtoarr('degrees west',NUMCHARS)
stationobs_units['surface_wind'] = windunits
stationobs_units['location_name'] = stringtoarr('None', NUMCHARS)
stationobs_units['temp_sounding'] = stringtoarr('Kelvin',NUMCHARS)
stationobs_units['press_sounding'] = stringtoarr('hPa',NUMCHARS)
statdat.units = stationobs_units
# close and reopen the file.
f.close()
f = Dataset('compound_example.nc')
print(f)
statdat = f.variables['station_obs']
print(statdat)
# print out data in variable.
print('data in a variable of compound type:')
print('----')
for data in statdat[:]:
    for name in statdat.dtype.names:
        if data[name].dtype.kind == 'S': # a string
            # convert array of characters back to a string for display.
            units = chartostring(statdat.units[name])
            print(name,': value =',chartostring(data[name]),\
                    ': units=',units)
        elif data[name].dtype.kind == 'V': # a nested compound type
            units_list = [chartostring(s) for s in tuple(statdat.units[name])]
            print(name,data[name].dtype.names,': value=',data[name],': units=',\
            units_list)
        else: # a numeric type.
            units = chartostring(statdat.units[name])
            print(name,': value=',data[name],': units=',units)
    print('----')
f.close()

f = Dataset('tst_vlen.nc','w')
vlen_t = f.createVLType(numpy.int32, 'phony_vlen')
x = f.createDimension('x',3)
y = f.createDimension('y',4)
vlvar = f.createVariable('phony_vlen_var', vlen_t, ('y','x'))
import random
data = numpy.empty(len(y)*len(x),object)
for n in range(len(y)*len(x)):
    data[n] = numpy.arange(random.randint(1,10),dtype='int32')+1
data = numpy.reshape(data,(len(y),len(x)))
vlvar[:] = data
print(vlvar)
print('vlen variable =\n',vlvar[:])
print(f)
print(f.variables['phony_vlen_var'])
print(f.vltypes['phony_vlen'])
z = f.createDimension('z', 10)
strvar = f.createVariable('strvar',str,'z')
chars = '1234567890aabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
data = numpy.empty(10,object)
for n in range(10):
    stringlen = random.randint(2,12)
    data[n] = ''.join([random.choice(chars) for i in range(stringlen)])
strvar[:] = data
print('variable-length string variable:\n',strvar[:])
print(f)
print(f.variables['strvar'])
f.close()

# Enum type example.
f = Dataset('clouds.nc','w')
# python dict describing the allowed values and their names.
enum_dict = {u'Altocumulus': 7, u'Missing': 255, u'Stratus': 2, u'Clear': 0,
u'Nimbostratus': 6, u'Cumulus': 4, u'Altostratus': 5, u'Cumulonimbus': 1,
u'Stratocumulus': 3}
# create the Enum type called 'cloud_t'.
cloud_type = f.createEnumType(numpy.uint8,'cloud_t',enum_dict)
print(cloud_type)
time = f.createDimension('time',None)
# create a 1d variable of type 'cloud_type' called 'primary_clouds'.
# The fill_value is set to the 'Missing' named value.
cloud_var = f.createVariable('primary_cloud',cloud_type,'time',\
fill_value=enum_dict['Missing'])
# write some data to the variable.
cloud_var[:] = [enum_dict['Clear'],enum_dict['Stratus'],enum_dict['Cumulus'],\
                enum_dict['Missing'],enum_dict['Cumulonimbus']]
# close file, reopen it.
f.close()
f = Dataset('clouds.nc')
cloud_var = f.variables['primary_cloud']
print(cloud_var)
print(cloud_var.datatype.enum_dict)
print(cloud_var[:])
f.close()
