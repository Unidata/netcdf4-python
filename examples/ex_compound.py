import netCDF4
import numpy as NP
# create a new file.
f = netCDF4.Dataset('tst_compound.nc','w')
# create an unlimited  dimension call 'station'
f.createDimension('station',False)
# define a compound data type (a list of 3-tuples containing
# the name of each member, it's primitive data type, and it's size).
# Only fixed size primitive data types allowed (no 'S').
# Members can be multi-dimensional arrays (in which case the third
# element is a shape tuple instead of a scalar).
datatype = [('latitude', 'f4',1), ('longitude', 'f4',1),('sfc_press','i4',1),
            ('temp_sounding','f4',10),('press_sounding','i4',10),
            ('2dfield','f8',(4,4)),('location_name','S1',80),]
# create a user-defined data type
table = f.createUserType(datatype,'compound','station_data')
# create a variable of this type.
statdat = f.createVariable('station_obs', table, ('station',))
# create a scalar variable of this type.
statdat_scalar = f.createVariable('station_obs1', table)
# create record array with data.
ra = NP.empty(1,statdat.dtype_base)
ra['latitude'] = 40.
ra['longitude'] = -105.
ra['sfc_press'] = 818
ra['temp_sounding'] = (280.3,272.,270.,269.,266.,258.,254.1,250.,245.5,240.)
ra['press_sounding'] = range(800,300,-50)
ra['2dfield'] = NP.ones((4,4),'f8')
# only fixed-size primitive data types can currenlty be used
# as compound data type members (although the library supports
# nested compound types).
# To store strings in a compound data type, each string must be stored as fixed-size
# (in this case 80) array of characters.
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
             range(900,400,-50),2.*NP.ones((4,4),'f8'),
             stringtoarr('New York, New York, USA',80))
statdat_scalar.assignValue(ra)
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
units_dict = {'latitude': 'degrees north', 'longitude': 'degrees east', 'sfc_press': 'Pascals', 'temp_sounding': 'Kelvin', 'press_sounding': 'Pascals','location_name': None,'2dfield': 'Whatsits'}
statdat.units = repr(units_dict)
# close the file.
f.close()
# open file, read data in.
f = netCDF4.Dataset('tst_compound.nc')
statdata = f.variables['station_obs']
# convert units string to a python dictionary.
statdata_units = eval(statdata.units)
# set options for printing of numpy arrays
#NP.set_printoptions(precision=1, linewidth=34)
# print out data in variable (including units attribute)
for data in statdata[:]:
    for item in statdata.dtype_base:
        name = item[0]
        type = item[1]
        if type == 'S1': # if array of chars, convert value to string.
            print name,': value =',data[name].tostring(),'units =',statdata_units[name]
        else:
            print name,': value =',data[name],'units =',statdata_units[name]
    print '----'
# read data from scalar variable
statdata_scalar = f.variables['station_obs1']
print statdata_scalar.shape, statdata_scalar.dtype_base
print statdata_scalar.getValue()
f.close()
