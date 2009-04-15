from netCDF4 import Dataset, CompoundType, chartostring
import numpy as np
f = Dataset('test_compound.nc','w')
d = f.createDimension('time',None)
dtype=np.dtype([('eastward', 'f4'), ('northward', 'f4')])
dtypec=np.dtype([('eastward', 'c',(8,)), ('northward', 'c',(8,))])
print dtypec
wind_vector_type = f.createCompoundType(dtype, 'wind_vector_t')
wind_vectorunits_type = f.createCompoundType(dtypec, 'wind_vectorunits_t')
print f._cmptypes
v = f.createVariable('wind',wind_vector_type, 'time')
g = f.createGroup('forecasts')
vv = g.createVariable('forecast_wind',wind_vector_type,'time')
data = np.zeros(10,dtype)
data['eastward']=1
data['northward']=-1.
missvals = np.zeros(1,dtype)
missvals['eastward']=9999.
missvals['northward']=-9999.
v.missing_values = missvals
chararr = np.array(list('%-08s'%'m/s'))
windunits = np.zeros(1,dtypec)
windunits['eastward'] = chararr
windunits['northward'] = chararr
v.units = windunits
v[:] = data
data['eastward']=-1
data['northward']=1.
vv[:] = data
vv.missing_values = missvals
vv.units = windunits
f.close()

f = Dataset('test_compound.nc')
v = f.variables['wind']
g = f.groups['forecasts']
vv = g.variables['forecast_wind']
print v.missing_values.dtype
print v.missing_values
print v.units.dtype, v.units.shape
print chartostring(v.units['eastward'])
print chartostring(v.units['northward'])
print vv.missing_values.dtype
print vv.missing_values
print vv.units.dtype, vv.units.shape
print chartostring(vv.units['eastward'])
print chartostring(vv.units['northward'])
f.close()
