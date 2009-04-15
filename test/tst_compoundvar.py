from netCDF4 import Dataset, CompoundType
import numpy as np
f = Dataset('test_compound.nc','w')
d = f.createDimension('phony_dim',None)
dtype1=np.dtype([('i', 'i2'), ('j', 'i8')])
dtype2=np.dtype([('xx', 'f4'), ('yy', 'f8', (3,2))])
dtype3=np.dtype([('xxx', dtype1), ('yyy', dtype2, (4,))])
dtype4=np.dtype([('x', 'f4'), ('y', 'f8', (2,3)), ('z', dtype3, (2,2))])
print dtype4.fields
# multiply nested compound types
cmptype1 = f.createCompoundType(dtype1, 'cmp1')
cmptype2 = f.createCompoundType(dtype2, 'cmp2')
cmptype3 = f.createCompoundType(dtype3, 'cmp3')
cmptype4 = f.createCompoundType(dtype4, 'cmp4')
v = f.createVariable('phony_compound_var',cmptype4, 'phony_dim')
data = np.zeros(10,dtype4)
data['x']=1
data['z']['xxx']['i'][:]=2
v[:] = data
data2 = v[:]
print data2['x']
print data2['z']['xxx']['i'].dtype
print data2['z']['xxx']['i'].shape
print data2['z']['xxx']['i'][:,0,0]
f.close()

print
f = Dataset('test_compound.nc')
v = f.variables['phony_compound_var']
print v.dtype
data2 = v[:]
print data2['x']
print data2['z']['xxx']['i'].dtype
print data2['z']['xxx']['i'].shape
print data2['z']['xxx']['i'][:,0,0]
f.close()
