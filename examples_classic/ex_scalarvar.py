import netCDF4_classic as netCDF4
# test scalar variable creation and retrieval.
# create a file (Dataset object, also the root group).
rootgrp = netCDF4.Dataset('test.nc', 'w')
# scalar variable.
temp = rootgrp.createVariable('temp','f4')
print "testing scalar variables, the next two lines should be '()'"
print temp.dimensions
print temp.shape
#temp[:] = 12.
temp.assignValue(12.)
rootgrp.close()
rootgrp = netCDF4.Dataset('test.nc', 'r')
temp = rootgrp.variables['temp']
#print temp[:]
print 'the next line should be 12'
print temp.getValue()
rootgrp.close()
