# use 'orthogonal indexing' feature to subselect data over CONUS.
import netCDF4
import numpy as np
import matplotlib.pyplot as plt

# use real data from CFS reanlysis.
# note:  we're reading GRIB2 data!
URL="http://nomads.ncdc.noaa.gov/thredds/dodsC/modeldata/cmd_flxf/2010/201007/20100701/flxf00.gdas.2010070100.grb2"
nc = netCDF4.Dataset(URL)
lats = nc.variables['lat'][:]; lons = nc.variables['lon'][:]
latselect = np.logical_and(lats>25,lats<50)
lonselect = np.logical_and(lons>230,lons<305)
data = nc.variables['Soil_moisture_content'][0,0,latselect,lonselect]
plt.contourf(data[::-1]) # flip latitudes so they go south -> north
plt.show()
