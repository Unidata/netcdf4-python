import pygrib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
for grb in pygrib.open('/Users/jsw/python/pygrib2/sampledata/dspr.temp.grb'):
    if grb['forecastTime'] == 12:
        break
data = grb['values']
print data.shape, data.min(), data.max()
lats, lons = grb.latlons()
llcrnrlon = lons[0,0]
llcrnrlat = lats[0,0]
urcrnrlon = lons[-1,-1]
urcrnrlat = lats[-1,-1]
print llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
rsphere = (grb.projparams['a'], grb.projparams['b'])
lat_ts = grb.projparams['lat_ts']
lon_0 = grb.projparams['lon_0']
projection = grb.projparams['proj']
fig=plt.figure()
ax = fig.add_axes([0.1,0.1,0.75,0.75])
m = Basemap(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,
            urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,rsphere=rsphere,lon_0=lon_0,
            lat_ts=lat_ts,resolution='h',projection=projection)
x,y = m(lons, lats)
cs = m.contourf(x,y,data,20,cmap=plt.cm.jet)
m.drawcoastlines()
m.drawstates()
m.drawcountries()
m.drawmeridians(np.arange(280,300,1),labels=[0,0,0,1])
m.drawparallels(np.arange(16,21,1),labels=[1,0,0,0])
# new axis for colorbar.
cax = plt.axes([0.875, 0.10, 0.03, 0.75])
plt.colorbar(cs, cax, format='%g') # draw colorbar
plt.axes(ax)  # make the original axes current again
plt.title('NDFD Temp Puerto Rico %d-h fcst from %d' %\
        (grb['forecastTime'],grb['dataDate']),fontsize=12)
plt.show()
