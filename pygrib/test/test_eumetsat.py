import pygrib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
grb = pygrib.open('/Users/jsw/python/pygrib2/sampledata/eumetsat_precip.grb')
grb.next()
fld = grb['values']
lats, lons = grb.latlons()
rsphere = (grb.projparams['a'], grb.projparams['b'])
lon_0 = grb.projparams['lon_0']
h = grb.projparams['h']
print grb.projparams
projection = grb.projparams['proj']

m = Basemap(lon_0=lon_0,satellite_height=h,\
            rsphere = rsphere,\
            resolution='l',area_thresh=10000.,projection='geos')
# plot every 50th point.
x, y = m(lons,lats)
m.scatter(x[::50,::50].flat,y[::50,::50].flat,1,marker='o',color='k',zorder=10)
m.drawcoastlines()
m.drawcountries()
#m.fillcontinents(color='coral')
m.drawcoastlines()
# contour data.
m.contourf(x,y,fld,20)
# pcolor image (slower)
#m.pcolor(x,y,fld)
m.drawparallels(np.arange(-80,81,20))
m.drawmeridians(np.arange(-90,90,20))
m.drawmapboundary()
plt.title('EUMETSAT geostationary projection grid')

plt.show()
