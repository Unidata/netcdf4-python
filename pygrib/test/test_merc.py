import pygrib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
grb = pygrib.open('/Users/jsw/python/pygrib2/sampledata/dspr.temp.grb')
grb.next()
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
m = Basemap(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,
            urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,rsphere=rsphere,lon_0=lon_0,
            lat_ts=lat_ts,resolution='i',projection=projection)
x,y = m(lons, lats)
m.scatter(x.flat,y.flat,1,marker='o',color='k',zorder=10)
m.drawcoastlines()
m.fillcontinents()
plt.title('Mercator Grid (Puerto Rico)')
plt.show()
