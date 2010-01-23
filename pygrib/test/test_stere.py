import pygrib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
for grb in pygrib.open('/Users/jsw/python/pygrib2/sampledata/ngm.grb'):
    if grb['parameterName'] == 'Pressure' and grb['typeOfLevel'] == 'surface':
        data = grb['values']
        lats,lons = grb.latlons()
        break
print lats.min(), lats.max()
print lons.min(), lons.max()
print lats[0,0],lons[0,0]
print lats[-1,-1],lons[-1,-1]
llcrnrlon = lons[0,0]
llcrnrlat = lats[0,0]
urcrnrlon = lons[-1,-1]
urcrnrlat = lats[-1,-1]
rsphere = (grb.projparams['a'], grb.projparams['b'])
lat_ts = grb.projparams['lat_ts']
lon_0 = grb.projparams['lon_0']
lat_0 = grb.projparams['lat_0']
projection = grb.projparams['proj']
m = Basemap(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,
            urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,rsphere=rsphere,lon_0=lon_0,
            lat_ts=lat_ts,lat_0=lat_0,resolution='l',projection=projection)
x,y = m(lons, lats)
m.scatter(x.flat,y.flat,3,marker='o',color='k',zorder=10)
m.drawcoastlines()
x,y = m(lons,lats)
m.contourf(x,y,data,15)
plt.title('Stereographic Model Grid')
plt.show()
