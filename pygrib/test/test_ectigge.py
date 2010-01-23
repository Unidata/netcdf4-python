import pygrib
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma
from mpl_toolkits.basemap import Basemap
for grb in pygrib.open('/Users/jsw/python/pygrib2/sampledata/ecmwf_tigge.grb'):
    if grb['parameterName'] == 'Soil moisture':
        fld = grb['values']
        lats,lons = grb.latlons()
        break
llcrnrlon = lons[0,0]
llcrnrlat = lats[0,0]
urcrnrlon = lons[-1,-1]
urcrnrlat = lats[-1,-1]
m = Basemap(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,
            urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,
            resolution='l',projection='cyl')
CS = m.contourf(lons,lats,fld,15,cmap=plt.cm.jet)
#im = m.pcolor(lons,lats,fld,cmap=plt.cm.jet,shading='flat')
ax = plt.gca()
pos = ax.get_position()
l, b, w, h = pos.bounds
cax = plt.axes([l+w+0.025, b, 0.025, h]) # setup colorbar axes
plt.colorbar(drawedges=True, cax=cax, format='%g') # draw colorbar
plt.axes(ax)  # make the original axes current again
m.drawcoastlines()
# draw parallels
delat = 30.
circles = np.arange(-90.,90.+delat,delat)
m.drawparallels(circles,labels=[1,0,0,0])
# draw meridians
delon = 60.
meridians = np.arange(0,360,delon)
m.drawmeridians(meridians,labels=[0,0,0,1])
plt.title(grb['parameterName']+' on ECMWF Reduced Gaussian Grid')
plt.show()
