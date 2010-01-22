import pygrib
import sys
import numpy as np
# lists information for grib file specified on command line.
filename = sys.argv[1]
n = 0
for grb in pygrib.open(filename):
    n = n + 1
    print '------message %d------' %n
    for k in grb.keys():
        if k.startswith('mars'): continue
        if k == 'values' or k == 'codedValues': continue
        print k,'=',grb[k]
    data = grb['values']
    print 'min/max of data = ',data.min(),data.max()
