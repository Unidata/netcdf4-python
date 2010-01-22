import pygrib
import sys
import numpy as np
filename = sys.argv[1]
n = 0
for grb in pygrib.open(filename):
    n = n + 1
    data = grb['values']
    print '------message %d------' %n
    for k in grb.keys():
        if k.startswith('mars'): continue
        if k == 'values' or k == 'codedValues': continue
        print k,'=',grb[k]
    print 'min/max of data = ',data.min(),data.max()
