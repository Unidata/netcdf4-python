import netCDF4
# test group creation.

FILE_NAME= "tst_grps1.nc"
DYNASTY="Tudor"
HENRY_VII="Henry_VII"
MARGARET="Margaret"
JAMES_V_OF_SCOTLAND="James_V_of_Scotland"
MARY_I_OF_SCOTLAND="Mary_I_of_Scotland"
JAMES_VI_OF_SCOTLAND_AND_I_OF_ENGLAND="James_VI_of_Scotland_and_I_of_England"

f  = netCDF4.Dataset(FILE_NAME, 'w')
f.history = 'created today'
g1 = f.createGroup(HENRY_VII)
g2 = g1.createGroup(MARGARET)
g3 = g2.createGroup(JAMES_V_OF_SCOTLAND)
g4 = g3.createGroup(MARY_I_OF_SCOTLAND)
g5 = g4.createGroup(JAMES_VI_OF_SCOTLAND_AND_I_OF_ENGLAND)
f.close()

# python generator to walk the Group tree.
def walktree(top):
    values = top.groups.values()
    yield values
    for value in top.groups.values():
        for children in walktree(value):
            yield  children
f = netCDF4.Dataset(FILE_NAME)
print f.path, f
for children in walktree(f):
     for child in children:
         print child.path, child
f.close()

FILE_NAME= "tst_grps2.nc"

f  = netCDF4.Dataset(FILE_NAME, 'w')
g1 = netCDF4.Group(f,DYNASTY)
g2 = g1.createGroup(HENRY_VII)
g3 = g1.createGroup(MARGARET)
g4 = g1.createGroup(JAMES_V_OF_SCOTLAND)
g5 = g1.createGroup(MARY_I_OF_SCOTLAND)
g6 = g1.createGroup(JAMES_VI_OF_SCOTLAND_AND_I_OF_ENGLAND)
f.close()

f = netCDF4.Dataset(FILE_NAME)
print f.path, f
for children in walktree(f):
     for child in children:
         print child.path, child
f.close()
