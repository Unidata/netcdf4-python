import sys
import unittest
import os
import tempfile
import numpy as NP
import netCDF4

# test group creation.

FILE_NAME1 = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
FILE_NAME2 = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
DYNASTY=u"Tudor"
HENRY_VII=u"Henry_VII"
MARGARET=u"Margaret"
JAMES_V_OF_SCOTLAND=u"James_V_of_Scotland"
MARY_I_OF_SCOTLAND=u"Mary_I_of_Scotland"
JAMES_VI_OF_SCOTLAND_AND_I_OF_ENGLAND=u"James_VI_of_Scotland_and_I_of_England"
names = [HENRY_VII,MARGARET,JAMES_V_OF_SCOTLAND,MARY_I_OF_SCOTLAND,JAMES_VI_OF_SCOTLAND_AND_I_OF_ENGLAND]
root = '/'
TREE1 = [root]
for n in range(1,len(names)+1):
    path = []
    for name in names[0:n]:
        path.append(root+name)
    TREE1.append(''.join(path))
TREE2 = [root,root+DYNASTY]
for name in names:
    TREE2.append(root+DYNASTY+root+name)
TREE2.sort()


# python generator to walk the Group tree.
def walktree(top):
    values = top.groups.values()
    yield values
    for value in top.groups.values():
        for children in walktree(value):
            yield  children

class GroupsTestCase(unittest.TestCase):

    def setUp(self):
        self.file1 = FILE_NAME1
        f  = netCDF4.Dataset(self.file1, 'w')
        g1 = f.createGroup(HENRY_VII)
        g2 = g1.createGroup(MARGARET)
        g3 = g2.createGroup(JAMES_V_OF_SCOTLAND)
        g4 = g3.createGroup(MARY_I_OF_SCOTLAND)
        g5 = g4.createGroup(JAMES_VI_OF_SCOTLAND_AND_I_OF_ENGLAND)
        f.close()
        self.file2 = FILE_NAME2
        f  = netCDF4.Dataset(self.file2, 'w')
        g1 = netCDF4.Group(f,DYNASTY)
        g2 = g1.createGroup(HENRY_VII)
        g3 = g1.createGroup(MARGARET)
        g4 = g1.createGroup(JAMES_V_OF_SCOTLAND)
        g5 = g1.createGroup(MARY_I_OF_SCOTLAND)
        g6 = g1.createGroup(JAMES_VI_OF_SCOTLAND_AND_I_OF_ENGLAND)
        f.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file1)
        os.remove(self.file2)

    def runTest(self):
        """testing groups"""
        f  = netCDF4.Dataset(self.file1, 'r')
        tree = [f.path]
        for children in walktree(f):
            for child in children:
                tree.append(child.path)
        f.close()
        assert tree == TREE1
        f  = netCDF4.Dataset(self.file2, 'r')
        tree = [f.path]
        for children in walktree(f):
            for child in children:
                tree.append(child.path)
        tree.sort()
        f.close()
        assert tree == TREE2

if __name__ == '__main__':
    unittest.main()
