import netCDF4
import numpy as np
import sys, unittest, os, tempfile

FILE_NAME = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
ATT1 = u'\u03a0\u03a3\u03a9'
ATT2 = u'x\xb0'
ATT3 = [u'\u03a0',u'\u03a3',u'\u03a9']
DIM_NAME = u'x\xb0'
VAR_NAME = u'Andr\xe9'

class UnicodeTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        f = netCDF4.Dataset(self.file,'w')
        f.attribute1 = ATT1
        f.attribute2 = ATT2
        f.attribute3 = ATT3
        d = f.createDimension(DIM_NAME, None)
        v = f.createVariable(VAR_NAME, np.float, (DIM_NAME,))
        f.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """testing unicode"""
        f  = netCDF4.Dataset(self.file, 'r')
        d = f.dimensions[DIM_NAME]
        v = f.variables[VAR_NAME]
        # check accessing individual attributes.
        assert f.attribute1 == ATT1
        assert f.attribute2 == ATT2
        assert f.attribute3 == ''.join(ATT3)
        f.close()

if __name__ == '__main__':
    unittest.main()
