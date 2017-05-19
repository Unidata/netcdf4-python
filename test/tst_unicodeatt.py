from netCDF4 import Dataset
import sys, unittest, os, tempfile

FILE_NAME = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name

class UnicodeAttTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        nc = Dataset(self.file,'w')
        # write as a utf-8 string
        nc.stratt = b'\xe6\xb7\xb1\xe5\x85\xa5 Python'.decode('utf-8')
        # write as raw bytes (decoded string is same as above with 'big5' encoding)
        nc.stratt2 = b'\xb2`\xa4J Python'
        # same as above, but attribute forced to be of type NC_STRING
        nc.setncattr_string('stratt3',b'\xb2`\xa4J Python')
        nc.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """testing unicode attributes"""
        nc  = Dataset(self.file, 'r')
        assert(nc.stratt.encode('utf-8') == b'\xe6\xb7\xb1\xe5\x85\xa5 Python')
        stratt2 = nc.getncattr('stratt2',encoding='big5') # decodes using big5
        stratt3 = nc.getncattr('stratt3',encoding='big5') # same as above
        assert(stratt2.encode('big5') == b'\xb2`\xa4J Python')
        assert(nc.stratt == stratt2) # decoded strings are the same
        assert(nc.stratt == stratt3) # decoded strings are the same
        nc.close()

if __name__ == '__main__':
    unittest.main()
