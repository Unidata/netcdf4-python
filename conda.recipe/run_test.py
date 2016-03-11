import os
import netCDF4

# Check OPeNDAP functionality.
url = 'http://geoport-dev.whoi.edu/thredds/dodsC/estofs/atlantic'
nc = netCDF4.Dataset(url)

# Check if it was compiled with cython.
assert nc.filepath() == url

# Run the unittests.
test_dir = os.path.join(os.environ['SRC_DIR'], 'test')
os.chdir(test_dir)
os.system('python run_all.py')
