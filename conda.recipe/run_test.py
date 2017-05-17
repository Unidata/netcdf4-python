import os
import netCDF4

# Run the unittests, skipping the opendap test.
test_dir = os.path.join('test')
os.chdir(test_dir)
os.environ['NO_NET']='1'
os.system('python run_all.py')
