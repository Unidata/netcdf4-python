import glob, os, sys
test_files = glob.glob('tst_*.py')
py_path = os.environ.get('PYTHONPATH')
if py_path is None:
    py_path = '.'
else:
    py_path = os.pathsep.join(['.',py_path])
os.environ['PYTHONPATH'] = py_path

for f in test_files:
    ff = os.path.join(sys.path[0],f)
    args = [sys.executable,ff,'-v']
    print "Running",ff
    status = os.spawnve(os.P_WAIT,sys.executable,args,os.environ)
    if status:
        print 'TEST FAILURE (status=%s)' % (status)
