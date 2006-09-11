import glob, os, sys
test_files = glob.glob('ex_*.py')
test_files = test_files + ['tutorial.py', 'bench.py']
py_path = os.environ.get('PYTHONPATH')
if py_path is None:
    py_path = '.'
else:
    py_path = os.pathsep.join(['.',py_path])
os.environ['PYTHONPATH'] = py_path

for f in test_files:
    print "**********************************************"
    ff = os.path.join(sys.path[0],f)
    args = [sys.executable,ff]
    print "Running",' '.join(args)
    status = os.spawnve(os.P_WAIT,sys.executable,args,os.environ)
    if status:
        print 'TEST FAILURE (status=%s)' % (status)
