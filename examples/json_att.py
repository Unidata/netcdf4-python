from netCDF4 import Dataset
import json
# example showing how python objects (lists, dicts, None, True)
# can be serialized as strings, saved as netCDF attributes,
# and then converted back to python objects using json.
ds = Dataset('json.nc', 'w')
ds.pythonatt1 =  json.dumps([u'foo', {u'bar': [u'baz', None, 1.0, 2]}])
ds.pythonatt2 = "true" # converted to bool
ds.pythonatt3 = "null" # converted to None
print(ds)
ds.close()
ds = Dataset('json.nc')
def convert_json(s):
    try:
        a = json.loads(s)
        return a
    except:
        return s
x = convert_json(ds.pythonatt1)
print(type(x))
print(x)
print(convert_json(ds.pythonatt2))
print(convert_json(ds.pythonatt3))
ds.close()
