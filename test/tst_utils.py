from numpy.testing import assert_equal
import netCDF4 as nc
import unittest
import numpy as np

class TestgetStartCountStride(unittest.TestCase):
    
    def test_basic(self):
        # Basic usage
        elem = [0, slice(None), slice(None)]
        start, count, stride, put_ind = nc._getStartCountStride(elem, (50, 4, 10))
        assert_equal(start, 0)
        assert_equal(count[..., 0], 1)
        assert_equal(count[..., 1], 4)
        assert_equal(count[..., 2], 10)
        assert_equal(stride, 1)
        assert_equal(put_ind[...,0], -1)
        assert_equal(put_ind[...,1], slice(None))
        assert_equal(put_ind[...,2], slice(None))
        assert_equal(nc._out_array_shape(count), (1, 4,10))
    
    def test_slice(self):
        # start and stop slice
        elem = [5, slice(None), slice(5, 8, 2)]
        start, count, stride, put_ind = nc._getStartCountStride(elem, (50, 4, 10))
        assert_equal(start[..., 0], 5)
        assert_equal(start[..., 1], 0)
        assert_equal(start[..., 2], 5)
        assert_equal(count[..., 0], 1)
        assert_equal(count[..., 1], 4)
        assert_equal(count[..., 2], 2)
        assert_equal(stride[..., 2], 2)
                
        assert_equal(nc._out_array_shape(count), (1, 4,2))

    def test_fancy(self):
        # Fancy indexing
        elem = [slice(None), [1,3,4], 8]
        start, count, stride, put_ind = nc._getStartCountStride(elem, (50, 4, 10))
        assert_equal(start[..., 0], 0)
        assert_equal(start[..., 1].squeeze(), [1,3,4])
        assert_equal(start[..., 2], 8)
        assert_equal(count[...,0], 50)
        assert_equal(count[...,1], 1)       
        assert_equal(count[...,2], 1)        
        assert_equal(put_ind[...,1].squeeze(), [0,1,2])
        
        assert_equal(nc._out_array_shape(count), (50, 3, 1))
        
        i = np.array([2,5,7],'i4')
        elem = [slice(None, -1,2),i,slice(None)]
        start, count, stride, put_ind = nc._getStartCountStride(elem, (9,10,11))
    
    
        try:
            elem = ( np.arange(6).reshape((3,2)), slice(None), slice(None) )
            start, count, stride, put_ind = nc._getStartCountStride(elem, (3,4,5))
        except IndexError:
            pass

        # this one should be converted to a slice
        elem = [slice(None), [1,3,5], 8]
        start, count, stride, put_ind = nc._getStartCountStride(elem, (50, 4, 10))
        assert_equal(put_ind[...,1].squeeze(), slice(None,None,None))
    
    
    def test_multiple_sequences(self):
        elem = [[4,6,7], [1,3, 4], slice(None)]
        start, count, stride, put_ind = nc._getStartCountStride(elem, (50, 4, 10))
        
        assert_equal(nc._out_array_shape(count), (3, 1, 10))
        
        assert_equal(start[..., 0].squeeze(), [4,6,7])
        assert_equal(start[..., 1].squeeze(),  [1,3, 4])
        assert_equal(start[..., 2], 0)
        assert_equal(count[...,0], 1)
        assert_equal(count[...,1], 1)       
        assert_equal(count[...,2], 10)        
        
        i = [1,3,4]
        elem = (i, i, i)
        start, count, stride, put_ind = nc._getStartCountStride(elem, (50, 4, 10))
        assert_equal(nc._out_array_shape(count), (3,1,1))
        
    def test_put_indices(self):
        elem = (1, slice(None), slice(None))
        start, count, stride, put_ind = nc._getStartCountStride(elem, (3,4,5))
        orig = np.arange(60).reshape((3,4,5))
        dest = np.empty(nc._out_array_shape(count))
        dest[tuple(put_ind[0,0,0])] = orig[tuple(elem)]
        
    def test_boolean(self):
        elem = (1, slice(None), np.array([True, True, False, False, True]))
        start, count, stride, put_ind = nc._getStartCountStride(elem, (50, 4,5))

        assert_equal(start[..., 2].squeeze(), [0,1,4])
        assert_equal(count[...,2], 1)
        
        assert_equal(nc._out_array_shape(count), (1, 4, 3))
    
        # Multiple booleans --- The behavior is different from NumPy in this case. 
        elem = (np.array([True, True, False]), np.array([True, True, False, True]), slice(None))
        start, count, stride, put_ind = nc._getStartCountStride(elem, (3,4,5))
        assert_equal(nc._out_array_shape(count), (2,3,5))
    
        try:
            elem = (np.array([True, True, False]), np.array([True, True, True, False]), slice(None))
        except IndexError:
            pass
    
    def test_1d(self):
        # Scalar
        elem = (0)
        start, count, stride, put_ind = nc._getStartCountStride(elem, (10,))
        assert_equal(start, 0)
        assert_equal(count, 1)
        assert_equal(stride, 1)
        assert_equal(put_ind, -1)
        
        # Slice
        elem = (slice(2,5,2))
        start, count, stride, put_ind = nc._getStartCountStride(elem, (10,))
        assert_equal(start, 2)
        assert_equal(count, 2)
        assert_equal(stride, 2)
        assert_equal(put_ind, slice(None))
        
        # Integer sequence
        elem = ([2,4,7])
        start, count, stride, put_ind = nc._getStartCountStride(elem, (10,))
        assert_equal(start.squeeze(), [2,4,7])
        assert_equal(count, 1)
        assert_equal(stride, 1)
        assert_equal(put_ind[:,0], [0,1,2])

        # Boolean slicing
        elem = (np.array([True, True, False, True, False]),)
        start, count, stride, put_ind = nc._getStartCountStride(elem, (5,))
        assert_equal(start.squeeze(), [0,1,3])
        assert_equal(count, 1)
        assert_equal(stride, 1)
        assert_equal(put_ind[:,0], [0,1,2])

        # Integer sequence simplification
        elem = ([2,3,4])
        start, count, stride, put_ind = nc._getStartCountStride(elem, (10,))
        assert_equal(start, 2)
        assert_equal(count, 3)
        assert_equal(stride, 1)
        assert_equal(put_ind, slice(None))

        # Boolean indices simplification
        elem = (np.array([False, True, True, True, False]))
        start, count, stride, put_ind = nc._getStartCountStride(elem, (5,))
        assert_equal(start, 1)
        assert_equal(count, 3)
        assert_equal(stride, 1)
        assert_equal(put_ind, slice(None))

        
class TestsetStartCountStride(unittest.TestCase):
    
    def test_basic(self):
        grp = FakeGroup({'x':False, 'y':False, 'time':True})
        elem=(slice(None), slice(None), 1)
        start, count, stride, put_ind = nc._getStartCountStride(elem, (22, 25, 1), ['x', 'y', 'time'], grp, (22,25))
    
        assert_equal(start[0][0][0], [0, 0, 1])
        assert_equal(count[0][0][0], (22, 25, 1))
        assert_equal(put_ind[0][0][0], (slice(None), slice(None), -1))
    
        elem=(slice(None), slice(None), slice(1, 4))
        start, count, stride, put_ind = nc._getStartCountStride(elem, (22, 25, 1), ['x', 'y', 'time'], grp, (22,25, 3))
    
        assert_equal(start[0][0][0], [0, 0, 1])
        assert_equal(count[0][0][0], (22, 25, 3))
        assert_equal(put_ind[0][0][0], (slice(None), slice(None), slice(1, 4)))
    
    
    
class FakeGroup():
    """Create a fake group instance by passing a dictionary of booleans
    keyed by dimension name."""
    def __init__(self, dimensions):
        self.dimensions = {}
        for k,v in dimensions.iteritems():
            self.dimensions[k] = FakeDimension(v)
    
class FakeDimension():
    def __init__(self, unlimited=False):
        self.unlimited = unlimited
    
    def isunlimited(self):
        return self.unlimited

if __name__=='__main__':
    unittest.main()
