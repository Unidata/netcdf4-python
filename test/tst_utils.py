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
        elem = [slice(None), [1,2,3], 8]
        start, count, stride, put_ind = nc._getStartCountStride(elem, (50, 4, 10))
        assert_equal(start[..., 0], 0)
        assert_equal(start[..., 1].squeeze(), [1,2,3])
        assert_equal(start[..., 2], 8)
        assert_equal(count[...,0], 50)
        assert_equal(count[...,1], 1)       
        assert_equal(count[...,2], 1)        
        assert_equal(put_ind[...,1].squeeze(), [0,1,2])
        
        assert_equal(nc._out_array_shape(count), (50, 3, 1))
        
        i = np.array([2,5,7],'i4')
        elem = [slice(None, -1,2),i,slice(None)]
        start, count, stride, put_ind = nc._getStartCountStride(elem, (9,10,11))
    
    def test_multiple_sequences(self):
        elem = [[4,5,6], [1,2, 3], slice(None)]
        start, count, stride, put_ind = nc._getStartCountStride(elem, (50, 4, 10))
        
        assert_equal(nc._out_array_shape(count), (3, 1, 10))
        
        assert_equal(start[..., 0].squeeze(), [4,5,6])
        assert_equal(start[..., 1].squeeze(),  [1,2, 3])
        assert_equal(start[..., 2], 0)
        assert_equal(count[...,0], 1)
        assert_equal(count[...,1], 1)       
        assert_equal(count[...,2], 10)        
        
        i = [1,2,3]
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
    
        elem = (np.array([True, True, False]), np.array([True, True, False, False]), slice(None))
        start, count, stride, put_ind = nc._getStartCountStride(elem, (3,4,5))
        assert_equal(nc._out_array_shape(count), (2,1,5))
    
        try:
            elem = (np.array([True, True, False]), np.array([True, True, True, False]), slice(None))
        except IndexError:
            pass
    
        try:
            elem = ( np.arange(6).reshape((3,2)), slice(None), slice(None) )
            start, count, stride, put_ind = nc._getStartCountStride(elem, (3,4,5))
        except IndexError:
            pass
    
    
        
if __name__=='__main__':
    unittest.main()
