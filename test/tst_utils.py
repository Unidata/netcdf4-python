from numpy.testing import assert_equal
from netCDF4.utils import _StartCountStride, _out_array_shape
import unittest
import numpy as np

class TestgetStartCountStride(unittest.TestCase):

    def test_basic(self):
        # Basic usage
        elem = [0, slice(None), slice(None)]
        start, count, stride, put_ind = _StartCountStride(elem, (50, 4, 10))
        assert_equal(start, 0)
        assert_equal(count[..., 0], 1)
        assert_equal(count[..., 1], 4)
        assert_equal(count[..., 2], 10)
        assert_equal(stride, 1)
        assert_equal(put_ind[...,0], -1)
        assert_equal(put_ind[...,1], slice(None))
        assert_equal(put_ind[...,2], slice(None))
        assert_equal(_out_array_shape(count), (1, 4,10))

    def test_slice(self):
        # start and stop slice
        elem = [5, slice(None), slice(5, 8, 2)]
        start, count, stride, put_ind = _StartCountStride(elem, (50, 4, 10))
        assert_equal(start[..., 0], 5)
        assert_equal(start[..., 1], 0)
        assert_equal(start[..., 2], 5)
        assert_equal(count[..., 0], 1)
        assert_equal(count[..., 1], 4)
        assert_equal(count[..., 2], 2)
        assert_equal(stride[..., 2], 2)

        assert_equal(_out_array_shape(count), (1, 4,2))

    def test_fancy(self):
        # Fancy indexing
        elem = [slice(None), [1,2,3], 8]
        start, count, stride, put_ind = _StartCountStride(elem, (50, 4, 10))
        assert_equal(start[..., 0], 0)
        assert_equal(start[..., 1].squeeze(), 1)
        assert_equal(start[..., 2], 8)
        assert_equal(count[...,0], 50)
        assert_equal(count[...,1], 3)
        assert_equal(count[...,2], 1)
        assert_equal(put_ind[...,1].squeeze(), slice(None, None, None))

        assert_equal(_out_array_shape(count), (50, 3, 1))

        i = np.array([2,5,7],'i4')
        elem = [slice(None, -1,2),i,slice(None)]
        start, count, stride, put_ind = _StartCountStride(elem, (9,10,11))


        try:
            elem = ( np.arange(6).reshape((3,2)), slice(None), slice(None) )
            start, count, stride, put_ind = _StartCountStride(elem, (3,4,5))
        except IndexError:
            pass

        # this one should be converted to a slice
        elem = [slice(None), [1,3,5], 8]
        start, count, stride, put_ind = _StartCountStride(elem, (50, 6, 10))
        # pull request #683 now does not convert integer sequences to strided
        # slices.
        #assert_equal(put_ind[...,1].squeeze(), slice(None,None,None))
        assert_equal(put_ind[...,1].squeeze(), [0,1,2])


    def test_multiple_sequences(self):
        elem = [[4,6,7], [1,2,3], slice(None)]
        start, count, stride, put_ind = _StartCountStride(elem, (50, 4, 10))

        assert_equal(_out_array_shape(count), (3, 3, 10))

        assert_equal(start[..., 0].squeeze(), [4,6,7])
        assert_equal(start[..., 1].squeeze(), [1,1,1])
        assert_equal(start[..., 2], 0)
        assert_equal(count[...,0], 1)
        assert_equal(count[...,1], 3)
        assert_equal(count[...,2], 10)

        i = [1,3,4]
        elem = (i, i, i)
        start, count, stride, put_ind = _StartCountStride(elem, (50, 5, 10))
        assert_equal(_out_array_shape(count), (3,3,3))

    def test_put_indices(self):
        elem = (1, slice(None), slice(None))
        start, count, stride, put_ind = _StartCountStride(elem, (3,4,5))
        orig = np.arange(60).reshape((3,4,5))
        dest = np.empty(_out_array_shape(count))
        dest[tuple(put_ind[0,0,0])] = orig[tuple(elem)]

    def test_boolean(self):
        elem = (1, slice(None), np.array([True, True, False, False, True]))
        start, count, stride, put_ind = _StartCountStride(elem, (50, 4,5))

        assert_equal(start[..., 2].squeeze(), [0,1,4])
        assert_equal(count[...,2], 1)

        assert_equal(_out_array_shape(count), (1, 4, 3))

        # Multiple booleans --- The behavior is different from NumPy in this case.
        elem = (np.array([True, True, False]), np.array([True, True, False, True]), slice(None))
        start, count, stride, put_ind = _StartCountStride(elem, (3,4,5))
        assert_equal(_out_array_shape(count), (2,3,5))

        try:
            elem = (np.array([True, True, False]), np.array([True, True, True, False]), slice(None))
        except IndexError:
            pass



    def test_1d(self):
        # Scalar
        elem = (0)
        start, count, stride, put_ind = _StartCountStride(elem, (10,))
        assert_equal(start, 0)
        assert_equal(count, 1)
        assert_equal(stride, 1)
        assert_equal(put_ind, -1)

        elem = (-1)
        start, count, stride, put_ind = _StartCountStride(elem, (10,))
        assert_equal(start, 9)
        assert_equal(count, 1)
        assert_equal(stride, 1)
        assert_equal(put_ind, -1)

        # test conversion of a integer index array to a slice
        elem = (np.array([0,1]))
        start, count, stride, put_ind = _StartCountStride(elem, (10,))
        assert_equal(start, 0)
        assert_equal(count, 2)
        assert_equal(stride, 1)
        assert_equal(put_ind[:,0], slice(None,None,None))

        # Slice
        elem = (slice(2,5,2))
        start, count, stride, put_ind = _StartCountStride(elem, (10,))
        assert_equal(start, 2)
        assert_equal(count, 2)
        assert_equal(stride, 2)
        assert_equal(put_ind, slice(None))

        # Integer sequence
        elem = ([2,4,7])
        start, count, stride, put_ind = _StartCountStride(elem, (10,))
        assert_equal(start.squeeze(), [2,4,7])
        assert_equal(count, 1)
        assert_equal(stride, 1)
        assert_equal(put_ind[:,0], [0,1,2])

        # Boolean slicing
        elem = (np.array([True, True, False, True, False]),)
        start, count, stride, put_ind = _StartCountStride(elem, (5,))
        assert_equal(start.squeeze(), [0,1,3])
        assert_equal(count, 1)
        assert_equal(stride, 1)
        assert_equal(put_ind[:,0], [0,1,2])

        # Integer sequence simplification
        elem = ([2,3,4])
        start, count, stride, put_ind = _StartCountStride(elem, (10,))
        assert_equal(start, 2)
        assert_equal(count, 3)
        assert_equal(stride, 1)
        assert_equal(put_ind, slice(None))

        # Boolean indices simplification
        elem = (np.array([False, True, True, True, False]))
        start, count, stride, put_ind = _StartCountStride(elem, (5,))
        assert_equal(start, 1)
        assert_equal(count, 3)
        assert_equal(stride, 1)
        assert_equal(put_ind, slice(None))

        # All False
        elem = (np.array([False, False, False, False]))
        start, count, stride, put_ind = _StartCountStride(elem, (4,))

        assert_equal(count, 0)
        assert_equal(_out_array_shape(count), (0,))

    def test_ellipsis(self):
        elem=(Ellipsis, slice(1, 4))
        start, count, stride, put_ind = _StartCountStride(elem, (22,25,4))
        assert_equal(start[0,0,0], [0, 0, 1])
        assert_equal(count[0,0,0], (22, 25, 3))
        assert_equal(put_ind[0,0,0], (slice(None), slice(None), slice(None)))

        elem=(Ellipsis, [15,16,17,18,19], slice(None), slice(None))
        start, count, stride, put_ind = _StartCountStride(elem, (2,10,20,10,10))
        assert_equal(start[0,0,0,0,0], [0, 0, 15, 0, 0])
        assert_equal(count[0,0,0,0,0], (2, 10, 5, 10, 10))
        assert_equal(put_ind[0,0,0,0,0], (slice(None), slice(None), slice(None), slice(None), slice(None)))
        
        try:
            elem=(Ellipsis, [15,16,17,18,19], slice(None))
            start, count, stride, put_ind = _StartCountStride(elem, (2,10,20,10,10))
            assert_equal(None, 'Should throw an exception')
        except IndexError as e:
            assert_equal(str(e), "integer index exceeds dimension size")
            
        try:
            elem=(Ellipsis, [15,16,17,18,19], Ellipsis)
            start, count, stride, put_ind = _StartCountStride(elem, (2,10, 20,10,10))
            assert_equal(None, 'Should throw an exception')
        except IndexError as e:
            assert_equal(str(e), "At most one ellipsis allowed in a slicing expression")
            
class TestsetStartCountStride(unittest.TestCase):

    def test_basic(self):

        grp = FakeGroup({'x':False, 'y':False, 'time':True})

        elem=(slice(None), slice(None), 1)
        start, count, stride, take_ind = _StartCountStride(elem, (22, 25, 1), ['x', 'y', 'time'], grp, (22,25), put=True)
        assert_equal(start[0][0][0], [0, 0, 1])
        assert_equal(count[0][0][0], (22, 25, 1))
        assert_equal(take_ind[0][0][0], (slice(None), slice(None), -1))

        elem=(slice(None), slice(None), slice(1, 4))
        start, count, stride, take_ind = _StartCountStride(elem, (22,25,1),\
            ['x', 'y', 'time'], grp, (22,25,3), put=True)
        assert_equal(start[0][0][0], [0, 0, 1])
        assert_equal(count[0][0][0], (22, 25, 3))
        assert_equal(take_ind[0][0][0], (slice(None), slice(None), slice(None)))

    def test_integer(self):
        grp = FakeGroup({'x':False, 'y':False})

        elem=([0,4,5], slice(20, None))
        start, count, stride, take_ind = _StartCountStride(elem, (22, 25), ['x', 'y'], grp, (3,5), put=True)
        assert_equal(start[0][0], (0, 20))
        assert_equal(start[1][0], (4, 20))
        assert_equal(start[2][0], (5, 20))
        assert_equal(count[0], np.array([[1,5],]))
        assert_equal(stride[0][0], (1, 1))
        assert_equal(take_ind[0][0], (0, slice(None)))
        assert_equal(take_ind[1][0], (1, slice(None)))
        assert_equal(take_ind[2][0], (2, slice(None)))

    def test_booleans(self):
        grp = FakeGroup({'x':False, 'y':False, 'z':False})

        elem=([0,4,5], np.array([False, True, False, True, True]), slice(None))
        start, count, stride, take_ind = _StartCountStride(elem, (10, 5, 12), ['x', 'y', 'z'], grp, (3, 3, 12), put=True)
        assert_equal(start[0][0][0], (0, 1, 0))
        assert_equal(start[1][0][0], (4, 1, 0))
        assert_equal(start[2][0][0], (5, 1, 0))
        assert_equal(start[0][1][0], (0, 3, 0))
        assert_equal(count[0][0][0], (1, 1, 12))
        assert_equal(stride[0][0][0], (1, 1, 1))
        assert_equal(take_ind[0][0][0], (0, 0, slice(None)))
        assert_equal(take_ind[1][0][0], (1, 0, slice(None)))
        assert_equal(take_ind[0][1][0], (0, 1, slice(None)))

    def test_unlim(self):
        grp = FakeGroup({'time':True,'x':False, 'y':False})

        elem = ([0,2,5], slice(None), slice(None))
        start, count, stride, take_ind = _StartCountStride(elem, (0, 6, 7),\
                ['time', 'x', 'y'], grp, (3, 6, 7), put=True)
        assert_equal(start[0][0][0], (0, 0, 0))
        assert_equal(start[2][0][0], (5, 0, 0))
        assert_equal(count[2][0][0], (1, 6, 7))
        assert_equal(take_ind[0][0][0], (0, slice(None), slice(None)))
        assert_equal(take_ind[2][0][0], (2, slice(None), slice(None)))


        # pull request #683 broke this, since _StartCountStride now uses
        # Dimension.__len__.
        #elem = (slice(None, None, 2), slice(None), slice(None))
        #start, count, stride, take_ind = _StartCountStride(elem, (0, 6, 7),\
        #        ['time', 'x', 'y'], grp, (10, 6, 7),put=True)
        #assert_equal(start[0][0][0], (0,0,0))
        #assert_equal(count[0][0][0], (5, 6, 7))
        #assert_equal(stride[0][0][0], (2, 1, 1))
        #assert_equal(take_ind[0][0][0], 3*(slice(None),))
     
    def test_ellipsis(self):
        grp = FakeGroup({'x':False, 'y':False, 'time':True})

        elem=(Ellipsis, slice(1, 4))
        start, count, stride, take_ind = _StartCountStride(elem, (22,25,1),\
            ['x', 'y', 'time'], grp, (22,25,3), put=True)
        assert_equal(start[0,0,0], [0, 0, 1])
        assert_equal(count[0,0,0], (22, 25, 3))
        assert_equal(take_ind[0,0,0], (slice(None), slice(None), slice(None)))
        
        grp = FakeGroup({'time':True, 'h':False, 'z':False, 'y':False, 'x':False})

        elem=(Ellipsis, [15,16,17,18,19], slice(None), slice(None))
        start, count, stride, take_ind = _StartCountStride(elem, (2,10,20,10,10),\
            ['time', 'h', 'z', 'y', 'x'], grp, (2,10,5,10,10), put=True)
        assert_equal(start[0,0,0,0,0], [0, 0, 15, 0, 0])
        assert_equal(count[0,0,0,0,0], [2, 10, 5, 10, 10])
        assert_equal(stride[0,0,0,0,0], [1, 1, 1, 1, 1])
        assert_equal(take_ind[0,0,0,0,0], (slice(None), slice(None), slice(None), slice(None), slice(None)))
        
        try:
            elem=(Ellipsis, [15,16,17,18,19], slice(None))
            start, count, stride, take_ind = _StartCountStride(elem, (2,10,20,10,10),\
               ['time', 'z', 'y', 'x'], grp, (2,10,5,10,10), put=True)
            assert_equal(None, 'Should throw an exception')
        except IndexError as e:
            assert_equal(str(e), "integer index exceeds dimension size")
            
        try:
            elem=(Ellipsis, [15,16,17,18,19], Ellipsis)
            start, count, stride, take_ind = _StartCountStride(elem, (2,10, 20,10,10),\
               ['time', 'z', 'y', 'x'], grp, (2,10,5,10,10), put=True)
            assert_equal(None, 'Should throw an exception')
        except IndexError as e:
            assert_equal(str(e), "At most one ellipsis allowed in a slicing expression")
       
class FakeGroup(object):
    """Create a fake group instance by passing a dictionary of booleans
    keyed by dimension name."""
    def __init__(self, dimensions):
        self.dimensions = {}
        for k,v in dimensions.items():
            self.dimensions[k] = FakeDimension(v)

class FakeDimension(object):
    def __init__(self, unlimited=False):
        self.unlimited = unlimited

    def isunlimited(self):
        return self.unlimited

if __name__=='__main__':
    unittest.main()
