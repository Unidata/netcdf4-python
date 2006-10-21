import numpy
import types

def _find_dim(grp, dimname):
    # find Dimension instance given group and name.
    # look in current group, and parents.
    group = grp
    dim = None
    while 1:
        try:
            dim = group.dimensions[dimname]
            break
        except:
            group = group.parent
    return dim

def _buildStartCountStride(elem, shape, dimensions, grp):

    # Create the 'start', 'count', 'slice' and 'stride' tuples that
    # will be passed to 'nc_get_var_0'/'nc_put_var_0'.
    #   start     starting indices along each dimension
    #   count     count of values along each dimension; a value of -1
    #             indicates that and index, not a slice, was applied to
    #             the dimension; in that case, the dimension should be
    #             dropped from the output array.
    #   stride    strides along each dimension

    # Adapted from pycdf (http://pysclint.sourceforge.net/pycdf)
    # by Andre Gosselin.

    # Handle a scalar variable as a 1-dimensional array of length 1.

    # this function is pure python.

    nDims = len(dimensions)
    if nDims == 0:
        nDims = 1
        shape = (1,)

    # Make sure the indexing expression does not exceed the variable
    # number of dimensions.
    if type(elem) == types.TupleType:
        if len(elem) > nDims:
            raise ValueError("slicing expression exceeds the number of dimensions of the variable")
    else:   # Convert single index to sequence
        elem = [elem]
        
    # 'elem' is a tuple whose element types can be one of:
    #    IntType      for standard indexing
    #    SliceType    for extended slicing (using 'start', 'stop' and 'step' attributes)
    #    EllipsisType for an ellipsis (...); at most one ellipsis can occur in the
    #                 slicing expression, otherwise the expressionis ambiguous
    # Recreate the 'elem' tuple, replacing a possible ellipsis with empty slices.
    hasEllipsis = 0
    newElem = []
    for e in elem:
        if type(e) == types.EllipsisType:
            if hasEllipsis:
                raise IndexError("at most one ellipsis allowed in a slicing expression")
            # The ellipsis stands for the missing dimensions.
            newElem.extend((slice(None, None, None),) * (nDims - len(elem) + 1))
        else:
            newElem.append(e)
    elem = newElem

    # Build arguments to "nc_get_var/nc_put_var".
    start = []
    count = []
    stride = []
    n = -1
    for e in elem:
        n = n+1

        if len(dimensions):
            dimname = dimensions[n]
            # is this dimension unlimited?
            # look in current group, and parents for dim.
            dim = _find_dim(grp, dimname)
            unlim = dim.isunlimited()
        else:
            unlim = False
        
        # Simple index
        if type(e) == types.IntType:
            isSlice = 0       # we do not deal with a slice
            # Respect standard python sequence indexing behavior.
            # Count from the dimension end if index is negative.
            # Consider as illegal an out of bound index, except for the
            # unlimited dimension.
            if e < 0 :
                e = e+shape[n]
            if e < 0 or (not unlim and e >= shape[n]):
                raise IndexError("index out of range")
            beg = e
            end = e + 1
            inc = 1
            
        # Slice index. Respect Python syntax for slice upper bounds,
        # which are not included in the resulting slice. Also, if the
        # upper bound exceed the dimension size, truncate it.
        elif type(e) == types.SliceType:
            isSlice = 1     # we deal with a slice
            # None means not specified
            if e.step is not None:
                inc = e.step
            else:
                inc = 1
            if shape[n]: 
                if unlim and e.stop > shape[n]:
                    beg, end, inc = e.indices(e.stop)
                else:
                    beg, end, inc = e.indices(shape[n])
            else:
                if inc > 0:
                    if e.stop is None:
                        raise IndexError('illegal slice')
                    else:
                        length = e.stop
                else:
                    if e.start is None:
                        raise IndexError('illegal slice')
                    else:
                        length = e.start+1
                beg, end, inc = e.indices(length)
                
        # Bug
        else:
            raise ValueError("Bug: unexpected element type to __getitem__")

        # Clip end index (except if unlimited dimension)
        # and compute number of elements to get.
        if not unlim and end > shape[n]:
            end = shape[n]
        if isSlice:       # we deal with a slice
            cnt = len(xrange(beg,end,inc))
        else: 
            cnt = -1
        start.append(beg)
        count.append(cnt)
        stride.append(inc)

    # Complete missing dimensions
    while n < nDims - 1:
        n = n+1
        start.append(0)
        count.append(shape[n])
        stride.append(1)

    # Done
    return start, count, stride

def _quantize(data,least_significant_digit):
    """

quantize data to improve compression. data is quantized using 
around(scale*data)/scale, where scale is 2**bits, and bits is determined 
from the least_significant_digit. For example, if 
least_significant_digit=1, bits will be 4.

This function is pure python.

    """
    precision = pow(10.,-least_significant_digit)
    exp = numpy.log10(precision)
    if exp < 0:
        exp = int(numpy.floor(exp))
    else:
        exp = int(numpy.ceil(exp))
    bits = numpy.ceil(numpy.log2(pow(10.,-exp)))
    scale = pow(2.,bits)
    return numpy.around(scale*data)/scale
