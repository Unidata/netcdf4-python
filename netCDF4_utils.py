import numpy
from numpy import ma
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
            try:
                group = group.parent
            except:
                raise ValueError("cannot find dimension %s in this group or parent groups" % dimname)
    return dim

def _buildStartCountStride(elem, shape, dimensions, grp, datashape=None):

    # Create the 'start', 'count', and 'stride' tuples that
    # will be passed to 'nc_get_var'/'nc_put_var'.
    #   start     starting indices along each dimension
    #   count     count of values along each dimension; a value of -1
    #             indicates that and index, not a slice, was applied to
    #             the dimension; in that case, the dimension should be
    #             dropped from the output array.
    #   stride    strides along each dimension

    # Adapted from pycdf (http://pysclint.sourceforge.net/pycdf)
    # by Andre Gosselin.

    # Handle a scalar variable as a 1-dimensional array of length 1.

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
    #                 slicing expression, otherwise the expression is ambiguous
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

    # if slice doesn't cover all dims, assume ellipsis for rest of dims.
    if len(elem) < len(shape):
        for n in range(len(elem)+1,len(shape)+1):
            elem.append(slice(None,None,None))

    # Build start, count, stride tuples.
    start = []
    count = []
    stride = []
    sliceout = []
    hasfancyindex = False
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

        # is this element of the tuple a sequence? (but not a string)
        try:
            e[:]
            if type(e) != types.StringType:
                isSequenceType = True
            else:
                isSequenceType = False
        except:
            isSequenceType = False
        
        # Slice index. Respect Python syntax for slice upper bounds,
        # which are not included in the resulting slice. Also, if the
        # upper bound exceed the dimension size, truncate it.
        if type(e) == types.SliceType:
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
                        if unlim and datashape is not None:
                            length = datashape[n]
                        else:
                            raise IndexError('illegal slice')
                    else:
                        length = e.stop
                else:
                    if e.start is None:
                        raise IndexError('illegal slice')
                    else:
                        length = e.start+1
                beg, end, inc = e.indices(length)
            sliceout.append(slice(None,None,None))
        elif isSequenceType: # a sequence for 'fancy indexing'
        # just grab all the data along this dimension
        # then slice the resulting numpy array with the sequence
            isSlice = 1
            hasfancyindex = 1
            beg, end, inc = 0, shape[n], 1
            sliceout.append(e)
        # it's a simple integer index
        elif type(e) == types.IntType:
            isSlice = 0       # not a slice
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
        # a rank-0 numpy integer array.
        elif hasattr(e,'dtype') and e.dtype.kind in ['i','u']:
            e = int(e)
            isSlice = 0       # not a slice
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
        else:
            raise IndexError('slice must be an integer, a sequence of integers, a slice object, or an integer array scalar')

        # Clip end index (except if unlimited dimension)
        # and compute number of elements to get.
        if not unlim and end > shape[n]:
            end = shape[n]
        if isSlice:   
            cnt = len(xrange(beg,end,inc))
        else: 
            cnt = -1 # -1 means a single element.
        start.append(beg)
        count.append(cnt)
        stride.append(inc)

    # Complete missing dimensions
    while n < nDims - 1:
        n = n+1
        start.append(0)
        count.append(shape[n])
        stride.append(1)
        sliceout.append(slice(None,None,None))

    # if no fancy indexing requested, just set sliceout to None.
    if not hasfancyindex: sliceout = None

    # Done
    return start, count, stride, sliceout

def _quantize(data,least_significant_digit):
    """
quantize data to improve compression. data is quantized using 
around(scale*data)/scale, where scale is 2**bits, and bits is determined 
from the least_significant_digit. For example, if 
least_significant_digit=1, bits will be 4.
    """
    precision = pow(10.,-least_significant_digit)
    exp = numpy.log10(precision)
    if exp < 0:
        exp = int(numpy.floor(exp))
    else:
        exp = int(numpy.ceil(exp))
    bits = numpy.ceil(numpy.log2(pow(10.,-exp)))
    scale = pow(2.,bits)
    datout = numpy.around(scale*data)/scale
    if hasattr(datout,'mask'):
        datout.set_fill_value(data.fill_value())
        return datout
    else:
        return datout
