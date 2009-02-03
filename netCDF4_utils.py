import numpy
import numpy as np
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

def _getStartCountStride(elem, shape):
    """Return start, count, stride and put_indices that store the information 
    needed to extract chunks of data from a netCDF variable and put those
    chunks in the output array. 
    
    This function is used to convert a NumPy index into a form that is 
    compatible with the nc_get_vars function. Specifically, it needs
    to interpret slices, ellipses, sequences of integers as well as
    sequences of booleans. Note that all the fancy indexing tricks
    implemented in NumPy are not supported. In particular, multidimensional
    indexing is not supported and will raise an IndexError. 

    Parameters
    ----------
    elem : tuple of integer, slice, ellipsis or sequence of integers. 
      The indexing information.
    shape : tuple
      The shape of the netCDF variable.
      
    Returns
    -------
    start : ndarray (..., n)
      A starting indices array of dimension n+1. The first n 
      dimensions identify different independent data chunks. The last dimension 
      can be read as the starting indices.
    count : ndarray (..., n)
      An array of dimension (n+1) storing the number of elements to get. 
    stride : ndarray (..., n)
      An array of dimension (n+1) storing the steps between each datum. 
    put_indices : ndarray (..., n)
      An array storing the indices describing the location of the 
      data chunk in the target array. 
      
    Notes:
    
    netCDF data is accessed via the function: 
       nc_get_vars(grpid, varid, start, count, stride, data)
       
    Assume that the variable has dimension n, then 
    
    start is a n-tuple that contains the indices at the beginning of data chunk.
    count is a n-tuple that contains the number of elements to be accessed. 
    stride is a n-tuple that contains the step length between each element. 
        
    Note that this function will only work when getting data out, not 
    setting it. 
    
    """
    # Adapted from pycdf (http://pysclint.sourceforge.net/pycdf)
    # by Andre Gosselin..
    # Modified by David Huard to handle efficiently fancy indexing with
    # sequences of integers. 
    
    nDims = len(shape)
    if nDims == 0:
        ndims = 1
        shape = (1,)
    
    # Make sure the indexing expression does not exceed the variable
    # number of dimensions.
    if np.iterable(elem):
        if len(elem) > nDims:
            raise ValueError("slicing expression exceeds the number of dimensions of the variable")
    else:   # Convert single index to sequence
        elem = [elem]
        
    # Replace ellipsis with slices.
    hasEllipsis = 0
    newElem = []
    for e in elem:
        if type(e) == types.EllipsisType:
            if hasEllipsis:
                raise IndexError("At most one ellipsis allowed in a slicing expression")
            # The ellipsis stands for the missing dimensions.
            newElem.extend((slice(None, None, None),) * (nDims - len(elem) + 1))
        else:
            newElem.append(e)
    elem = newElem

    # If slice doesn't cover all dims, assume ellipsis for rest of dims.
    if len(elem) < len(shape):
        for n in range(len(elem)+1,len(shape)+1):
            elem.append(slice(None,None,None))  

    # Compute the dimensions of the start, count, stride and put_indices arrays.
    # The number of elements in the first n dimensions corresponds to the 
    # number of times the _get method will be called. 
    sdim = []
    ind_dim = None
    for i, e in enumerate(elem):
        # Raise error if multidimensional indexing is used. 
        if np.ndim(e) > 1:
            raise IndexError("Index cannot be multidimensional.")
        
        # Slices
        if type(e) is types.SliceType:
            sdim.append(1)
            
        # Booleans --- Same shape as data along corresponding dimension
        elif getattr(getattr(e, 'dtype', None), 'kind', None) == 'b':
            if shape[i] != len(e):
                raise IndexError, 'Boolean array must have the same shape as the data along this dimension.'
            elif ind_dim is None:
                sdim.append(e.sum())
                ind_dim = i
            elif e.sum() == 1 or e.sum() == sdim[ind_dim]:
                sdim.append(1)
            else:
                raise IndexError, "Boolean arrays must have the same number of True elements."
            
        # Sequence of indices
        # If multiple sequences are used, they must have the same length. 
        elif np.iterable(e):
            if ind_dim is None:
                sdim.append(np.alen(e))
                ind_dim = i
            elif np.alen(e) == 1 or np.alen(e) == sdim[ind_dim]:
                sdim.append(1)
            else:
                raise IndexError, "Indice mismatch. Indices must have the same length."
        # Scalar
        else:
            sdim.append(1)
        
    # Create the start, count, stride and put_indices arrays. 
    
    sdim.append(max(nDims, 1))
    start = np.empty(sdim, dtype=int)
    count = np.empty(sdim, dtype=int)
    stride = np.empty(sdim, dtype=int)
    put_indices = np.empty(sdim, dtype=object)
    
    for i, e in enumerate(elem):
        #    SLICE    #
        if type(e) is types.SliceType:
            beg, end, inc = e.indices(shape[i])
            n = len(xrange(beg,end,inc))
            
            start[...,i] = beg
            count[...,i] = n
            stride[...,i] = inc
            put_indices[...,i] = slice(None)
            
            
        #    STRING    #
        elif type(e) is str:
            raise IndexError("Index cannot be a string.")

        #    ITERABLE    #
        elif np.iterable(e) and np.array(e).dtype.kind in 'ib':  # Sequence of integers or booleans
        
            #    BOOLEAN ARRAY   #
            if type(e) == np.ndarray and e.dtype.kind == 'b':
                e = np.arange(len(e))[e]
                
                # Originally, I thought boolean indexing worked differently than 
                # integer indexing, namely that we could select the rows and columns 
                # independently. 
                #start[...,i] = np.apply_along_axis(lambda x: np.array(e)*x, i, np.ones(sdim[:-1]))
                #put_indices[...,i] = np.apply_along_axis(lambda x: np.arange(sdim[i])*x, i, np.ones(sdim[:-1], int))
                
                
            # Sequence of INTEGER INDICES
            
            start[...,i] = np.apply_along_axis(lambda x: np.array(e)*x, ind_dim, np.ones(sdim[:-1]))
            if i == ind_dim:
                put_indices[...,i] = np.apply_along_axis(lambda x: np.arange(sdim[i])*x, ind_dim, np.ones(sdim[:-1], int))
            else:
                put_indices[...,i] = -1

            count[...,i] = 1
            stride[...,i] = 1
            
            
        #    SCALAR INTEGER    #
        elif np.alen(e)==1 and np.dtype(type(e)).kind is 'i': 
            if e >= 0: 
                start[...,i] = e
            elif e < 0 and (-e < shape[i]) :
                start[...,i] = e+shape[n]
            else:
                raise IndexError("Index out of range")
            
            count[...,i] = 1
            stride[...,i] = 1
            put_indices[...,i] = -1    # Use -1 instead of 0 to indicate that 
                                       # this dimension shall be squeezed. 
            
            
    return start, count, stride, put_indices#, out_shape

def _out_array_shape(count):
    """Return the output array shape given the count array created by getStartCountStride"""
    
    s = list(count.shape[:-1])
    out = []
    
    for i, n in enumerate(s):
        if n == 1:
            c = count[..., i].ravel()[0] # All elements should be identical.
            out.append(c)
        else:
            out.append(n)
    return out

