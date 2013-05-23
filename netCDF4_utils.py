import numpy as np
from numpy import ma
import warnings


def _sortbylist(A,B):
    # sort one list (A) using the values from another list (B)
    return [A[i] for i in sorted(range(len(A)), key=B.__getitem__)]

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

def _quantize(data,least_significant_digit):
    """
quantize data to improve compression. data is quantized using
around(scale*data)/scale, where scale is 2**bits, and bits is determined
from the least_significant_digit. For example, if
least_significant_digit=1, bits will be 4.
    """
    precision = pow(10.,-least_significant_digit)
    exp = np.log10(precision)
    if exp < 0:
        exp = int(np.floor(exp))
    else:
        exp = int(np.ceil(exp))
    bits = np.ceil(np.log2(pow(10.,-exp)))
    scale = pow(2.,bits)
    datout = np.around(scale*data)/scale
    if ma.isMA(datout):
        datout.set_fill_value(data.fill_value)
        return datout
    else:
        return datout

def _StartCountStride(elem, shape, dimensions=None, grp=None, datashape=None):
    """Return start, count, stride and indices needed to store/extract data
    into/from a netCDF variable.

    This function is used to convert a NumPy index into a form that is
    compatible with the nc_get_vars function. Specifically, it needs
    to interpret slices, ellipses, sequences of integers as well as
    sequences of booleans.

    Note that all the fancy indexing tricks
    implemented in NumPy are not supported. In particular, multidimensional
    indexing is not supported and will raise an IndexError. Note also that
    boolean indexing does not work as in NumPy. In NumPy, booleans arrays
    behave identically to integer indices. For netCDF variables, we thought
    it would be useful to use a different logic, namely dimension independence.
    What this means is that you can do:
    >>> v[lat>60, lon<180, :]
    to fetch the elements of v obeying conditions on latitude and longitude.

    This function is used both by the __setitem__ and __getitem__ method of
    the Variable class. Although the behavior is similar in both cases, there
    are some differences to be noted.

    Parameters
    ----------
    elem : tuple of integer, slice, ellipsis or sequence of integers.
      The indexing information for the netCDF Variable: Variable[elem]
    shape : tuple
      The current shape of the netCDF variable.
    dimensions : sequence
      The name of the dimensions. This is only useful to find out
      whether or not some dimensions are unlimited. Only needed within
      __setitem__.
    grp  : netCDF Group
      The netCDF group to which the variable being set belongs to.
      Only needed within __setitem__.
    datashape : sequence
      The shape of the data that is being stored. Only needed within
      __setitem__.

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
    indices : ndarray (..., n)
      An array storing the indices describing the location of the
      data chunk in the target/source array (__getitem__/__setitem__).

    Notes:

    netCDF data is accessed via the function:
       nc_get_vars(grpid, varid, start, count, stride, data)

    Assume that the variable has dimension n, then

    start is a n-tuple that contains the indices at the beginning of data chunk.
    count is a n-tuple that contains the number of elements to be accessed.
    stride is a n-tuple that contains the step length between each element.

    """
    # Adapted from pycdf (http://pysclint.sourceforge.net/pycdf)
    # by Andre Gosselin..
    # Modified by David Huard to handle efficiently fancy indexing with
    # sequences of integers or booleans.

    nDims = len(shape)
    if nDims == 0:
        nDims = 1
        shape = (1,)

    # When a single array or (non-tuple) sequence of integers is given
    # as a slice, assume it applies to the first dimension,
    # and use ellipsis for remaining dimensions.
    if np.iterable(elem):
        if type(elem) == np.ndarray or (type(elem) != tuple and \
            np.array([_is_int(e) for e in elem]).all()):
            elem = [elem]
            for n in range(len(elem)+1,nDims+1):
                elem.append(slice(None,None,None))
    else:   # Convert single index to sequence
        elem = [elem]

    hasEllipsis = 0
    newElem = []
    for e in elem:
        # Raise error if multidimensional indexing is used.
        if np.ndim(e) > 1:
            raise IndexError("Index cannot be multidimensional.")
        # Replace ellipsis with slices.
        if type(e) == type(Ellipsis):
            if hasEllipsis:
                raise IndexError("At most one ellipsis allowed in a slicing expression")
            # The ellipsis stands for the missing dimensions.
            newElem.extend((slice(None, None, None),) * (nDims - len(elem) + 1))
        # Replace boolean array with slice object if possible.
        elif getattr(getattr(e, 'dtype', None), 'kind', None) == 'b':
            el = e.tolist()
            if any(el):
                start = el.index(True)
                el.reverse()
                stop = len(el)-el.index(True)
                step = False
                if e[start:stop].all():
                    step = 1
                else:
                    n1 = start+1
                    ee = e[n1]
                    estart = e[start]
                    while ee != estart:
                        n1 = n1 + 1
                        ee = e[n1]
                    step = n1-start
                    # check to make sure e[start:stop:step] are all True,
                    # and other elements in e[start:stop] are all False.
                    ii = range(start,stop,step)
                    for i in range(start,stop):
                        if i not in ii:
                            if e[i]: step = False
                        else:
                            if not e[i]: step = False
                if step: # it step False, can't convert to slice.
                    newElem.append(slice(start,stop,step))
                else:
                    newElem.append(e)
            else:
                newElem.append(slice(0,0))

        # Replace sequence of indices with slice object if possible.
        elif np.iterable(e) and len(e) > 1:
            start = e[0]
            stop = e[-1]+1
            step = e[1]-e[0]
            try:
                ee = range(start,stop,step)
            except ValueError: # start, stop or step is not valid for a range
                ee = False
            if ee and len(e) == len(ee) and (e == np.arange(start,stop,step)).all():
                newElem.append(slice(start,stop,step))
            else:
                newElem.append(e)
        else:
            newElem.append(e)
    elem = newElem

    # If slice doesn't cover all dims, assume ellipsis for rest of dims.
    if len(elem) < nDims:
        for n in range(len(elem)+1,nDims+1):
            elem.append(slice(None,None,None))

    # make sure there are not too many dimensions in slice.
    if len(elem) > nDims:
        raise ValueError("slicing expression exceeds the number of dimensions of the variable")

    # Compute the dimensions of the start, count, stride and indices arrays.
    # The number of elements in the first n dimensions corresponds to the
    # number of times the _get method will be called.
    sdim = []
    ind_dim = None
    for i, e in enumerate(elem):

        # Slices
        if type(e) == slice:
            sdim.append(1)

        # Booleans --- Same shape as data along corresponding dimension
        elif getattr(getattr(e, 'dtype', None), 'kind', None) == 'b':
            if shape[i] != len(e):
                msg="""
Boolean array must have the same shape as the data along this dimension."""
                raise IndexError(msg)
            sdim.append(e.sum())

        # Sequence of indices
        # If multiple sequences are used, they must have the same length.
        elif np.iterable(e):
            if ind_dim is None:
                sdim.append(np.alen(e))
                ind_dim = i
            elif np.alen(e) == 1 or np.alen(e) == sdim[ind_dim]:
                sdim.append(1)
            else:
                raise IndexError("Indice mismatch. Indices must have the same length.")
        # Scalar
        else:
            sdim.append(1)

    # Create the start, count, stride and indices arrays.

    sdim.append(max(nDims, 1))
    start = np.empty(sdim, dtype=int)
    count = np.empty(sdim, dtype=int)
    stride = np.empty(sdim, dtype=int)
    indices = np.empty(sdim, dtype=object)

    for i, e in enumerate(elem):

        # if dimensions and grp are given, set unlim flag for this dimension.
        if (dimensions is not None and grp is not None) and len(dimensions):
            dimname = dimensions[i]
            # is this dimension unlimited?
            # look in current group, and parents for dim.
            dim = _find_dim(grp, dimname)
            unlim = dim.isunlimited()
        else:
            unlim = False

        #    SLICE    #
        if type(e) == slice:

            # determine length parameter for slice.indices.

            # shape[i] can be zero for unlim dim that hasn't been written to
            # yet.
            # length of slice may be longer than current shape
            # if dimension is unlimited.
            if unlim and e.stop is not None and e.stop > shape[i]:
                length = e.stop
            elif unlim and e.stop is None and datashape != ():
                if e.start is None:
                    length = datashape[i]
                else:
                    length = e.start+datashape[i]
            else:
                if unlim and datashape == ():
                    # writing scalar along unlimited dimension using slicing
                    # syntax (var[:] = 1, when var.shape = ())
                    length = 1
                else:
                    length = shape[i]

            beg, end, inc = e.indices(length)
            n = len(range(beg,end,inc))

            start[...,i] = beg
            count[...,i] = n
            stride[...,i] = inc
            indices[...,i] = slice(None)

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
                start[...,i] = np.apply_along_axis(lambda x: np.array(e)*x, i, np.ones(sdim[:-1]))
                indices[...,i] = np.apply_along_axis(lambda x: np.arange(sdim[i])*x, i, np.ones(sdim[:-1], int))


            # Sequence of INTEGER INDICES
            else:
                start[...,i] = np.apply_along_axis(lambda x: np.array(e)*x, ind_dim, np.ones(sdim[:-1]))
                if i == ind_dim:
                    indices[...,i] = np.apply_along_axis(lambda x: np.arange(sdim[i])*x, ind_dim, np.ones(sdim[:-1], int))
                else:
                    indices[...,i] = -1

            count[...,i] = 1
            stride[...,i] = 1


        #    SCALAR INTEGER    #
        elif np.alen(e)==1 and np.dtype(type(e)).kind == 'i':
            if e >= 0:
                start[...,i] = e
            elif e < 0 and (-e < shape[i]) :
                start[...,i] = e+shape[i]
            else:
                raise IndexError("Index out of range")

            count[...,i] = 1
            stride[...,i] = 1
            indices[...,i] = -1    # Use -1 instead of 0 to indicate that
                                       # this dimension shall be squeezed.

    return start, count, stride, indices#, out_shape

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

def _is_container(a):
    # is object container-like?  (can test for
    # membership with "is in", but not a string)
    try: 1 in a
    except: return False
    if type(a) == type(basestring): return False
    return True

def _is_int(a):
    try:
        return int(a) == a
    except:
        return False

def _tostr(s):
    try:
        ss = str(s)
    except:
        ss = s
    return ss
