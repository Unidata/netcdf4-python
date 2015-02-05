import numpy as np
from numpy import ma
import sys

python3 = sys.version_info[0] > 2
if python3:
    # no unicode type in python 3, use bytes instead when testing
    # for a string-like object
    unicode = str
try:
    bytes
except NameError:
    # no bytes type in python < 2.6
    bytes = str


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

def _walk_grps(topgrp):
    """Iterate through all (sub-) groups of topgrp, similar to os.walktree.

    """
    grps = topgrp.groups.values()
    yield grps
    for grp in topgrp.groups.values():
        for children in _walk_grps(grp):
            yield children

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

def _StartCountStride(elem, shape, dimensions=None, grp=None, datashape=None,\
        put=False):
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
    put : True|False (default False).  If called from __setitem__, put is True.

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

    # replace sequence of integer indices with boolean arrays
    newElem = []
    IndexErrorMsg=\
    "only integers, slices (`:`), ellipsis (`...`), and 1-d integer or boolean arrays are valid indices"
    for i, e in enumerate(elem):
        # string-like object try to cast to int
        # needs to be done first, since strings are iterable and
        # hard to distinguish from something castable to an iterable numpy array.
        if type(e) in [str,bytes,unicode]:
            try:
                e = int(e)
            except:
                raise IndexError(IndexErrorMsg)
        ea = np.asarray(e)
        # Raise error if multidimensional indexing is used.
        if ea.ndim > 1:
            raise IndexError("Index cannot be multidimensional")
        # set unlim to True if dimension is unlimited and put==True
        # (called from __setitem__)
        if put and (dimensions is not None and grp is not None) and len(dimensions):
            dimname = dimensions[i]
            # is this dimension unlimited?
            # look in current group, and parents for dim.
            dim = _find_dim(grp, dimname)
            unlim = dim.isunlimited()
        else:
            unlim = False
        # an iterable (non-scalar) integer array.
        if np.iterable(ea) and ea.dtype.kind == 'i':
            # convert negative indices in 1d array to positive ones.
            ea = np.where(ea < 0, ea + shape[i], ea)
            if np.any(ea < 0):
                raise IndexErro("integer index out of range")
            if not np.all(np.diff(ea) > 0): # same but cheaper than np.all(np.unique(ea) == ea)
                # raise an error when new indexing behavior is different
                # (i.e. when integer sequence not sorted, or there are
                # duplicate indices in the sequence)
                msg = "integer sequences in slices must be sorted and cannot have duplicates"
                raise IndexError(msg)
            # convert to boolean array.
            # if unlim, let boolean array be longer than current dimension
            # length.
            elen = shape[i]
            if unlim:
                elen = max(ea.max()+1,elen)
            else:
                if ea.max()+1 > elen:
                    msg="integer index exceeds dimension size"
                    raise IndexError(msg)
            eb = np.zeros(elen,np.bool)
            eb[ea] = True
            newElem.append(eb)
        # an iterable (non-scalar) boolean array
        elif np.iterable(ea) and ea.dtype.kind =='b':
            # check that boolen array not too long
            if not unlim and shape[i] != len(ea):
                msg="""
Boolean array must have the same shape as the data along this dimension."""
                raise IndexError(msg)
            newElem.append(ea)
        # integer scalar
        elif ea.dtype.kind == 'i':
            newElem.append(e)
        # slice or ellipsis object
        elif type(e) == slice or type(e) == type(Ellipsis):
            newElem.append(e)
        else:  # castable to a scalar int, otherwise invalid
            try:
                e = int(e)
                newElem.append(e)
            except:
                raise IndexError(IndexErrorMsg)
    elem = newElem

    # replace Ellipsis and boolean arrays with slice objects, if possible.
    hasEllipsis = False
    newElem = []
    for e in elem:
        ea = np.asarray(e)
        # Replace ellipsis with slices.
        if type(e) == type(Ellipsis):
            if hasEllipsis:
                raise IndexError("At most one ellipsis allowed in a slicing expression")
            # The ellipsis stands for the missing dimensions.
            newElem.extend((slice(None, None, None),) * (nDims - len(elem) + 1))
            hasEllipsis = True
        # Replace boolean array with slice object if possible.
        elif ea.dtype.kind == 'b':
            if ea.any():
                indices = np.flatnonzero(ea)
                start = indices[0]
                stop = indices[-1] + 1
                if len(indices) >= 2:
                    step = indices[1] - indices[0]
                else:
                    step = None
                if np.array_equal(indices, np.arange(start, stop, step)):
                    newElem.append(slice(start, stop, step))
                else:
                    newElem.append(ea)
            else:
                newElem.append(slice(0, 0))
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
    for i, e in enumerate(elem):
        # at this stage e is a slice, a scalar integer, or a 1d boolean array.
        # Booleans --- _get call for each True value
        if np.asarray(e).dtype.kind == 'b':
            sdim.append(e.sum())
        # Scalar int or slice, just a single _get call
        else:
            sdim.append(1)

    # Create the start, count, stride and indices arrays.

    sdim.append(max(nDims, 1))
    start = np.empty(sdim, dtype=int)
    count = np.empty(sdim, dtype=int)
    stride = np.empty(sdim, dtype=int)
    indices = np.empty(sdim, dtype=object)

    for i, e in enumerate(elem):

        ea = np.asarray(e)

        # set unlim to True if dimension is unlimited and put==True
        # (called from __setitem__). Note: grp and dimensions must be set.
        if put and (dimensions is not None and grp is not None) and len(dimensions):
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
            # if dimension is unlimited (and we are writing, not reading).
            if unlim and e.stop is not None and e.stop > shape[i]:
                length = e.stop
            elif unlim and e.stop is None and datashape != ():
                if e.start is None:
                    length = datashape[i]
                else:
                    length = e.start+datashape[i]
            else:
                if unlim and datashape == () and len(dim) == 0:
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

        #    BOOLEAN ITERABLE    #
        elif ea.dtype.kind == 'b':
            e = np.arange(len(e))[e] # convert to integer array
            start[...,i] = np.apply_along_axis(lambda x: e*x, i, np.ones(sdim[:-1]))
            indices[...,i] = np.apply_along_axis(lambda x: np.arange(sdim[i])*x, i, np.ones(sdim[:-1], int))

            count[...,i] = 1
            stride[...,i] = 1


        #   all that's left is SCALAR INTEGER    #
        else:
            if e >= 0:
                start[...,i] = e
            elif e < 0 and (-e <= shape[i]) :
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
