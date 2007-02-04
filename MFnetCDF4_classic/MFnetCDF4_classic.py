"""
Module for reading multi-file netCDF Datasets, making variables
spanning multiple files appear as if they were in one file.

Datasets must be in C{NETCDF4_CLASSIC, NETCDF3_CLASSIC or NETCDF3_64BIT}
format (C{NETCDF4} Datasets won't work).

Adapted from U{pycdf <http://pysclint.sourceforge.net/pycdf>} by Andre Gosselin.

Example usage:

>>> import MFnetCDF4_classic, netCDF4_classic, numpy
>>> # create a series of netCDF files with a variable sharing
>>> # the same unlimited dimension.
>>> for nfile in range(10):
>>>     f = netCDF4_classic.Dataset('mftest'+repr(nfile)+'.nc','w')
>>>     f.createDimension('x',None)
>>>     x = f.createVariable('x','i',('x',))
>>>     x[0:10] = numpy.arange(nfile*10,10*(nfile+1))
>>>     f.close()
>>> # now read all those files in at once, in one Dataset.
>>> f = MFnetCDF4_classic.Dataset('mftest*nc')
>>> print f.variables['x'][:]
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74
 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99]
"""

import netCDF4_classic
import numpy
from glob import glob

__version__ = "0.6"

class Dataset(netCDF4_classic.Dataset): 
    """
class for reading a multi-file netCDF dataset.
    """

    def __init__(self, files, check=False):
        """
Open a Dataset spanning multiple files, making it look as if it was a 
single file. Variables in the list of files that share the same unlimited 
dimension are aggregated. 

Adapted from U{pycdf <http://pysclint.sourceforge.net/pycdf>} by Andre Gosselin.

Usage:

nc = MFnetCDF4_classic.Dataset(files, check=False)

@param files: either a sequence of netCDF files or a string with a 
wildcard (converted to a sorted list of files using glob)  The first file 
in the list will become the "master" file, defining all the record 
variables (variables with an unlimited dimension) which may span 
subsequent files. Attribute access returns attributes only from "master" 
file. The files are always opened in read-only mode.

@param check: True if you want to do consistency checking to ensure the 
correct variables structure for all of the netcdf files.  Checking makes 
the initialization of the MFnetCDF4_classic instance much slower. Default is 
False.
       """

        # Open the master file in the base class, so that the CDFMF instance
        # can be used like a CDF instance.
        if isinstance(files, str):
            files = sorted(glob(files))
        
        master = files[0]

        # Open the master again, this time as a classic CDF instance. This will avoid
        # calling methods of the CDFMF subclass when querying the master file.
        cdfm = netCDF4_classic.Dataset(master)
        # copy attributes from master.
        for name, value in cdfm.__dict__.items():
            self.__dict__[name] = value

        # Make sure the master defines an unlimited dimension.
        unlimDimId = None
        for dimname,dim in cdfm.dimensions.items():
            if dim.isunlimited():
                unlimDimId = dim
                unlimDimName = dimname
        if unlimDimId is None:
            raise IOError("master dataset %s does not have an unlimited dimension" % master)

        # Get info on all record variables defined in the master.
        # Make sure the master defines at least one record variable.
        masterRecVar = {}
        for vName,v in cdfm.variables.items():
            dims = v.dimensions
            shape = v.shape
            type = v.dtype
            # Be carefull: we may deal with a scalar (dimensionless) variable.
            # Unlimited dimension always occupies index 0.
            if (len(dims) > 0 and unlimDimName == dims[0]):
                masterRecVar[vName] = (dims, shape, type)
        if len(masterRecVar) == 0:
            raise IOError("master dataset %s does not have any record variable" % master)

        # Create the following:
        #   cdf       list of Dataset instances
        #   cdfVLen   list unlimited dimension lengths in each CDF instance
        #   cdfRecVar dictionnary indexed by the record var names; each key holds
        #             a list of the corresponding Variable instance, one for each
        #             cdf file of the file set
        cdf = [cdfm]
        self._cdf = cdf        # Store this now, because dim() method needs it
        cdfVLen = [len(unlimDimId)]
        cdfRecVar = {}
        for v in masterRecVar.keys():
            cdfRecVar[v] = [cdfm.variables[v]]
        
        # Open each remaining file in read-only mode.
        # Make sure each file defines the same record variables as the master
        # and that the variables are defined in the same way (name, shape and type)
        for f in files[1:]:
            part = netCDF4_classic.Dataset(f)
            varInfo = part.variables
            for v in masterRecVar.keys():
                if check:
                    # Make sure master rec var is also defined here.
                    if v not in varInfo.keys():
                        raise IOError("record variable %s not defined in %s" % (v, f))

                    # Make sure it is a record var.
                    vInst = part.variables[v]
                    if not part.dimensions[vInst.dimensions[0]].isunlimited():
                        raise MFnetCDF4_classic("variable %s is not a record var inside %s" % (v, f))

                    masterDims, masterShape, masterType = masterRecVar[v][:3]
                    extDims, extShape, extType = varInfo[v][:3]
                    extDims = varInfo[v].dimensions
                    extShape = varInfo[v].shape
                    extType = varInfo[v].dtype
                    # Check that dimension names are identical.
                    if masterDims != extDims:
                        raise IOError("variable %s : dimensions mismatch between "
                                       "master %s (%s) and extension %s (%s)" %
                                       (v, master, masterDims, f, extDims))

                    # Check that the ranks are identical, and the dimension lengths are
                    # identical (except for that of the unlimited dimension, which of
                    # course may vary.
                    if len(masterShape) != len(extShape):
                        raise IOError("variable %s : rank mismatch between "
                                       "master %s (%s) and extension %s (%s)" %
                                       (v, master, len(masterShape), f, len(extShape)))
                    if masterShape[1:] != extShape[1:]:
                        raise IOError("variable %s : shape mismatch between "
                                       "master %s (%s) and extension %s (%s)" %
                                       (v, master, masterShape, f, extShape))

                    # Check that the data types are identical.
                    if masterType != extType:
                        raise IOError("variable %s : data type mismatch between "
                                       "master %s (%s) and extension %s (%s)" %
                                       (v, master, masterType, f, extType))

                    # Everythig ok.
                    cdfRecVar[v].append(vInst)
                else:
                    # No making sure of anything -- assume this is ok..
                    vInst = part.variables[v]
                    cdfRecVar[v].append(vInst)

            cdf.append(part)
            cdfVLen.append(len(part.dimensions[unlimDimName]))

        # Attach attributes to the MFnetCDF4_classic.Dataset instance.
        # A local __setattr__() method is required for them.
        self._files = files            # list of cdf file names in the set
        self._cdfVLen = cdfVLen              # list of unlimited lengths
        self._cdfTLen = reduce(lambda x, y: x + y, cdfVLen) # total length
        self._cdfRecVar = cdfRecVar          # dictionary of Variable instances for all
                                             # the record variables
        self._dims = cdfm.dimensions
        for dimname, dim in self._dims.items():
            if dim.isunlimited():
                self._dims[dimname] = _Dimension(dimname, dim, self._cdfVLen, self._cdfTLen)
        self._vars = cdfm.variables
        for varname,var in self._vars.items():
            if varname in self._cdfRecVar.keys():
                self._vars[varname] = _Variable(self, varname, var, unlimDimName)
        self._file_format = []
        for dset in self._cdf:
            self._file_format.append(dset.file_format)

    def __setattr__(self, name, value):
        """override base class attribute creation"""
        self.__dict__[name] = value

    def __getattribute__(self, name):
        if name in ['variables','dimensions','file_format']: 
            if name == 'dimensions': return self._dims
            if name == 'variables': return self._vars
            if name == 'file_format': return self._file_format
        else:
            return netCDF4_classic.Dataset.__getattribute__(self, name)

    def ncattrs(self):
        return self._cdf[0].__dict__.keys()

    def close(self):
        for dset in self._cdf:
            dset.close()

class _Dimension(object):
    def __init__(self, dimname, dim, dimlens, dimtotlen):
        self.dimlens = dimlens
        self.dimtotlen = dimtotlen
    def __len__(self):
        return self.dimtotlen
    def isunlimited(self):
        return True

class _Variable(object):
    def __init__(self, dset, varname, var, recdimname):
        self.dimensions = var.dimensions 
        self._dset = dset
        self._mastervar = var
        self._recVar = dset._cdfRecVar[varname]
        self._recdimname = recdimname
        self._recLen = dset._cdfVLen
        self.dtype = var.dtype
        # copy attributes from master.
        for name, value in var.__dict__.items():
            self.__dict__[name] = value
    def typecode(self):
        return self.dtype
    def ncattrs(self):
        return self._mastervar.__dict__.keys()
    def __getattr__(self,name):
        if name == 'shape': return self._shape()
        return self.__dict__[name]
    def _shape(self):
        recdimlen = len(self._dset.dimensions[self._recdimname])
        return (recdimlen,) + self._mastervar.shape[1:]
    def __getitem__(self, elem):
        """Get records from a concatenated set of variables."""
        # Number of variables making up the MFVariable.Variable.
        nv = len(self._recLen)
        # Parse the slicing expression, needed to properly handle
        # a possible ellipsis.
        start, count, stride = netCDF4_classic._buildStartCountStride(elem, self.shape, self.dimensions, self._dset)
        # make sure count=-1 becomes count=1
        count = [abs(cnt) for cnt in count]
        if (numpy.array(stride) < 0).any():
            raise IndexError('negative strides not allowed when slicing MFVariable Variable instance')
        # Start, stop and step along 1st dimension, eg the unlimited
        # dimension.
        sta = start[0]
        step = stride[0]
        stop = sta + count[0] * step
        
        # Build a list representing the concatenated list of all records in
        # the MFVariable variable set. The list is composed of 2-elem lists
        # each holding:
        #  the record index inside the variables, from 0 to n
        #  the index of the Variable instance to which each record belongs
        idx = []    # list of record indices
        vid = []    # list of Variable indices
        for n in range(nv):
            k = self._recLen[n]     # number of records in this variable
            idx.extend(range(k))
            vid.extend([n] * k)

        # Merge the two lists to get a list of 2-elem lists.
        # Slice this list along the first dimension.
        lst = zip(idx, vid).__getitem__(slice(sta, stop, step))

        # Rebuild the slicing expression for dimensions 1 and ssq.
        newSlice = [slice(None, None, None)]
        for n in range(1, len(start)):   # skip dimension 0
            newSlice.append(slice(start[n],
                                  start[n] + count[n] * stride[n], stride[n]))
            
        # Apply the slicing expression to each var in turn, extracting records
        # in a list of arrays.
        lstArr = []
        for n in range(nv):
            # Get the list of indices for variable 'n'.
            idx = [i for i,numv in lst if numv == n]
            if idx:
                # Rebuild slicing expression for dimension 0.
                newSlice[0] = slice(idx[0], idx[-1] + 1, step)
                # Extract records from the var, and append them to a list
                # of arrays.
                lstArr.append(netCDF4_classic.Variable.__getitem__(self._recVar[n], tuple(newSlice)))
        
        # Return the extracted records as a unified array.
        if lstArr:
            lstArr = numpy.concatenate(lstArr)
        return numpy.squeeze(lstArr)
