# size_t, ptrdiff_t are defined in stdlib.h
cdef extern from "stdlib.h":
    ctypedef long size_t
    ctypedef long ptrdiff_t

# hdf5 version info.
cdef extern from "H5public.h":
    ctypedef int herr_t
    int H5get_libversion( unsigned int *majnum, unsigned int *minnum, unsigned int *relnum ) nogil
 
cdef extern from *:
    ctypedef char* const_char_ptr "const char*"
 
# netcdf functions.
cdef extern from "netcdf.h":
    ctypedef int nclong
    ctypedef int nc_type
    ctypedef struct nc_vlen_t:
        size_t len                 # Length of VL data (in base type units) 
        void *p                    # Pointer to VL data 
    float NC_FILL_FLOAT
    long NC_FILL_INT
    double NC_FILL_DOUBLE
    char NC_FILL_CHAR
    long long NC_FILL_INT64
    unsigned long NC_FILL_UINT
    unsigned long long NC_FILL_UINT64
    cdef enum:
        NC_NAT # NAT = 'Not A Type' (c.f. NaN) 
        NC_BYTE # signed 1 byte integer 
        NC_CHAR # ISO/ASCII character 
        NC_SHORT # signed 2 byte integer 
        NC_INT # signed 4 byte integer 
        NC_LONG # deprecated, but required for backward compatibility. 
        NC_FLOAT # single precision floating point number 
        NC_DOUBLE # double precision floating point number 
        NC_UBYTE # unsigned 1 byte int 
        NC_USHORT # unsigned 2-byte int 
        NC_UINT # unsigned 4-byte int 
        NC_INT64 # signed 8-byte int 
        NC_UINT64 # unsigned 8-byte int 
        NC_STRING # string 
        NC_VLEN # used internally for vlen types 
        NC_OPAQUE # used internally for opaque types 
        NC_COMPOUND # used internally for compound types 
        NC_ENUM # used internally for enum types.
        # Use these 'mode' flags for nc_open.
        NC_NOWRITE # default is read only 
        NC_WRITE # read & write 
        # Use these 'mode' flags for nc_create.
        NC_CLOBBER
        NC_NOCLOBBER # Don't destroy existing file on create 
        NC_64BIT_OFFSET # Use large (64-bit) file offsets 
        NC_64BIT_DATA # Use cdf-5 format 
        NC_NETCDF4 # Use netCDF-4/HDF5 format 
        NC_CLASSIC_MODEL # Enforce strict netcdf-3 rules. 
        # Use these 'mode' flags for both nc_create and nc_open.
        NC_SHARE # Share updates, limit cacheing 
        # The following flag currently is ignored, but use in
        # nc_open() or nc_create() may someday support use of advisory
        # locking to prevent multiple writers from clobbering a file 
        NC_LOCK  # Use locking if available 
        #       Default fill values, used unless _FillValue attribute is set.
        # These values are stuffed into newly allocated space as appropriate.
        # The hope is that one might use these to notice that a particular datum
        # has not been set.
        NC_FILL_BYTE    
        #NC_FILL_CHAR    
        NC_FILL_SHORT   
        #NC_FILL_INT     
        #NC_FILL_FLOAT   
        #NC_FILL_DOUBLE  
        NC_FILL_UBYTE  
        NC_FILL_USHORT 
        #NC_FILL_UINT   
        #NC_FILL_INT64  
        #NC_FILL_UINT64 
        # These represent the max and min values that can be stored in a
        # netCDF file for their associated types. Recall that a C compiler
        # may define int to be any length it wants, but a NC_INT is *always*
        # a 4 byte signed int. On a platform with has 64 bit ints, there will
        # be many ints which are outside the range supported by NC_INT. But
        # since NC_INT is an external format, it has to mean the same thing
        # everywhere. 
        NC_MAX_BYTE 
        NC_MIN_BYTE 
        NC_MAX_CHAR 
        NC_MAX_SHORT
        NC_MIN_SHORT
        NC_MAX_INT 
        NC_MIN_INT 
        NC_MAX_FLOAT 
        NC_MIN_FLOAT 
        NC_MAX_DOUBLE8 
        NC_MIN_DOUBLE
        NC_MAX_UBYTE 
        NC_MAX_USHORT 
        NC_MAX_UINT 
        NC_MAX_INT64
        NC_MIN_INT64
        NC_MAX_UINT64 
        X_INT64_MAX   
        X_INT64_MIN   
        X_UINT64_MAX  
        # The above values are defaults.
        # If you wish a variable to use a different value than the above
        # defaults, create an attribute with the same type as the variable
        # and the following reserved name. The value you give the attribute
        # will be used as the fill value for that variable.
        _FillValue
        NC_FILL
        NC_NOFILL
        # Starting with version 3.6, there are different format netCDF
        # files. 4.0 instroduces the third one. These defines are only for
        # the nc_set_default_format function.
        NC_FORMAT_CLASSIC 
        NC_FORMAT_64BIT   
        NC_FORMAT_64BIT_OFFSET
        NC_FORMAT_64BIT_DATA
        NC_FORMAT_NETCDF4 
        NC_FORMAT_NETCDF4_CLASSIC  
        NC_FORMAT_NC3
        NC_FORMAT_NC_HDF4
        NC_FORMAT_NC_HDF5
        NC_FORMAT_DAP2
        NC_FORMAT_DAP4
        NC_FORMAT_PNETCDF
        NC_FORMAT_UNDEFINED
        NC_SIZEHINT_DEFAULT 
        NC_ALIGN_CHUNK 
        # 'size' argument to ncdimdef for an unlimited dimension
        NC_UNLIMITED 
        # attribute id to put/get a global attribute
        NC_GLOBAL 
        # These maximums are enforced by the interface, to facilitate writing
        # applications and utilities.  However, nothing is statically allocated to
        # these sizes internally.
        NC_MAX_DIMS
        NC_MAX_ATTRS
        NC_MAX_VARS
        NC_MAX_NAME
        NC_MAX_VAR_DIMS        
        #   Algorithms for netcdf-4 chunking. 
        NC_CHUNK_SEQ 
        NC_CHUNK_SUB 
        NC_CHUNK_SIZES   
        NC_CHUNKED
        NC_CONTIGUOUS
        # The netcdf version 3 functions all return integer error status.
        # These are the possible values, in addition to certain
        # values from the system errno.h.
        NC_ISSYSERR       
        NC_NOERR       
        NC2_ERR         
        NC_EBADID
        NC_ENFILE
        NC_EEXIST
        NC_EINVAL
        NC_EPERM
        NC_ENOTINDEFINE
        NC_EINDEFINE    
        NC_EINVALCOORDS 
        NC_EMAXDIMS     
        NC_ENAMEINUSE   
        NC_ENOTATT             
        NC_EMAXATTS     
        NC_EBADTYPE    
        NC_EBADDIM     
        NC_EUNLIMPOS   
        NC_EMAXVARS     
        NC_ENOTVAR        
        NC_EGLOBAL        
        NC_ENOTNC         
        NC_ESTS                
        NC_EMAXNAME            
        NC_EUNLIMIT            
        NC_ENORECVARS          
        NC_ECHAR       
        NC_EEDGE       
        NC_ESTRIDE     
        NC_EBADNAME                           
        # N.B. following must match value in ncx.h 
        NC_ERANGE         # Math result not representable 
        NC_ENOMEM         # Memory allocation (malloc) failure 
        NC_EVARSIZE        # One or more variable sizes violate format constraints  
        NC_EDIMSIZE        # Invalid dimension size 
        NC_ETRUNC          # NetCDFFile likely truncated or possibly corrupted 
        # The following was added in support of netcdf-4. Make all netcdf-4
        # error codes < -100 so that errors can be added to netcdf-3 if
        # needed. 
        NC4_FIRST_ERROR 
        NC_EHDFERR      
        NC_ECANTREAD    
        NC_ECANTWRITE   
        NC_ECANTCREATE  
        NC_EFILEMETA    
        NC_EDIMMETA     
        NC_EATTMETA     
        NC_EVARMETA     
        NC_ENOCOMPOUND  
        NC_EATTEXISTS   
        NC_ENOTNC4      
        NC_ESTRICTNC3     
        NC_ENOTNC3      
        NC_ENOPAR         
        NC_EPARINIT     
        NC_EBADGRPID    
        NC_EBADTYPID    
        NC_ETYPDEFINED  
        NC_EBADFIELD    
        NC_EBADCLASS    
        NC4_LAST_ERROR  
        NC_ENDIAN_NATIVE 
        NC_ENDIAN_LITTLE 
        NC_ENDIAN_BIG 
    const_char_ptr *nc_inq_libvers() nogil
    const_char_ptr *nc_strerror(int ncerr)
    int nc_create(char *path, int cmode, int *ncidp) nogil
    int nc_open(char *path, int mode, int *ncidp) nogil
    int nc_inq_path(int ncid, size_t *pathlen, char *path) nogil
    int nc_inq_format_extended(int ncid, int *formatp, int* modep) nogil
    int nc_inq_ncid(int ncid, char *name, int *grp_ncid) nogil
    int nc_inq_grps(int ncid, int *numgrps, int *ncids) nogil
    int nc_inq_grpname(int ncid, char *name) nogil
    int nc_inq_grp_parent(int ncid, int *parent_ncid) nogil
    int nc_inq_varids(int ncid, int *nvars, int *varids) nogil
    int nc_inq_dimids(int ncid, int *ndims, int *dimids, int include_parents) nogil
    int nc_def_grp(int parent_ncid, char *name, int *new_ncid) nogil
    int nc_def_compound(int ncid, size_t size, char *name, nc_type *typeidp) nogil
    int nc_insert_compound(int ncid, nc_type xtype, char *name, 
                   size_t offset, nc_type field_typeid) nogil
    int nc_insert_array_compound(int ncid, nc_type xtype, char *name, 
                         size_t offset, nc_type field_typeid,
                         int ndims, int *dim_sizes) nogil
    int nc_inq_type(int ncid, nc_type xtype, char *name, size_t *size) nogil
    int nc_inq_compound(int ncid, nc_type xtype, char *name, size_t *size, 
                size_t *nfieldsp) nogil
    int nc_inq_compound_name(int ncid, nc_type xtype, char *name) nogil
    int nc_inq_compound_size(int ncid, nc_type xtype, size_t *size) nogil
    int nc_inq_compound_nfields(int ncid, nc_type xtype, size_t *nfieldsp) nogil
    int nc_inq_compound_field(int ncid, nc_type xtype, int fieldid, char *name,
                      size_t *offsetp, nc_type *field_typeidp, int *ndimsp, 
                      int *dim_sizesp) nogil
    int nc_inq_compound_fieldname(int ncid, nc_type xtype, int fieldid, 
                          char *name) nogil
    int nc_inq_compound_fieldindex(int ncid, nc_type xtype, char *name, 
                           int *fieldidp) nogil
    int nc_inq_compound_fieldoffset(int ncid, nc_type xtype, int fieldid, 
                            size_t *offsetp) nogil
    int nc_inq_compound_fieldtype(int ncid, nc_type xtype, int fieldid, 
                          nc_type *field_typeidp) nogil
    int nc_inq_compound_fieldndims(int ncid, nc_type xtype, int fieldid, 
                           int *ndimsp) nogil
    int nc_inq_compound_fielddim_sizes(int ncid, nc_type xtype, int fieldid, 
                               int *dim_sizes) nogil
    int nc_def_vlen(int ncid, char *name, nc_type base_typeid, nc_type *xtypep) nogil
    int nc_inq_vlen(int ncid, nc_type xtype, char *name, size_t *datum_sizep, 
            nc_type *base_nc_typep) nogil
    int nc_inq_user_type(int ncid, nc_type xtype, char *name, size_t *size, 
                     nc_type *base_nc_typep, size_t *nfieldsp, int *classp) nogil
    int nc_inq_typeids(int ncid, int *ntypes, int *typeids) nogil
    int nc_put_att(int ncid, int varid, char *name, nc_type xtype, 
               size_t len, void *op) nogil
    int nc_get_att(int ncid, int varid, char *name, void *ip) nogil
    int nc_get_att_string(int ncid, int varid, char *name, char **ip) nogil
    int nc_put_att_string(int ncid, int varid, char *name, size_t len, const char **op) nogil
    int nc_def_opaque(int ncid, size_t size, char *name, nc_type *xtypep) nogil
    int nc_inq_opaque(int ncid, nc_type xtype, char *name, size_t *sizep) nogil
    int nc_put_att_opaque(int ncid, int varid, char *name,
                      size_t len, void *op) nogil
    int nc_get_att_opaque(int ncid, int varid, char *name, 
                      void *ip) nogil
    int nc_put_cmp_att_opaque(int ncid, nc_type xtype, int fieldid, 
                          char *name, size_t len, void *op) nogil
    int nc_get_cmp_att_opaque(int ncid, nc_type xtype, int fieldid, 
                          char *name, void *ip) nogil
    int nc_put_var1(int ncid, int varid, size_t *indexp,
                void *op) nogil
    int nc_get_var1(int ncid, int varid,  size_t *indexp,
                void *ip) nogil
    int nc_put_vara(int ncid, int varid,  size_t *startp, 
                size_t *countp, void *op) nogil
    int nc_get_vara(int ncid, int varid,  size_t *startp, 
                size_t *countp, void *ip) nogil
    int nc_put_vars(int ncid, int varid,  size_t *startp, 
                size_t *countp, ptrdiff_t *stridep,
                void *op) nogil
    int nc_get_vars(int ncid, int varid,  size_t *startp, 
                size_t *countp, ptrdiff_t *stridep,
                void *ip) nogil
    int nc_put_varm(int ncid, int varid,  size_t *startp, 
                size_t *countp, ptrdiff_t *stridep,
                ptrdiff_t *imapp, void *op) nogil
    int nc_get_varm(int ncid, int varid,  size_t *startp, 
                size_t *countp, ptrdiff_t *stridep,
                ptrdiff_t *imapp, void *ip) nogil
    int nc_put_var(int ncid, int varid,  void *op) nogil
    int nc_get_var(int ncid, int varid,  void *ip) nogil
    int nc_def_var_deflate(int ncid, int varid, int shuffle, int deflate, 
	           	   int deflate_level) nogil
    int nc_def_var_fletcher32(int ncid, int varid, int fletcher32) nogil
    int nc_inq_var_fletcher32(int ncid, int varid, int *fletcher32p) nogil
    int nc_def_var_chunking(int ncid, int varid, int contiguous, size_t *chunksizesp) nogil
    int nc_def_var_fill(int ncid, int varid, int no_fill, void *fill_value) nogil
    int nc_def_var_endian(int ncid, int varid, int endian) nogil
    int nc_inq_var_chunking(int ncid, int varid, int *contiguousp, size_t *chunksizesp) nogil
    int nc_inq_var_deflate(int ncid, int varid, int *shufflep, 
      		   int *deflatep, int *deflate_levelp) nogil
    int nc_inq_var_fill(int ncid, int varid, int *no_fill, void *fill_value) nogil
    int nc_inq_var_endian(int ncid, int varid, int *endianp) nogil
    int nc_set_fill(int ncid, int fillmode, int *old_modep) nogil 
    int nc_set_default_format(int format, int *old_formatp) nogil
    int nc_redef(int ncid) nogil
    int nc_enddef(int ncid) nogil
    int nc_sync(int ncid) nogil
    int nc_abort(int ncid) nogil
    int nc_close(int ncid) nogil
    int nc_inq(int ncid, int *ndimsp, int *nvarsp, int *nattsp, int *unlimdimidp) nogil
    int nc_inq_ndims(int ncid, int *ndimsp) nogil 
    int nc_inq_nvars(int ncid, int *nvarsp) nogil
    int nc_inq_natts(int ncid, int *nattsp) nogil 
    int nc_inq_unlimdim(int ncid, int *unlimdimidp) nogil
    int nc_inq_unlimdims(int ncid, int *nunlimdimsp, int *unlimdimidsp) nogil
    int nc_inq_format(int ncid, int *formatp) nogil
    int nc_def_dim(int ncid, char *name, size_t len, int *idp) nogil
    int nc_inq_dimid(int ncid, char *name, int *idp) nogil
    int nc_inq_dim(int ncid, int dimid, char *name, size_t *lenp) nogil
    int nc_inq_dimname(int ncid, int dimid, char *name) nogil
    int nc_inq_dimlen(int ncid, int dimid, size_t *lenp) nogil
    int nc_rename_dim(int ncid, int dimid, char *name) nogil
    int nc_inq_att(int ncid, int varid, char *name,
               nc_type *xtypep, size_t *lenp) nogil
    int nc_inq_attid(int ncid, int varid, char *name, int *idp) nogil
    int nc_inq_atttype(int ncid, int varid, char *name, nc_type *xtypep) nogil
    int nc_inq_attlen(int ncid, int varid, char *name, size_t *lenp) nogil
    int nc_inq_attname(int ncid, int varid, int attnum, char *name) nogil
    int nc_copy_att(int ncid_in, int varid_in, char *name, int ncid_out, int varid_out)
    int nc_rename_att(int ncid, int varid, char *name, char *newname) nogil
    int nc_del_att(int ncid, int varid, char *name) nogil
    int nc_put_att_text(int ncid, int varid, char *name,
                    size_t len, char *op) nogil
    int nc_get_att_text(int ncid, int varid, char *name, char *ip) nogil
    int nc_def_var(int ncid, char *name, nc_type xtype, int ndims, 
               int *dimidsp, int *varidp) nogil
    int nc_inq_var(int ncid, int varid, char *name, nc_type *xtypep, 
               int *ndimsp, int *dimidsp, int *nattsp) nogil
    int nc_inq_varid(int ncid, char *name, int *varidp) nogil
    int nc_inq_varname(int ncid, int varid, char *name) nogil
    int nc_inq_vartype(int ncid, int varid, nc_type *xtypep) nogil
    int nc_inq_varndims(int ncid, int varid, int *ndimsp) nogil
    int nc_inq_vardimid(int ncid, int varid, int *dimidsp) nogil
    int nc_inq_varnatts(int ncid, int varid, int *nattsp) nogil
    int nc_rename_var(int ncid, int varid, char *name) nogil
    int nc_free_vlen(nc_vlen_t *vl) nogil
    int nc_free_vlens(size_t len, nc_vlen_t *vl) nogil
    int nc_free_string(size_t len, char **data) nogil
    int nc_get_chunk_cache(size_t *sizep, size_t *nelemsp, float *preemptionp) nogil
    int nc_set_chunk_cache(size_t size, size_t nelems, float preemption) nogil
    int nc_set_var_chunk_cache(int ncid, int varid, size_t size, size_t nelems, float preemption) nogil
    int nc_get_var_chunk_cache(int ncid, int varid, size_t *sizep, size_t *nelemsp, float *preemptionp) nogil
    int nc_def_enum(int ncid, nc_type base_typeid, char *name, nc_type *typeidp) nogil
    int nc_insert_enum(int ncid, nc_type xtype, char *name, void *value) nogil
    int nc_inq_enum(int ncid, nc_type xtype, char *name, nc_type *base_nc_typep,\
	    size_t *base_sizep, size_t *num_membersp) nogil
    int nc_inq_enum_member(int ncid, nc_type xtype, int idx, char *name, void *value) nogil
    int nc_inq_enum_ident(int ncid, nc_type xtype, long long value, char *identifier) nogil


cdef extern from "mpi-compat.h":
    pass


# taken from numpy.pxi in numpy 1.0rc2.
cdef extern from "numpy/arrayobject.h":
    ctypedef int npy_intp 
    ctypedef extern class numpy.ndarray [object PyArrayObject]:
        pass
    npy_intp PyArray_SIZE(ndarray arr) nogil
    npy_intp PyArray_ISCONTIGUOUS(ndarray arr) nogil
    npy_intp PyArray_ISALIGNED(ndarray arr) nogil
    void* PyArray_DATA(ndarray) nogil
    char* PyArray_BYTES(ndarray) nogil
    npy_intp* PyArray_STRIDES(ndarray) nogil
    void import_array()


include "parallel_support_imports.pxi"

# Compatibility shims
cdef extern from "netcdf-compat.h":
    int nc_rename_grp(int grpid, char *name) nogil
    int nc_set_alignment(int threshold, int alignment)
    int nc_get_alignment(int *threshold, int *alignment)
    int nc_rc_set(char* key, char* value) nogil

    int nc_open_mem(const char *path, int mode, size_t size, void* memory, int *ncidp) nogil
    int nc_create_mem(const char *path, int mode, size_t initialize, int *ncidp) nogil
    ctypedef struct NC_memio:
        size_t size
        void* memory
        int flags
    int nc_close_memio(int ncid, NC_memio* info) nogil

    # Quantize shims
    int nc_def_var_quantize(int ncid, int varid, int quantize_mode, int nsd) nogil
    int nc_inq_var_quantize(int ncid, int varid, int *quantize_modep, int *nsdp) nogil

    # Filter shims
    int nc_inq_filter_avail(int ncid, unsigned filterid) nogil

    int nc_def_var_szip(int ncid, int varid, int options_mask,
                        int pixels_per_bloc) nogil
    int nc_inq_var_szip(int ncid, int varid, int *options_maskp,
                        int *pixels_per_blockp) nogil

    int nc_def_var_zstandard(int ncid, int varid, int level) nogil
    int nc_inq_var_zstandard(int ncid, int varid, int* hasfilterp, int *levelp) nogil

    int nc_def_var_bzip2(int ncid, int varid, int level) nogil
    int nc_inq_var_bzip2(int ncid, int varid, int* hasfilterp, int *levelp) nogil

    int nc_def_var_blosc(int ncid, int varid, unsigned subcompressor, unsigned level,
                         unsigned blocksize, unsigned addshuffle) nogil
    int nc_inq_var_blosc(int ncid, int varid, int* hasfilterp, unsigned* subcompressorp,
                         unsigned* levelp, unsigned* blocksizep,
                         unsigned* addshufflep) nogil

    # Parallel shims
    int nc_create_par(char *path, int cmode, MPI_Comm comm, MPI_Info info, int *ncidp) nogil
    int nc_open_par(char *path, int mode, MPI_Comm comm, MPI_Info info, int *ncidp) nogil
    int nc_var_par_access(int ncid, int varid, int par_access) nogil

    cdef enum:
        HAS_RENAME_GRP
        HAS_NC_INQ_PATH
        HAS_NC_INQ_FORMAT_EXTENDED
        HAS_NC_OPEN_MEM
        HAS_NC_CREATE_MEM
        HAS_CDF5_FORMAT
        HAS_PARALLEL_SUPPORT
        HAS_PARALLEL4_SUPPORT
        HAS_PNETCDF_SUPPORT
        HAS_SZIP_SUPPORT
        HAS_QUANTIZATION_SUPPORT
        HAS_ZSTANDARD_SUPPORT
        HAS_BZIP2_SUPPORT
        HAS_BLOSC_SUPPORT
        HAS_SET_ALIGNMENT
        HAS_NCFILTER
        HAS_NCRCSET

        NC_NOQUANTIZE
        NC_QUANTIZE_BITGROOM
        NC_QUANTIZE_GRANULARBR
        NC_QUANTIZE_BITROUND

        H5Z_FILTER_SZIP
        H5Z_FILTER_ZSTD
        H5Z_FILTER_BZIP2
        H5Z_FILTER_BLOSC

        NC_COLLECTIVE
        NC_INDEPENDENT

        NC_MPIIO
        NC_MPIPOSIX
        NC_PNETCDF


# Declarations for handling complex numbers
cdef extern from "nc_complex/nc_complex.h":
  bint pfnc_var_is_complex(int ncid, int varid) nogil
  bint pfnc_var_is_complex_type(int ncid, int varid) nogil

  int pfnc_get_complex_dim(int ncid, int* nc_dim) nogil
  int pfnc_inq_var_complex_base_type(int ncid, int varid, int* nc_typeid) nogil

  int pfnc_inq_varndims (int ncid, int varid, int *ndimsp) nogil
  int pfnc_inq_vardimid (int ncid, int varid, int *dimidsp) nogil

  int pfnc_def_var(int ncid, char *name, nc_type xtype, int ndims,
                   int *dimidsp, int *varidp) nogil

  int pfnc_get_vars(int ncid, int varid, size_t *startp,
                    size_t *countp, ptrdiff_t *stridep,
                    void *ip) nogil

  int pfnc_put_vars(int ncid, int varid, size_t *startp,
                    size_t *countp, ptrdiff_t *stridep,
                    void *op) nogil

  cdef enum:
      PFNC_DOUBLE_COMPLEX
      PFNC_DOUBLE_COMPLEX_DIM
      PFNC_FLOAT_COMPLEX
      PFNC_FLOAT_COMPLEX_DIM
