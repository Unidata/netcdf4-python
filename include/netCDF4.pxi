# size_t, ptrdiff_t are defined in stdlib.h
cdef extern from "stdlib.h":
    ctypedef long size_t
    ctypedef long ptrdiff_t

# hdf5 version info.
cdef extern from "H5public.h":
    cdef char *H5_VERS_INFO
    cdef char *H5_VERS_SUBRELEASE
    cdef enum:
        H5_VERS_MAJOR
        H5_VERS_MINOR
        H5_VERS_RELEASE
 
# netcdf version info.
#cdef extern from "netcdf_meta.h":
#    cdef char *NC_VERSION_NOTE
#    cdef enum:
#        NC_VERSION_MAJOR
#        NC_VERSION_MINOR
#        NC_VERSION_PATCH

cdef extern from *:
    ctypedef char* const_char_ptr "const char*"
 
# netcdf functions.
cdef extern from "netcdf.h":
    ctypedef int nclong
    ctypedef int nc_type
    ctypedef struct nc_vlen_t:
        size_t len                 # Length of VL data (in base type units) 
        void *p                    # Pointer to VL data 
# default fill values.
# could define these in the anonymous enum, but then they
# would be assumed to be integers.
#define NC_FILL_BYTE	((signed char)-127)
#define NC_FILL_CHAR	((char)0)
#define NC_FILL_SHORT	((short)-32767)
#define NC_FILL_INT	(-2147483647L)
#define NC_FILL_FLOAT	(9.9692099683868690e+36f) /* near 15 * 2^119 */
#define NC_FILL_DOUBLE	(9.9692099683868690e+36)
#define NC_FILL_UBYTE   (255)
#define NC_FILL_USHORT  (65535)
#define NC_FILL_UINT    (4294967295U)
#define NC_FILL_INT64   ((long long)-9223372036854775806)
#define NC_FILL_UINT64  ((unsigned long long)18446744073709551614)
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
        NC_NETCDF4 # Use netCDF-4/HDF5 format 
        NC_CLASSIC_MODEL # Enforce strict netcdf-3 rules. 
        # Use these 'mode' flags for both nc_create and nc_open.
        NC_SHARE # Share updates, limit cacheing 
        NC_MPIIO
        NC_MPIPOSIX
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
        # Let nc__create() or nc__open() figure out
        # as suitable chunk size.
        NC_SIZEHINT_DEFAULT 
        # In nc__enddef(), align to the chunk size.
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
        NC_SZIP_EC_OPTION_MASK  # entropy encoding
        NC_SZIP_NN_OPTION_MASK  # nearest neighbor encoding
    const_char_ptr *nc_inq_libvers() nogil
    const_char_ptr *nc_strerror(int ncerr)
    int nc_create(char *path, int cmode, int *ncidp)
    int nc__create(char *path, int cmode, size_t initialsz, size_t *chunksizehintp, int *ncidp)
    int nc_open(char *path, int mode, int *ncidp)
    int nc__open(char *path, int mode, size_t *chunksizehintp, int *ncidp)
    int nc_inq_path(int ncid, size_t *pathlen, char *path) nogil
    int nc_inq_format_extended(int ncid, int *formatp, int* modep) nogil
    int nc_inq_ncid(int ncid, char *name, int *grp_ncid) nogil
    int nc_inq_grps(int ncid, int *numgrps, int *ncids) nogil
    int nc_inq_grpname(int ncid, char *name) nogil
    int nc_inq_grp_parent(int ncid, int *parent_ncid) nogil
    int nc_inq_varids(int ncid, int *nvars, int *varids) nogil
    int nc_inq_dimids(int ncid, int *ndims, int *dimids, int include_parents) nogil
    int nc_def_grp(int parent_ncid, char *name, int *new_ncid)
    int nc_def_compound(int ncid, size_t size, char *name, nc_type *typeidp)
    int nc_insert_compound(int ncid, nc_type xtype, char *name, 
                   size_t offset, nc_type field_typeid)
    int nc_insert_array_compound(int ncid, nc_type xtype, char *name, 
                         size_t offset, nc_type field_typeid,
                         int ndims, int *dim_sizes)
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
    int nc_def_vlen(int ncid, char *name, nc_type base_typeid, nc_type *xtypep)
    int nc_inq_vlen(int ncid, nc_type xtype, char *name, size_t *datum_sizep, 
            nc_type *base_nc_typep) nogil
    int nc_inq_user_type(int ncid, nc_type xtype, char *name, size_t *size, 
                     nc_type *base_nc_typep, size_t *nfieldsp, int *classp) nogil
    int nc_inq_typeids(int ncid, int *ntypes, int *typeids) nogil
    int nc_put_att(int ncid, int varid, char *name, nc_type xtype, 
               size_t len, void *op)
    int nc_get_att(int ncid, int varid, char *name, void *ip) nogil
    int nc_get_att_string(int ncid, int varid, char *name, char **ip) nogil
    int nc_put_att_string(int ncid, int varid, char *name, size_t len, char **op) nogil
    int nc_def_opaque(int ncid, size_t size, char *name, nc_type *xtypep)
    int nc_inq_opaque(int ncid, nc_type xtype, char *name, size_t *sizep)
    int nc_put_att_opaque(int ncid, int varid, char *name,
                      size_t len, void *op)
    int nc_get_att_opaque(int ncid, int varid, char *name, 
                      void *ip)
    int nc_put_cmp_att_opaque(int ncid, nc_type xtype, int fieldid, 
                          char *name, size_t len, void *op)
    int nc_get_cmp_att_opaque(int ncid, nc_type xtype, int fieldid, 
                          char *name, void *ip)
    int nc_put_var1(int ncid, int varid, size_t *indexp,
                void *op)
    int nc_get_var1(int ncid, int varid,  size_t *indexp,
                void *ip)
    int nc_put_vara(int ncid, int varid,  size_t *startp, 
                size_t *countp, void *op) 
    int nc_get_vara(int ncid, int varid,  size_t *startp, 
                size_t *countp, void *ip) nogil
    int nc_put_vars(int ncid, int varid,  size_t *startp, 
                size_t *countp, ptrdiff_t *stridep,
                void *op) 
    int nc_get_vars(int ncid, int varid,  size_t *startp, 
                size_t *countp, ptrdiff_t *stridep,
                void *ip) nogil
    int nc_put_varm(int ncid, int varid,  size_t *startp, 
                size_t *countp, ptrdiff_t *stridep,
                ptrdiff_t *imapp, void *op)
    int nc_get_varm(int ncid, int varid,  size_t *startp, 
                size_t *countp, ptrdiff_t *stridep,
                ptrdiff_t *imapp, void *ip)
    int nc_put_var(int ncid, int varid,  void *op)
    int nc_get_var(int ncid, int varid,  void *ip)
    int nc_def_var_deflate(int ncid, int varid, int shuffle, int deflate, 
	           	   int deflate_level)
    int nc_def_var_fletcher32(int ncid, int varid, int fletcher32)
    int nc_inq_var_fletcher32(int ncid, int varid, int *fletcher32p) nogil
    int nc_def_var_chunking(int ncid, int varid, int contiguous, size_t *chunksizesp)
    int nc_def_var_fill(int ncid, int varid, int no_fill, void *fill_value)
    int nc_def_var_endian(int ncid, int varid, int endian)
    int nc_inq_var_chunking(int ncid, int varid, int *contiguousp, size_t *chunksizesp) nogil
    int nc_inq_var_deflate(int ncid, int varid, int *shufflep, 
      		   int *deflatep, int *deflate_levelp) nogil
    int nc_inq_var_fill(int ncid, int varid, int *no_fill, void *fill_value) nogil
    int nc_inq_var_endian(int ncid, int varid, int *endianp) nogil
    int nc_set_fill(int ncid, int fillmode, int *old_modep)
    int nc_set_default_format(int format, int *old_formatp)
    int nc_redef(int ncid)
    int nc__enddef(int ncid, size_t h_minfree, size_t v_align,
            size_t v_minfree, size_t r_align)
    int nc_enddef(int ncid)
    int nc_sync(int ncid)
    int nc_abort(int ncid)
    int nc_close(int ncid)
    int nc_inq(int ncid, int *ndimsp, int *nvarsp, int *nattsp, int *unlimdimidp) nogil
    int nc_inq_ndims(int ncid, int *ndimsp) nogil
    int nc_inq_nvars(int ncid, int *nvarsp) nogil
    int nc_inq_natts(int ncid, int *nattsp) nogil
    int nc_inq_unlimdim(int ncid, int *unlimdimidp) nogil
    int nc_inq_unlimdims(int ncid, int *nunlimdimsp, int *unlimdimidsp) nogil
    int nc_inq_format(int ncid, int *formatp) nogil
    int nc_def_dim(int ncid, char *name, size_t len, int *idp)
    int nc_inq_dimid(int ncid, char *name, int *idp) nogil
    int nc_inq_dim(int ncid, int dimid, char *name, size_t *lenp) nogil
    int nc_inq_dimname(int ncid, int dimid, char *name) nogil
    int nc_inq_dimlen(int ncid, int dimid, size_t *lenp) nogil
    int nc_rename_dim(int ncid, int dimid, char *name)
    int nc_inq_att(int ncid, int varid, char *name,
               nc_type *xtypep, size_t *lenp) nogil
    int nc_inq_attid(int ncid, int varid, char *name, int *idp) nogil
    int nc_inq_atttype(int ncid, int varid, char *name, nc_type *xtypep) nogil
    int nc_inq_attlen(int ncid, int varid, char *name, size_t *lenp) nogil
    int nc_inq_attname(int ncid, int varid, int attnum, char *name) nogil
    int nc_copy_att(int ncid_in, int varid_in, char *name, int ncid_out, int varid_out)
    int nc_rename_att(int ncid, int varid, char *name, char *newname)
    int nc_del_att(int ncid, int varid, char *name)
    int nc_put_att_text(int ncid, int varid, char *name,
                    size_t len, char *op)
    int nc_get_att_text(int ncid, int varid, char *name, char *ip) nogil
    int nc_put_att_uchar(int ncid, int varid, char *name, nc_type xtype,
                     size_t len, unsigned char *op)
    int nc_get_att_uchar(int ncid, int varid, char *name, unsigned char *ip)
    int nc_put_att_schar(int ncid, int varid, char *name, nc_type xtype,
                     size_t len, signed char *op)
    int nc_get_att_schar(int ncid, int varid, char *name, signed char *ip)
    int nc_put_att_short(int ncid, int varid, char *name, nc_type xtype,
                     size_t len, short *op)
    int nc_get_att_short(int ncid, int varid, char *name, short *ip)
    int nc_put_att_int(int ncid, int varid, char *name, nc_type xtype,
                   size_t len, int *op)
    int nc_get_att_int(int ncid, int varid, char *name, int *ip)
    int nc_put_att_long(int ncid, int varid, char *name, nc_type xtype,
                    size_t len, long *op)
    int nc_get_att_long(int ncid, int varid, char *name, long *ip)
    int nc_put_att_float(int ncid, int varid, char *name, nc_type xtype,
                     size_t len, float *op)
    int nc_get_att_float(int ncid, int varid, char *name, float *ip)
    int nc_put_att_double(int ncid, int varid, char *name, nc_type xtype,
                      size_t len, double *op)
    int nc_get_att_double(int ncid, int varid, char *name, double *ip)
    int nc_put_att_ushort(int ncid, int varid, char *name, nc_type xtype,
                      size_t len, unsigned short *op)
    int nc_get_att_ushort(int ncid, int varid, char *name, unsigned short *ip)
    int nc_put_att_uint(int ncid, int varid, char *name, nc_type xtype,
                    size_t len, unsigned int *op)
    int nc_get_att_uint(int ncid, int varid, char *name, unsigned int *ip)
    int nc_put_att_longlong(int ncid, int varid, char *name, nc_type xtype,
                     size_t len, long long *op)
    int nc_get_att_longlong(int ncid, int varid, char *name, long long *ip)
    int nc_put_att_ulonglong(int ncid, int varid, char *name, nc_type xtype,
                         size_t len, unsigned long long *op)
    int nc_get_att_ulonglong(int ncid, int varid, char *name, 
                         unsigned long long *ip)
    int nc_def_var(int ncid, char *name, nc_type xtype, int ndims, 
               int *dimidsp, int *varidp)
    int nc_inq_var(int ncid, int varid, char *name, nc_type *xtypep, 
               int *ndimsp, int *dimidsp, int *nattsp) nogil
    int nc_inq_varid(int ncid, char *name, int *varidp) nogil
    int nc_inq_varname(int ncid, int varid, char *name) nogil
    int nc_inq_vartype(int ncid, int varid, nc_type *xtypep) nogil
    int nc_inq_varndims(int ncid, int varid, int *ndimsp) nogil
    int nc_inq_vardimid(int ncid, int varid, int *dimidsp) nogil
    int nc_inq_varnatts(int ncid, int varid, int *nattsp) nogil
    int nc_rename_var(int ncid, int varid, char *name)
    int nc_copy_var(int ncid_in, int varid, int ncid_out)
    int nc_put_var1_text(int ncid, int varid, size_t *indexp, char *op)
    int nc_get_var1_text(int ncid, int varid, size_t *indexp, char *ip)
    int nc_put_var1_uchar(int ncid, int varid, size_t *indexp,
                      unsigned char *op)
    int nc_get_var1_uchar(int ncid, int varid, size_t *indexp,
                      unsigned char *ip)
    int nc_put_var1_schar(int ncid, int varid, size_t *indexp,
                      signed char *op)
    int nc_get_var1_schar(int ncid, int varid, size_t *indexp,
                      signed char *ip)
    int nc_put_var1_short(int ncid, int varid, size_t *indexp,
                      short *op)
    int nc_get_var1_short(int ncid, int varid, size_t *indexp,
                      short *ip)
    int nc_put_var1_int(int ncid, int varid, size_t *indexp, int *op)
    int nc_get_var1_int(int ncid, int varid, size_t *indexp, int *ip)
    int nc_put_var1_long(int ncid, int varid, size_t *indexp, long *op)
    int nc_get_var1_long(int ncid, int varid, size_t *indexp, long *ip)
    int nc_put_var1_float(int ncid, int varid, size_t *indexp, float *op)
    int nc_get_var1_float(int ncid, int varid, size_t *indexp, float *ip)
    int nc_put_var1_double(int ncid, int varid, size_t *indexp, double *op)
    int nc_get_var1_double(int ncid, int varid, size_t *indexp, double *ip)
    int nc_put_var1_ubyte(int ncid, int varid, size_t *indexp, 
                      unsigned char *op)
    int nc_get_var1_ubyte(int ncid, int varid, size_t *indexp, 
                      unsigned char *ip)
    int nc_put_var1_ushort(int ncid, int varid, size_t *indexp, 
                       unsigned short *op)
    int nc_get_var1_ushort(int ncid, int varid, size_t *indexp, 
                       unsigned short *ip)
    int nc_put_var1_uint(int ncid, int varid, size_t *indexp, 
                     unsigned int *op)
    int nc_get_var1_uint(int ncid, int varid, size_t *indexp, 
                     unsigned int *ip)
    int nc_put_var1_longlong(int ncid, int varid, size_t *indexp, 
                         long long *op)
    int nc_get_var1_longlong(int ncid, int varid, size_t *indexp, 
                      long long *ip)
    int nc_put_var1_ulonglong(int ncid, int varid, size_t *indexp, 
                       unsigned long long *op)
    int nc_get_var1_ulonglong(int ncid, int varid, size_t *indexp, 
                       unsigned long long *ip)
    int nc_put_vara_text(int ncid, int varid,
            size_t *startp, size_t *countp, char *op)
    int nc_get_vara_text(int ncid, int varid,
            size_t *startp, size_t *countp, char *ip)
    int nc_put_vara_uchar(int ncid, int varid,
            size_t *startp, size_t *countp, unsigned char *op)
    int nc_get_vara_uchar(int ncid, int varid, size_t *startp, 
                      size_t *countp, unsigned char *ip)
    int nc_put_vara_schar(int ncid, int varid, size_t *startp, 
                      size_t *countp, signed char *op)
    int nc_get_vara_schar(int ncid, int varid, size_t *startp, 
                      size_t *countp, signed char *ip)
    int nc_put_vara_short(int ncid, int varid, size_t *startp, 
                      size_t *countp, short *op)
    int nc_get_vara_short(int ncid, int varid, size_t *startp, 
                      size_t *countp, short *ip)
    int nc_put_vara_int(int ncid, int varid, size_t *startp, 
                    size_t *countp, int *op)
    int nc_get_vara_int(int ncid, int varid, size_t *startp, 
                    size_t *countp, int *ip)
    int nc_put_vara_long(int ncid, int varid, size_t *startp, 
                     size_t *countp, long *op)
    int nc_get_vara_long(int ncid, int varid,
            size_t *startp, size_t *countp, long *ip)
    int nc_put_vara_float(int ncid, int varid,
            size_t *startp, size_t *countp, float *op)
    int nc_get_vara_float(int ncid, int varid,
            size_t *startp, size_t *countp, float *ip)
    int nc_put_vara_double(int ncid, int varid, size_t *startp, 
                       size_t *countp, double *op)
    int nc_get_vara_double(int ncid, int varid, size_t *startp, 
                       size_t *countp, double *ip)
    int nc_put_vara_ubyte(int ncid, int varid, size_t *startp, 
                      size_t *countp, unsigned char *op)
    int nc_get_vara_ubyte(int ncid, int varid, size_t *startp, 
                      size_t *countp, unsigned char *ip)
    int nc_put_vara_ushort(int ncid, int varid, size_t *startp, 
                       size_t *countp, unsigned short *op)
    int nc_get_vara_ushort(int ncid, int varid, size_t *startp, 
                       size_t *countp, unsigned short *ip)
    int nc_put_vara_uint(int ncid, int varid, size_t *startp, 
                     size_t *countp, unsigned int *op)
    int nc_get_vara_uint(int ncid, int varid, size_t *startp, 
                     size_t *countp, unsigned int *ip)
    int nc_put_vara_longlong(int ncid, int varid, size_t *startp, 
                      size_t *countp, long long *op)
    int nc_get_vara_longlong(int ncid, int varid, size_t *startp, 
                      size_t *countp, long long *ip)
    int nc_put_vara_ulonglong(int ncid, int varid, size_t *startp, 
                       size_t *countp, unsigned long long *op)
    int nc_get_vara_ulonglong(int ncid, int varid, size_t *startp, 
                       size_t *countp, unsigned long long *ip)
    int nc_put_vars_text(int ncid, int varid,
            size_t *startp, size_t *countp, ptrdiff_t *stridep,
            char *op)
    int nc_get_vars_text(int ncid, int varid,
            size_t *startp, size_t *countp, ptrdiff_t *stridep,
            char *ip)
    int nc_put_vars_uchar(int ncid, int varid,
            size_t *startp, size_t *countp, ptrdiff_t *stridep,
            unsigned char *op)
    int nc_get_vars_uchar(int ncid, int varid,
            size_t *startp, size_t *countp, ptrdiff_t *stridep,
            unsigned char *ip)
    int nc_put_vars_schar(int ncid, int varid,
            size_t *startp, size_t *countp, ptrdiff_t *stridep,
            signed char *op)
    int nc_get_vars_schar(int ncid, int varid,
            size_t *startp, size_t *countp, ptrdiff_t *stridep,
            signed char *ip)
    int nc_put_vars_short(int ncid, int varid,
            size_t *startp, size_t *countp, ptrdiff_t *stridep,
            short *op)
    int nc_get_vars_short(int ncid, int varid, size_t *startp, 
                      size_t *countp, ptrdiff_t *stridep,
                      short *ip)
    int nc_put_vars_int(int ncid, int varid,
            size_t *startp, size_t *countp, ptrdiff_t *stridep,
            int *op)
    int nc_get_vars_int(int ncid, int varid,
            size_t *startp, size_t *countp, ptrdiff_t *stridep,
            int *ip)
    int nc_put_vars_long(int ncid, int varid,
            size_t *startp, size_t *countp, ptrdiff_t *stridep,
            long *op) 
    int nc_get_vars_long(int ncid, int varid,
            size_t *startp, size_t *countp, ptrdiff_t *stridep,
            long *ip)
    int nc_put_vars_float(int ncid, int varid,
            size_t *startp, size_t *countp, ptrdiff_t *stridep,
            float *op) 
    int nc_get_vars_float(int ncid, int varid,
            size_t *startp, size_t *countp, ptrdiff_t *stridep,
            float *ip)
    int nc_put_vars_double(int ncid, int varid,
            size_t *startp, size_t *countp, ptrdiff_t *stridep,
            double *op)
    int nc_get_vars_double(int ncid, int varid, size_t *startp, 
                       size_t *countp, ptrdiff_t *stridep,
                       double *ip)
    int nc_put_vars_ubyte(int ncid, int varid, size_t *startp, 
                      size_t *countp, ptrdiff_t *stridep, 
                      unsigned char *op)
    int nc_get_vars_ubyte(int ncid, int varid, size_t *startp, 
                      size_t *countp, ptrdiff_t *stridep, 
                      unsigned char *ip)
    int nc_put_vars_ushort(int ncid, int varid, size_t *startp, 
                       size_t *countp, ptrdiff_t *stridep, 
                       unsigned short *op)
    int nc_get_vars_ushort(int ncid, int varid, size_t *startp, 
                       size_t *countp, ptrdiff_t *stridep, 
                       unsigned short *ip)
    int nc_put_vars_uint(int ncid, int varid, size_t *startp, 
                     size_t *countp, ptrdiff_t *stridep, 
                     unsigned int *op)
    int nc_get_vars_uint(int ncid, int varid, size_t *startp, 
                     size_t *countp, ptrdiff_t *stridep, 
                     unsigned int *ip)
    int nc_put_vars_longlong(int ncid, int varid, size_t *startp, 
                      size_t *countp, ptrdiff_t *stridep, 
                      long long *op)
    int nc_get_vars_longlong(int ncid, int varid, size_t *startp, 
                      size_t *countp, ptrdiff_t *stridep, 
                      long long *ip)
    int nc_put_vars_ulonglong(int ncid, int varid, size_t *startp, 
                       size_t *countp, ptrdiff_t *stridep, 
                       unsigned long long *op)
    int nc_get_vars_ulonglong(int ncid, int varid, size_t *startp, 
                       size_t *countp, ptrdiff_t *stridep, 
                       unsigned long long *ip)
    int nc_put_varm_text(int ncid, int varid, size_t *startp, 
                     size_t *countp, ptrdiff_t *stridep,
                     ptrdiff_t *imapp, char *op)
    int nc_get_varm_text(int ncid, int varid, size_t *startp, 
                     size_t *countp, ptrdiff_t *stridep,
                     ptrdiff_t *imapp, char *ip)
    int nc_put_varm_uchar(int ncid, int varid, size_t *startp, 
                      size_t *countp, ptrdiff_t *stridep,
                      ptrdiff_t *imapp, unsigned char *op)
    int nc_get_varm_uchar(int ncid, int varid, size_t *startp, 
                      size_t *countp, ptrdiff_t *stridep,
                      ptrdiff_t *imapp, unsigned char *ip)
    int nc_put_varm_schar(int ncid, int varid, size_t *startp, 
                      size_t *countp, ptrdiff_t *stridep,
                      ptrdiff_t *imapp, signed char *op)
    int nc_get_varm_schar(int ncid, int varid, size_t *startp, 
                      size_t *countp, ptrdiff_t *stridep,
                      ptrdiff_t *imapp, signed char *ip)
    int nc_put_varm_short(int ncid, int varid, size_t *startp, 
                      size_t *countp, ptrdiff_t *stridep,
                      ptrdiff_t *imapp, short *op)
    int nc_get_varm_short(int ncid, int varid, size_t *startp, 
                      size_t *countp, ptrdiff_t *stridep,
                      ptrdiff_t *imapp, short *ip)
    int nc_put_varm_int(int ncid, int varid, size_t *startp, 
                    size_t *countp, ptrdiff_t *stridep,
                    ptrdiff_t *imapp, int *op)
    int nc_get_varm_int(int ncid, int varid, size_t *startp, 
                    size_t *countp, ptrdiff_t *stridep,
                    ptrdiff_t *imapp, int *ip)
    int nc_put_varm_long(int ncid, int varid, size_t *startp, 
                     size_t *countp, ptrdiff_t *stridep,
                     ptrdiff_t *imapp, long *op)
    int nc_get_varm_long(int ncid, int varid, size_t *startp, 
                     size_t *countp, ptrdiff_t *stridep,
                     ptrdiff_t *imapp, long *ip)
    int nc_put_varm_float(int ncid, int varid,size_t *startp, 
                      size_t *countp, ptrdiff_t *stridep,
                      ptrdiff_t *imapp, float *op)
    int nc_get_varm_float(int ncid, int varid,size_t *startp, 
                      size_t *countp, ptrdiff_t *stridep,
                      ptrdiff_t *imapp, float *ip)
    int nc_put_varm_double(int ncid, int varid, size_t *startp, 
                       size_t *countp, ptrdiff_t *stridep,
                       ptrdiff_t *imapp, double *op)
    int nc_get_varm_double(int ncid, int varid, size_t *startp, 
                       size_t *countp, ptrdiff_t *stridep,
                       ptrdiff_t * imapp, double *ip)
    int nc_put_varm_ubyte(int ncid, int varid, size_t *startp, 
                      size_t *countp, ptrdiff_t *stridep, 
                      ptrdiff_t * imapp, unsigned char *op)
    int nc_get_varm_ubyte(int ncid, int varid, size_t *startp, 
                      size_t *countp, ptrdiff_t *stridep, 
                      ptrdiff_t * imapp, unsigned char *ip)
    int nc_put_varm_ushort(int ncid, int varid, size_t *startp, 
                       size_t *countp, ptrdiff_t *stridep, 
                       ptrdiff_t * imapp, unsigned short *op)
    int nc_get_varm_ushort(int ncid, int varid, size_t *startp, 
                       size_t *countp, ptrdiff_t *stridep, 
                       ptrdiff_t * imapp, unsigned short *ip)
    int nc_put_varm_uint(int ncid, int varid, size_t *startp, 
                     size_t *countp, ptrdiff_t *stridep, 
                     ptrdiff_t * imapp, unsigned int *op)
    int nc_get_varm_uint(int ncid, int varid, size_t *startp, 
                     size_t *countp, ptrdiff_t *stridep, 
                     ptrdiff_t * imapp, unsigned int *ip)
    int nc_put_varm_longlong(int ncid, int varid, size_t *startp, 
                      size_t *countp, ptrdiff_t *stridep, 
                      ptrdiff_t * imapp, long long *op)
    int nc_get_varm_longlong(int ncid, int varid, size_t *startp, 
                      size_t *countp, ptrdiff_t *stridep, 
                      ptrdiff_t * imapp, long long *ip)
    int nc_put_varm_ulonglong(int ncid, int varid, size_t *startp, 
                       size_t *countp, ptrdiff_t *stridep, 
                       ptrdiff_t * imapp, unsigned long long *op)
    int nc_get_varm_ulonglong(int ncid, int varid, size_t *startp, 
                       size_t *countp, ptrdiff_t *stridep, 
                       ptrdiff_t * imapp, unsigned long long *ip)
    int nc_put_var_text(int ncid, int varid, char *op)
    int nc_get_var_text(int ncid, int varid, char *ip)
    int nc_put_var_uchar(int ncid, int varid, unsigned char *op)
    int nc_get_var_uchar(int ncid, int varid, unsigned char *ip)
    int nc_put_var_schar(int ncid, int varid, signed char *op)
    int nc_get_var_schar(int ncid, int varid, signed char *ip)
    int nc_put_var_short(int ncid, int varid, short *op)
    int nc_get_var_short(int ncid, int varid, short *ip)
    int nc_put_var_int(int ncid, int varid, int *op)
    int nc_get_var_int(int ncid, int varid, int *ip)
    int nc_put_var_long(int ncid, int varid, long *op)
    int nc_get_var_long(int ncid, int varid, long *ip)
    int nc_put_var_float(int ncid, int varid, float *op)
    int nc_get_var_float(int ncid, int varid, float *ip)
    int nc_put_var_double(int ncid, int varid, double *op)
    int nc_get_var_double(int ncid, int varid, double *ip)
    int nc_put_var_ubyte(int ncid, int varid, unsigned char *op)
    int nc_get_var_ubyte(int ncid, int varid, unsigned char *ip)
    int nc_put_var_ushort(int ncid, int varid, unsigned short *op)
    int nc_get_var_ushort(int ncid, int varid, unsigned short *ip)
    int nc_put_var_uint(int ncid, int varid, unsigned int *op)
    int nc_get_var_uint(int ncid, int varid, unsigned int *ip)
    int nc_put_var_longlong(int ncid, int varid, long long *op)
    int nc_get_var_longlong(int ncid, int varid, long long *ip)
    int nc_put_var_ulonglong(int ncid, int varid, unsigned long long *op)
    int nc_get_var_ulonglong(int ncid, int varid, unsigned long long *ip)
    # set logging verbosity level.
    void nc_set_log_level(int new_level)
    int nc_show_metadata(int ncid)
    int nc_free_vlen(nc_vlen_t *vl)
    int nc_free_vlens(size_t len, nc_vlen_t *vl)
    int nc_free_string(size_t len, char **data)
    int nc_set_chunk_cache(size_t size, size_t nelems, float preemption)
    int nc_get_chunk_cache(size_t *sizep, size_t *nelemsp, float *preemptionp)
    int nc_set_var_chunk_cache(int ncid, int varid, size_t size, size_t nelems, float preemption)
    int nc_get_var_chunk_cache(int ncid, int varid, size_t *sizep, size_t *nelemsp, float *preemptionp) nogil
    int nc_rename_grp(int grpid, char *name)
    int nc_def_enum(int ncid, nc_type base_typeid, char *name, nc_type *typeidp)
    int nc_insert_enum(int ncid, nc_type xtype, char *name, void *value)
    int nc_inq_enum(int ncid, nc_type xtype, char *name, nc_type *base_nc_typep,\
	    size_t *base_sizep, size_t *num_membersp) nogil
    int nc_inq_enum_member(int ncid, nc_type xtype, int idx, char *name, void *value) nogil

    int nc_inq_enum_ident(int ncid, nc_type xtype, long long value, char *identifier) nogil


IF HAS_NC_OPEN_MEM:
    cdef extern from "netcdf_mem.h":
        int nc_open_mem(const char *path, int mode, size_t size, void* memory, int *ncidp)

# taken from numpy.pxi in numpy 1.0rc2.
cdef extern from "numpy/arrayobject.h":
    ctypedef int npy_intp 
    ctypedef extern class numpy.ndarray [object PyArrayObject]:
        cdef char *data
        cdef int nd
        cdef npy_intp *dimensions
        cdef npy_intp *strides
        cdef object base
#       cdef dtype descr
        cdef int flags
    npy_intp PyArray_SIZE(ndarray arr)
    npy_intp PyArray_ISCONTIGUOUS(ndarray arr)
    npy_intp PyArray_ISALIGNED(ndarray arr)
    void import_array()
