#ifndef NETCDF_COMPAT_H
#define NETCDF_COMPAT_H

#include <netcdf.h>
#include <netcdf_meta.h>

#define NC_VERSION_EQ(MAJOR, MINOR, PATCH) \
  ((NC_VERSION_MAJOR == (MAJOR)) && \
   (NC_VERSION_MINOR == (MINOR)) && \
   (NC_VERSION_PATCH == (PATCH)))

#define NC_VERSION_GT(MAJOR, MINOR, PATCH) \
  (NC_VERSION_MAJOR > (MAJOR) || \
   (NC_VERSION_MAJOR == (MAJOR) && \
    (NC_VERSION_MINOR > (MINOR) || \
     (NC_VERSION_MINOR == (MINOR) && \
      (NC_VERSION_PATCH > (PATCH))))))

#define NC_VERSION_GE(MAJOR, MINOR, PATCH) \
  (NC_VERSION_GT(MAJOR, MINOR, PATCH) || \
   NC_VERSION_EQ(MAJOR, MINOR, PATCH))

#if NC_VERSION_GE(4, 3, 0)
#define HAS_RENAME_GRP 1
#else
#define HAS_RENAME_GRP 0
static inline int nc_rename_grp(int grpid, const char* name) { return NC_EINVAL; }
#endif

#if NC_VERSION_GE(4, 1, 2)
#define HAS_NC_INQ_PATH 1
#else
#define HAS_NC_INQ_PATH 0
static inline int nc_inq_path(int ncid, size_t *pathlen, char *path) {
  *pathlen = 0; *path = "\0"; return NC_EINVAL;
}
#endif

#if NC_VERSION_GE(4, 3, 1)
#define HAS_NC_INQ_FORMAT_EXTENDED 1
#else
#define HAS_NC_INQ_FORMAT_EXTENDED 0
static inline int nc_inq_format_extended(int ncid, int *formatp, int* modep) {
  *formatp = 0; *modep = 0; return NC_EINVAL;
}
#endif

#if NC_VERSION_GE(4, 9, 0)
#define HAS_SET_ALIGNMENT 1
#else
#define HAS_SET_ALIGNMENT 0
static inline int nc_set_alignment(int threshold, int alignment) { return NC_EINVAL; }
static inline int nc_get_alignment(int* thresholdp, int* alignmentp) {
  *thresholdp = 0; *alignmentp = 0; return NC_EINVAL;
}
#endif

#if NC_VERSION_GE(4, 9, 0)
#define HAS_NCRCSET 1
#else
#define HAS_NCRCSET 0
static inline int nc_rc_set(const char* key, const char* value) { return NC_EINVAL; }
#endif

#if NC_VERSION_GE(4, 4, 0)
#include <netcdf_mem.h>
#define HAS_NC_OPEN_MEM 1
#else
#define HAS_NC_OPEN_MEM 0
static inline int nc_open_mem(const char *path, int mode, size_t size, void* memory, int *ncidp) { return NC_EINVAL; }
#endif

#if NC_VERSION_GE(4, 6, 2)
#define HAS_NC_CREATE_MEM 1
#else
#define HAS_NC_CREATE_MEM 0
static inline int nc_create_mem(const char *path, int mode, size_t initialize, int *ncidp) { return NC_EINVAL; }
typedef struct NC_memio {
  size_t size;
  void* memory;
  int flags;
} NC_memio;
static inline int nc_close_memio(int ncid, NC_memio* info) { return NC_EINVAL; }
#endif

#if defined(NC_HAS_CDF5) && NC_HAS_CDF5
#define HAS_CDF5_FORMAT 1
#else
# ifndef NC_HAS_CDF5
# define NC_64BIT_DATA    0x0020
# define NC_CDF5          NC_64BIT_DATA
# define NC_FORMAT_64BIT_OFFSET    (2)
# define NC_FORMAT_64BIT_DATA      (5)
# endif
#define HAS_CDF5_FORMAT 0
#endif

#if defined(NC_HAS_PARALLEL) && NC_HAS_PARALLEL
#include <netcdf_par.h>
#define HAS_PARALLEL_SUPPORT 1
#else
#define HAS_PARALLEL_SUPPORT 0
typedef int MPI_Comm;
typedef int MPI_Info;
static inline int nc_create_par(const char *path, int cmode, MPI_Comm comm, MPI_Info info, int *ncidp) { return NC_EINVAL; }
static inline int nc_open_par(const char *path, int mode, MPI_Comm comm, MPI_Info info, int *ncidp) { return NC_EINVAL; }
static inline int nc_var_par_access(int ncid, int varid, int par_access) { return NC_EINVAL; }
# ifndef NC_INDEPENDENT
#  define NC_INDEPENDENT 0
#  define NC_COLLECTIVE 1
# endif
# ifndef NC_MPIIO
#  define NC_MPIIO 0x2000
#  define NC_MPIPOSIX NC_MPIIO
#  define NC_PNETCDF (NC_MPIIO)
# endif
#endif

#if defined(NC_HAS_PARALLEL4) && NC_HAS_PARALLEL4
#define HAS_PARALLEL4_SUPPORT 1
#else
#define HAS_PARALLEL4_SUPPORT 0
#endif

#if defined(NC_HAS_PNETCDF) && NC_HAS_PNETCDF
#define HAS_PNETCDF_SUPPORT 1
#else
#define HAS_PNETCDF_SUPPORT 0
#endif

#if NC_VERSION_GE(4, 7, 0)
#include <netcdf_filter.h>
#endif

#if NC_VERSION_GE(4, 9, 0)
#define HAS_NCFILTER 1
#else
#define HAS_NCFILTER 0
static inline int nc_inq_filter_avail(int ncid, unsigned filterid) { return -136; }
#endif

#if defined(NC_HAS_SZIP) && NC_HAS_SZIP
#define HAS_SZIP_SUPPORT 1
#else
#define HAS_SZIP_SUPPORT 0
# ifndef NC_HAS_SZIP
static inline int nc_def_var_szip(int ncid, int varid, int options_mask, int pixels_per_bloc) { return NC_EINVAL; }
# endif
# ifndef H5Z_FILTER_SZIP
#  define H5Z_FILTER_SZIP 4
# endif
#endif

#if defined(NC_HAS_QUANTIZE) && NC_HAS_QUANTIZE
#define HAS_QUANTIZATION_SUPPORT 1
#else
#define HAS_QUANTIZATION_SUPPORT 0
# ifndef NC_HAS_QUANTIZE
static inline int nc_def_var_quantize(int ncid, int varid, int quantize_mode, int nsd) { return NC_EINVAL; }
static inline int nc_inq_var_quantize(int ncid, int varid, int *quantize_modep, int *nsdp) { return NC_EINVAL; }
# define NC_NOQUANTIZE 0
# define NC_QUANTIZE_BITGROOM 1
# define NC_QUANTIZE_GRANULARBR 2
# define NC_QUANTIZE_BITROUND 3
# endif
#endif

#if defined(NC_HAS_ZSTD) && NC_HAS_ZSTD
#define HAS_ZSTANDARD_SUPPORT 1
#else
# ifndef NC_HAS_ZSTD
static inline int nc_def_var_zstandard(int ncid, int varid, int level) { return NC_EINVAL; }
static inline int nc_inq_var_zstandard(int ncid, int varid, int* hasfilterp, int *levelp) { return NC_EINVAL; }
# define H5Z_FILTER_ZSTD 32015
# endif
#define HAS_ZSTANDARD_SUPPORT 0
#endif

#if defined(NC_HAS_BZ2) && NC_HAS_BZ2
#define HAS_BZIP2_SUPPORT 1
#else
# ifndef NC_HAS_BZ2
static inline int nc_def_var_bzip2(int ncid, int varid, int level) { return NC_EINVAL; }
static inline int nc_inq_var_bzip2(int ncid, int varid, int* hasfilterp, int *levelp) { return NC_EINVAL; }
# define H5Z_FILTER_BZIP2 307
# endif
#define HAS_BZIP2_SUPPORT 0
#endif

#if defined(NC_HAS_BLOSC) && NC_HAS_BLOSC
#define HAS_BLOSC_SUPPORT 1
#else
# ifndef NC_HAS_BLOSC
static inline int nc_def_var_blosc(int ncid, int varid, unsigned subcompressor, unsigned level, unsigned blocksize, unsigned addshuffle) {
  return NC_EINVAL;
}
static inline int nc_inq_var_blosc(int ncid, int varid, int* hasfilterp, unsigned* subcompressorp, unsigned* levelp, unsigned* blocksizep, unsigned* addshufflep) {
  return NC_EINVAL;
}
# define H5Z_FILTER_BLOSC 32001
# endif
#define HAS_BLOSC_SUPPORT 0
#endif

#endif /* NETCDF_COMPAT_H */
