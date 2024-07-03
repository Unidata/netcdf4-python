#include "nc_complex/nc_complex.h"

#include <ctype.h>
#include <netcdf.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "nc_complex_version.h"

// NOLINTBEGIN(bugprone-assignment-in-if-condition)
#define CHECK(func)           \
    do {                      \
        int res;              \
        if ((res = (func))) { \
            return res;       \
        }                     \
    } while (0)
// NOLINTEND(bugprone-assignment-in-if-condition)

// Vector of ones for get/put_var1 functions
static const size_t coord_one[NC_MAX_VAR_DIMS] = {1};

static const char* double_complex_struct_name = "_PFNC_DOUBLE_COMPLEX_TYPE";
static const char* float_complex_struct_name = "_PFNC_FLOAT_COMPLEX_TYPE";

#define COMPLEX_DIM_NAME "_pfnc_complex"
static const char* complex_dim_name = COMPLEX_DIM_NAME;

static const char* known_dim_names[] = {COMPLEX_DIM_NAME, "complex", "ri"};
static const size_t num_known_dim_names =
    sizeof(known_dim_names) / sizeof(known_dim_names[0]);

static const char pfnc_libvers[] = NC_COMPLEX_GIT_VERSION;

const char* pfnc_inq_libvers(void) {
    return pfnc_libvers;
}

bool pfnc_var_is_complex(int ncid, int varid) {
    return pfnc_var_is_complex_type(ncid, varid)
        || pfnc_var_has_complex_dimension(ncid, varid);
}

int pfnc_complex_base_type(int ncid, nc_type nc_typeid, nc_type* base_type_id) {
    if (nc_typeid < NC_MAX_ATOMIC_TYPE) {
        *base_type_id = nc_typeid;
        return NC_NOERR;
    }

    // TODO: This should probably handle vlens too

    return nc_inq_compound_field(
        ncid, nc_typeid, 0, NULL, NULL, base_type_id, NULL, NULL
    );
}

int pfnc_inq_var_complex_base_type(int ncid, int varid, nc_type* nc_typeid) {
    nc_type var_type_id;
    CHECK(nc_inq_vartype(ncid, varid, &var_type_id));
    return pfnc_complex_base_type(ncid, var_type_id, nc_typeid);
}

/// Return true if a compound type is compatible with a known convention
bool compound_type_is_compatible(int ncid, nc_type nc_typeid) {
    // Does the name matching a known convention?
    char name[NC_MAX_NAME + 1];
    nc_inq_compound_name(ncid, nc_typeid, name);
    if (name == double_complex_struct_name) {
        return true;
    }

    // Does it have exactly two fields?
    size_t num_fields;
    nc_inq_compound_nfields(ncid, nc_typeid, &num_fields);
    if (num_fields != 2) {
        return false;
    }

    // As far as I can tell, all conventions put the real part first and
    // the imaginary part second. I'm pretty sure all native language
    // types are also this way round. That means we don't have to worry
    // about trying both combinations!
    char real_name[NC_MAX_NAME + 1];
    size_t real_offset;
    nc_type real_field_type;
    int real_rank;
    nc_inq_compound_field(
        ncid, nc_typeid, 0, real_name, &real_offset, &real_field_type, &real_rank, NULL
    );

    // If it's not a floating type, we're not interested
    if (!(real_field_type == NC_FLOAT || real_field_type == NC_DOUBLE)) {
        return false;
    }
    // Also needs to be scalar
    if (real_rank != 0) {
        return false;
    }

    // Now check names. For now, just check it starts with "r", in any case
    if (tolower(real_name[0]) != 'r') {
        return false;
    }

    char imag_name[NC_MAX_NAME + 1];
    size_t imag_offset;
    nc_type imag_field_type;
    int imag_rank;
    nc_inq_compound_field(
        ncid, nc_typeid, 1, imag_name, &imag_offset, &imag_field_type, &imag_rank, NULL
    );

    // Both component types better match
    if (imag_field_type != real_field_type) {
        return false;
    }
    if (imag_rank != 0) {
        return false;
    }
    if (tolower(imag_name[0]) != 'i') {
        return false;
    }

    return true;
}

/// Return true if file already has a complex type with the given base type
bool file_has_complex_struct(int ncid, nc_type* typeidp, nc_type base_type) {
    // Simplest case, check for our type name
    int err = nc_inq_typeid(ncid, double_complex_struct_name, typeidp);
    if (err == NC_NOERR) {
        return true;
    }

    int ntypes;
    err = nc_inq_typeids(ncid, &ntypes, NULL);
    if (err != NC_NOERR) {
        return false;
    }

    bool result = false;

    nc_type* typeids = malloc((size_t)ntypes * sizeof(nc_type));
    err = nc_inq_typeids(ncid, NULL, typeids);
    if (err != NC_NOERR) {
        goto cleanup;
    }

    for (size_t i = 0; i < (size_t)ntypes; i++) {
        if (compound_type_is_compatible(ncid, typeids[i])) {
            nc_type base_type_id;
            err = pfnc_complex_base_type(ncid, typeids[i], &base_type_id);
            if (err != NC_NOERR) {
                goto cleanup;
            }
            if (base_type_id == base_type) {
                *typeidp = typeids[i];
                result = true;
                goto cleanup;
            }
        }
    }
cleanup:
    free(typeids);
    return result;
}

/// Return true if a given dimension matches a known convention
bool pfnc_is_complex_dim(int ncid, int dim_id) {
    size_t length;
    nc_inq_dimlen(ncid, dim_id, &length);

    // Definitely can only be exactly two. Note that we can't catch
    // unlimited dimensions that only have two records so far.
    if (length != 2) {
        return false;
    }

    // Not sure if this is the best way, but here we are.
    char name[NC_MAX_NAME + 1];
    nc_inq_dimname(ncid, dim_id, name);

    const size_t name_length = strlen(name);

    // Check against known names of complex dimensions
    for (size_t i = 0; i < num_known_dim_names; i++) {
        if (strncmp(name, known_dim_names[i], name_length) == 0) {
            return true;
        }
    }

    return false;
}

/// Return true if a variable uses the dimension-convention
bool pfnc_var_has_complex_dimension(int ncid, int nc_varid) {
    int num_dims;
    nc_inq_varndims(ncid, nc_varid, &num_dims);

    int* dim_ids = (int*)malloc((size_t)num_dims * sizeof(int));
    nc_inq_vardimid(ncid, nc_varid, dim_ids);

    // Now we check if any of the dimensions match one of our known
    // conventions. Do we need to check all of them, or just the
    // first/last?
    for (int i = 0; i < num_dims; i++) {
        if (pfnc_is_complex_dim(ncid, dim_ids[i])) {
            free(dim_ids);
            return true;
        }
    }

    free(dim_ids);
    return false;
}

/// Return true if a netCDF datatype is a compound type
bool is_compound_type(int ncid, int type_id) {
    // There appears to be no API for detecting whether a type ID is a
    // primitive type, so we have to check ourselves
    if (type_id <= NC_MAX_ATOMIC_TYPE) {
        return false;
    }

    int class_type;
    nc_inq_user_type(ncid, type_id, NULL, NULL, NULL, NULL, &class_type);
    return class_type == NC_COMPOUND;
}

/// Copy an array meant for a complex-dimensioned variable
size_t* copy_complex_dim_size_t_array(
    const size_t* old_array, int numdims, size_t complex_dim_value
) {
    size_t* new_buffer = NULL;

    if (old_array != NULL) {
        new_buffer = (size_t*)malloc(sizeof(size_t) * (size_t)numdims);

        size_t last_dim = (size_t)(numdims - 1);
        for (size_t i = 0; i < last_dim; i++) {
            new_buffer[i] = old_array[i];
        }

        new_buffer[last_dim] = complex_dim_value;
    }
    return new_buffer;
}

ptrdiff_t* copy_complex_dim_ptrdiff_t_array(
    const ptrdiff_t* old_array, int numdims, ptrdiff_t complex_dim_value
) {
    ptrdiff_t* new_buffer = NULL;

    if (old_array != NULL) {
        new_buffer = (ptrdiff_t*)malloc(sizeof(ptrdiff_t) * (size_t)numdims);

        size_t last_dim = (size_t)(numdims - 1);
        for (size_t i = 0; i < last_dim; i++) {
            new_buffer[i] = old_array[i];
        }

        new_buffer[last_dim] = complex_dim_value;
    }
    return new_buffer;
}

bool pfnc_var_is_complex_type(int ncid, int varid) {
    nc_type var_type_id;
    if (nc_inq_vartype(ncid, varid, &var_type_id)) {
        return false;
    }

    if (is_compound_type(ncid, var_type_id)) {
        return compound_type_is_compatible(ncid, var_type_id);
    }
    return false;
}

size_t complex_type_size(nc_type base_type) {
    switch (base_type) {
    case NC_FLOAT:
        return sizeof(float_complex);
    case NC_DOUBLE:
        return sizeof(double_complex);
    default:
        return 0;
    }
}

size_t base_type_size(nc_type base_type) {
    switch (base_type) {
    case NC_FLOAT:
        return sizeof(float);
    case NC_DOUBLE:
        return sizeof(double);
    default:
        return 0;
    }
}

int get_or_make_complex_struct(
    int ncid, nc_type* nc_typeid, nc_type base_type, const char* struct_name
) {
    // TODO: Error if not netCDF4

    if (file_has_complex_struct(ncid, nc_typeid, base_type)) {
        return NC_NOERR;
    }

    const size_t complex_size = complex_type_size(base_type);
    if (complex_size == 0) {
        return NC_EBADTYPE;
    }
    const size_t base_size = base_type_size(base_type);
    if (base_size == 0) {
        return NC_EBADTYPE;
    }

    CHECK(nc_def_compound(ncid, complex_size, struct_name, nc_typeid));
    CHECK(nc_insert_compound(ncid, *nc_typeid, "r", 0, base_type));
    CHECK(nc_insert_compound(ncid, *nc_typeid, "i", base_size, base_type));

    return NC_NOERR;
}

int pfnc_get_double_complex_typeid(int ncid, nc_type* nc_typeid) {
    return get_or_make_complex_struct(
        ncid, nc_typeid, NC_DOUBLE, double_complex_struct_name
    );
}

int pfnc_get_float_complex_typeid(int ncid, nc_type* nc_typeid) {
    return get_or_make_complex_struct(
        ncid, nc_typeid, NC_FLOAT, float_complex_struct_name
    );
}

int pfnc_get_complex_dim(int ncid, int* nc_dim) {
    int num_dims;
    CHECK(nc_inq_ndims(ncid, &num_dims));

    int* dim_ids = (int*)malloc((size_t)num_dims * sizeof(int));
    int ierr = nc_inq_dimids(ncid, NULL, dim_ids, true);
    if (ierr != NC_NOERR) {
        goto cleanup;
    }

    // Now we check if any of the dimensions match one of our known
    // conventions. Do we need to check all of them, or just the
    // first/last?
    for (int i = 0; i < num_dims; i++) {
        if (pfnc_is_complex_dim(ncid, dim_ids[i])) {
            *nc_dim = dim_ids[i];
            goto cleanup;
        }
    }

    ierr = nc_def_dim(ncid, complex_dim_name, 2, nc_dim);

cleanup:
    free(dim_ids);
    return ierr;
}

int pfnc_put_vara_double_complex(
    int ncid,
    int varid,
    const size_t* startp,
    const size_t* countp,
    const double_complex* op
) {
    return pfnc_put_vars_double_complex(ncid, varid, startp, countp, NULL, op);
}

int pfnc_get_vara_double_complex(
    int ncid, int varid, const size_t* startp, const size_t* countp, double_complex* ip
) {
    return pfnc_get_vars_double_complex(ncid, varid, startp, countp, NULL, ip);
}

int pfnc_put_vars_double_complex(
    int ncid,
    int varid,
    const size_t* startp,
    const size_t* countp,
    const ptrdiff_t* stridep,
    const double_complex* op
) {
    if (!pfnc_var_is_complex(ncid, varid)) {
        return NC_EBADTYPE;
    }

    // TODO: handle converting different float sizes

    // Check if we can get away without fudging count/start sizes
    if (((startp == NULL) && (countp == NULL) && (stridep == NULL))
        || !pfnc_var_has_complex_dimension(ncid, varid)) {
        return nc_put_vars(ncid, varid, startp, countp, stridep, op);
    }

    // The real variable has a complex dimension, but we're pretending
    // it doesn't, so now we need start/count arrays of the real size

    int numdims = 0;
    CHECK(nc_inq_varndims(ncid, varid, &numdims));

    // Copy start/count buffers, appending an extra element for the
    // complex dimension. This dimension starts at 0 and has 2 elements
    size_t* start_buffer = copy_complex_dim_size_t_array(startp, numdims, 0);
    size_t* count_buffer = copy_complex_dim_size_t_array(countp, numdims, 2);
    ptrdiff_t* stride_buffer = copy_complex_dim_ptrdiff_t_array(stridep, numdims, 1);

    const int ierr =
        nc_put_vars(ncid, varid, start_buffer, count_buffer, stride_buffer, op);

    if (start_buffer != NULL) {
        free(start_buffer);
    }
    if (count_buffer != NULL) {
        free(count_buffer);
    }
    if (stride_buffer != NULL) {
        free(stride_buffer);
    }
    return ierr;
}

int pfnc_get_vars_double_complex(
    int ncid,
    int varid,
    const size_t* startp,
    const size_t* countp,
    const ptrdiff_t* stridep,
    double_complex* ip
) {
    if (!pfnc_var_is_complex(ncid, varid)) {
        return NC_EBADTYPE;
    }

    // TODO: handle converting different float sizes

    // Check if we can get away without fudging count/start sizes
    if (((startp == NULL) && (countp == NULL) && (stridep == NULL))
        || !pfnc_var_has_complex_dimension(ncid, varid)) {
        return nc_get_vars(ncid, varid, startp, countp, stridep, ip);
    }

    // The real variable has a complex dimension, but we're pretending
    // it doesn't, so now we need start/count arrays of the real size

    int numdims = 0;
    CHECK(nc_inq_varndims(ncid, varid, &numdims));

    // Copy start/count buffers, appending an extra element for the
    // complex dimension. This dimension starts at 0 and has 2 elements
    size_t* start_buffer = copy_complex_dim_size_t_array(startp, numdims, 0);
    size_t* count_buffer = copy_complex_dim_size_t_array(countp, numdims, 2);
    ptrdiff_t* stride_buffer = copy_complex_dim_ptrdiff_t_array(stridep, numdims, 1);

    const int ierr =
        nc_get_vars(ncid, varid, start_buffer, count_buffer, stride_buffer, ip);

    if (start_buffer != NULL) {
        free(start_buffer);
    }
    if (count_buffer != NULL) {
        free(count_buffer);
    }
    if (stride_buffer != NULL) {
        free(stride_buffer);
    }
    return ierr;
}

int pfnc_put_var1_double_complex(
    int ncid, int varid, const size_t* indexp, const double_complex* data
) {
    return pfnc_put_vara_double_complex(ncid, varid, indexp, coord_one, data);
}

int pfnc_get_var1_double_complex(
    int ncid, int varid, const size_t* indexp, double_complex* data
) {
    return pfnc_get_vara_double_complex(ncid, varid, indexp, coord_one, data);
}

int pfnc_put_vara_float_complex(
    int ncid,
    int varid,
    const size_t* startp,
    const size_t* countp,
    const float_complex* op
) {
    return pfnc_put_vars_float_complex(ncid, varid, startp, countp, NULL, op);
}

int pfnc_get_vara_float_complex(
    int ncid, int varid, const size_t* startp, const size_t* countp, float_complex* ip
) {
    return pfnc_get_vars_float_complex(ncid, varid, startp, countp, NULL, ip);
}

int pfnc_put_vars_float_complex(
    int ncid,
    int varid,
    const size_t* startp,
    const size_t* countp,
    const ptrdiff_t* stridep,
    const float_complex* op
) {
    if (!pfnc_var_is_complex(ncid, varid)) {
        return NC_EBADTYPE;
    }

    // TODO: handle converting different float sizes

    // Check if we can get away without fudging count/start sizes
    if (((startp == NULL) && (countp == NULL) && (stridep == NULL))
        || !pfnc_var_has_complex_dimension(ncid, varid)) {
        return nc_put_vars(ncid, varid, startp, countp, stridep, op);
    }

    // The real variable has a complex dimension, but we're pretending
    // it doesn't, so now we need start/count arrays of the real size

    int numdims = 0;
    CHECK(nc_inq_varndims(ncid, varid, &numdims));

    // Copy start/count buffers, appending an extra element for the
    // complex dimension. This dimension starts at 0 and has 2 elements
    size_t* start_buffer = copy_complex_dim_size_t_array(startp, numdims, 0);
    size_t* count_buffer = copy_complex_dim_size_t_array(countp, numdims, 2);
    ptrdiff_t* stride_buffer = copy_complex_dim_ptrdiff_t_array(stridep, numdims, 1);

    const int ierr =
        nc_put_vars(ncid, varid, start_buffer, count_buffer, stride_buffer, op);

    if (start_buffer != NULL) {
        free(start_buffer);
    }
    if (count_buffer != NULL) {
        free(count_buffer);
    }
    if (stride_buffer != NULL) {
        free(stride_buffer);
    }
    return ierr;
}

int pfnc_get_vars_float_complex(
    int ncid,
    int varid,
    const size_t* startp,
    const size_t* countp,
    const ptrdiff_t* stridep,
    float_complex* ip
) {
    if (!pfnc_var_is_complex(ncid, varid)) {
        return NC_EBADTYPE;
    }

    // TODO: handle converting different float sizes

    // Check if we can get away without fudging count/start sizes
    if (((startp == NULL) && (countp == NULL) && (stridep == NULL))
        || !pfnc_var_has_complex_dimension(ncid, varid)) {
        return nc_get_vars(ncid, varid, startp, countp, stridep, ip);
    }

    // The real variable has a complex dimension, but we're pretending
    // it doesn't, so now we need start/count arrays of the real size

    int numdims = 0;
    CHECK(nc_inq_varndims(ncid, varid, &numdims));

    // Copy start/count buffers, appending an extra element for the
    // complex dimension. This dimension starts at 0 and has 2 elements
    size_t* start_buffer = copy_complex_dim_size_t_array(startp, numdims, 0);
    size_t* count_buffer = copy_complex_dim_size_t_array(countp, numdims, 2);
    ptrdiff_t* stride_buffer = copy_complex_dim_ptrdiff_t_array(stridep, numdims, 1);

    const int ierr =
        nc_get_vars(ncid, varid, start_buffer, count_buffer, stride_buffer, ip);

    if (start_buffer != NULL) {
        free(start_buffer);
    }
    if (count_buffer != NULL) {
        free(count_buffer);
    }
    if (stride_buffer != NULL) {
        free(stride_buffer);
    }
    return ierr;
}

int pfnc_put_var1_float_complex(
    int ncid, int varid, const size_t* indexp, const float_complex* data
) {
    return pfnc_put_vara_float_complex(ncid, varid, indexp, coord_one, data);
}

int pfnc_get_var1_float_complex(
    int ncid, int varid, const size_t* indexp, float_complex* data
) {
    return pfnc_get_vara_float_complex(ncid, varid, indexp, coord_one, data);
}

int pfnc_def_var(
    int ncid,
    const char* name,
    nc_type xtype,
    int ndims,
    const int* dimidsp,
    int* varidp
) {
    // If it's not a complex number, we don't need to do anything
    if (!(xtype == PFNC_DOUBLE_COMPLEX || xtype == PFNC_DOUBLE_COMPLEX_DIM
          || xtype == PFNC_FLOAT_COMPLEX || xtype == PFNC_FLOAT_COMPLEX_DIM)) {
        return nc_def_var(ncid, name, xtype, ndims, dimidsp, varidp);
    }

    const bool base_is_double =
        (xtype == PFNC_DOUBLE_COMPLEX || xtype == PFNC_DOUBLE_COMPLEX_DIM);

    // Check the format used by this file. If it's some variation on the
    // classic model, then we have to use a complex dimension. Also,
    // NcZarr, for some reason doesn't support compound types (yet?).
    // I _think_ DAP supports compound types
    int format = 0;
    int mode = 0;
    CHECK(nc_inq_format_extended(ncid, &format, &mode));

    if ((format == NC_FORMAT_CLASSIC || format == NC_FORMAT_NETCDF4_CLASSIC)
        || (mode == NC_FORMATX_NCZARR)) {
        xtype = base_is_double ? PFNC_DOUBLE_COMPLEX_DIM : PFNC_FLOAT_COMPLEX_DIM;
    }

    if (xtype == PFNC_DOUBLE_COMPLEX_DIM || xtype == PFNC_FLOAT_COMPLEX_DIM) {
        // Using a complex dimension. We need to get the complex dimension
        // used in this file and append it to the list of dimensions
        // passed in by the user

        int complex_dim = 0;
        CHECK(pfnc_get_complex_dim(ncid, &complex_dim));

        int new_dims = ndims + 1;
        int* dim_ids_buffer = (int*)malloc((size_t)new_dims * sizeof(int));
        for (size_t i = 0; i < (size_t)ndims; i++) {
            dim_ids_buffer[i] = dimidsp[i];
        }
        dim_ids_buffer[ndims] = complex_dim;

        const nc_type base_type = base_is_double ? NC_DOUBLE : NC_FLOAT;

        const int ierr =
            nc_def_var(ncid, name, base_type, new_dims, dim_ids_buffer, varidp);
        free(dim_ids_buffer);
        return ierr;
    }

    // Using a complex type. We need to get the complex type used in
    // this file and pass that as `xtype`
    nc_type complex_type = 0;
    if (base_is_double) {
        CHECK(pfnc_get_double_complex_typeid(ncid, &complex_type));
    } else {
        CHECK(pfnc_get_float_complex_typeid(ncid, &complex_type));
    }

    return nc_def_var(ncid, name, complex_type, ndims, dimidsp, varidp);
}

int pfnc_inq_var(
    int ncid,
    int varid,
    char* name,
    nc_type* xtypep,
    int* ndimsp,
    int* dimidsp,
    int* nattsp
) {
    if (!pfnc_var_has_complex_dimension(ncid, varid)) {
        return nc_inq_var(ncid, varid, name, xtypep, ndimsp, dimidsp, nattsp);
    }

    // Tricky bit: if variable has complex dimension, and user used
    // pfnc_inq_varndims, then dimidsp is one smaller than netCDF thinks
    // it should be. So we'll have to allocate our own array of the
    // correct size and copy out of that.

    // This buffer will point to either the user's array, or our own one
    int* buffer = dimidsp;
    int numdims = 0;

    if (dimidsp != NULL) {
        CHECK(nc_inq_varndims(ncid, varid, &numdims));
        buffer = (int*)malloc(sizeof(int) * (size_t)numdims);
    }

    int ierr = nc_inq_var(ncid, varid, name, xtypep, &numdims, buffer, nattsp);

    if (ierr != NC_NOERR) {
        goto cleanup;
    }

    if (dimidsp != NULL) {
        if (numdims <= 0) {
            // This should never happen
            goto cleanup;
        }
        const size_t other_dims = (size_t)(numdims - 1);
        for (size_t i = 0; i < other_dims; i++) {
            dimidsp[i] = buffer[i];
        }
    }

    if (ndimsp != NULL) {
        *ndimsp = numdims - 1;
    }

cleanup:
    free(buffer);
    return ierr;
}

int pfnc_def_var_chunking(int ncid, int varid, int storage, const size_t* chunksizesp) {
    if (chunksizesp == NULL || !pfnc_var_has_complex_dimension(ncid, varid)) {
        return nc_def_var_chunking(ncid, varid, storage, chunksizesp);
    }

    // The real variable has a complex dimension, but we're pretending
    // it doesn't, so now we need start/count arrays of the real size

    int numdims = 0;
    CHECK(nc_inq_varndims(ncid, varid, &numdims));

    // Copy chunksize buffer, appending an extra element for the
    // complex dimension
    size_t* chunk_buffer = copy_complex_dim_size_t_array(chunksizesp, numdims, 2);

    const int ierr = nc_def_var_chunking(ncid, varid, storage, chunk_buffer);
    free(chunk_buffer);
    return ierr;
}

int pfnc_inq_var_chunking(int ncid, int varid, int* storagep, size_t* chunksizesp) {
    if (chunksizesp == NULL || !pfnc_var_has_complex_dimension(ncid, varid)) {
        return nc_inq_var_chunking(ncid, varid, storagep, chunksizesp);
    }

    int numdims = 0;

    CHECK(nc_inq_varndims(ncid, varid, &numdims));

    // Copy chunksize buffer, appending an extra element for the
    // complex dimension
    size_t* chunk_buffer = copy_complex_dim_size_t_array(chunksizesp, numdims, 2);

    const int ierr = nc_inq_var_chunking(ncid, varid, storagep, chunk_buffer);

    if (ierr != NC_NOERR) {
        goto cleanup;
    }

    const size_t other_dims = (size_t)(numdims - 1);
    for (size_t i = 0; i < other_dims; i++) {
        chunksizesp[i] = chunk_buffer[i];
    }

cleanup:
    free(chunk_buffer);
    return ierr;
}

int pfnc_get_vara(
    int ncid, int varid, const size_t* startp, const size_t* countp, void* ip
) {
    if (pfnc_var_is_complex(ncid, varid)) {
        nc_type base_type;
        CHECK(pfnc_inq_var_complex_base_type(ncid, varid, &base_type));
        switch (base_type) {
        case NC_DOUBLE:
            return pfnc_get_vara_double_complex(ncid, varid, startp, countp, ip);
        case NC_FLOAT:
            return pfnc_get_vara_float_complex(ncid, varid, startp, countp, ip);
        default:
            return NC_EBADTYPE;
        }
    }

    return nc_get_vara(ncid, varid, startp, countp, ip);
}

int pfnc_put_vara(
    int ncid, int varid, const size_t* startp, const size_t* countp, const void* op
) {
    if (pfnc_var_is_complex(ncid, varid)) {
        nc_type base_type;
        CHECK(pfnc_inq_var_complex_base_type(ncid, varid, &base_type));
        switch (base_type) {
        case NC_DOUBLE:
            return pfnc_put_vara_double_complex(ncid, varid, startp, countp, op);
        case NC_FLOAT:
            return pfnc_put_vara_float_complex(ncid, varid, startp, countp, op);
        default:
            return NC_EBADTYPE;
        }
    }
    return nc_put_vara(ncid, varid, startp, countp, op);
}

int pfnc_put_vars(
    int ncid,
    int varid,
    const size_t* startp,
    const size_t* countp,
    const ptrdiff_t* stridep,
    const void* op
) {
    if (pfnc_var_is_complex(ncid, varid)) {
        nc_type base_type;
        CHECK(pfnc_inq_var_complex_base_type(ncid, varid, &base_type));
        switch (base_type) {
        case NC_DOUBLE:
            return pfnc_put_vars_double_complex(
                ncid, varid, startp, countp, stridep, op
            );
        case NC_FLOAT:
            return pfnc_put_vars_float_complex(
                ncid, varid, startp, countp, stridep, op
            );
        default:
            return NC_EBADTYPE;
        }
    }
    return nc_put_vars(ncid, varid, startp, countp, stridep, op);
}

int pfnc_get_vars(
    int ncid,
    int varid,
    const size_t* startp,
    const size_t* countp,
    const ptrdiff_t* stridep,
    void* ip
) {
    if (pfnc_var_is_complex(ncid, varid)) {
        nc_type base_type;
        CHECK(pfnc_inq_var_complex_base_type(ncid, varid, &base_type));
        switch (base_type) {
        case NC_DOUBLE:
            return pfnc_get_vars_double_complex(
                ncid, varid, startp, countp, stridep, ip
            );
        case NC_FLOAT:
            return pfnc_get_vars_float_complex(
                ncid, varid, startp, countp, stridep, ip
            );
        default:
            return NC_EBADTYPE;
        }
    }
    return nc_get_vars(ncid, varid, startp, countp, stridep, ip);
}
