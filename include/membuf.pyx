# Buffer code found here similar to
# http://stackoverflow.com/a/28166272/428751
# Allows to return a malloced python buffer,
# which will be freed when the python object is garbage collected.
from cpython.buffer cimport PyBuffer_FillInfo, PyBuffer_Release
from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdlib cimport free
from libc.stdint cimport uintptr_t

ctypedef void dealloc_callback(const void *p, size_t l, void *arg)

cdef void free_buf(const void *p, size_t l, void *arg):
    free(<void *>p)

# this is the function used to create a memory view from
# a raw pointer.
cdef makebuf(void *p, size_t l):
    assert p!=NULL, "invalid NULL buffer pointer"
    return memoryview( MemBuf_init(p, l, &free_buf, NULL) )

cdef class MemBuf:
    cdef const void *p
    cdef size_t l
    cdef dealloc_callback *dealloc_cb_p
    cdef void *dealloc_cb_arg

    def __len__(self):
        return self.l

    def __repr__(self):
        return "MemBuf(%#x)" % (<uintptr_t> self.p)

    cdef const void *get_mem(self):
        return self.p

    def __getbuffer__(self, Py_buffer *buf, int flags):
        cdef int ret,readonly
        readonly=1
        ret=PyBuffer_FillInfo(buf, self, <void *>self.p, self.l, readonly, flags)

    def __releasebuffer__(self, Py_buffer *buf):
        #PyBuffer_Release(buf) 
        pass

    def __dealloc__(self):
        if self.dealloc_cb_p != NULL:
            self.dealloc_cb_p(self.p, self.l, self.dealloc_cb_arg)

    # not really needed if MemBuf converted to memoryview
    def tobytes(self):
        return PyBytes_FromStringAndSize(<char *>self.p, self.l)

# Call this instead of constructing a MemBuf directly.  The __cinit__
# and __init__ methods can only take Python objects, so the real
# constructor is here.  See:
# https://mail.python.org/pipermail/cython-devel/2012-June/002734.html
cdef MemBuf MemBuf_init(const void *p, size_t l,
                        dealloc_callback *dealloc_cb_p,
                        void *dealloc_cb_arg):
    cdef MemBuf ret = MemBuf()
    ret.p = p
    ret.l = l
    ret.dealloc_cb_p = dealloc_cb_p
    ret.dealloc_cb_arg = dealloc_cb_arg
    return ret
