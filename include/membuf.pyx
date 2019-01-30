# Creates a memoryview from a malloced C pointer,
# which will be freed when the python object is garbage collected.
# Code found here is derived from
# http://stackoverflow.com/a/28166272/428751
from cpython.buffer cimport PyBuffer_FillInfo
from libc.stdlib cimport free

# create a python memoryview object from a raw pointer.
cdef memview_fromptr(void *memory, size_t size):
    cdef _MemBuf buf = _MemBuf()
    buf.memory = memory # malloced void pointer
    buf.size = size # size of pointer in bytes
    return memoryview(buf)

# private extension type that implements buffer protocal.
cdef class _MemBuf:
    cdef const void *memory
    cdef size_t size
    def __getbuffer__(self, Py_buffer *buf, int flags):
        PyBuffer_FillInfo(buf, self, <void *>self.memory, self.size, 1, flags)
    def __releasebuffer__(self, Py_buffer *buf):
        # why doesn't this do anything??
        pass
    def __dealloc__(self):
        free(self.memory)
