import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from libc.stdlib cimport rand, RAND_MAX


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void maximum_path_each(int[:,:] path, float[:,:,:] pi, int t_x, int t_y) nogil:
  cdef int idx
  cdef int j = t_y -1
  cdef int i = t_x -1
  cdef float[:] prob 
  cdef float x


  path[i,j] = 1
  while i>=0 and j >=0:
    if i==0 and j ==0:
      break

    prob = pi[i,j]
    # Generate a random float 
    x = rand()/(RAND_MAX+1.0)
    if x < prob[0]:
      i-= 1
      path[i,j] = 1
    else:
      i-=1
      j-=1
      path[i,j] = 1


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void maximum_path_c(int[:,:,:] paths, float[:,:,:,:] pi, int[:] t_xs, int[:] t_ys) nogil:
  cdef int batch = pi.shape[0]
  cdef int i

  for i in prange(batch, nogil=True):
    maximum_path_each(paths[i], pi[i], t_xs[i], t_ys[i])
