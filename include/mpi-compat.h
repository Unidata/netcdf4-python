/* Author:  Lisandro Dalcin   */
/* Contact: dalcinl@gmail.com */

#ifndef MPI_COMPAT_H
#define MPI_COMPAT_H

#include "netcdf-compat.h"

#if HAS_PARALLEL_SUPPORT

#include <mpi.h>

#ifdef MSMPI_VER
#define PyMPI_HAVE_MPI_Message 1
#endif

#if (MPI_VERSION < 3) && !defined(PyMPI_HAVE_MPI_Message)
typedef void *PyMPI_MPI_Message;
#define MPI_Message PyMPI_MPI_Message
#endif

#if (MPI_VERSION < 4) && !defined(PyMPI_HAVE_MPI_Session)
typedef void *PyMPI_MPI_Session;
#define MPI_Session PyMPI_MPI_Session
#endif

#endif /* HAS_PARALLEL_SUPPORT */

#endif/*MPI_COMPAT_H*/
