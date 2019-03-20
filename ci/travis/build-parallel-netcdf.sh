#!/bin/bash

set -e

pushd /tmp
if [ -n "${PNETCDF_VERSION}" ]; then
	echo "Using downloaded PnetCDF version ${PNETCDF_VERSION}"
	wget https://parallel-netcdf.github.io/Release/pnetcdf-${PNETCDF_VERSION}.tar.gz
	tar -xzf pnetcdf-${PNETCDF_VERSION}.tar.gz
	pushd pnetcdf-${PNETCDF_VERSION}
	./configure --prefix $NETCDF_DIR --enable-shared --disable-fortran --disable-cxx
	NETCDF_EXTRA_CONFIG="--enable-pnetcdf"
	make -j 2
	make install
	popd
fi
echo "Using downloaded netCDF version ${NETCDF_VERSION} with parallel capabilities enabled"
if [ ${NETCDF_VERSION} == "GITMASTER" ]; then
   git clone http://github.com/Unidata/netcdf-c netcdf-c
   pushd netcdf-c
   autoreconf -i
else
   wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-c-${NETCDF_VERSION}.tar.gz
   tar -xzf netcdf-c-${NETCDF_VERSION}.tar.gz
   pushd netcdf-c-${NETCDF_VERSION}
fi
# for Ubuntu xenial
export CPPFLAGS="-I/usr/include/hdf5/mpich -I${NETCDF_DIR}/include"
export LDFLAGS="-L${NETCDF_DIR}/lib"
export LIBS="-lhdf5_mpich_hl -lhdf5_mpich -lm -lz"
./configure --prefix $NETCDF_DIR --enable-netcdf-4 --enable-shared --disable-dap --enable-parallel4 $NETCDF_EXTRA_CONFIG
make -j 2
make install
popd
