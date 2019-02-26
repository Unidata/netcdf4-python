#!/bin/bash

set -e

echo "Using downloaded netCDF version ${NETCDF_VERSION} with parallel capabilities enabled"
pushd /tmp
if [ ${NETCDF_VERSION} == "GITMASTER" ]; then
   git clone http://github.com/Unidata/netcdf-c netcdf-c
   pushd netcdf-c
   autoreconf -i
else
   wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-c-${NETCDF_VERSION}.tar.gz
   tar -xzvf netcdf-c-${NETCDF_VERSION}.tar.gz
   pushd netcdf-c-${NETCDF_VERSION}
fi
# for Ubuntu xenial
export CPPFLAGS="-I/usr/include/hdf5/openmpi"
export LIBS="-lhdf5_openmpihl -lhdf5_openmpi -lm -lz"
./configure --prefix $NETCDF_DIR --enable-netcdf-4 --enable-shared --disable-dap --enable-parallel
make -j 2
make install
popd
