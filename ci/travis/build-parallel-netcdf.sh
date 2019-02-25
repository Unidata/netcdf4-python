#!/bin/bash

set -e

echo "Using downloaded netCDF version ${NETCDF_VERSION} with parallel capabilities enabled"
pushd /tmp
wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-c-${NETCDF_VERSION}.tar.gz
tar -xzvf netcdf-${NETCDF_VERSION}.tar.gz
pushd netcdf-${NETCDF_VERSION}
./configure --prefix $NETCDF_DIR --enable-netcdf-4 --enable-shared --disable-dap  --enable-parallel
make -j 2
make install
popd
