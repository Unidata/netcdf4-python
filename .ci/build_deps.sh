#!/usr/bin/bash

set -ex


download_and_build_netcdf() {
  if [ ! -d "netcdf-c" ]; then
      netcdf_url=https://github.com/Unidata/netcdf-c
      netcdf_src=netcdf-c
      netcdf_build=netcdf-build

      git clone ${netcdf_url} -b v4.9.2 ${netcdf_src}

      cmake ${netcdf_src} -B ${netcdf_build} \
            -DENABLE_NETCDF4=on \
            -DENABLE_HDF5=on \
            -DENABLE_DAP=on \
            -DENABLE_TESTS=off \
            -DENABLE_PLUGIN_INSTALL=off \
            -DBUILD_SHARED_LIBS=on \
            -DCMAKE_BUILD_TYPE=Release

      cmake --build ${netcdf_build} \
            --target install
fi
}

download_and_build_netcdf
