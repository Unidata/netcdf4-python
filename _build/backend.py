# -*- coding: utf-8 -*-
import logging
import os
from setuptools.build_meta import *
import subprocess


fmt = logging.Formatter("\n=== NETCDF4 BUILD DEBUG: %(message)s\n")
hdlr = logging.StreamHandler()
hdlr.setFormatter(fmt)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(hdlr)


def netcdf_has_parallel_support():
    netcdf4_dir = os.environ.get("NETCDF4_DIR")
    ncconfig = (
        os.path.join(netcdf4_dir, "bin", "nc-config") if netcdf4_dir else "nc-config"
    )
    log.debug(f"{ncconfig = }")
    process = subprocess.run([ncconfig, "--has-parallel4"], capture_output=True)
    out = process.stdout.decode("utf-8").rstrip()
    return out == "yes"


def get_requires_for_build_editable(config_settings=None):
    return ["mpi4py>=3.1"] if netcdf_has_parallel_support() else []


def get_requires_for_build_wheel(config_settings=None):
    return ["mpi4py>=3.1"] if netcdf_has_parallel_support() else []


def get_requires_for_build_sdist(config_settings=None):
    return ["mpi4py>=3.1"] if netcdf_has_parallel_support() else []
