# -*- coding: utf-8 -*-
from setuptools.build_meta import *

import utils


def get_requires_for_build_editable(config_settings=None):
    return ["mpi4py>=3.1,<4.1"] if utils.netcdf4_has_parallel_support() else []


def get_requires_for_build_sdist(config_settings=None):
    return ["mpi4py>=3.1,<4.1"] if utils.netcdf4_has_parallel_support() else []


def get_requires_for_build_wheel(config_settings=None):
    return ["mpi4py>=3.1,<4.1"] if utils.netcdf4_has_parallel_support() else []
