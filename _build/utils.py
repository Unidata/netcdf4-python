# -*- coding: utf-8 -*-
"""
This module contains a streamlined version of some utilities defined in `setup.py`, to be at
disposal of in-tree build backends.
"""

import configparser
import os
import subprocess


PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OPEN_KWARGS = {"encoding": "utf-8"}


def get_config_flags(command: list[str], flag: str) -> list[str]:
    """Pull out specific flags from a config command (pkg-config or nc-config)"""
    flags = subprocess.run(command, capture_output=True, text=True)
    return [arg[2:] for arg in flags.stdout.split() if arg.startswith(flag)]


def is_netcdf4_include_dir(inc_dir: str) -> bool:
    try:
        f = open(os.path.join(inc_dir, "netcdf.h"), **OPEN_KWARGS)
    except OSError:
        return False

    for line in f:
        if line.startswith("nc_inq_compound"):
            return True
    return False


def get_netcdf4_include_dir():
    netcdf4_dir = os.environ.get("NETCDF4_DIR")
    netcdf4_incdir = os.environ.get("NETCDF4_INCDIR")

    if bool(int(os.environ.get("USE_SETUPCFG", 1))) and os.path.exists(
        setup_cfg := os.path.join(PROJECT_ROOT_DIR, "setup.cfg")
    ):
        config = configparser.ConfigParser()
        config.read(setup_cfg)

        netcdf4_dir = config.get("directories", "NETCDF4_DIR", fallback=netcdf4_dir)
        netcdf4_incdir = config.get(
            "directories", "NETCDF4_INCDIR", fallback=netcdf4_incdir
        )

        # make sure USE_NCCONFIG from environment takes precedence over use_ncconfig from setup.cfg
        #  (issue #341)
        if "USE_NCCONFIG" in os.environ:
            use_ncconfig = bool(int(os.environ.get("USE_NCCONFIG", 0)))
        else:
            use_ncconfig = config.getboolean("options", "use_ncconfig", fallback=None)

        ncconfig = config.get("options", "ncconfig", fallback=None)
    else:
        use_ncconfig = None
        ncconfig = None

    try:
        if ncconfig is None:
            if netcdf4_dir is not None:
                ncconfig = os.path.join(netcdf4_dir, "bin", "nc-config")
            else:  # otherwise, just hope it's in the users PATH
                ncconfig = "nc-config"
        has_ncconfig = subprocess.call([ncconfig, "--libs"]) == 0
    except OSError:
        has_ncconfig = False

    # if nc-config exists, and use_ncconfig not set, try to use it
    if use_ncconfig is None and has_ncconfig:
        use_ncconfig = has_ncconfig

    dirs_to_search = []
    if os.environ.get("CONDA_PREFIX"):
        dirs_to_search.append(os.environ["CONDA_PREFIX"])  # linux,macosx
        dirs_to_search.append(
            os.path.join(os.environ["CONDA_PREFIX"], "Library")
        )  # windows
    dirs_to_search += [
        os.path.expanduser("~"),
        "/usr/local",
        "/sw",
        "/opt",
        "/opt/local",
        "/opt/homebrew",
        "/usr",
    ]

    if netcdf4_incdir is None and netcdf4_dir is None:
        if use_ncconfig and has_ncconfig:
            inc_dirs = get_config_flags([ncconfig, "--cflags"], "-I")
        else:
            inc_dirs = [os.path.join(dir_, "include") for dir_ in dirs_to_search]

        for inc_dir in inc_dirs:
            if is_netcdf4_include_dir(inc_dir):
                netcdf4_incdir = inc_dir
                break

        if netcdf4_incdir is None:
            raise ValueError("Did not find netCDF version 4 headers.")
    else:
        if netcdf4_incdir is None:
            netcdf4_incdir = os.path.join(netcdf4_dir, "include")
        if not is_netcdf4_include_dir(netcdf4_incdir):
            raise ValueError(
                f"Did not find netCDF version 4 headers under `{netcdf4_incdir}`."
            )

    return netcdf4_incdir


def netcdf4_has_parallel_support() -> bool:
    netcdf4_incdir = get_netcdf4_include_dir()
    if os.path.exists(ncmetapath := os.path.join(netcdf4_incdir, "netcdf_meta.h")):
        with open(ncmetapath) as f:
            for line in f:
                if line.startswith("#define NC_HAS_PARALLEL"):
                    try:
                        return bool(int(line.split()[2]))
                    except ValueError:
                        pass
            return False
    else:
        return False
