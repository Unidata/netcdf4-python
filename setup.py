import os, sys, subprocess, glob
import os.path as osp
import pathlib
import shutil
import configparser
from setuptools import setup, Extension
from setuptools.dist import Distribution
from typing import List

open_kwargs = {'encoding': 'utf-8'}


def check_hdf5version(hdf5_includedir):
    try:
        f = open(os.path.join(hdf5_includedir, 'H5public.h'), **open_kwargs)
    except OSError:
        return None
    hdf5_version = None
    for line in f:
        if line.startswith('#define H5_VERS_INFO'):
            hdf5_version = line.split('"')[1]
    return hdf5_version

def get_hdf5_version(direc):
    # check to see if hdf5 headers in direc, return version number or None
    hdf5_version = None
    print(f"checking {direc}...")
    hdf5_version = check_hdf5version(direc)
    if hdf5_version is None:
        print(f'hdf5 headers not found in {direc}')
        return None
    else:
        print(f'{hdf5_version} headers found in {direc}')
        return hdf5_version

def check_ifnetcdf4(netcdf4_includedir):
    try:
        f = open(os.path.join(netcdf4_includedir, 'netcdf.h'), **open_kwargs)
    except OSError:
        return False
    isnetcdf4 = False
    for line in f:
        if line.startswith('nc_inq_compound'):
            isnetcdf4 = True
    return isnetcdf4


def check_has_parallel_support(inc_dirs: list) -> bool:
    has_parallel_support = False

    for d in inc_dirs:
        ncmetapath = os.path.join(d,'netcdf_meta.h')
        if not os.path.exists(ncmetapath):
            continue

        with open(ncmetapath) as f:
            for line in f:
                if line.startswith('#define NC_HAS_PARALLEL'):
                    try:
                        has_parallel_support = bool(int(line.split()[2]))
                    except ValueError:
                        pass

    return has_parallel_support


def getnetcdfvers(libdirs):
    """
    Get the version string for the first netcdf lib found in libdirs.
    (major.minor.release). If nothing found, return None.
    """

    import os, re, sys, ctypes

    if sys.platform.startswith('win'):
        regexp = re.compile('^netcdf.dll$')
    elif sys.platform.startswith('cygwin'):
        bindirs = []
        for d in libdirs:
            bindirs.append(os.path.dirname(d) + '/bin')
        regexp = re.compile(r'^cygnetcdf-\d.dll')
    elif sys.platform.startswith('darwin'):
        regexp = re.compile(r'^libnetcdf.dylib')
    else:
        regexp = re.compile(r'^libnetcdf.so')

    if sys.platform.startswith('cygwin'):
        dirs = bindirs
    else:
        dirs = libdirs
    for d in dirs:
        try:
            candidates = [x for x in os.listdir(d) if regexp.match(x)]
            if len(candidates) != 0:
                candidates.sort(
                    key=lambda x: len(x))  # Prefer libfoo.so to libfoo.so.X.Y.Z
                path = os.path.abspath(os.path.join(d, candidates[0]))
            lib = ctypes.cdll.LoadLibrary(path)
            inq_libvers = lib.nc_inq_libvers
            inq_libvers.restype = ctypes.c_char_p
            vers = lib.nc_inq_libvers()
            return vers.split()[0]
        except Exception:
            pass  # We skip invalid entries, because that's what the C compiler does

    return None

def extract_version(CYTHON_FNAME):
    version = None
    with open(CYTHON_FNAME) as fi:
        for line in fi:
            if (line.startswith('__version__')):
                _, version = line.split('=')
                version = version.strip()[1:-1]  # Remove quotation characters.
                break
    return version


HDF5_dir = os.environ.get('HDF5_DIR')
HDF5_incdir = os.environ.get('HDF5_INCDIR')
HDF5_libdir = os.environ.get('HDF5_LIBDIR')
netCDF4_dir = os.environ.get('NETCDF4_DIR')
netCDF4_incdir = os.environ.get('NETCDF4_INCDIR')
netCDF4_libdir = os.environ.get('NETCDF4_LIBDIR')
szip_dir = os.environ.get('SZIP_DIR')
szip_libdir = os.environ.get('SZIP_LIBDIR')
szip_incdir = os.environ.get('SZIP_INCDIR')
hdf4_dir = os.environ.get('HDF4_DIR')
hdf4_libdir = os.environ.get('HDF4_LIBDIR')
hdf4_incdir = os.environ.get('HDF4_INCDIR')
jpeg_dir = os.environ.get('JPEG_DIR')
jpeg_libdir = os.environ.get('JPEG_LIBDIR')
jpeg_incdir = os.environ.get('JPEG_INCDIR')
curl_dir = os.environ.get('CURL_DIR')
curl_libdir = os.environ.get('CURL_LIBDIR')
curl_incdir = os.environ.get('CURL_INCDIR')
mpi_incdir = os.environ.get('MPI_INCDIR')

USE_NCCONFIG = bool(int(os.environ.get('USE_NCCONFIG', 0)))
# override use of setup.cfg with env var.
USE_SETUPCFG = bool(int(os.environ.get('USE_SETUPCFG', 1)))

setup_cfg = 'setup.cfg'
# contents of setup.cfg will override env vars, unless
# USE_SETUPCFG evaluates to False.
ncconfig = None
use_ncconfig = None
if USE_SETUPCFG and os.path.exists(setup_cfg):
    print('reading from setup.cfg...')
    config = configparser.ConfigParser()
    config.read(setup_cfg)
    HDF5_dir = config.get("directories", "HDF5_dir", fallback=HDF5_dir)
    HDF5_libdir = config.get("directories", "HDF5_libdir", fallback=HDF5_libdir)
    HDF5_incdir = config.get("directories", "HDF5_incdir", fallback=HDF5_incdir)
    netCDF4_dir = config.get("directories", "netCDF4_dir", fallback=netCDF4_dir)
    netCDF4_libdir = config.get("directories", "netCDF4_libdir", fallback=netCDF4_libdir)
    netCDF4_incdir = config.get("directories", "netCDF4_incdir", fallback=netCDF4_incdir)
    szip_dir = config.get("directories", "szip_dir", fallback=szip_dir)
    szip_libdir = config.get("directories", "szip_libdir", fallback=szip_libdir)
    szip_incdir = config.get("directories", "szip_incdir", fallback=szip_incdir)
    hdf4_dir = config.get("directories", "hdf4_dir", fallback=hdf4_dir)
    hdf4_libdir = config.get("directories", "hdf4_libdir", fallback=hdf4_libdir)
    hdf4_incdir = config.get("directories", "hdf4_incdir", fallback=hdf4_incdir)
    jpeg_dir = config.get("directories", "jpeg_dir", fallback=jpeg_dir)
    jpeg_libdir = config.get("directories", "jpeg_libdir", fallback=jpeg_libdir)
    jpeg_incdir = config.get("directories", "jpeg_incdir", fallback=jpeg_incdir)
    curl_dir = config.get("directories", "curl_dir", fallback=curl_dir)
    curl_libdir = config.get("directories", "curl_libdir", fallback=curl_libdir)
    curl_incdir = config.get("directories", "curl_incdir", fallback=curl_incdir)
    mpi_incdir = config.get("directories","mpi_incdir", fallback=mpi_incdir)
    use_ncconfig = config.getboolean("options", "use_ncconfig", fallback=use_ncconfig)
    ncconfig = config.get("options", "ncconfig", fallback=ncconfig)

try:
    if ncconfig is None:
        if netCDF4_dir is not None:
            ncconfig = os.path.join(netCDF4_dir, 'bin/nc-config')
        else:  # otherwise, just hope it's in the users PATH.
            ncconfig = 'nc-config'
    HAS_NCCONFIG = subprocess.call([ncconfig, '--libs']) == 0
except OSError:
    HAS_NCCONFIG = False

# make sure USE_NCCONFIG from environment takes
# precendence over use_ncconfig from setup.cfg (issue #341).
if use_ncconfig and not USE_NCCONFIG:
    USE_NCCONFIG = use_ncconfig
elif not USE_NCCONFIG:
    # if nc-config exists, and USE_NCCONFIG not set, try to use it.
    USE_NCCONFIG = HAS_NCCONFIG

try:
    HAS_PKG_CONFIG = subprocess.call(['pkg-config', '--libs', 'hdf5']) == 0
except OSError:
    HAS_PKG_CONFIG = False


def config_flags(command: List[str], flag: str) -> list:
    """Pull out specific flags from a config command (pkg-config or nc-config)"""
    flags = subprocess.run(command, capture_output=True, text=True)
    return [arg[2:] for arg in flags.stdout.split() if arg.startswith(flag)]


def _populate_hdf5_info(dirstosearch, inc_dirs, libs, lib_dirs):
    global HDF5_incdir, HDF5_dir, HDF5_libdir

    nohdf5dirs = HDF5_incdir is None and HDF5_libdir is None and  HDF5_dir is None
    if HAS_PKG_CONFIG and nohdf5dirs:
        # if HDF5 dirs not specified, and pkg-config available, use it
        inc_dirs.extend(config_flags(["pkg-config", "--cflags", "hdf5"], "-I"))
        libs.extend(config_flags(["pkg-config", "--libs", "hdf5"], "-l"))
        lib_dirs.extend(config_flags(["pkg-config", "--libs", "hdf5"], "-L"))
    else:
        if HDF5_incdir is None and HDF5_dir is None:
            print("    HDF5_DIR environment variable not set, checking some standard locations ..")
            for direc in dirstosearch:
                hdf5_version = get_hdf5_version(os.path.join(direc, 'include'))
                if hdf5_version is None:
                    continue
                else:
                    HDF5_dir = direc
                    HDF5_incdir = os.path.join(direc, 'include')
                    print(f'{hdf5_version} found in {HDF5_dir}')
                    break
            if HDF5_dir is None:
                raise ValueError('did not find HDF5 headers')
        else:
            if HDF5_incdir is None:
                HDF5_incdir = os.path.join(HDF5_dir, 'include')
            hdf5_version = get_hdf5_version(HDF5_incdir)
            if hdf5_version is None:
                raise ValueError(f'did not find HDF5 headers in {HDF5_incdir}')
            print(f'{hdf5_version} found in {HDF5_dir}')

        if HDF5_libdir is None and HDF5_dir is not None:
            HDF5_libdir = os.path.join(HDF5_dir, 'lib')

        if HDF5_libdir is not None: lib_dirs.append(HDF5_libdir)
        if HDF5_incdir is not None: inc_dirs.append(HDF5_incdir)

        libs.extend(['hdf5_hl', 'hdf5'])


dirstosearch = []
if os.environ.get("CONDA_PREFIX"):
    dirstosearch.append(os.environ["CONDA_PREFIX"]) # linux,macosx
    dirstosearch.append(os.path.join(os.environ["CONDA_PREFIX"],'Library')) # windows
dirstosearch += [os.path.expanduser('~'), '/usr/local', '/sw', '/opt',
                '/opt/local', '/opt/homebrew', '/usr']

# try nc-config first
if USE_NCCONFIG and HAS_NCCONFIG and ncconfig is not None:
    print(f'using {ncconfig}...')
    libs = config_flags([ncconfig, "--libs"], "-l")
    lib_dirs = config_flags([ncconfig, "--libs"], "-L")
    inc_dirs = config_flags([ncconfig, '--cflags'], "-I")

    # check to see if hdf5 found in directories returned by nc-config
    hdf5_version = None
    for direc in inc_dirs:
        hdf5_version = get_hdf5_version(direc)
        if hdf5_version is not None:
            break
    # if hdf5 not found, search other standard locations (including those specified in env vars).
    if hdf5_version is None:
        print('nc-config did provide path to HDF5 headers, search standard locations...')
        _populate_hdf5_info(dirstosearch, inc_dirs, libs, lib_dirs)

# If nc-config doesn't work, fall back on brute force method.
else:
    lib_dirs = []
    inc_dirs = []
    libs = []

    # _populate_hdf5_info will use HDF5_dir, HDF5_libdir and HDF5_incdir if they are set.
    # otherwise pkg-config will be tried, and if that fails, dirstosearch will be searched.
    _populate_hdf5_info(dirstosearch, inc_dirs, libs, lib_dirs)

    if netCDF4_incdir is None and netCDF4_dir is None:
        print("NETCDF4_DIR environment variable not set, checking standard locations..")
        for direc in dirstosearch:
            print(f'checking {direc}...')
            isnetcdf4 = check_ifnetcdf4(os.path.join(direc, 'include'))
            if not isnetcdf4:
                continue
            else:
                netCDF4_dir = direc
                netCDF4_incdir = os.path.join(direc, 'include')
                print(f'netCDF4 found in {netCDF4_dir}')
                break
        if netCDF4_dir is None:
            raise ValueError('did not find netCDF version 4 headers')
    else:
        if netCDF4_incdir is None:
            netCDF4_incdir = os.path.join(netCDF4_dir, 'include')
        isnetcdf4 = check_ifnetcdf4(netCDF4_incdir)
        if not isnetcdf4:
            raise ValueError(
                'did not find netCDF version 4 headers %s' % netCDF4_incdir)

    if netCDF4_libdir is None and netCDF4_dir is not None:
        netCDF4_libdir = os.path.join(netCDF4_dir, 'lib')

    if sys.platform == 'win32':
        libs.extend(['netcdf', 'zlib'])
    else:
        libs.extend(['netcdf', 'z'])

    if netCDF4_libdir is not None: lib_dirs.append(netCDF4_libdir)
    if netCDF4_incdir is not None: inc_dirs.append(netCDF4_incdir)

    # add szip to link if desired.
    if szip_libdir is None and szip_dir is not None:
        szip_libdir = os.path.join(szip_dir, 'lib')
    if szip_incdir is None and szip_dir is not None:
        szip_incdir = os.path.join(szip_dir, 'include')
    if szip_incdir is not None and szip_libdir is not None:
        if sys.platform == 'win32':
            libs.append('szip')
        else:
            libs.append('sz')

        lib_dirs.append(szip_libdir)
        inc_dirs.append(szip_incdir)
    # add hdf4 to link if desired.
    if hdf4_libdir is None and hdf4_dir is not None:
        hdf4_libdir = os.path.join(hdf4_dir, 'lib')
    if hdf4_incdir is None and hdf4_dir is not None:
        hdf4_incdir = os.path.join(hdf4_dir, 'include')
    if hdf4_incdir is not None and hdf4_libdir is not None:
        libs.append('mfhdf')
        libs.append('df')
        lib_dirs.append(hdf4_libdir)
        inc_dirs.append(hdf4_incdir)
    # add jpeg to link if desired.
    if jpeg_libdir is None and jpeg_dir is not None:
        jpeg_libdir = os.path.join(jpeg_dir, 'lib')
    if jpeg_incdir is None and jpeg_dir is not None:
        jpeg_incdir = os.path.join(jpeg_dir, 'include')
    if jpeg_incdir is not None and jpeg_libdir is not None:
        libs.append('jpeg')
        lib_dirs.append(jpeg_libdir)
        inc_dirs.append(jpeg_incdir)
    # add curl to link if desired.
    if curl_libdir is None and curl_dir is not None:
        curl_libdir = os.path.join(curl_dir, 'lib')
    if curl_incdir is None and curl_dir is not None:
        curl_incdir = os.path.join(curl_dir, 'include')
    if curl_incdir is not None and curl_libdir is not None:
        libs.append('curl')
        lib_dirs.append(curl_libdir)
        inc_dirs.append(curl_incdir)

if sys.platform == 'win32':
    runtime_lib_dirs = []
else:
    runtime_lib_dirs = lib_dirs

# Do not require numpy for just querying the package
# Taken from the h5py setup file.
if any('--' + opt in sys.argv for opt in Distribution.display_option_names +
        ['help-commands', 'help']) or sys.argv[1] == 'egg_info':
    pass
else:
    # append numpy include dir.
    import numpy
    inc_dirs.append(numpy.get_include())

# get netcdf library version.
netcdf_lib_version = getnetcdfvers(lib_dirs)
if netcdf_lib_version is None:
    print('unable to detect netcdf library version')
else:
    netcdf_lib_version = str(netcdf_lib_version)
    print(f'using netcdf library version {netcdf_lib_version}')

DEFINE_MACROS = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
netcdf4_src_root = osp.join(osp.join('src','netCDF4'), '_netCDF4')
netcdf4_src_c = netcdf4_src_root + '.c'
netcdf4_src_pyx = netcdf4_src_root + '.pyx'
if 'sdist' not in sys.argv[1:] and 'clean' not in sys.argv[1:] and '--version' not in sys.argv[1:]:
    print('using Cython to compile netCDF4.pyx...')
    # remove _netCDF4.c file if it exists, so cython will recompile _netCDF4.pyx.
    # run for build *and* install (issue #263). Otherwise 'pip install' will
    # not regenerate _netCDF4.c, even if the C lib supports the new features.
    if len(sys.argv) >= 2:
        if os.path.exists(netcdf4_src_c):
            os.remove(netcdf4_src_c)

    # for netcdf 4.4.x CDF5 format is always enabled.
    if netcdf_lib_version is not None and\
       (netcdf_lib_version > "4.4" and netcdf_lib_version < "4.5"):
        has_cdf5_format = True

    has_parallel_support = check_has_parallel_support(inc_dirs)
    has_has_not = "has" if has_parallel_support else "does not have"
    print(f"netcdf lib {has_has_not} parallel functions")

    if has_parallel_support:
        import mpi4py
        inc_dirs.append(mpi4py.get_include())
        # mpi_incdir should not be needed if using nc-config
        # (should be included in nc-config --cflags)
        if mpi_incdir is not None:
            inc_dirs.append(mpi_incdir)

        # Name of file containing imports required for parallel support
        parallel_support_imports = "parallel_support_imports.pxi.in"
    else:
        parallel_support_imports = "no_parallel_support_imports.pxi.in"

    # Copy the specific version of the file containing parallel
    # support imports
    shutil.copyfile(
        osp.join("include", parallel_support_imports),
        osp.join("include", "parallel_support_imports.pxi")
    )

    nc_complex_dir = pathlib.Path("./external/nc_complex")
    source_files = [
        netcdf4_src_pyx,
        str(nc_complex_dir / "src/nc_complex.c"),
    ]
    include_dirs = inc_dirs + [
        "include",
        str(nc_complex_dir / "include"),
        str(nc_complex_dir / "include/generated_fallbacks"),
    ]
    DEFINE_MACROS += [("NC_COMPLEX_NO_EXPORT", "1")]

    ext_modules = [Extension("netCDF4._netCDF4",
                             source_files,
                             define_macros=DEFINE_MACROS,
                             libraries=libs,
                             library_dirs=lib_dirs,
                             include_dirs=include_dirs,
                             runtime_library_dirs=runtime_lib_dirs)]
    # set language_level directive to 3
    for e in ext_modules:
        e.cython_directives = {'language_level': "3"} #
else:
    ext_modules = None

# if NETCDF_PLUGIN_DIR set, install netcdf-c compression plugins inside package
# (should point to location of lib__nc* files built by netcdf-c)
copied_plugins=False
if os.environ.get("NETCDF_PLUGIN_DIR"):
    plugin_dir = os.environ.get("NETCDF_PLUGIN_DIR")
    plugins = glob.glob(os.path.join(plugin_dir, "lib__nc*"))
    if not plugins:
        print('no plugin files in NETCDF_PLUGIN_DIR, not installing...')
        data_files = []
    else:
        data_files = plugins
        print(f'installing netcdf compression plugins from {plugin_dir} ...')
        sofiles = [os.path.basename(sofilepath) for sofilepath in data_files]
        print(repr(sofiles))
        if 'sdist' not in sys.argv[1:] and 'clean' not in sys.argv[1:] and '--version' not in sys.argv[1:]:
            for f in data_files:
                shutil.copy(f, osp.join(os.getcwd(),osp.join(osp.join('src','netCDF4'),'plugins')))
            copied_plugins=True
else:
    print('NETCDF_PLUGIN_DIR not set, no netcdf compression plugins installed')
    data_files = []

# See pyproject.toml for project metadata
setup(
    name="netCDF4",  # need by GitHub dependency graph
    version=extract_version(netcdf4_src_pyx),
    ext_modules=ext_modules,
)

# remove plugin files copied from outside source tree
if copied_plugins:
    for f in sofiles:
        filepath = osp.join(osp.join(osp.join('src','netCDF4'),'plugins'),f)
        if os.path.exists(filepath):
            os.remove(filepath)
