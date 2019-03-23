import os, sys, subprocess
import os.path as osp
from setuptools import setup, Extension
from distutils.dist import Distribution

setuptools_extra_kwargs = {
    "install_requires": ["numpy>=1.7","cftime"],
    "setup_requires": ['setuptools>=18.0', "cython>=0.19"],
    "entry_points": {
        'console_scripts': [
            'ncinfo = netCDF4.utils:ncinfo',
            'nc4tonc3 = netCDF4.utils:nc4tonc3',
            'nc3tonc4 = netCDF4.utils:nc3tonc4',
        ]
    },
}

if sys.version_info[0] < 3:
    import ConfigParser as configparser

    open_kwargs = {}
else:
    import configparser

    open_kwargs = {'encoding': 'utf-8'}


def check_hdf5version(hdf5_includedir):
    try:
        f = open(os.path.join(hdf5_includedir, 'H5public.h'), **open_kwargs)
    except IOError:
        return None
    hdf5_version = None
    for line in f:
        if line.startswith('#define H5_VERS_INFO'):
            hdf5_version = line.split('"')[1]
    return hdf5_version


def check_ifnetcdf4(netcdf4_includedir):
    try:
        f = open(os.path.join(netcdf4_includedir, 'netcdf.h'), **open_kwargs)
    except IOError:
        return False
    isnetcdf4 = False
    for line in f:
        if line.startswith('nc_inq_compound'):
            isnetcdf4 = True
    return isnetcdf4


def check_api(inc_dirs):
    has_rename_grp = False
    has_nc_inq_path = False
    has_nc_inq_format_extended = False
    has_cdf5_format = False
    has_nc_open_mem = False
    has_nc_create_mem = False
    has_parallel4_support = False
    has_pnetcdf_support = False

    for d in inc_dirs:
        try:
            f = open(os.path.join(d, 'netcdf.h'), **open_kwargs)
        except IOError:
            continue

        has_nc_open_mem = os.path.exists(os.path.join(d, 'netcdf_mem.h'))

        for line in f:
            if line.startswith('nc_rename_grp'):
                has_rename_grp = True
            if line.startswith('nc_inq_path'):
                has_nc_inq_path = True
            if line.startswith('nc_inq_format_extended'):
                has_nc_inq_format_extended = True
            if line.startswith('#define NC_FORMAT_64BIT_DATA'):
                has_cdf5_format = True

        if has_nc_open_mem:
            try:
                f = open(os.path.join(d, 'netcdf_mem.h'), **open_kwargs)
            except IOError:
                continue
            for line in f:
                if line.startswith('EXTERNL int nc_create_mem'):
                    has_nc_create_mem = True

        ncmetapath = os.path.join(d,'netcdf_meta.h')
        if os.path.exists(ncmetapath):
            for line in open(ncmetapath):
                if line.startswith('#define NC_HAS_CDF5'):
                    has_cdf5_format = bool(int(line.split()[2]))
                elif line.startswith('#define NC_HAS_PARALLEL4'):
                    has_parallel4_support = bool(int(line.split()[2]))
                elif line.startswith('#define NC_HAS_PNETCDF'):
                    has_pnetcdf_support = bool(int(line.split()[2]))
        break

    return has_rename_grp, has_nc_inq_path, has_nc_inq_format_extended, \
           has_cdf5_format, has_nc_open_mem, has_nc_create_mem, \
           has_parallel4_support, has_pnetcdf_support


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

USE_NCCONFIG = os.environ.get('USE_NCCONFIG')
if USE_NCCONFIG is not None:
    USE_NCCONFIG = bool(int(USE_NCCONFIG))
USE_SETUPCFG = os.environ.get('USE_SETUPCFG')
# override use of setup.cfg with env var.
if USE_SETUPCFG is not None:
    USE_SETUPCFG = bool(int(USE_SETUPCFG))
else:
    USE_SETUPCFG = True

setup_cfg = 'setup.cfg'
# contents of setup.cfg will override env vars, unless
# USE_SETUPCFG evaluates to True. Exception is use_ncconfig,
# which does not take precedence ofver USE_NCCONFIG env var.
ncconfig = None
use_ncconfig = None
if USE_SETUPCFG and os.path.exists(setup_cfg):
    sys.stdout.write('reading from setup.cfg...\n')
    config = configparser.SafeConfigParser()
    config.read(setup_cfg)
    try:
        HDF5_dir = config.get("directories", "HDF5_dir")
    except:
        pass
    try:
        HDF5_libdir = config.get("directories", "HDF5_libdir")
    except:
        pass
    try:
        HDF5_incdir = config.get("directories", "HDF5_incdir")
    except:
        pass
    try:
        netCDF4_dir = config.get("directories", "netCDF4_dir")
    except:
        pass
    try:
        netCDF4_libdir = config.get("directories", "netCDF4_libdir")
    except:
        pass
    try:
        netCDF4_incdir = config.get("directories", "netCDF4_incdir")
    except:
        pass
    try:
        szip_dir = config.get("directories", "szip_dir")
    except:
        pass
    try:
        szip_libdir = config.get("directories", "szip_libdir")
    except:
        pass
    try:
        szip_incdir = config.get("directories", "szip_incdir")
    except:
        pass
    try:
        hdf4_dir = config.get("directories", "hdf4_dir")
    except:
        pass
    try:
        hdf4_libdir = config.get("directories", "hdf4_libdir")
    except:
        pass
    try:
        hdf4_incdir = config.get("directories", "hdf4_incdir")
    except:
        pass
    try:
        jpeg_dir = config.get("directories", "jpeg_dir")
    except:
        pass
    try:
        jpeg_libdir = config.get("directories", "jpeg_libdir")
    except:
        pass
    try:
        jpeg_incdir = config.get("directories", "jpeg_incdir")
    except:
        pass
    try:
        curl_dir = config.get("directories", "curl_dir")
    except:
        pass
    try:
        curl_libdir = config.get("directories", "curl_libdir")
    except:
        pass
    try:
        curl_incdir = config.get("directories", "curl_incdir")
    except:
        pass
    try:
        mpi_incdir = config.get("directories","mpi_incdir")
    except:
        pass
    try:
        use_ncconfig = config.getboolean("options", "use_ncconfig")
    except:
        pass
    try:
        ncconfig = config.get("options", "ncconfig")
    except:
        pass

# make sure USE_NCCONFIG from environment takes
# precendence over use_ncconfig from setup.cfg (issue #341).
if USE_NCCONFIG is None and use_ncconfig is not None:
    USE_NCCONFIG = use_ncconfig
elif USE_NCCONFIG is None:
    USE_NCCONFIG = False

# if USE_NCCONFIG set, and nc-config works, use it.
if USE_NCCONFIG:
    # if NETCDF4_DIR env var is set, look for nc-config in NETCDF4_DIR/bin.
    if ncconfig is None:
        if netCDF4_dir is not None:
            ncconfig = os.path.join(netCDF4_dir, 'bin/nc-config')
        else:  # otherwise, just hope it's in the users PATH.
            ncconfig = 'nc-config'
    try:
        retcode = subprocess.call([ncconfig, '--libs'], stdout=subprocess.PIPE)
    except:
        retcode = 1
else:
    retcode = 1

try:
    HAS_PKG_CONFIG = subprocess.call(['pkg-config', '--libs', 'hdf5'],
                                     stdout=subprocess.PIPE) == 0
except OSError:
    HAS_PKG_CONFIG = False

def _populate_hdf5_info(dirstosearch, inc_dirs, libs, lib_dirs):
    global HDF5_incdir, HDF5_dir, HDF5_libdir

    if HAS_PKG_CONFIG:
        dep = subprocess.Popen(['pkg-config', '--cflags', 'hdf5'],
                               stdout=subprocess.PIPE).communicate()[0]
        inc_dirs.extend([str(i[2:].decode()) for i in dep.split() if
                         i[0:2].decode() == '-I'])
        dep = subprocess.Popen(['pkg-config', '--libs', 'hdf5'],
                               stdout=subprocess.PIPE).communicate()[0]
        libs.extend(
            [str(l[2:].decode()) for l in dep.split() if l[0:2].decode() == '-l'])
        lib_dirs.extend(
            [str(l[2:].decode()) for l in dep.split() if l[0:2].decode() == '-L'])
        dep = subprocess.Popen(['pkg-config', '--cflags', 'hdf5'],
                               stdout=subprocess.PIPE).communicate()[0]
        inc_dirs.extend(
            [str(i[2:].decode()) for i in dep.split() if i[0:2].decode() == '-I'])
    else:
        if HDF5_incdir is None and HDF5_dir is None:
            sys.stdout.write("""
    HDF5_DIR environment variable not set, checking some standard locations ..\n""")
            for direc in dirstosearch:
                sys.stdout.write('checking %s ...\n' % direc)
                hdf5_version = check_hdf5version(os.path.join(direc, 'include'))
                if hdf5_version is None:
                    continue
                else:
                    HDF5_dir = direc
                    HDF5_incdir = os.path.join(direc, 'include')
                    sys.stdout.write('%s found in %s\n' %
                                    (hdf5_version,HDF5_dir))
                    break
            if HDF5_dir is None:
                raise ValueError('did not find HDF5 headers')
        else:
            if HDF5_incdir is None:
                HDF5_incdir = os.path.join(HDF5_dir, 'include')
            hdf5_version = check_hdf5version(HDF5_incdir)
            if hdf5_version is None:
                raise ValueError('did not find HDF5 headers in %s' % HDF5_incdir)
            else:
                sys.stdout.write('%s found in %s\n' %
                                (hdf5_version,HDF5_dir))

        if HDF5_libdir is None and HDF5_dir is not None:
            HDF5_libdir = os.path.join(HDF5_dir, 'lib')

        if HDF5_libdir is not None: lib_dirs.append(HDF5_libdir)
        if HDF5_incdir is not None: inc_dirs.append(HDF5_incdir)

        libs.extend(['hdf5_hl', 'hdf5'])


dirstosearch = [os.path.expanduser('~'), '/usr/local', '/sw', '/opt',
                '/opt/local', '/usr']

if not retcode:  # Try nc-config.
    sys.stdout.write('using nc-config ...\n')
    dep = subprocess.Popen([ncconfig, '--libs'],
                           stdout=subprocess.PIPE).communicate()[0]
    libs = [str(l[2:].decode()) for l in dep.split() if l[0:2].decode() == '-l']
    lib_dirs = [str(l[2:].decode()) for l in dep.split() if
                l[0:2].decode() == '-L']
    dep = subprocess.Popen([ncconfig, '--cflags'],
                           stdout=subprocess.PIPE).communicate()[0]
    inc_dirs = [str(i[2:].decode()) for i in dep.split() if
                i[0:2].decode() == '-I']

    _populate_hdf5_info(dirstosearch, inc_dirs, libs, lib_dirs)
elif HAS_PKG_CONFIG:  # Try pkg-config.
    sys.stdout.write('using pkg-config ...\n')
    dep = subprocess.Popen(['pkg-config', '--libs', 'netcdf'],
                           stdout=subprocess.PIPE).communicate()[0]
    libs = [str(l[2:].decode()) for l in dep.split() if l[0:2].decode() == '-l']
    lib_dirs = [str(l[2:].decode()) for l in dep.split() if
                l[0:2].decode() == '-L']

    inc_dirs = []
    _populate_hdf5_info(dirstosearch, inc_dirs, libs, lib_dirs)
# If nc-config and pkg-config both didn't work (it won't on Windows), fall back on brute force method.
else:
    lib_dirs = []
    inc_dirs = []
    libs = []

    _populate_hdf5_info(dirstosearch, inc_dirs, libs, lib_dirs)

    if netCDF4_incdir is None and netCDF4_dir is None:
        sys.stdout.write("""
NETCDF4_DIR environment variable not set, checking standard locations.. \n""")
        for direc in dirstosearch:
            sys.stdout.write('checking %s ...\n' % direc)
            isnetcdf4 = check_ifnetcdf4(os.path.join(direc, 'include'))
            if not isnetcdf4:
                continue
            else:
                netCDF4_dir = direc
                netCDF4_incdir = os.path.join(direc, 'include')
                sys.stdout.write('netCDF4 found in %s\n' % netCDF4_dir)
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
    sys.stdout.write('unable to detect netcdf library version\n')
else:
    netcdf_lib_version = str(netcdf_lib_version)
    sys.stdout.write('using netcdf library version %s\n' % netcdf_lib_version)

cmdclass = {}
netcdf4_src_root = osp.join('netCDF4', '_netCDF4')
netcdf4_src_c = netcdf4_src_root + '.c'
if 'sdist' not in sys.argv[1:] and 'clean' not in sys.argv[1:]:
    sys.stdout.write('using Cython to compile netCDF4.pyx...\n')
    # remove _netCDF4.c file if it exists, so cython will recompile _netCDF4.pyx.
    # run for build *and* install (issue #263). Otherwise 'pip install' will
    # not regenerate _netCDF4.c, even if the C lib supports the new features.
    if len(sys.argv) >= 2:
        if os.path.exists(netcdf4_src_c):
            os.remove(netcdf4_src_c)
    # this determines whether renameGroup and filepath methods will work.
    has_rename_grp, has_nc_inq_path, has_nc_inq_format_extended, \
    has_cdf5_format, has_nc_open_mem, has_nc_create_mem, \
    has_parallel4_support, has_pnetcdf_support = check_api(inc_dirs)
    # for netcdf 4.4.x CDF5 format is always enabled.
    if netcdf_lib_version is not None and\
       (netcdf_lib_version > "4.4" and netcdf_lib_version < "4.5"):
        has_cdf5_format = True

    # disable parallel support if mpi4py not available.
    try:
        import mpi4py
    except ImportError:
        has_parallel4_support = False
        has_pnetcdf_support = False

    f = open(osp.join('include', 'constants.pyx'), 'w')
    if has_rename_grp:
        sys.stdout.write('netcdf lib has group rename capability\n')
        f.write('DEF HAS_RENAME_GRP = 1\n')
    else:
        sys.stdout.write('netcdf lib does not have group rename capability\n')
        f.write('DEF HAS_RENAME_GRP = 0\n')

    if has_nc_inq_path:
        sys.stdout.write('netcdf lib has nc_inq_path function\n')
        f.write('DEF HAS_NC_INQ_PATH = 1\n')
    else:
        sys.stdout.write('netcdf lib does not have nc_inq_path function\n')
        f.write('DEF HAS_NC_INQ_PATH = 0\n')

    if has_nc_inq_format_extended:
        sys.stdout.write('netcdf lib has nc_inq_format_extended function\n')
        f.write('DEF HAS_NC_INQ_FORMAT_EXTENDED = 1\n')
    else:
        sys.stdout.write(
            'netcdf lib does not have nc_inq_format_extended function\n')
        f.write('DEF HAS_NC_INQ_FORMAT_EXTENDED = 0\n')

    if has_nc_open_mem:
        sys.stdout.write('netcdf lib has nc_open_mem function\n')
        f.write('DEF HAS_NC_OPEN_MEM = 1\n')
    else:
        sys.stdout.write('netcdf lib does not have nc_open_mem function\n')
        f.write('DEF HAS_NC_OPEN_MEM = 0\n')

    if has_nc_create_mem:
        sys.stdout.write('netcdf lib has nc_create_mem function\n')
        f.write('DEF HAS_NC_CREATE_MEM = 1\n')
    else:
        sys.stdout.write('netcdf lib does not have nc_create_mem function\n')
        f.write('DEF HAS_NC_CREATE_MEM = 0\n')

    if has_cdf5_format:
        sys.stdout.write('netcdf lib has cdf-5 format capability\n')
        f.write('DEF HAS_CDF5_FORMAT = 1\n')
    else:
        sys.stdout.write('netcdf lib does not have cdf-5 format capability\n')
        f.write('DEF HAS_CDF5_FORMAT = 0\n')

    if has_parallel4_support:
        sys.stdout.write('netcdf lib has netcdf4 parallel functions\n')
        f.write('DEF HAS_PARALLEL4_SUPPORT = 1\n')
    else:
        sys.stdout.write('netcdf lib does not have netcdf4 parallel functions\n')
        f.write('DEF HAS_PARALLEL4_SUPPORT = 0\n')

    if has_pnetcdf_support:
        sys.stdout.write('netcdf lib has pnetcdf parallel functions\n')
        f.write('DEF HAS_PNETCDF_SUPPORT = 1\n')
    else:
        sys.stdout.write('netcdf lib does not have pnetcdf parallel functions\n')
        f.write('DEF HAS_PNETCDF_SUPPORT = 0\n')

    f.close()

    if has_parallel4_support or has_pnetcdf_support:
        inc_dirs.append(mpi4py.get_include())
        # mpi_incdir should not be needed if using nc-config
        # (should be included in nc-config --cflags)
        if mpi_incdir is not None: inc_dirs.append(mpi_incdir)

    ext_modules = [Extension("netCDF4._netCDF4",
                             [netcdf4_src_root + '.pyx'],
                             libraries=libs,
                             library_dirs=lib_dirs,
                             include_dirs=inc_dirs + ['include'],
                             runtime_library_dirs=runtime_lib_dirs)]
else:
    ext_modules = None

setup(name="netCDF4",
      cmdclass=cmdclass,
      version="1.5.0",
      long_description="netCDF version 4 has many features not found in earlier versions of the library, such as hierarchical groups, zlib compression, multiple unlimited dimensions, and new data types.  It is implemented on top of HDF5.  This module implements most of the new features, and can read and write netCDF files compatible with older versions of the library.  The API is modelled after Scientific.IO.NetCDF, and should be familiar to users of that module.\n\nThis project is hosted on a `GitHub repository <https://github.com/Unidata/netcdf4-python>`_ where you may access the most up-to-date source.",
      author="Jeff Whitaker",
      author_email="jeffrey.s.whitaker@noaa.gov",
      url="http://github.com/Unidata/netcdf4-python",
      download_url="http://python.org/pypi/netCDF4",
      platforms=["any"],
      license="OSI Approved",
      description="Provides an object-oriented python interface to the netCDF version 4 library.",
      keywords=['numpy', 'netcdf', 'data', 'science', 'network', 'oceanography',
                'meteorology', 'climate'],
      classifiers=["Development Status :: 3 - Alpha",
                   "Programming Language :: Python :: 2",
                   "Programming Language :: Python :: 2.6",
                   "Programming Language :: Python :: 2.7",
                   "Programming Language :: Python :: 3",
                   "Programming Language :: Python :: 3.3",
                   "Programming Language :: Python :: 3.4",
                   "Programming Language :: Python :: 3.5",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved",
                   "Topic :: Software Development :: Libraries :: Python Modules",
                   "Topic :: System :: Archiving :: Compression",
                   "Operating System :: OS Independent"],
      packages=['netCDF4'],
      ext_modules=ext_modules,
      **setuptools_extra_kwargs)
