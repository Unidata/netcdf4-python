import os, sys, subprocess, glob
import os.path as osp
import shutil
import configparser
from setuptools import setup, Extension
from setuptools.dist import Distribution

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
    sys.stdout.write('checking %s ...\n' % direc)
    hdf5_version = check_hdf5version(direc)
    if hdf5_version is None:
        sys.stdout.write('hdf5 headers not found in %s\n' % direc)
        return None
    else:
        sys.stdout.write('%s headers found in %s\n' %
                        (hdf5_version,direc))
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

def check_api(inc_dirs,netcdf_lib_version):
    has_rename_grp = False
    has_nc_inq_path = False
    has_nc_inq_format_extended = False
    has_cdf5_format = False
    has_nc_open_mem = False
    has_nc_create_mem = False
    has_parallel_support = False
    has_parallel4_support = False
    has_pnetcdf_support = False
    has_szip_support = False
    has_quantize = False
    has_zstandard = False
    has_bzip2 = False
    has_blosc = False
    has_ncfilter = False
    has_set_alignment = False
    has_nc_rc_set = False

    for d in inc_dirs:
        try:
            f = open(os.path.join(d, 'netcdf.h'), **open_kwargs)
        except OSError:
            continue

        has_nc_open_mem = os.path.exists(os.path.join(d, 'netcdf_mem.h'))
        has_nc_filter = os.path.exists(os.path.join(d, 'netcdf_filter.h'))

        for line in f:
            if line.startswith('nc_rename_grp'):
                has_rename_grp = True
            if line.startswith('nc_inq_path'):
                has_nc_inq_path = True
            if line.startswith('nc_inq_format_extended'):
                has_nc_inq_format_extended = True
            if line.startswith('#define NC_FORMAT_64BIT_DATA'):
                has_cdf5_format = True
            if line.startswith('nc_def_var_quantize'):
                has_quantize = True
            if line.startswith('nc_set_alignment'):
                has_set_alignment = True
            if line.startswith('EXTERNL int nc_rc_set'):
                has_nc_rc_set = True

        if has_nc_open_mem:
            try:
                f = open(os.path.join(d, 'netcdf_mem.h'), **open_kwargs)
            except OSError:
                continue
            for line in f:
                if line.startswith('EXTERNL int nc_create_mem'):
                    has_nc_create_mem = True

        if has_nc_filter:
            try:
                f = open(os.path.join(d, 'netcdf_filter.h'), **open_kwargs)
            except OSError:
                continue
            for line in f:
                if line.startswith('EXTERNL int nc_def_var_zstandard'):
                    has_zstandard = True
                if line.startswith('EXTERNL int nc_def_var_bzip2'):
                    has_bzip2 = True
                if line.startswith('EXTERNL int nc_def_var_blosc'):
                    has_blosc = True
                if line.startswith('EXTERNL int nc_inq_filter_avail'):
                    has_ncfilter = True

        ncmetapath = os.path.join(d,'netcdf_meta.h')
        if os.path.exists(ncmetapath):
            for line in open(ncmetapath):
                if line.startswith('#define NC_HAS_CDF5'):
                    try:
                        has_cdf5_format = bool(int(line.split()[2]))
                    except ValueError:
                        pass  # keep default False if value cannot be parsed
                if line.startswith('#define NC_HAS_PARALLEL'):
                    try:
                        has_parallel_support = bool(int(line.split()[2]))
                    except ValueError:
                        pass
                if line.startswith('#define NC_HAS_PARALLEL4'):
                    try:
                        has_parallel4_support = bool(int(line.split()[2]))
                    except ValueError:
                        pass
                if line.startswith('#define NC_HAS_PNETCDF'):
                    try:
                        has_pnetcdf_support = bool(int(line.split()[2]))
                    except ValueError:
                        pass
                if line.startswith('#define NC_HAS_SZIP_WRITE'):
                    try:
                        has_szip_support = bool(int(line.split()[2]))
                    except ValueError:
                        pass

        # NC_HAS_PARALLEL4 missing in 4.6.1 (issue #964)
        if not has_parallel4_support and has_parallel_support and not has_pnetcdf_support:
            has_parallel4_support = True
        # for 4.6.1, if NC_HAS_PARALLEL=NC_HAS_PNETCDF=1, guess that
        # parallel HDF5 is enabled (must guess since there is no
        # NC_HAS_PARALLEL4)
        elif netcdf_lib_version == "4.6.1" and not has_parallel4_support and has_parallel_support:
            has_parallel4_support = True
        break

    return has_rename_grp, has_nc_inq_path, has_nc_inq_format_extended, \
           has_cdf5_format, has_nc_open_mem, has_nc_create_mem, \
           has_parallel4_support, has_pnetcdf_support, has_szip_support, has_quantize, \
           has_zstandard, has_bzip2, has_blosc, has_set_alignment, has_ncfilter, \
           has_nc_rc_set


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
# USE_SETUPCFG evaluates to False.
ncconfig = None
use_ncconfig = None
if USE_SETUPCFG and os.path.exists(setup_cfg):
    sys.stdout.write('reading from setup.cfg...\n')
    config = configparser.ConfigParser()
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

try:
    if ncconfig is None:
        if netCDF4_dir is not None:
            ncconfig = os.path.join(netCDF4_dir, 'bin/nc-config')
        else:  # otherwise, just hope it's in the users PATH.
            ncconfig = 'nc-config'
    HAS_NCCONFIG = subprocess.call([ncconfig, '--libs'],
                                     stdout=subprocess.PIPE) == 0
except OSError:
    HAS_NCCONFIG = False

# make sure USE_NCCONFIG from environment takes
# precendence over use_ncconfig from setup.cfg (issue #341).
if USE_NCCONFIG is None and use_ncconfig is not None:
    USE_NCCONFIG = use_ncconfig
elif USE_NCCONFIG is None:
    # if nc-config exists, and USE_NCCONFIG not set, try to use it.
    if HAS_NCCONFIG: USE_NCCONFIG=True
#elif USE_NCCONFIG is None:
#    USE_NCCONFIG = False # don't try to use nc-config if USE_NCCONFIG not set

try:
    HAS_PKG_CONFIG = subprocess.call(['pkg-config', '--libs', 'hdf5'],
                                     stdout=subprocess.PIPE) == 0
except OSError:
    HAS_PKG_CONFIG = False

def _populate_hdf5_info(dirstosearch, inc_dirs, libs, lib_dirs):
    global HDF5_incdir, HDF5_dir, HDF5_libdir

    nohdf5dirs = HDF5_incdir is None and HDF5_libdir is None and  HDF5_dir is None
    if HAS_PKG_CONFIG and nohdf5dirs:
        # if HDF5 dirs not specified, and pkg-config available, use it
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
                hdf5_version = get_hdf5_version(os.path.join(direc, 'include'))
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
            hdf5_version = get_hdf5_version(HDF5_incdir)
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


dirstosearch = []
if os.environ.get("CONDA_PREFIX"):
    dirstosearch.append(os.environ["CONDA_PREFIX"]) # linux,macosx
    dirstosearch.append(os.path.join(os.environ["CONDA_PREFIX"],'Library')) # windows
dirstosearch += [os.path.expanduser('~'), '/usr/local', '/sw', '/opt',
                '/opt/local', '/opt/homebrew', '/usr']

# try nc-config first
if USE_NCCONFIG and HAS_NCCONFIG:  # Try nc-config.
    sys.stdout.write('using %s...\n' % ncconfig)
    dep = subprocess.Popen([ncconfig, '--libs'],
                           stdout=subprocess.PIPE).communicate()[0]
    libs = [str(l[2:].decode()) for l in dep.split() if l[0:2].decode() == '-l']
    lib_dirs = [str(l[2:].decode()) for l in dep.split() if
                l[0:2].decode() == '-L']
    dep = subprocess.Popen([ncconfig, '--cflags'],
                           stdout=subprocess.PIPE).communicate()[0]
    inc_dirs = [str(i[2:].decode()) for i in dep.split() if
                i[0:2].decode() == '-I']

    # check to see if hdf5 found in directories returned by nc-config
    hdf5_version = None
    for direc in inc_dirs:
        hdf5_version = get_hdf5_version(direc)
        if hdf5_version is not None:
            break
    # if hdf5 not found, search other standard locations (including those specified in env vars).
    if hdf5_version is None:
        sys.stdout.write('nc-config did provide path to HDF5 headers, search standard locations...')
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
DEFINE_MACROS = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
netcdf4_src_root = osp.join(osp.join('src','netCDF4'), '_netCDF4')
netcdf4_src_c = netcdf4_src_root + '.c'
netcdf4_src_pyx = netcdf4_src_root + '.pyx'
if 'sdist' not in sys.argv[1:] and 'clean' not in sys.argv[1:] and '--version' not in sys.argv[1:]:
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
    has_parallel4_support, has_pnetcdf_support, has_szip_support, has_quantize, \
    has_zstandard, has_bzip2, has_blosc, has_set_alignment, has_ncfilter, has_nc_rc_set = \
    check_api(inc_dirs,netcdf_lib_version)
    # for netcdf 4.4.x CDF5 format is always enabled.
    if netcdf_lib_version is not None and\
       (netcdf_lib_version > "4.4" and netcdf_lib_version < "4.5"):
        has_cdf5_format = True

    # disable parallel support if mpi4py not available.
    #try:
    #    import mpi4py
    #except ImportError:
    #    f.write('disabling mpi parallel support because mpi4py not found\n')
    #    has_parallel4_support = False
    #    has_pnetcdf_support = False

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

    if has_quantize:
        sys.stdout.write('netcdf lib has bit-grooming/quantization functions\n')
        f.write('DEF HAS_QUANTIZATION_SUPPORT = 1\n')
    else:
        sys.stdout.write('netcdf lib does not have bit-grooming/quantization functions\n')
        f.write('DEF HAS_QUANTIZATION_SUPPORT = 0\n')

    if has_zstandard:
        sys.stdout.write('netcdf lib has zstandard compression functions\n')
        f.write('DEF HAS_ZSTANDARD_SUPPORT = 1\n')
    else:
        sys.stdout.write('netcdf lib does not have zstandard compression functions\n')
        f.write('DEF HAS_ZSTANDARD_SUPPORT = 0\n')

    if has_bzip2:
        sys.stdout.write('netcdf lib has bzip2 compression functions\n')
        f.write('DEF HAS_BZIP2_SUPPORT = 1\n')
    else:
        sys.stdout.write('netcdf lib does not have bzip2 compression functions\n')
        f.write('DEF HAS_BZIP2_SUPPORT = 0\n')

    if has_blosc:
        sys.stdout.write('netcdf lib has blosc compression functions\n')
        f.write('DEF HAS_BLOSC_SUPPORT = 1\n')
    else:
        sys.stdout.write('netcdf lib does not have blosc compression functions\n')
        f.write('DEF HAS_BLOSC_SUPPORT = 0\n')

    if has_szip_support:
        sys.stdout.write('netcdf lib has szip compression functions\n')
        f.write('DEF HAS_SZIP_SUPPORT = 1\n')
    else:
        sys.stdout.write('netcdf lib does not have szip compression functions\n')
        f.write('DEF HAS_SZIP_SUPPORT = 0\n')

    if has_set_alignment:
        sys.stdout.write('netcdf lib has nc_set_alignment function\n')
        f.write('DEF HAS_SET_ALIGNMENT = 1\n')
    else:
        sys.stdout.write('netcdf lib does not have nc_set_alignment function\n')
        f.write('DEF HAS_SET_ALIGNMENT = 0\n')

    if has_ncfilter:
        sys.stdout.write('netcdf lib has nc_inq_filter_avail function\n')
        f.write('DEF HAS_NCFILTER = 1\n')
    else:
        sys.stdout.write('netcdf lib does not have nc_inq_filter_avail function\n')
        f.write('DEF HAS_NCFILTER = 0\n')

    if has_nc_rc_set:
        sys.stdout.write('netcdf lib has nc_rc_set function\n')
        f.write('DEF HAS_NCRCSET = 1\n')
    else:
        sys.stdout.write('netcdf lib does not have nc_rc_set function\n')
        f.write('DEF HAS_NCRCSET = 0\n')

    f.close()

    if has_parallel4_support or has_pnetcdf_support:
        import mpi4py
        inc_dirs.append(mpi4py.get_include())
        # mpi_incdir should not be needed if using nc-config
        # (should be included in nc-config --cflags)
        if mpi_incdir is not None: inc_dirs.append(mpi_incdir)

    ext_modules = [Extension("netCDF4._netCDF4",
                             [netcdf4_src_pyx],
                             define_macros=DEFINE_MACROS,
                             libraries=libs,
                             library_dirs=lib_dirs,
                             include_dirs=inc_dirs + ['include'],
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
        sys.stdout.write('no plugin files in NETCDF_PLUGIN_DIR, not installing..\n')
        data_files = []
    else:
        data_files = plugins
        sys.stdout.write('installing netcdf compression plugins from %s ...\n' % plugin_dir)
        sofiles = [os.path.basename(sofilepath) for sofilepath in data_files]
        sys.stdout.write(repr(sofiles)+'\n')
        if 'sdist' not in sys.argv[1:] and 'clean' not in sys.argv[1:] and '--version' not in sys.argv[1:]:
            for f in data_files:
                shutil.copy(f, osp.join(os.getcwd(),osp.join(osp.join('src','netCDF4'),'plugins')))
            copied_plugins=True
else:
    sys.stdout.write('NETCDF_PLUGIN_DIR not set, no netcdf compression plugins installed\n')
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
