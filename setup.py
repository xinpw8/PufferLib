# Debug command:
#    DEBUG=1 python setup.py build_ext --inplace --force
#    CUDA_VISIBLE_DEVICES=None LD_PRELOAD=$(gcc -print-file-name=libasan.so) python3.12 -m pufferlib.clean_pufferl eval --train.device cpu

from setuptools import find_packages, find_namespace_packages, setup, Extension
import numpy
import os
import glob
import urllib.request
import zipfile
import tarfile
import platform
import shutil

def _find_lammps_include_dir():
    # Allow explicit override
    include_dir = os.getenv('LAMMPS_INCLUDE_DIR')
    if include_dir and os.path.exists(os.path.join(include_dir, 'lammps', 'library.h')):
        return include_dir

    lammps_dir = os.getenv('LAMMPS_DIR')
    if lammps_dir:
        candidate = os.path.join(lammps_dir, 'include')
        if os.path.exists(os.path.join(candidate, 'lammps', 'library.h')):
            return candidate

    for base in ('/usr/local/include', '/usr/include'):
        if os.path.exists(os.path.join(base, 'lammps', 'library.h')):
            return base

    return None

def _find_lammps_lib_dir():
    # Allow explicit override
    lib_dir = os.getenv('LAMMPS_LIB_DIR')
    if lib_dir and os.path.isdir(lib_dir):
        return lib_dir

    lammps_dir = os.getenv('LAMMPS_DIR')
    if lammps_dir:
        for candidate in (os.path.join(lammps_dir, 'lib'), os.path.join(lammps_dir, 'lib64')):
            if os.path.isdir(candidate):
                return candidate

    for base in ('/usr/local/lib', '/usr/lib', '/usr/lib64'):
        if os.path.isdir(base):
            return base
    return None

from setuptools.command.build_ext import build_ext
from torch.utils import cpp_extension
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
    ROCM_HOME
)

# build cuda extension if torch can find CUDA or HIP/ROCM in the system
# may require `uv pip install --no-build-isolation` or `python setup.py build_ext --inplace`
BUID_CUDA_EXT = bool(CUDA_HOME or ROCM_HOME)

# Build with DEBUG=1 to enable debug symbols
DEBUG = os.getenv("DEBUG", "0") == "1"
NO_OCEAN = os.getenv("NO_OCEAN", "0") == "1"
NO_TRAIN = os.getenv("NO_TRAIN", "0") == "1"

# Build raylib for your platform
RAYLIB_URL = 'https://github.com/raysan5/raylib/releases/download/5.5/'
RAYLIB_NAME = 'raylib-5.5_macos' if platform.system() == "Darwin" else 'raylib-5.5_linux_amd64'
RLIGHTS_URL = 'https://raw.githubusercontent.com/raysan5/raylib/refs/heads/master/examples/shaders/rlights.h'

def download_raylib(platform, ext):
    if not os.path.exists(platform):
        print(f'Downloading Raylib {platform}')
        urllib.request.urlretrieve(RAYLIB_URL + platform + ext, platform + ext)
        if ext == '.zip':
            with zipfile.ZipFile(platform + ext, 'r') as zip_ref:
                zip_ref.extractall()
        else:
            with tarfile.open(platform + ext, 'r') as tar_ref:
                tar_ref.extractall()

        os.remove(platform + ext)
        urllib.request.urlretrieve(RLIGHTS_URL, platform + '/include/rlights.h')

if not NO_OCEAN:
    download_raylib('raylib-5.5_webassembly', '.zip')
    download_raylib(RAYLIB_NAME, '.tar.gz')

BOX2D_URL = 'https://github.com/capnspacehook/box2d/releases/latest/download/'
BOX2D_NAME = 'box2d-macos-arm64' if platform.system() == "Darwin" else 'box2d-linux-amd64'

def download_box2d(platform):
    if not os.path.exists(platform):
        ext = ".tar.gz"

        print(f'Downloading Box2D {platform}')
        urllib.request.urlretrieve(BOX2D_URL + platform + ext, platform + ext)
        with tarfile.open(platform + ext, 'r') as tar_ref:
            tar_ref.extractall()

        os.remove(platform + ext)

if not NO_OCEAN:
    download_box2d('box2d-web')
    download_box2d(BOX2D_NAME)

# Shared compile args for all platforms
extra_compile_args = [
    '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
    '-DPLATFORM_DESKTOP',
    '-fpermissive',
]
extra_link_args = [
    '-fwrapv'
]
cxx_args = [
    '-fdiagnostics-color=always',
]
nvcc_args = []

if DEBUG:
    extra_compile_args += [
        '-O0',
        '-g',
        '-fsanitize=address,undefined,bounds,pointer-overflow,leak',
        '-fno-omit-frame-pointer',
    ]
    extra_link_args += [
        '-g',
        '-fsanitize=address,undefined,bounds,pointer-overflow,leak',
    ]
    cxx_args += [
        '-O0',
        '-g',
    ]
    nvcc_args += [
        '-O0',
        '-g',
    ]
else:
    extra_compile_args += [
        '-O2',
        '-flto',
    ]
    extra_link_args += [
        '-O2',
    ]
    cxx_args += [
        '-O3',
    ]
    nvcc_args += [
        '-O3',
    ]

system = platform.system()
if system == 'Linux':
    extra_compile_args += [
        '-Wno-alloc-size-larger-than',
        '-Wno-implicit-function-declaration',
        '-fmax-errors=3',
    ]
    extra_link_args += [
        '-Bsymbolic-functions',
    ]
elif system == 'Darwin':
    extra_compile_args += [
        '-Wno-error=int-conversion',
        '-Wno-error=incompatible-function-pointer-types',
        '-Wno-error=implicit-function-declaration',
    ]
    extra_link_args += [
        '-framework', 'Cocoa',
        '-framework', 'OpenGL',
        '-framework', 'IOKit',
    ]
else:
    raise ValueError(f'Unsupported system: {system}')

# Default Gym/Gymnasium/PettingZoo versions
# Gym:
# - 0.26 still has deprecation warnings and is the last version of the package
# - 0.25 adds a breaking API change to reset, step, and render_modes
# - 0.24 is broken
# - 0.22-0.23 triggers deprecation warnings by calling its own functions
# - 0.21 is the most stable version
# - <= 0.20 is missing dict methods for gym.spaces.Dict
# - 0.18-0.21 require setuptools<=65.5.0

# Extensions 
class BuildExt(build_ext):
    def run(self):
        # Propagate any build_ext options (e.g., --inplace, --force) to subcommands
        build_ext_opts = self.distribution.command_options.get('build_ext', {})
        if build_ext_opts:
            # Copy flags so build_torch and build_c respect inplace/force
            self.distribution.command_options['build_torch'] = build_ext_opts.copy()
            self.distribution.command_options['build_c'] = build_ext_opts.copy()

        # Run the torch and C builds (which will handle copying when inplace is set)
        self.run_command('build_torch')
        self.run_command('build_c')

class CBuildExt(build_ext):
    def run(self, *args, **kwargs):
        self.extensions = [e for e in self.extensions if e.name != "pufferlib._C"]
        super().run(*args, **kwargs)

class TorchBuildExt(cpp_extension.BuildExtension):
    def run(self):
        self.extensions = [e for e in self.extensions if e.name == "pufferlib._C"]
        super().run()

INCLUDE = [f'{BOX2D_NAME}/include', f'{BOX2D_NAME}/src', 'pufferlib/extensions']
RAYLIB_A = f'{RAYLIB_NAME}/lib/libraylib.a'
extension_kwargs = dict(
    include_dirs=INCLUDE,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    extra_objects=[RAYLIB_A],
)

# Find C extensions
c_extensions = []
if not NO_OCEAN:
    c_extension_paths = []
    for pattern in ('binding.cpp', 'binding.c'):
        c_extension_paths.extend(
            glob.glob(f'pufferlib/ocean/**/{pattern}', recursive=True)
        )
        c_extension_paths.extend(
            glob.glob(f'pufferlib/pufferlib/ocean/**/{pattern}', recursive=True)
        )

    # Deduplicate while preserving order
    seen_paths = set()
    c_extension_paths = [p for p in c_extension_paths if not (p in seen_paths or seen_paths.add(p))]
    
    # Filter out backup directories
    c_extension_paths = [p for p in c_extension_paths if 'backup' not in p]

    for path in c_extension_paths:
        ext_dir  = os.path.dirname(path)
        ext_name = os.path.splitext(path)[0].replace('/', '.')

        # Collect all C/C++ translation units in the same directory so
        # symbols defined outside binding.cpp (e.g., enable_stockfish_black)
        # are linked into the shared object.
        extra_sources = []
        for pattern in ('*.cpp', '*.c'):
            for p in glob.glob(os.path.join(ext_dir, pattern)):
                # The standalone chess.cpp contains a full GUI + main() and
                # re-defines every symbol from chess.h.  Including it in the
                # Python extension causes duplicate-symbol link errors.
                if 'chess.cpp' in p and '/chess/' in p.replace('\\', '/'):
                    continue  # skip – binding.cpp already includes chess.h
                # Skip chess_original.cpp as it also re-defines symbols from chess.h
                if 'chess_original.cpp' in p and '/chess/' in p.replace('\\', '/'):
                    continue  # skip – binding.cpp already includes chess.h
                # Skip game_replay_tool.cpp as it's a standalone executable
                if 'game_replay_tool.cpp' in p:
                    continue  # skip – standalone tool with main()
                if p.endswith('.c'):
                    try:
                        with open(p, 'r', encoding='utf-8', errors='ignore') as fh:
                            contents = fh.read()
                    except OSError:
                        contents = ''
                    if 'int main' in contents:
                        continue  # skip demo binaries with their own entry point
                if p == path:
                    continue  # never re-add binding.cpp itself
                extra_sources.append(p)

        is_c_extension = path.endswith('.c')
        language = 'c' if is_c_extension else 'c++'
        compile_args = list(extension_kwargs.get('extra_compile_args', []))
        if is_c_extension:
            compile_args = compile_args + ['-std=gnu99']
        else:
            compile_args = compile_args + ['-std=c++17', '-x', 'c++']

        include_dirs = list(extension_kwargs.get('include_dirs', []))
        link_args = list(extension_kwargs.get('extra_link_args', []))
        extra_objects = list(extension_kwargs.get('extra_objects', []))

        c_ext = Extension(
            ext_name,
            sources=[path] + extra_sources,
            language=language,
            extra_compile_args=compile_args,
            include_dirs=include_dirs,
            extra_link_args=link_args,
            extra_objects=extra_objects,
        )

        c_extensions.append(c_ext)

    # Remember extension directories so they install as namespace packages
    c_extension_paths = [os.path.dirname(p) for p in c_extension_paths]

    # Configure optional extensions and skip those with missing system deps.
    build_matsci = os.getenv('BUILD_MATSCI', '0') == '1'
    filtered_extensions = []
    for c_ext in c_extensions:
        if "impulse_wars" in c_ext.name:
            print(f"Adding {c_ext.name} to extra objects")
            c_ext.extra_objects.append(f'{BOX2D_NAME}/libbox2d.a')
            # TODO: Figure out why this is necessary for some users
            impulse_include = 'pufferlib/ocean/impulse_wars/include'
            if impulse_include not in c_ext.include_dirs:
                c_ext.include_dirs.append(impulse_include)

        if 'matsci' in c_ext.name:
            lammps_include = _find_lammps_include_dir()
            lammps_lib = _find_lammps_lib_dir()

            if lammps_include is None and not build_matsci:
                print('Skipping matsci extension: LAMMPS headers not found. '
                      'Set BUILD_MATSCI=1 and LAMMPS_INCLUDE_DIR=/path/to/include to build.')
                continue

            if lammps_include is not None:
                c_ext.include_dirs.append(lammps_include)
            else:
                # BUILD_MATSCI=1 but include dir not found; let compiler error be explicit.
                c_ext.include_dirs.append('/usr/local/include')

            if lammps_lib is not None:
                c_ext.extra_link_args.extend([f'-L{lammps_lib}', '-llammps'])
            else:
                # Best-effort default
                c_ext.extra_link_args.extend(['-L/usr/local/lib', '-llammps'])

        filtered_extensions.append(c_ext)

    c_extensions = filtered_extensions

# Define cmdclass outside of setup to add dynamic commands
cmdclass = {
    "build_ext": BuildExt,
    "build_torch": TorchBuildExt,
    "build_c": CBuildExt,
}

if not NO_OCEAN:
    def create_env_build_class(full_name):
        class EnvBuildExt(build_ext):
            def run(self):
                self.extensions = [e for e in self.extensions if e.name == full_name]
                super().run()
        return EnvBuildExt

    # Add a build_<env> command for each env
    for c_ext in c_extensions:
        env_name = c_ext.name.split('.')[-2]
        cmdclass[f"build_{env_name}"] = create_env_build_class(c_ext.name)


# Check if CUDA compiler is available. You need cuda dev, not just runtime.
torch_extensions = []
if not NO_TRAIN:
    torch_sources = [
        "pufferlib/extensions/pufferlib.cpp",
    ]
    if BUID_CUDA_EXT:
        extension = CUDAExtension
        torch_sources.append("pufferlib/extensions/cuda/pufferlib.cu")
    else:
        extension = CppExtension

    torch_extensions = [
       extension(
            "pufferlib._C",
            torch_sources,
            extra_compile_args = {
                "cxx": cxx_args,
                "nvcc": nvcc_args,
            }
        ),
    ]

# Prevent Conda from injecting garbage compile flags
from distutils.sysconfig import get_config_vars
cfg_vars = get_config_vars()
for key in ('CC', 'CXX', 'LDSHARED'):
    if cfg_vars[key]:
        cfg_vars[key] = cfg_vars[key].replace('-B /root/anaconda3/compiler_compat', '')
        cfg_vars[key] = cfg_vars[key].replace('-pthread', '')
        cfg_vars[key] = cfg_vars[key].replace('-fno-strict-overflow', '')

for key, value in cfg_vars.items():
    if value and '-fno-strict-overflow' in str(value):
        cfg_vars[key] = value.replace('-fno-strict-overflow', '')

install_requires = [
    'setuptools',
    'numpy<2.0',
    'shimmy[gym-v21]',
    'gym==0.23',
    'gymnasium>=0.29.1',
    'pettingzoo>=1.24.1',
]

if not NO_TRAIN:
    install_requires += [
        'torch',
        'psutil',
        'nvidia-ml-py',
        'rich',
        'rich_argparse',
        'imageio',
        'gpytorch',
        'scikit-learn',
        'heavyball>=2.2.0', # contains relevant fixes compared to 1.7.2 and 2.1.1
        'neptune',
        'wandb',
    ]

setup(
    version="3.0.0",
    packages=find_namespace_packages() + find_packages() + c_extension_paths + ['pufferlib/extensions'],
    package_data={
        "pufferlib": [RAYLIB_NAME + '/lib/libraylib.a']
    },
    include_package_data=True,
    install_requires=install_requires,
    ext_modules = c_extensions + torch_extensions,
    cmdclass=cmdclass,
    include_dirs=[numpy.get_include(), RAYLIB_NAME + '/include'],
    entry_points={
        'console_scripts': [
            'puffer = pufferlib.pufferl:main',
        ],
    },
)
#stable_baselines3
#supersuit==3.3.5
#'git+https://github.com/oxwhirl/smac.git',

#curl -L -o smac.zip https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
#unzip -P iagreetotheeula smac.zip 
#curl -L -o maps.zip https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip
#unzip maps.zip && mv SMAC_Maps/ StarCraftII/Maps/