# setup.py

from setuptools import setup, Extension
# from Cython.Build import cythonize # Defer import until needed
import numpy as np
import os
import sys
import platform
import subprocess
import warnings
from pathlib import Path

# --- Configuration ---
VERSION = "0.3.0" # <<< ENSURE THIS MATCHES pyproject.toml / __init__.py
PROJECT_ROOT = Path(__file__).parent.resolve() # Get absolute path to project root

# --- Environment Variable Checks ---
FORCE_BUILD_GSL = os.environ.get("BUILD_GSL", "").lower() == "1"
DISABLE_BUILD_GSL = os.environ.get("BUILD_GSL", "").lower() == "0"
FORCE_BUILD_MKL = os.environ.get("BUILD_MKL", "").lower() == "1"
DISABLE_BUILD_MKL = os.environ.get("BUILD_MKL", "").lower() == "0"

print(f"--- Build Configuration (Version: {VERSION}) ---")
print(f"Project Root: {PROJECT_ROOT}")
if FORCE_BUILD_GSL: print("Attempting to force GSL build (BUILD_GSL=1)")
if DISABLE_BUILD_GSL: print("Skipping GSL build (BUILD_GSL=0)")
if FORCE_BUILD_MKL: print("Attempting to force MKL build (BUILD_MKL=1)")
if DISABLE_BUILD_MKL: print("Skipping MKL build (BUILD_MKL=0)")

# --- Platform Detection ---
is_windows = platform.system() == 'Windows'
is_mac = platform.system() == 'Darwin'
is_linux = platform.system() == 'Linux'
is_apple_silicon = is_mac and platform.machine() == 'arm64'

# --- Build Flags ---
common_compile_args = ['-O3', '-Wall', '-std=c99']
if not is_windows: common_compile_args.extend(['-fPIC', '-Wno-unused-function', '-Wno-unreachable-code'])
common_link_args = []; openmp_compile_args = []; openmp_link_args = []
print(f"\n--- Determining platform-specific build flags for {platform.system()} {platform.machine()} ---")

if is_windows:
    warnings.warn("Windows build support is experimental."); openmp_compile_args = ['/openmp']
elif is_mac:
    cc_path = os.environ.get("CC", "clang"); compiler_supports_omp = False
    try:
        p = subprocess.run([cc_path, "-E", "-fopenmp", "-xc", "-"], input="int main() { return 0; }", capture_output=True, text=True, check=False, timeout=5)
        if p.returncode == 0 and "error:" not in p.stderr and "unsupported option" not in p.stderr and "invalid argument" not in p.stderr:
             compiler_supports_omp = True; print(f"Compiler '{cc_path}' appears to support -fopenmp.")
             openmp_compile_args = ['-fopenmp']; openmp_link_args = ['-fopenmp']
        else:
            print(f"Compiler '{cc_path}' may not support -fopenmp directly ({p.stderr[:150]}...). Checking Homebrew libomp.")
            brew_prefixes = ['/usr/local/opt/libomp', '/opt/homebrew/opt/libomp']
            omp_prefix_found = next((prefix for prefix in brew_prefixes if os.path.isdir(os.path.join(prefix, 'include')) and os.path.isdir(os.path.join(prefix, 'lib'))), None)
            if omp_prefix_found:
                print(f"Found Homebrew libomp: {omp_prefix_found}")
                openmp_compile_args = ['-Xpreprocessor', '-fopenmp', f'-I{omp_prefix_found}/include']
                openmp_link_args = [f'-L{omp_prefix_found}/lib', '-lomp', f'-Wl,-rpath,{omp_prefix_found}/lib']
            else: warnings.warn("OpenMP support not detected via -fopenmp or Homebrew libomp. Features may be disabled.", ImportWarning)
    except Exception as omp_e: warnings.warn(f"Error checking OpenMP support: {omp_e}. Features may be disabled.", ImportWarning)
else: # Linux
    openmp_compile_args = ['-fopenmp']; openmp_link_args = ['-fopenmp']

base_extra_compile_args = common_compile_args + openmp_compile_args
base_extra_link_args = common_link_args + openmp_link_args
print(f"Using base compile args: {base_extra_compile_args}"); print(f"Using base link args: {base_extra_link_args}")

# --- Extension List ---
ext_modules = []; gsl_configured_successfully = False; mkl_configured_successfully = False

# --- Helper Functions ---
def check_command_exists(cmd):
    try: subprocess.run([cmd, '--version'], check=True, capture_output=True, text=True, timeout=2); return True
    except Exception: return False
def check_path_exists(path_list): return next((p for p in path_list if p and os.path.exists(p)), None)
def find_library_path(lib_name, lib_dirs):
    """Checks for library files (like lib<lib_name>.so) in a list of directories."""
    lib_patterns = [f"lib{lib_name}.so", f"lib{lib_name}.dylib"]
    if is_windows: lib_patterns.extend([f"{lib_name}.lib", f"{lib_name}.dll"])
    for d in lib_dirs:
        if d and os.path.isdir(d):
            for pattern in lib_patterns:
                if os.path.exists(os.path.join(d, pattern)):
                    return d # Return the directory containing the library
    return None

# --- MKL Discovery Logic ---
def find_mkl():
    print("\n--- Attempting MKL config ---")
    mkl_include_dir = None; mkl_library_dir = None; method_used = None
    link_libs = ['mkl_rt']; compile_args = []; link_args = []
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        print(f"INFO: Searching for MKL in Conda environment: {conda_prefix}")
        potential_include = os.path.join(conda_prefix, 'include'); potential_lib = os.path.join(conda_prefix, 'lib')
        mkl_header_path = os.path.join(potential_include, 'mkl.h'); mkl_lib_dir_found = find_library_path("mkl_rt", [potential_lib])
        if os.path.exists(mkl_header_path) and mkl_lib_dir_found:
            print(f"INFO: Found MKL headers/library in Conda env: {potential_include}, {mkl_lib_dir_found}")
            mkl_include_dir, mkl_library_dir, method_used = potential_include, mkl_lib_dir_found, "CONDA_PREFIX"
            if not is_windows: link_args.append(f"-Wl,-rpath,{mkl_library_dir}")
        else: print("INFO: Conda environment detected, but MKL headers/libs not found within it.")
    if not mkl_include_dir:
        mkl_root = os.environ.get('MKLROOT')
        if mkl_root:
            print(f"INFO: Found MKLROOT environment variable: {mkl_root}")
            potential_include = os.path.join(mkl_root, 'include')
            potential_lib_paths = [os.path.join(mkl_root, d) for d in ['lib', 'lib64', os.path.join('lib', 'intel64')] if os.path.isdir(os.path.join(mkl_root, d))]
            mkl_header_path = os.path.join(potential_include, 'mkl.h'); mkl_lib_dir_found = find_library_path("mkl_rt", potential_lib_paths)
            if os.path.exists(mkl_header_path) and mkl_lib_dir_found:
                 mkl_include_dir, mkl_library_dir, method_used = potential_include, mkl_lib_dir_found, "MKLROOT"
                 print(f"INFO: Found MKL headers/library via MKLROOT: {mkl_include_dir}, {mkl_library_dir}")
                 if not is_windows: link_args.append(f"-Wl,-rpath,{mkl_library_dir}")
            elif not os.path.exists(mkl_header_path): print(f"WARNING: MKLROOT set, but mkl.h not found in {potential_include}")
            elif not mkl_lib_dir_found: print(f"WARNING: MKLROOT set, but mkl_rt library not found in expected paths: {potential_lib_paths}")
            else: print("WARNING: MKLROOT set, but headers or libraries not found in expected locations.")
            if not (mkl_include_dir and mkl_library_dir): mkl_include_dir, mkl_library_dir, link_args, method_used = None, None, [], None
        else: print("INFO: MKLROOT environment variable not set.")
    if mkl_include_dir and mkl_library_dir:
        macros = [('HAVE_MKL', '1')]; print(f"MKL Configured ({method_used}).")
        # --- Manually add MKL linker flags to extra_link_args ---
        l_flag_dir = f"-L{mkl_library_dir}"
        if l_flag_dir not in link_args:
            link_args.append(l_flag_dir)
        for lib in link_libs:
             l_flag_lib = f"-l{lib}"
             if l_flag_lib not in link_args:
                 link_args.append(l_flag_lib)
        # -------------------------------------------------------
        return {"include_dirs": [mkl_include_dir], "library_dirs": [mkl_library_dir], "libraries": link_libs, "define_macros": macros, "extra_compile_args": compile_args, "extra_link_args": link_args, "method": method_used}
    else: print("MKL configuration failed."); return None

# --- GSL Config (Revised with linker flags fix) ---
def find_gsl():
    print("\n--- Attempting GSL config ---")
    gsl_cfg = {'libs': [], 'lib_dirs': [], 'inc_dirs': [], 'macros': [], 'method': None, 'extra_compile_args': [], 'extra_link_args': []}
    pkg_config_path = os.environ.get("PKG_CONFIG_PATH", "")
    try:
        # --- Try pkg-config first ---
        if check_command_exists('pkg-config') and subprocess.run(['pkg-config', '--exists', 'gsl'], check=False, env={**os.environ, "PKG_CONFIG_PATH": pkg_config_path}).returncode == 0:
            print("Trying GSL via pkg-config.")
            cflags_p = subprocess.run(['pkg-config', '--cflags-only-I', 'gsl'], env={**os.environ, "PKG_CONFIG_PATH": pkg_config_path}, capture_output=True, text=True, check=False)
            libs_p = subprocess.run(['pkg-config', '--libs-only-L', 'gsl'], env={**os.environ, "PKG_CONFIG_PATH": pkg_config_path}, capture_output=True, text=True, check=False)
            if cflags_p.returncode == 0 and cflags_p.stdout.strip() and libs_p.returncode == 0 and libs_p.stdout.strip():
                print("Found GSL include and library paths via pkg-config.")
                gsl_cfg['method'] = "pkg-config"; cflags = subprocess.check_output(['pkg-config', '--cflags', 'gsl'], env={**os.environ, "PKG_CONFIG_PATH": pkg_config_path}).strip().decode('utf-8')
                libs = subprocess.check_output(['pkg-config', '--libs', 'gsl'], env={**os.environ, "PKG_CONFIG_PATH": pkg_config_path}).strip().decode('utf-8'); lib_dirs_str = libs_p.stdout.strip()
                gsl_cfg['inc_dirs'].extend([f[2:] for f in cflags.split() if f.startswith('-I')]); gsl_cfg['lib_dirs'].extend([f[2:] for f in lib_dirs_str.split() if f.startswith('-L')])
                gsl_cfg['libs'].extend([f[2:] for f in libs.split() if f.startswith('-l')]); gsl_cfg['extra_link_args'].extend([f for f in libs.split() if not (f.startswith('-L') or f.startswith('-l'))])
            else: print("pkg-config found 'gsl' but did not return both include and library paths. Trying other methods."); # Add details if needed
        # --- If pkg-config failed or didn't provide full paths, try GSL_HOME ---
        if gsl_cfg['method'] is None and 'GSL_HOME' in os.environ:
            gh = os.environ['GSL_HOME']; print(f"Trying GSL_HOME: {gh}"); potential_inc = os.path.join(gh, 'include')
            potential_lib_dirs = [os.path.join(gh, d) for d in ['lib', 'lib64'] if os.path.isdir(os.path.join(gh, d))]
            if os.path.isdir(potential_inc) and potential_lib_dirs:
                print(f" GSL_HOME: Found include: {potential_inc}, potential libs: {potential_lib_dirs}"); found_lib_dir = find_library_path("gsl", potential_lib_dirs)
                if found_lib_dir:
                    print(f" GSL_HOME: Confirmed 'libgsl' in: {found_lib_dir}"); hdr_path = os.path.join(potential_inc, 'gsl', 'gsl_math.h')
                    if os.path.exists(hdr_path): print(f" GSL_HOME: Confirmed header: {hdr_path}"); gsl_cfg['inc_dirs'].append(potential_inc); gsl_cfg['lib_dirs'].append(found_lib_dir); gsl_cfg['method'] = "GSL_HOME"
                    else: print(f"Warning: GSL_HOME setup issue: Header '{hdr_path}' not found.")
                else: print(f"Warning: GSL_HOME setup issue: 'libgsl' not found in {potential_lib_dirs}.")
            else: print(f"Warning: GSL_HOME ('{gh}') invalid or missing 'include'/'lib' dirs.")
        # --- If still not found, try standard paths ---
        if gsl_cfg['method'] is None:
            print("Trying standard paths for GSL..."); std_inc = ["/usr/local/include", "/opt/local/include", "/usr/include", os.path.expanduser("~/.local/include")]
            std_lib = ["/usr/local/lib64", "/usr/local/lib", "/opt/local/lib", "/usr/lib64", "/usr/lib", os.path.expanduser("~/.local/lib")]
            found_inc_dir = next((p for p in std_inc if os.path.exists(os.path.join(p, 'gsl', 'gsl_math.h'))), None); lib_gsl_dir = find_library_path("gsl", std_lib)
            if found_inc_dir and lib_gsl_dir:
                print(f"Found GSL in standard paths (Inc: {found_inc_dir}, Lib: {lib_gsl_dir})."); gsl_cfg['inc_dirs'].append(found_inc_dir); gsl_cfg['lib_dirs'].append(lib_gsl_dir); gsl_cfg['method'] = "Standard Paths"
                if not find_library_path("gslcblas", [lib_gsl_dir]): warnings.warn(f"Standard Paths: Found libgsl in {lib_gsl_dir}, but not libgslcblas.")
            else: print("GSL not found in standard paths."); # Add details if needed

        # --- Final Checks and Configuration ---
        if gsl_cfg['method'] is None: raise FileNotFoundError("GSL could not be configured.")
        hdr = os.path.join('gsl', 'gsl_math.h')
        if not any(os.path.exists(os.path.join(d, hdr)) for d in gsl_cfg['inc_dirs']): raise FileNotFoundError(f"GSL header '{hdr}' still not found in final paths: {gsl_cfg['inc_dirs']}.")
        if not find_library_path("gsl", gsl_cfg['lib_dirs']): warnings.warn(f"Core GSL lib 'libgsl' not found in final paths: {gsl_cfg['lib_dirs']}. Link may fail.")

        # Add NumPy and package includes
        gsl_cfg['inc_dirs'].append(np.get_include()); gsl_cfg['inc_dirs'].append(str(PROJECT_ROOT / "ridge_inference")); gsl_cfg['inc_dirs'].append(str(PROJECT_ROOT / "src"))
        gsl_cfg['inc_dirs'] = sorted(list(set(gsl_cfg['inc_dirs']))); gsl_cfg['lib_dirs'] = sorted(list(set(gsl_cfg['lib_dirs'])))

        # Set default libs if needed, check existence
        if not gsl_cfg['libs']: gsl_cfg['libs'] = ["gsl", "gslcblas"]
        found_gslcblas = find_library_path("gslcblas", gsl_cfg['lib_dirs'])
        if not found_gslcblas and "gslcblas" in gsl_cfg['libs']: print("INFO: libgslcblas not found, removing from default link list."); gsl_cfg['libs'].remove("gslcblas")
        if not is_windows: gsl_cfg['libs'].append("m")
        gsl_cfg['libs'] = sorted(list(set(gsl_cfg['libs']))) # Final list of libraries to link

        # --- *** Force linker flags into extra_link_args *** ---
        for lib_dir in gsl_cfg['lib_dirs']:
            l_flag = f"-L{lib_dir}"
            if l_flag not in gsl_cfg['extra_link_args']:
                 gsl_cfg['extra_link_args'].append(l_flag)
        for lib in gsl_cfg['libs']:
            l_flag = f"-l{lib}"
            if l_flag not in gsl_cfg['extra_link_args']:
                gsl_cfg['extra_link_args'].append(l_flag)
        # --- ************************************************ ---

        # Add rpath for non-Windows (keep this as well for runtime lookup)
        if not is_windows:
            for lib_dir in gsl_cfg['lib_dirs']:
                rpath_arg = f"-Wl,-rpath,{lib_dir}"
                if rpath_arg not in gsl_cfg['extra_link_args']:
                     gsl_cfg['extra_link_args'].append(rpath_arg)

        # Check for OpenBLAS (optional)
        if find_library_path("openblas", gsl_cfg['lib_dirs']): print("Found OpenBLAS alongside GSL. Enabling GSL BLAS thread macros."); gsl_cfg['macros'].append(('HAVE_OPENBLAS', '1'))
        elif os.environ.get("USE_OPENBLAS", "0") == "1": print("USE_OPENBLAS=1 set. Enabling GSL BLAS thread macros."); gsl_cfg['macros'].append(('HAVE_OPENBLAS', '1'))

        # Ensure extra_link_args are unique at the end
        gsl_cfg['extra_link_args'] = sorted(list(set(gsl_cfg['extra_link_args'])))

        print(f"GSL Configured ({gsl_cfg['method']}). Inc: {gsl_cfg['inc_dirs']}, Lib Dirs: {gsl_cfg['lib_dirs']}, Libs: {gsl_cfg['libs']}, Link Args: {gsl_cfg['extra_link_args']}, Macros: {gsl_cfg['macros']}")
        return gsl_cfg

    except Exception as e: print(f"GSL configuration failed: {e}"); return None


# --- Find Libraries ---
gsl_config = None
mkl_config = None

if not DISABLE_BUILD_GSL:
    gsl_config = find_gsl()
    if gsl_config: gsl_configured_successfully = True
    elif FORCE_BUILD_GSL: print("\nERROR: GSL config failed but BUILD_GSL=1 was set.", file=sys.stderr); sys.exit(1)
    else: warnings.warn("INFO: GSL extension config failed. Skipping GSL backend build.", ImportWarning)

if not DISABLE_BUILD_MKL:
    mkl_config = find_mkl()
    if mkl_config: mkl_configured_successfully = True
    elif FORCE_BUILD_MKL: print("\nERROR: MKL config failed but BUILD_MKL=1 was set.", file=sys.stderr); sys.exit(1)
    else: warnings.warn("INFO: MKL extension config failed. Skipping MKL backend build.", ImportWarning)

# --- Define Extensions ---
package_dir = PROJECT_ROOT / "ridge_inference"; src_dir = PROJECT_ROOT / "src"
common_include_dirs = sorted(list(set([np.get_include(), str(package_dir), str(src_dir), "ridge_inference", "src"])))
print(f"DEBUG: Common include directories set to: {common_include_dirs}")

# Use a function to create extensions to avoid errors if config is None
def create_extension(name, sources, config, common_includes, base_compile_args, base_link_args):
    if not config: return None
    try:
        final_include_dirs = sorted(list(set(config.get("include_dirs", []) + common_includes)))
        print(f"DEBUG: Final include_dirs for {name}: {final_include_dirs}")
        # Combine base args with config-specific args, ensuring uniqueness
        final_compile_args = list(dict.fromkeys(base_compile_args + config.get("extra_compile_args", [])))
        final_link_args = list(dict.fromkeys(base_link_args + config.get("extra_link_args", [])))

        return Extension(
            name, sources=sources,
            include_dirs=final_include_dirs,
            library_dirs=config.get("library_dirs", []), # Keep for potential system search paths
            libraries=config.get("libraries", []),       # Keep for potential system search paths
            define_macros=config.get("define_macros", []),
            extra_compile_args=final_compile_args, # Use combined args
            extra_link_args=final_link_args,       # Use combined args
            language="c",
        )
    except KeyError as e: # Should not happen with .get()
        print(f"ERROR: KeyError '{e}' while creating Extension '{name}'. Config was: {config}")
        return None
    except Exception as e:
        print(f"ERROR: Unexpected error '{e}' while creating Extension '{name}'.")
        return None

# Create GSL extension if configured
gsl_extension = create_extension(
    "ridge_inference.ridge_gsl",
    ["ridge_inference/ridge_gsl.pyx", "src/ridge_gsl.c"],
    gsl_config, common_include_dirs, base_extra_compile_args, base_extra_link_args
)
if gsl_extension: ext_modules.append(gsl_extension)

# Create MKL extension if configured
mkl_extension = create_extension(
    "ridge_inference.ridge_mkl",
    ["ridge_inference/ridge_mkl.pyx", "src/ridge_mkl.c"],
    mkl_config, common_include_dirs, base_extra_compile_args, base_extra_link_args
)
if mkl_extension: ext_modules.append(mkl_extension)


if not ext_modules and not (DISABLE_BUILD_GSL and DISABLE_BUILD_MKL):
    if not ((FORCE_BUILD_GSL and not gsl_config) or (FORCE_BUILD_MKL and not mkl_config)):
       warnings.warn("Could not find prerequisites (GSL or MKL) to build any backend.", RuntimeWarning)

# --- Cythonize Extensions ---
cythonized_modules = []
if ext_modules:
    print(f"\n--- Cythonizing {len(ext_modules)} configured extension(s) ---")
    try:
        from Cython.Build import cythonize
        cythonized_modules = cythonize(
            ext_modules, language_level=3,
            compiler_directives={'embedsignature': True, 'binding': True},
            force=True
        )
        print("Cythonization successful.")
        if not cythonized_modules and ext_modules:
             warnings.warn("Cythonization reported success but produced no modules.", RuntimeWarning)
             cythonized_modules = []
        elif len(cythonized_modules) != len(ext_modules):
             warnings.warn(f"Cythonization might have failed for some modules. Expected {len(ext_modules)}, got {len(cythonized_modules)}.", RuntimeWarning)
    except ImportError:
        print("\nERROR: Cython is required to build extensions. Please install Cython.", file=sys.stderr)
        cythonized_modules = []
        warnings.warn("Cython not found. C extensions will not be built.", RuntimeWarning)
    except Exception as e:
         print(f"\nERROR: Cythonization failed: {e}", file=sys.stderr)
         cythonized_modules = []
         warnings.warn(f"Cythonization failed ({type(e).__name__}). C extensions might not be built.", RuntimeWarning)
else:
    print("\nINFO: No C extensions configured. Skipping Cythonization.")

# --- Configure setup() ---
this_directory = Path(__file__).parent
try:
    with open(this_directory / "README.md", encoding="utf-8") as f: long_description = f.read()
except FileNotFoundError: long_description = "High-performance ridge regression. See package documentation."

install_requires=[
    'numpy>=1.20', 'scipy>=1.6', 'pandas>=1.3.0', 'psutil>=5.9.0', 'tqdm>=4.62.0',
]

setup(
    version=VERSION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=cythonized_modules,
    package_data={'ridge_inference': ['py.typed']},
    zip_safe=False,
    name="ridge-inference",
    packages=["ridge_inference"],
    include_package_data=True,
    install_requires=install_requires,
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'ridge-inference-main=ridge_inference.cli:run_main',
            'ridge-inference-secact=ridge_inference.cli:run_secact',
        ],
    },
)

# --- Final Status Message ---
print("\n--- Setup script finished ---")
built_ext_names = [ext.name for ext in cythonized_modules] if cythonized_modules else []
if not DISABLE_BUILD_GSL:
    if gsl_configured_successfully:
        if "ridge_inference.ridge_gsl" in built_ext_names: print("INFO: GSL extension configured and appears in built modules list.")
        else: print("WARNING: GSL configured but 'ridge_inference.ridge_gsl' not found in list of successfully built/cythonized modules. Check compilation errors.")
    else: print("INFO: GSL extension NOT configured/built (configuration failed).")
elif DISABLE_BUILD_GSL: print("INFO: GSL extension build explicitly disabled via BUILD_GSL=0.")
if not DISABLE_BUILD_MKL:
    if mkl_configured_successfully:
        if "ridge_inference.ridge_mkl" in built_ext_names: print("INFO: MKL extension configured and appears in built modules list.")
        else: print("WARNING: MKL configured but 'ridge_inference.ridge_mkl' not found in list of successfully built/cythonized modules. Check compilation errors.")
    else: print("INFO: MKL extension NOT configured/built (configuration failed).")
elif DISABLE_BUILD_MKL: print("INFO: MKL extension build explicitly disabled via BUILD_MKL=0.")
if not built_ext_names and (gsl_configured_successfully or mkl_configured_successfully): print("WARNING: One or more C extensions were configured, but the list of built modules is empty. Cythonization or Compilation likely failed.")
elif not built_ext_names and not (DISABLE_BUILD_GSL and DISABLE_BUILD_MKL): print("INFO: No C extensions were configured or built.")
if built_ext_names: print(f"Build process will now attempt compilation for: {built_ext_names}")
else: print("No C extensions were passed to the C compiler phase.")