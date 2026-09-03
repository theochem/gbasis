"""
pytest configuration for gbasis test suite.

Adds MinGW64 bin directory to PATH on Windows CI so that MinGW runtime
DLLs (libgcc, libstdc++, libwinpthread) are found when importing the
libcint extension module.
"""
import os
import sys

# On Windows CI with MinGW, add MinGW bin to PATH so libgcc DLLs are found
if sys.platform == "win32":
    mingw_bin = r"D:\a\_temp\msys64\mingw64\bin"
    if os.path.isdir(mingw_bin) and mingw_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = mingw_bin + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(mingw_bin)

def pytest_configure(config):
    """Add MinGW64 bin to PATH before test collection on Windows CI.

    Checks both the GitHub Actions MSYS2 path and the local MSYS2 install
    path so the hook works in both CI and local Windows environments.
    """    
    import os, sys
    if sys.platform == "win32":
        for mingw_bin in [
            r"D:\a\_temp\msys64\mingw64\bin",
            r"C:\msys64\mingw64\bin",
        ]:
            if os.path.isdir(mingw_bin):
                os.environ["PATH"] = mingw_bin + os.pathsep + os.environ.get("PATH", "")
                if hasattr(os, 'add_dll_directory'):
                    os.add_dll_directory(mingw_bin)
                break
