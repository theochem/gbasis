
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
    """Add MinGW bin to PATH before test collection on Windows."""
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
