import os
import runpy
import sys

if os.name == "nt" and (dll_directory := os.environ.get("APITOFSIM_DLL_DIRECTORY")):
    dll_directory_handle = os.add_dll_directory(dll_directory)

stubgen = sys.argv.pop(1)
runpy.run_path(stubgen, run_name="__main__")
