# We need to install the pure Python sources, but we also need them in the build dir for the tests
apitofsim_python_py_src = [
'api.py',
'cli.py',
'db.py',
'config.py',
'__init__.py',
'__main__.py',
'py.typed',
]
apitofsim_python_py_builddir = []
foreach f : apitofsim_python_py_src
  apitofsim_python_py_builddir += fs.copyfile(f)
endforeach
# This file is not installed -- it will be replaced with the .so
apitofsim_python_py_builddir += fs.copyfile('apitofsimraw.py')
python.install_sources(apitofsim_python_py_src, subdir: 'apitofsim', pure : false)
python.install_sources([
'ingest/__init__.py',
'ingest/gaussian.py',
'ingest/orca.py',
'ingest/legacy.py',
], subdir: 'apitofsim/ingest', pure : false)
