# We need to install the pure Python sources, but we also need them in the build dir for the tests
apitofsim_python_py_src = [
'api.py',
'cli.py',
'config.py',
'generate.py',
'marimo.py',
'plotting.py',
'transmission.py',
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
'ingest/common.py',
'ingest/csv.py',
], subdir: 'apitofsim/ingest', pure : false)
python.install_sources([
'workflow/__init__.py',
'workflow/base.py',
'workflow/db.py',
'workflow/db_utils.py',
'workflow/ingest.py',
'workflow/runners.py',
'workflow/sql_files.py',
], subdir: 'apitofsim/workflow', pure : false)

python.install_sources([
'workflow/sql/experiment.sql',
'workflow/sql/experiment_report.sql',
'workflow/sql/histograms.sql',
'workflow/sql/pathway.sql',
'workflow/sql/pathway_report.sql',
'workflow/sql/realizations.sql',
'workflow/sql/event_report.sql',
], subdir: 'apitofsim/workflow/sql', pure : false)
