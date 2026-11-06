# We need to install the pure Python sources, but we also need them in the build dir for the tests
apitofsim_python_py_src = [
'api.py',
'recording.py',
'event_conversion.py',
'events.py',
'collision.py',
'cli.py',
'config.py',
'generate.py',
'marimo.py',
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
'workflow/sql_files_nofk.py',
], subdir: 'apitofsim/workflow', pure : false)

python.install_sources([
'plotting/__init__.py',
'plotting/input.py',
'plotting/report.py',
'plotting/survivals.py',
'plotting/spectrogram.py',
'plotting/events.py',
], subdir: 'apitofsim/plotting', pure : false)

python.install_sources([
'workflow/sql/experiment.sql',
'workflow/sql/experiment_report.sql',
'workflow/sql/histograms.sql',
'workflow/sql/vib_energy_index.sql',
'workflow/sql/rf_basis.sql',
'workflow/sql/fragmentation_dos.sql',
'workflow/sql/pathway.sql',
'workflow/sql/pathway_report.sql',
'workflow/sql/event_recording.sql',
'workflow/sql/event_columns.sql',
'workflow/sql/event_views.sql',
], subdir: 'apitofsim/workflow/sql', pure : false)

install_data(
  'licenses/NOTICE.txt',
  'licenses/concurrentqueue.LICENSE.txt',
  'licenses/eigen.APACHE.LICENSE.txt',
  'licenses/eigen.BSD.LICENSE.txt',
  'licenses/eigen.MINPACK.LICENSE.txt',
  'licenses/eigen.MPL2.LICENSE.txt',
  'licenses/flatcc.LICENSE.txt',
  'licenses/lightweightsemaphore.LICENSE.txt',
  'licenses/magic_enum.LICENSE.txt',
  'licenses/nanoarrow.LICENSE.txt',
  'licenses/nanobind.LICENSE.txt',
  'licenses/robin-map.LICENSE.txt',
  install_dir: python.get_install_dir(pure: false) / 'apitofsim' / 'licenses',
)
