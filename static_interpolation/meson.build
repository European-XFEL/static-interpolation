fs = import('fs')
py_mod = import('python')

py = py_mod.find_installation(required: true, pure: false)
py_dep = py.dependency()


python_sources = [
  'coordinate_mappers.py',	
  'utils.py',
  'data_structures.py',
  'config.py',
  'planning.py',
  'engines.py',
  'interpolators.py',
  '__init__.py'
]

py.install_sources(
  python_sources,
  pure: false,
  subdir: 'extra_pieces/interpolation'
)