import os


def make_path(*paths):
  path = os.path.join(*[str(path) for path in paths])
  path = os.path.realpath(path)
  return path
