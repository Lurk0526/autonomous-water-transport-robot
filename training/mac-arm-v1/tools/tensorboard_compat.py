"""Offline TensorBoard 2.19 launcher for setuptools without pkg_resources.

Only the entry-point and version interfaces used by this installed TensorBoard
are adapted. No installed library or training configuration is modified.
"""
import importlib.metadata
import sys
import types
from packaging.version import parse

try:
    import pkg_resources
except ModuleNotFoundError:
    compat = types.ModuleType('pkg_resources')
    compat.parse_version = parse
    compat.iter_entry_points = lambda group: (
        types.SimpleNamespace(resolve=entry.load)
        for entry in importlib.metadata.entry_points(group=group)
    )
    sys.modules['pkg_resources'] = compat

from tensorboard.main import run_main

if __name__ == '__main__':
    run_main()
