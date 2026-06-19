"""Make the ``fitcsg`` package importable when running pytest from the repo root
without installing it."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
