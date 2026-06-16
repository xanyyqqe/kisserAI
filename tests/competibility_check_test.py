#!/usr/bin/env python3
import importlib
import sys
import subprocess
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCKER_DIR = PROJECT_ROOT / "docker"
if str(DOCKER_DIR) not in sys.path:
    sys.path.insert(0, str(DOCKER_DIR))

from compatibility_check import KissAIDependencyChecker

checker = KissAIDependencyChecker("requirements.txt")
checker.run() #works correctly with global requirements.txt file