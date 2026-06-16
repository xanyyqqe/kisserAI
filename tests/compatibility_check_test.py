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


def test_compatibility_check(file:str = "requirements.txt"):

    from compatibility_check import KissAIDependencyChecker

    checker = KissAIDependencyChecker(file)
    checker.run()


def test_compatibility_check_no_file(file:str = "requirements.txt"):

    from compatibility_check import KissAIDependencyChecker

    checker = KissAIDependencyChecker()
    checker.run()


def test_compatibility_check_bad_file(file:str = "requirements.txt"):

    from compatibility_check import KissAIDependencyChecker

    checker = KissAIDependencyChecker("absent_file.txt")
    checker.run()


def test_compatibility_with_docker(file:str = "requirements.txt"):
    
    from compatibility_check import KissAIDependencyChecker

    checker = KissAIDependencyChecker("requirements.txt")
    checker.run_in_docker()