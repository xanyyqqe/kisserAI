#!/usr/bin/env python3
import importlib
import sys
import subprocess
import os
import re

class KissAIDependencyChecker:

    IMPORT_NAMES = {
        'scikit-learn': 'sklearn',
        'scikit-image': 'skimage',
        'scikit-optimize': 'skopt',
        'scikit-learn-extra': 'sklearn_extra',

        'pytorch': 'torch',
        'pytorch-lightning': 'pytorch_lightning',
        'torch-geometric': 'torch_geometric',
        'torch-scatter': 'torch_scatter',
        'torch-sparse': 'torch_sparse',
        'torch-cluster': 'torch_cluster',
        'torch-spline-conv': 'torch_spline_conv',

        'sentence-transformers': 'sentence_transformers',
        'huggingface-hub': 'huggingface_hub',

        'opencv-python': 'cv2',
        'opencv-contrib-python': 'cv2',
        'pillow': 'PIL',

        'mlflow-skinny': 'mlflow',

        'ray-tune': 'ray.tune',

        'pyyaml': 'yaml',
        'python-dotenv': 'dotenv',
        'clickhouse-driver': 'clickhouse_driver',

        'pytest-cov': 'pytest_cov',
    }

    def __init__(self, requirements_file: str = "requirements.txt"):
        self.requirements_file = requirements_file
        self.requirements = []
        self.installed = {}
        self.missing = []
        self.version_mismatches = []
        self.import_success = []
        self.import_fail = []

    def load_requirements(self):
        if not os.path.exists(self.requirements_file):
            print(f"[KissAI] File not found: {self.requirements_file}")
            return False
        if os.path.isdir(self.requirements_file):
            print(f"[KissAI] Path is a directory, expected a requirements file: {self.requirements_file}")
            return False

        reqs = []
        with open(self.requirements_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    match = re.match(r'^([a-zA-Z0-9_.-]+)([=<>~!]+)([a-zA-Z0-9_.-]+)$', line)
                    if match:
                        reqs.append((match.group(1).lower(), match.group(3), match.group(2)))
                    else:
                        reqs.append((line.lower(), None, None))

        self.requirements = reqs
        print(f"[KissAI] {len(self.requirements)} dependencies are found")
        return True

    def get_installed(self):
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'list', '--format=freeze'],
            capture_output=True, text=True, check=True
        )

        installed = {}
        for line in result.stdout.strip().split('\n'):
            if '==' in line:
                pkg, ver = line.split('==')
                installed[pkg.lower()] = ver

        self.installed = installed
        print(f"[KissAI] {len(self.installed)} dependencies are installed")
        return True

    def get_import_name(self, package_name: str) -> str:
        if package_name.lower() in self.IMPORT_NAMES:
            return self.IMPORT_NAMES[package_name.lower()]
        return package_name.replace('-', '_')

    def _check_version(self, installed: str, required: str, operator: str) -> bool:

        if operator == '>=':
            return installed >= required
        elif operator == '<=':
            return installed <= required
        elif operator == '==':
            return installed == required
        elif operator == '>':
            return installed > required
        elif operator == '<':
            return installed < required
        elif operator == '~=':
            return installed.split('.')[0] == required.split('.')[0]
        else:
            return installed == required

    def check_requirements(self):
        for pkg, ver, op in self.requirements:
            if pkg not in self.installed:
                self.missing.append(pkg)
                print(f"[WARNING!] {pkg} is missing")
            elif ver:
                installed_ver = self.installed[pkg]
                if not self._check_version(installed_ver, ver, op):
                    self.version_mismatches.append(f"{pkg}: {op}{ver} is required, {installed_ver} is installed")
                    print(f"[WARNING!] {pkg}: requires {op}{ver}, but {installed_ver} is installed")
                else:
                    print(f"[KissAI] {pkg} - OK")
            else:
                print(f"[KissAI] {pkg} - OK")

    def test_imports(self):
        print(f"\n")
        for pkg, _, _ in self.requirements:
            import_name = self.get_import_name(pkg)

            try:
                module = importlib.import_module(import_name)
                version = getattr(module, '__version__', 'unknown')
                print(f"{pkg} - {version}")
                self.import_success.append(pkg)
            except ImportError:
                print(f"{pkg} - не найден")
                self.import_fail.append(pkg)
            except Exception as e:
                print(f"{pkg} - ошибка импорта: {e}")
                self.import_fail.append(pkg)

    def run_in_docker(self, dockerfile_path: str = "Dockerfile.compatibility"):

        try:
            print("[KissAI] Building Docker image...")
            build_result = subprocess.run(
                ["docker", "build", "-t", "kissai-dependency-checker", "-f", dockerfile_path, "."],
                capture_output=True,
                text=True,
                check=True
            )
            print(build_result.stdout)

            print("[KissAI] Running Docker container...")
            run_result = subprocess.run(
                ["docker", "run", "--rm", "kissai-dependency-checker"],
                capture_output=True,
                text=True,
                check=True
            )
            print(run_result.stdout)

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Docker command failed: {e.stderr}")
            return False

        return True

    def run(self):
        print(f"\n[KissAI] Dependencies compatibility check: {self.requirements_file}")
        print(f"Python: {sys.version}\n")

        if not self.load_requirements():
            print("[KissAI] No dependencies to check")

        self.get_installed()
        self.check_requirements()
        self.test_imports()

        print("\n" + "="*50)
        if not self.missing and not self.version_mismatches and not self.import_fail:
            print("[KissAI] All dependencies are installed correctly")

        else:
            if self.missing:
                print(f"Missing dependencies: {', '.join(self.missing)}")
            if self.version_mismatches:
                print(f"Versions mismatching: {', '.join(self.version_mismatches)}")
            if self.import_fail:
                print(f"Import errors: {', '.join(self.import_fail)}")

def main():
    req_file = sys.argv[1] if len(sys.argv) > 1 else "requirements.txt"
    checker = KissAIDependencyChecker(req_file)

    if "--docker" in sys.argv:
        checker.run_in_docker()
    else:
        checker.run()

if __name__ == "__main__":
    main()