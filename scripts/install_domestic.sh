#!/usr/bin/env bash
# Install the pinned CUDA 12.6 build through domestic mirrors.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="$repo_dir/.venv/bin/python"
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 environment required"'

export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_EXTRA_INDEX_URL=""
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$repo_dir/data/pip-cache}"
mkdir -p "$repo_dir/tmp/setup"
install_record_dir="$(mktemp -d "$repo_dir/tmp/setup/domestic-install-XXXXXXXX")"

if ! "$python_bin" - <<'PY'
import importlib.metadata as metadata
try:
    assert metadata.version("torch") == "2.8.0+cu126"
    assert metadata.version("torchvision") == "0.23.0+cu126"
except (AssertionError, metadata.PackageNotFoundError):
    raise SystemExit(1)
PY
then
    "$python_bin" -m pip install \
        'https://mirror.nju.edu.cn/pytorch/whl/cu126/torch-2.8.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=ce6e6a1f4803ad62d1fe51cec3fe5ca14bcd8bc7cace7b09d5590f8147fa16ad' \
        'https://mirror.nju.edu.cn/pytorch/whl/cu126/torchvision-0.23.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=7351ffe7a49fe8ef391c5eef3fc4f6c1e6f15d0e25e83b41e9a07a23c201c57f' \
        --report "$install_record_dir/torch-install.json"
fi
"$python_bin" -m pip install -r requirements.txt \
    --report "$install_record_dir/project-install.json"
"$python_bin" -m pip check
"$python_bin" -m pip freeze --all > "$install_record_dir/packages.txt"
printf 'Installation records: %s\n' "$install_record_dir"
