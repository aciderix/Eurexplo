#!/usr/bin/env bash
# Push the PMU v3 orchestrator notebook to Kaggle as a private kernel.
#
# Prerequisites on your laptop:
#   pip install kaggle>=1.8
#   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
#   chmod 600 ~/.kaggle/kaggle.json          # (or use KAGGLE_API_TOKEN env var)
#
# One-off setup:
#   cp kaggle_runner/kernel-metadata.json  /tmp/pmu-kernel/
#   cp kaggle_runner/notebook.ipynb        /tmp/pmu-kernel/
#   # edit /tmp/pmu-kernel/kernel-metadata.json → replace the "id" with your slug
#   bash kaggle_runner/push_kernel.sh /tmp/pmu-kernel
#
# Subsequent pushes: same command re-uses the folder and bumps the kernel
# version on Kaggle.

set -euo pipefail

DIR="${1:-.}"
cd "$DIR"

if [[ ! -f kernel-metadata.json ]]; then
    echo "ERR: no kernel-metadata.json in $DIR" >&2
    exit 1
fi
if [[ ! -f notebook.ipynb ]]; then
    echo "ERR: no notebook.ipynb in $DIR" >&2
    exit 1
fi

if grep -q "REPLACE_WITH_YOUR_USERNAME" kernel-metadata.json; then
    echo "ERR: edit kernel-metadata.json and replace REPLACE_WITH_YOUR_USERNAME" >&2
    exit 1
fi

echo "Pushing kernel from $(pwd) …"
kaggle kernels push -p .
echo "Done. Kernel status:"
SLUG=$(python3 -c "import json; print(json.load(open('kernel-metadata.json'))['id'])")
kaggle kernels status "$SLUG"
