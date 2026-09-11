#!/bin/bash
# Run from the Mac; SSH credentials stay on this machine.
set -euo pipefail
project_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "$project_dir/.venv-pruning/bin/python" ]]; then
    python_bin="$project_dir/.venv-pruning/bin/python"
else
    python_bin="$(command -v python3)"
fi
exec "$python_bin" "$project_dir/scripts/vast_rebuild.py" "$@"
