#!/usr/bin/env bash
# SIGMA bootstrap: create a size-capped environment from a bare clone.
#
#   bash apps/sigma/scripts/bootstrap.sh
#
# The venv lives OUTSIDE the repository (default /home/user/sigma-venv) so that
# nothing generated is ever committed. The script ends by printing the artifact
# audit: it exits non-zero if the 125 MB budget is exceeded (PRD section 10).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV="${SIGMA_VENV:-/home/user/sigma-venv}"
APP="$ROOT/apps/sigma"

echo "[bootstrap] repo     = $ROOT"
echo "[bootstrap] venv     = $VENV"

[ -d "$VENV" ] || python3 -m venv "$VENV"

echo "[bootstrap] installing LangGraph from the vendored monorepo + web layer"
"$VENV/bin/pip" install -q --disable-pip-version-check -U pip wheel
"$VENV/bin/pip" install -q --disable-pip-version-check \
  -e "$ROOT/libs/checkpoint" \
  -e "$ROOT/libs/checkpoint-sqlite" \
  -e "$ROOT/libs/prebuilt" \
  -e "$ROOT/libs/langgraph"
"$VENV/bin/pip" install -q --disable-pip-version-check \
  "fastapi==0.115.12" "uvicorn==0.34.3" "httpx==0.28.1"
# SIGMA itself is installed last: its editable hook depends on the packages above.
"$VENV/bin/pip" install -q --disable-pip-version-check -e "$APP"

# Some pip/hatchling combinations silently skip the editable .pth; fail loudly
# instead of shipping an environment where `import sigma` does not work.
"$VENV/bin/python" -c "import sigma" 2>/dev/null || {
  echo "[bootstrap] editable install incomplete -> retrying"
  "$VENV/bin/pip" install -q --disable-pip-version-check --force-reinstall --no-deps -e "$APP"
}
"$VENV/bin/python" -c "import sigma, langgraph, fastapi; print('[bootstrap] imports OK')"

echo "[bootstrap] pruning build/test-only dependencies"
"$VENV/bin/python" "$APP/scripts/prune_venv.py" "$VENV"

echo "[bootstrap] verifying the runtime"
"$VENV/bin/python" -m unittest discover -s "$APP/tests" -t "$APP" -q

echo "[bootstrap] auditing artifact size"
"$VENV/bin/python" "$APP/scripts/audit_size.py"

echo "[bootstrap] done"
echo "[bootstrap] start the console with: $VENV/bin/sigma serve --host 0.0.0.0 --port 8000"
