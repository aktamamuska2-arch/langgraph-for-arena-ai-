#!/usr/bin/env bash
# SIGMA bootstrap: minimal, size-capped dependency install into a venv OUTSIDE the repo.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV="${SIGMA_VENV:-/home/user/sigma-venv}"
echo "[bootstrap] repo=$ROOT venv=$VENV"
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --disable-pip-version-check -U pip wheel
# Core: LangGraph straight from the vendored monorepo (no PyPI copy), then a thin web/API layer.
"$VENV/bin/pip" install -q --disable-pip-version-check \
  -e "$ROOT/libs/checkpoint" \
  -e "$ROOT/libs/checkpoint-sqlite" \
  -e "$ROOT/libs/prebuilt" \
  -e "$ROOT/libs/langgraph" \
  -e "$ROOT/apps/sigma"
# Web API + ASGI server + test runner. Deliberately NO streamlit/pandas/numpy/LLM SDKs (size budget).
"$VENV/bin/pip" install -q --disable-pip-version-check "fastapi==0.115.12" "uvicorn==0.34.3" "pytest==8.4.1" "httpx==0.28.1"
echo "[bootstrap] done"
"$VENV/bin/python" -c "import langgraph, fastapi, sys; print('langgraph', langgraph.__file__); print('py', sys.version.split()[0])"
