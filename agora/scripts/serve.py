#!/usr/bin/env python3
"""Launch the interactive console.

python scripts/serve.py --host 0.0.0.0 --port 8010
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ASEOWS console.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)
    print(f"ASEOWS console -> http://{args.host}:{args.port}")
    uvicorn.run(
        "web.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
        app_dir=str(ROOT),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
