#!/usr/bin/env python3
"""Render a filmstrip of the world coalescing into a trade network.

    python scripts/render_atlas.py --out docs/atlas.png

Produces a single image with four frames (early, forming, connected, mature)
that documents the emergence story used in the README.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw  # noqa: E402

from agora.model import AgoraModel  # noqa: E402
from agora.render import render_frame  # noqa: E402
from agora.schemas import ExperimentSpec  # noqa: E402

CAPTIONS = {
    30: "t=30   villages and their first links",
    120: "t=120  corridors thicken",
    260: "t=260  the market spans the island",
    460: "t=460  mature network, paved routes",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--radius", type=float, default=17.8)
    parser.add_argument("--out", default=str(ROOT / "docs" / "atlas.png"))
    args = parser.parse_args(argv)

    spec = ExperimentSpec(economy={"discovery_radius": args.radius}, horizon=460)
    model = AgoraModel(spec)
    frames: list[tuple[int, Image.Image]] = []
    for tick in sorted(CAPTIONS):
        while model.tick_count < tick:
            model.step()
        frames.append((tick, render_frame(model, scale=5, hud=False)))

    width, height = frames[0][1].size
    pad = 10
    sheet = Image.new("RGB", (width * 2 + pad * 3, height * 2 + pad * 3 + 26 * 2), (6, 11, 16))
    draw = ImageDraw.Draw(sheet)
    for index, (tick, image) in enumerate(frames):
        col, row = index % 2, index // 2
        x = pad + col * (width + pad)
        y = pad + row * (height + pad + 26)
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + height + 6), CAPTIONS[tick], fill=(150, 190, 200))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"wrote {out} ({sheet.size[0]}x{sheet.size[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
