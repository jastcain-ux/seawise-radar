#!/usr/bin/env python3
"""HRRR total cloud cover, rendered as a translucent layer.

A scalar field like reflectivity, so unlike wind it ships as pictures and
reuses the radar renderer's Lambert-to-mercator sampling directly.

**Capped well short of opaque, deliberately.** A cloud layer was tried once
before on this project and rejected: as a base-map replacement it put a grey
sheet over everything and told a boater nothing. That objection is weaker for a
layer someone chooses to switch on, but it is not wrong — so cover maxes out
around half opacity, and the coastline, the place names and the radar
underneath stay readable through the thickest overcast.

Hourly only: TCDC is not in the sub-hourly files, and cloud does not change
meaningfully in fifteen minutes anyway.
"""
import argparse, datetime, json, os, sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import (NATIONAL_BBOX, NATIONAL_ID, NATIONAL_SIZE,  # noqa: E402
                   CLOUD_REGIONAL_SIZE, REGIONAL_KEY, is_regional, regional_cells)
import render                                        # noqa: E402
import wind                                          # noqa: E402

LEAD_MINUTES = list(range(0, 6 * 60 + 1, 60))
FIELD, LEVEL = "TCDC", "entire atmosphere"

# The ceiling. Full overcast is common and a layer that blanks the map on a
# common day is a layer people switch off once and never switch on again.
MAX_ALPHA = 0.55


# Alpha steps. A smooth gradient is what made the first version 570 KB a cell
# against the radar's 58 — PNG compresses runs of identical bytes, and a
# continuous ramp has none. Sixteen levels are indistinguishable on a
# translucent wash and compress like a poster.
ALPHA_STEPS = 16

# The model's coverage ends in a fade over its last grid cells, not a cut: on
# the national picture's 5 km pixels the cut was a staircase across the Gulf
# from 24.4N at 97.5W to 23.2N at 82W (B-262; radar build 7, item 14). Three
# cells is about 9 km of open water.
EDGE_FADE_CELLS = 3


def colourise(cover):
    """Percent cloud cover to a soft white wash, as luminance plus alpha.

    Slightly blue-grey rather than pure white: pure white reads as fog sitting
    on the water, and this is cloud between the boater and the sun. The colour
    never varies, so the image carries one grey channel and an alpha channel
    rather than three colour channels that are all constant.
    """
    pct = np.clip(np.nan_to_num(cover, nan=0.0), 0, 100)
    # Below a tenth is a clear sky with a wisp in it; drawing it only makes the
    # map look dirty.
    alpha = np.where(pct < 10, 0.0, (pct / 100.0) * MAX_ALPHA)
    # Rounded to the step, then held under the cap: round(0.55 x 16) / 16 is
    # 0.5625, so the top step sat over MAX_ALPHA (and would at 32 steps too).
    stepped = np.minimum(np.round(alpha * ALPHA_STEPS) / ALPHA_STEPS, MAX_ALPHA)
    h, w = pct.shape
    out = np.zeros((h, w, 2), dtype=np.uint8)
    out[..., 0] = 246
    out[..., 1] = (stepped * 255).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="public/cloud")
    args = ap.parse_args()

    day, run = render.chosen_run()
    os.makedirs(args.out, exist_ok=True)
    frames, regional_frames, total_in, total_out = [], [], 0, 0

    for minutes in LEAD_MINUTES:
        try:
            values, meta, n = wind.read_field(day, run, minutes, FIELD, LEVEL)
        except Exception as e:
            print(f"  +{minutes:>3} min: {e}", file=sys.stderr)
            continue
        total_in += n
        when = render.valid_time(meta)

        # The regional cells, plus one whole-domain frame.
        #
        # Cloud drew a single cell chosen by the boater's own spot, so any view
        # wider than that cell ended at a dead straight line with clear sky
        # beyond it — found on Jason's phone 2026-09-02. It is a field, like
        # wind: it paints everywhere inside its cell, so the cell's edge *is*
        # the picture.
        boxes = list(render.cells()) + regional_cells() + [(NATIONAL_ID, NATIONAL_BBOX)]

        for name, bbox in boxes:
            # Half the radar's resolution. Cloud is a far smoother field than
            # reflectivity — there is no storm edge to keep sharp — so the
            # extra pixels carried no information and quadrupled the bytes.
            if name == NATIONAL_ID:
                size = (NATIONAL_SIZE[0] // 2, NATIONAL_SIZE[1] // 2)
            elif is_regional(name):
                size = CLOUD_REGIONAL_SIZE          # the regional tier (item 11)
            else:
                size = (render.FORECAST_CELL_SIZE[0] // 2, render.FORECAST_CELL_SIZE[1] // 2)
            image = render.render(values, meta, bbox, size, paint=colourise,
                                  edge_cells=EDGE_FADE_CELLS)
            filename = f"{name}-cloud-{minutes:04d}.png"
            path = os.path.join(args.out, filename)
            image.save(path, optimize=True)
            total_out += os.path.getsize(path)
            (regional_frames if is_regional(name) else frames).append({
                "cell": name, "leadMinutes": minutes, "image": filename,
                "validTime": when.strftime("%Y-%m-%dT%H:%M:%SZ") if when else None,
                "west": bbox[0], "south": bbox[1], "east": bbox[2], "north": bbox[3],
            })

    manifest = {
        "model": "HRRR", "field": "tcdc", "unit": "percent",
        "runTime": f"{day[:4]}-{day[4:6]}-{day[6:]}T{run:02d}:00:00Z",
        "generatedAt": datetime.datetime.now(datetime.timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "maxAlpha": MAX_ALPHA,
        "alphaSteps": ALPHA_STEPS,
        "edgeFadeCells": EDGE_FADE_CELLS,
        "frames": frames,
        # The regional tier (radar build 7, item 11): the cloud and wind manifests carry each frame's box
        # inside `frames`, where the installed builds would draw these at 12 to 20 degrees, so they sit
        # under their own key instead.
        REGIONAL_KEY: {"frames": regional_frames},
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, separators=(",", ":"))

    print(f"{len(frames)} cloud frames")
    print(f"  in {total_in/1048576:.0f} MB, out {total_out/1024:.0f} KB")


if __name__ == "__main__":
    main()
