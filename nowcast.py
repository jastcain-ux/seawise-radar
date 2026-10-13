#!/usr/bin/env python3
"""Extrapolated measured radar: the last two MRMS frames, motion-tracked forward.

The reference apps' forecast stays crisp for the first hour or two because it
is not a model at all — it is the newest measured frame moved along the motion
seen between it and the one before. Ours went to HRRR's 3 km grid at
+15 minutes, which is why our forecast half could never look as sharp as our
measured half however well it was drawn. D-64.

**What this is, and is not.** It moves rain; it does not grow or kill it. A
steady system tracks for an hour or two; developing or collapsing convection
will not, and every app's nowcast shares that limit. The app captions these
frames "projected from the last hour" and never "forecast" — D-57. The manifest
says `"source": "extrapolation"` so nothing downstream can mistake it.

**Motion.** Block-wise phase correlation between the two newest frames on a
pooled grid: one vector per ~250 km block, smoothed, with blocks that have too
little echo to correlate inheriting their neighbours'. A single national vector
was tried first in thought and rejected — a front over Texas and a sea-breeze
line over Florida do not move together.

**Cost.** Wide tier at half resolution plus the national frame, eight leads.
The close cells are deliberately absent: an advected field is already a smear
of the source, and 480 px/degree of a smear costs render time for nothing. The
app's tier chooser then picks the wide frame at close zoom, which is the right
picture. Stamp-keyed on the newest MRMS frame, so a ten-minute tick that brings
no new frame renders nothing.
"""
import argparse, datetime, json, os, sys
from multiprocessing import Pool

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import observed                                            # noqa: E402
import render as forecast                                   # noqa: E402
from cells import (CELL_ORIGINS, CELL_SIZE, CELL_SPAN, MID_ORIGINS,  # noqa: E402
                   MID_SPAN, MID_SIZE, NATIONAL_BBOX, NATIONAL_ID, NATIONAL_SIZE)

LEAD_MINUTES = [15, 30, 45, 60, 75, 90, 105, 120]
# Pooling before correlation: the mosaic is 0.01 degrees and storm motion is a
# few kilometres a minute, so tracking at full resolution costs memory for no
# precision. Four cells is ~4 km per pooled pixel.
#
# It was 16, and a dry run on a synthetic storm caught it before the runner
# did: a true shift of 6 rows and 10 columns came back as 0 and 16, because a
# phase-correlation peak lands on whole pooled pixels and 16 km is coarser
# than most storm motion in ten minutes. The runner would have published
# confidently wrong extrapolations, and nothing on screen would have said so.
POOL = 4
# Correlation blocks, in pooled pixels: ~250 km across, ~16 of them over CONUS.
BLOCK = 64
# Below this fraction of echo a block cannot be correlated and borrows motion.
MIN_ECHO_FRACTION = 0.02
# Same tiers as the measured half, and the same sizes for the wide and
# national ones.
#
# The first version halved everything on the theory that an advected field
# is a smear of its source. The first real frames disproved it — the +15 and
# +60 crops carried MRMS-sharp cores and edges — and Jason saw the cost on
# the phone: the measured half draws close cells at 480 px/degree, so the
# moment the scrubber crossed NOW the picture dropped to 100 px/degree, five
# times coarser. "Not good."
#
# Close cells come in at half the measured cell's pixel size, 240 px/degree,
# and only for the first hour: that is three times sharper than before at a
# quarter of the pixels, and the near leads are where a boater is actually
# reading detail. The whole step was 51 seconds a run before this.
NOWCAST_MID_SIZE = MID_SIZE
NOWCAST_NATIONAL_SIZE = NATIONAL_SIZE
NOWCAST_CLOSE_SIZE = (CELL_SIZE[0] // 2, CELL_SIZE[1] // 2)
CLOSE_LEADS = {15, 30, 45, 60}


def _pool(field):
    h, w = field.shape
    h2, w2 = h - h % POOL, w - w % POOL
    lit = np.where(field > observed.NO_ECHO, field, 0.0)[:h2, :w2]
    return lit.reshape(h2 // POOL, POOL, w2 // POOL, POOL).mean(axis=(1, 3))


def _phase_shift(a, b):
    """Shift (dy, dx) that best moves `a` onto `b`, by phase correlation,
    refined to a fraction of a pixel by a parabola through the peak and its
    two neighbours on each axis."""
    fa, fb = np.fft.fft2(a), np.fft.fft2(b)
    cross = fb * np.conj(fa)
    denom = np.abs(cross)
    denom[denom == 0] = 1.0
    corr = np.fft.ifft2(cross / denom).real
    H, W = corr.shape
    py, px = np.unravel_index(np.argmax(corr), corr.shape)

    def refine(c_minus, c_0, c_plus):
        d = c_minus - 2 * c_0 + c_plus
        return 0.0 if abs(d) < 1e-12 else 0.5 * (c_minus - c_plus) / d

    dy = py + refine(corr[(py - 1) % H, px], corr[py, px], corr[(py + 1) % H, px])
    dx = px + refine(corr[py, (px - 1) % W], corr[py, px], corr[py, (px + 1) % W])
    if dy > H / 2: dy -= H
    if dx > W / 2: dx -= W
    return dy, dx, float(corr.max())


# Windows overlap: each 64 pooled pixels wide, centred every 32. A storm then
# has a window centred near it rather than one it happens to fall into, so its
# own motion dominates at its own position.
STRIDE = 32


def motion_field(prev, curr, minutes_apart):
    """Per-pixel (dy, dx) in grid cells per minute, on the full mosaic grid.

    Each overlapping window that holds enough echo is tracked by phase
    correlation; the vectors are then spread across the grid by an
    echo-weighted normalised convolution. Two things that were wrong before,
    caught by a synthetic two-storm test rather than on the runner:

    - Empty windows used to *borrow* the nearest tracked vector, and the field
      was then smoothed. A storm in a corner window had neighbours nearer to a
      different storm, borrowed that storm's opposite motion, and the smoothing
      averaged its own motion down to a fifth. Weighting by how much echo
      supported each vector means a storm's own window decides at the storm,
      and an empty region takes a smooth blend of whatever motion is nearest
      *in proportion to its evidence* — never a hard borrow.
    - Windows were tiled edge to edge, so a storm's position relative to a
      window edge changed the answer. Overlapping windows centre one near it.
    """
    pa, pb = _pool(prev), _pool(curr)
    H, W = pa.shape
    ys = list(range(0, max(H - BLOCK, 0) + 1, STRIDE)) or [0]
    xs = list(range(0, max(W - BLOCK, 0) + 1, STRIDE)) or [0]
    vy = np.zeros((len(ys), len(xs)))
    vx = np.zeros_like(vy)
    wt = np.zeros_like(vy)
    for i, by in enumerate(ys):
        for j, bx in enumerate(xs):
            a = pa[by:by + BLOCK, bx:bx + BLOCK]
            b = pb[by:by + BLOCK, bx:bx + BLOCK]
            echo = float((a > 0).mean()) if a.size else 0.0
            if echo < MIN_ECHO_FRACTION:
                continue
            dy, dx, _ = _phase_shift(a, b)
            vy[i, j], vx[i, j], wt[i, j] = dy, dx, echo
    if not wt.any():
        return None
    # Normalised convolution: weighted vectors over weights, both blurred by
    # the same kernel, so a tracked window's motion holds exactly at its own
    # centre and fades into its neighbours' with distance and evidence.
    sigma = 1.0
    num_y = ndimage.gaussian_filter(vy * wt, sigma, mode="nearest")
    num_x = ndimage.gaussian_filter(vx * wt, sigma, mode="nearest")
    den = ndimage.gaussian_filter(wt, sigma, mode="nearest")
    # Where no evidence reaches at all, widen until something does rather than
    # invent zero motion: a field that stops dead at the edge of the tracked
    # area would tear the picture there.
    for _ in range(6):
        gap = den < 1e-6
        if not gap.any():
            break
        num_y = np.where(gap, ndimage.gaussian_filter(num_y, 2.0, mode="nearest"), num_y)
        num_x = np.where(gap, ndimage.gaussian_filter(num_x, 2.0, mode="nearest"), num_x)
        den = np.where(gap, ndimage.gaussian_filter(den, 2.0, mode="nearest"), den)
    den = np.where(den < 1e-9, 1e-9, den)
    vy, vx = num_y / den, num_x / den

    # Back to the full grid, in cells per minute. Window (i, j) is centred at
    # pooled (ys[i] + BLOCK/2, xs[j] + BLOCK/2); map that lattice onto the grid.
    scale = POOL / max(minutes_apart, 1.0)
    cy = (np.array(ys) + BLOCK / 2.0) * POOL
    cx = (np.array(xs) + BLOCK / 2.0) * POOL
    gy = np.interp(np.arange(prev.shape[0]), cy, np.arange(len(ys)))
    gx = np.interp(np.arange(prev.shape[1]), cx, np.arange(len(xs)))
    GY, GX = np.meshgrid(gy, gx, indexing="ij")
    vy_full = ndimage.map_coordinates(vy, [GY, GX], order=1, mode="nearest") * scale
    vx_full = ndimage.map_coordinates(vx, [GY, GX], order=1, mode="nearest") * scale
    return vy_full, vx_full


def advect(field, vy, vx, minutes):
    """The field carried along its motion for `minutes`. Semi-Lagrangian: each
    output cell looks back along the flow for its value."""
    H, W = field.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    src_y = yy - vy * minutes
    src_x = xx - vx * minutes
    lit = np.where(field > observed.NO_ECHO, field, observed.NO_ECHO)
    return ndimage.map_coordinates(lit, [src_y, src_x], order=1,
                                   mode="constant", cval=observed.NO_ECHO)


_FIELD = _META = _OUT = _STAMP = None
_TOPS = None          # the projected field's cores, once per lead (item 19)


def _render_one(cell):
    name, bbox = cell
    try:
        if name == NATIONAL_ID:
            px = NOWCAST_NATIONAL_SIZE
        elif name.startswith("wide-"):
            px = NOWCAST_MID_SIZE
        else:
            px = NOWCAST_CLOSE_SIZE
        image = observed.render(_FIELD, _META, bbox, px, tops=_TOPS,
                                block=observed.NATIONAL_BLOCK_CELLS if name == NATIONAL_ID else None)
        image.save(os.path.join(_OUT, f"{name}-{_STAMP}.png"), optimize=True)
        return name, True, None
    except Exception as e:                                  # noqa: BLE001
        return name, False, str(e)


def tiers(minutes):
    """The frames to render for one lead: close cells for the first hour, the
    wide tier and the national frame for all of them."""
    out = []
    if minutes in CLOSE_LEADS:
        out += [(n, (w, s, w + CELL_SPAN[0], s + CELL_SPAN[1])) for n, w, s in CELL_ORIGINS]
    out += [(f"wide-{n}", (w, s, w + MID_SPAN[0], s + MID_SPAN[1])) for n, w, s in MID_ORIGINS]
    out.append((NATIONAL_ID, NATIONAL_BBOX))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/nowcast")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    steps = observed.wanted_keys()
    if len(steps) < 2:
        raise SystemExit("fewer than two MRMS frames; nothing to extrapolate")
    (t_prev, k_prev), (t_curr, k_curr) = steps[-2], steps[-1]
    stamp = t_curr.strftime("%Y%m%dT%H%M") + "Z"
    manifest_path = os.path.join(args.out, "manifest.json")

    # Nothing new since last run: the manifest already describes this stamp.
    if os.path.exists(manifest_path):
        try:
            if json.load(open(manifest_path)).get("runTime") == \
               t_curr.isoformat().replace("+00:00", "Z"):
                print(f"nowcast for {stamp} already published; nothing to do")
                return
        except Exception:                                   # noqa: BLE001
            pass

    prev, meta_prev, _ = observed.read_frame(k_prev)
    curr, meta, _ = observed.read_frame(k_curr)
    if prev.shape != curr.shape:
        raise SystemExit("MRMS grid changed between frames; refusing to track")
    apart = (t_curr - t_prev).total_seconds() / 60.0
    motion = motion_field(prev, curr, apart)
    if motion is None:
        raise SystemExit("no echo anywhere to track; nothing to extrapolate")
    vy, vx = motion
    print(f"tracked {stamp}: frames {apart:.0f} min apart, "
          f"median motion {np.median(np.hypot(vy, vx)) * 60 * 1.1:.0f} km/h")

    frames = []
    seen_cells = {}
    global _FIELD, _META, _OUT, _STAMP, _TOPS
    _META, _OUT = meta, args.out
    for minutes in LEAD_MINUTES:
        wanted = [c for c in tiers(minutes) if not args.only or c[0] == args.only]
        for n, b in wanted:
            seen_cells[n] = b
        _FIELD = advect(curr, vy, vx, minutes)
        _TOPS = observed.grid_tops(_FIELD, meta)
        _STAMP = f"{stamp}-{minutes:03d}"
        when = t_curr + datetime.timedelta(minutes=minutes)
        with Pool() as pool:
            results = pool.map(_render_one, wanted)
        for name, ok, err in results:
            if ok:
                frames.append({"cell": name, "leadMinutes": minutes,
                               "validTime": when.isoformat().replace("+00:00", "Z"),
                               "image": f"{name}-{_STAMP}.png"})
            else:
                print(f"  +{minutes:3d} {name}: SKIPPED: {err}", file=sys.stderr)
        print(f"  +{minutes:3d} min -> {sum(1 for _, ok, _ in results if ok)} frames")

    if not frames:
        raise SystemExit("no nowcast frames rendered; leaving the previous manifest in place")

    manifest = {
        "model": "MRMS extrapolation",
        "field": "composite reflectivity",
        "source": "extrapolation",
        "runTime": t_curr.isoformat().replace("+00:00", "Z"),
        "generatedAt": datetime.datetime.now(datetime.timezone.utc)
            .isoformat(timespec="seconds").replace("+00:00", "Z"),
        "size": {"width": NOWCAST_MID_SIZE[0], "height": NOWCAST_MID_SIZE[1]},
        "dbzFloor": forecast.RAMP[0][0],
        "rampSteps": forecast.RAMP_STEPS,
        "quantiser": forecast.QUANTISER,
        "palette": forecast.PALETTE,
        "smoothing": observed.SMOOTH_CELLS,
        "alpha": forecast.ALPHA,
        "cells": [{"id": n, "bbox": {"west": b[0], "south": b[1], "east": b[2], "north": b[3]}}
                  for n, b in seen_cells.items()],
        "frames": frames,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"{len(LEAD_MINUTES)} leads, {len(seen_cells)} tiers, {len(frames)} nowcast frames")


if __name__ == "__main__":
    main()
