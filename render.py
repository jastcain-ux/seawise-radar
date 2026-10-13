#!/usr/bin/env python3
"""
Render NOAA HRRR forecast reflectivity to PNG frames SeaWise can draw.

Why this exists
---------------
Radar measures what is already in the air. NOAA publishes no *forecast*
reflectivity as ready-made map tiles — it exists only as HRRR model output in
GRIB2 on NOMADS. Every consumer app showing "future radar" is either
extrapolating recent frames (good for about an hour) or rendering this same
model, usually bought from a vendor.

The data is free and public domain. The only missing piece was something to
turn the files into images, which is all this is.

What it does
------------
Finds the newest HRRR run, downloads *only* the composite reflectivity record
from each forecast hour using the GRIB index sidecar (about 0.3 MB each rather
than the ~130 MB full file), reprojects from the model's Lambert grid to web
mercator, and writes one national PNG per hour plus a manifest.

Honesty rules that are not optional
-----------------------------------
- A frame that could not be produced is ABSENT from the manifest, never
  substituted with a neighbouring hour. The app says "no frame for this hour"
  rather than showing 3 PM's weather under a 6 PM label.
- The manifest carries the model run time, so the app can show how old the
  forecast is. A six-hour outlook from a run that stopped updating four hours
  ago is not a six-hour outlook.
- Below 5 dBZ is transparent. Drizzle nobody can feel must not paint the bay.
"""
import argparse, datetime, json, math, multiprocessing, os, sys, tempfile, urllib.request

import numpy as np
import eccodes
from scipy import ndimage
from PIL import Image
from pyproj import CRS, Transformer

from cells import (FORECAST_CELL_ORIGINS, FORECAST_CELL_SPAN,
                   FORECAST_CELL_SIZE, NATIONAL_BBOX, NATIONAL_ID,
                   NATIONAL_SIZE)
from smoothing import shape  # noqa: E402
from edges import edge_weight  # noqa: E402
from peaks import band_tops, pixels_of, halo_bbox  # noqa: E402

NOMADS = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod"
MERCATOR_R = 20037508.342789244

# The frames the app scrubs through. HRRR runs out to 18 hours; SeaWise shows
# six, matching the radar timeline it already has.
# Every 15 minutes out to 8 hours.
#
# QUARTER-HOURS because that is HRRR's own sub-hourly cadence — these are real
# model steps, not interpolated ones. An hourly step was too coarse to watch a
# squall line approach: at 25 knots a storm crosses fifteen miles between
# frames, which on a 60-mile map is a jump rather than a movement.
#
# NINE HOURS, although the app only scrubs six. The newest complete run is
# always one to two hours behind the wall clock, so a series starting at run+1h
# ran out before the end of a six-hour scrubber and the last steps fell back to
# the wind field — a flat blue wash that read as "the radar turned blue".
#
# Three hours of margin is deliberate: past three hours the manifest reads as
# stale and the app says so out loud, so covering further would only be padding
# a window the screen has already disowned.
#
# Frames are only downloaded when scrubbed to, so the extra ones cost storage
# rather than the boater's data.
FORECAST_MINUTES = list(range(15, 9 * 60 + 1, 15))

# CONUS, a little wider than the model grid so nothing is clipped at the edges.
CONUS_BBOX = (-127.0, 21.0, -65.0, 50.0)

# Regional cells, the same correction the measured half already had.
#
# A national frame cannot serve the zoom a boater actually uses: 35 miles of
# water out of a country-wide picture is a few dozen pixels, and blown up to a
# phone it is mush. That was fixed for the measured frames and never applied
# here, which is why the forecast half looked so much worse than the past half
# on the same screen — about seven times harder magnified before it was drawn.
#
# The cells are COARSER than the measured ones on purpose. MRMS is a 1 km
# mosaic and earns 0.2 km/px; HRRR is a 3 km model, so past roughly 200 px per
# degree the extra pixels carry no extra information and only cost bytes and
# render time. Ten by seven degrees at 2000 x 1400 is ~200 px/degree — about
# six times what the national frame gave, and honest about the model's limit.
#
# Fewer, bigger cells also keep the run affordable: every lead time is rendered
# into every cell, so 36 steps against the measured half's 25 cells would be
# 900 frames an hour.
# The forecast layout lives in `cells.py` beside the measured one.
# Written as a 256-colour PNG, for the same reason the measured frames are: a
# third of the bytes, and indistinguishable side by side because the picture
# only ever contains ramp colours at a fixed set of alphas.
PALETTE_COLOURS = 256
# The fast octree with Floyd-Steinberg dithering.
#
# Median-cut was tried first and failed the whole run: Pillow's median-cut
# only accepts RGB, and every frame carries an alpha fade. The two manual runs
# that would have published it both died in the measured step, so nothing
# published at all — which is the one outcome worse than a coarse palette.
#
# The octree is what we had; the dither is new. Octree still chooses the 256
# entries (badly, on slow gradients — 57 of 256 used on a 96-step frame), but
# dithering turns what steps remain into noise the eye reads as gradient
# rather than bands. Legacy constants, not the `Image.Dither` enum, because the
# runner's Pillow is whatever pip gives it and the enum is newer than some of
# those. If this still collapses the ramp, the next step is full-colour PNGs
# for the forecast and nowcast tiers at about three times the bytes.
# "rgba": no palette and no dither. The dither was the grain Jason saw in every
# storm core — a 12x12 window held 11 colours where a smooth picture holds 3
# (D-65). A frame costs about three times the bytes; the picture is the product.
QUANTISER = "rgba"
# Fraction of one model cell to smooth by after the cubic upsample; the table
# and the reason are in `smoothing.py`.
SMOOTH_CELLS = 0.7
# Flat colour bands at the ramp's own stops instead of a continuous blend.
# On a smoothed field the band edges are clean curves, not the staircase the
# 24-step ramp drew on a raw one; The Weather Channel's picture is banded and
# Jason picked it against the blended version (D-66).
BANDS = True
# Named so the workflow re-renders when the colour scheme itself changes.
# The name carries the shaping revision too, because both renderers' redraw
# gates compare it: "-r2" is the point-bump peak restoration that replaced the
# maximum-filter plateaus (L-69). Bump it whenever the picture's rule changes
# without a manifest key of its own, or the old frames stay published.
PALETTE = "twc-2-bands-r4"   # r4: every core's top kept, the light showers kept (build 7, item 19); r3: peaks restored from 35 dBZ, not 40


# Set before the pool forks, read inside the workers. See `observed.py`.
_VALUES = _META = _OUT = _MINUTES = None
_TOPS = None          # (lats, lons) of the step's cores, found once on the model grid

# The restore floor: where the ramp leaves green (D-66, amended 2026-09-02).
RESTORE_FLOOR_DBZ = 35.0
# Each picture is shaped with this many pixels of margin on every side and
# cropped back, so its edge pixels are shaped as the middle ones are
# (item 19): four sigma plus the bump's reach is under 24 px on every tier.
SHAPE_HALO_PX = 24


def _render_one(cell):
    """Render one cell of the lead time the pool was forked for."""
    name, bbox, size = cell
    try:
        image = render(_VALUES, _META, bbox, size, tops=_TOPS)
        image.save(os.path.join(_OUT, f"{name}-refc-{_MINUTES:04d}.png"), optimize=True)
        return name, True, None
    except Exception as e:            # noqa: BLE001 - reported, never substituted
        return name, False, str(e)


def cells():
    """Every forecast cell as (id, bbox), west/south/east/north."""
    return [(name, (west, south, west + FORECAST_CELL_SPAN[0],
                    south + FORECAST_CELL_SPAN[1]))
            for name, west, south in FORECAST_CELL_ORIGINS]
# Finer than the model's 3 km grid (this is ~1.4 km/px), because the app
# upscales a national frame down to one bay and the smoothing has to have
# something to work with. Going finer than this buys nothing: the information
# ceiling is the model, not the raster.
FRAME_SIZE = (4096, 2304)

# NWS-style reflectivity ramp, in dBZ. The first stop is where painting starts:
# below it the frame stays transparent.
# NOAA's own reflectivity palette, sampled from the legend the measured mosaic
# publishes, at the same thresholds.
#
# Matching it is not decoration. The scrubber crosses from NOAA's measured
# mosaic into these frames at "now", and the two halves were painted from
# different ramps with different floors — the mosaic from 5 dBZ, these from 15.
# Dragging past now therefore made rain *vanish*: a boater watching a line
# approach saw it thin out at the present moment, which reads as clearing that
# is not happening. That is false calm at the exact instant of most interest.
#
# The floor comes back to 5 for the same reason. It was raised to 15 to hide
# model speckle, and that was solving the wrong problem — the speckle was
# posterisation, fixed properly by RAMP_STEPS and EDGE_FADE_STEPS below.
# The Weather Channel's scheme, on Jason's ruling (D-63), not NOAA's legend.
#
# No blue band. The floor is 15 dBZ — real rain, not drizzle — because the
# cyan-and-blue haze NOAA paints below that read as blur on the phone and hid
# the shape of the weather. Then green through yellow, orange and red as
# intensity climbs, saturated, and nothing past deep red: the purples and the
# cyan at the top of NOAA's scale mark hail cores and never earned their place
# on a boater's screen. Every value maps to a nearer colour than before; nothing
# is averaged (D-44's no-blur half stands).
RAMP = [
    (15, ( 74, 222, 128)), (20, ( 34, 197,  94)), (25, ( 22, 163,  74)),
    (30, ( 21, 128,  61)), (35, (250, 204,  21)), (40, (245, 158,  11)),
    (45, (249, 115,  22)), (50, (239,  68,  68)), (55, (220,  38,  38)),
    (60, (185,  28,  28)), (65, (153,  27,  27)), (70, (127,  29,  29)),
    (80, (127,  29,  29)),
]
ALPHA = 255   # opaque. Jason, 2026-09-02, against The Weather Channel side by side (D-65)
# The outermost returns fade in rather than starting at full opacity, so the
# edge of a cell is an edge rather than a cliff. Real precipitation does not
# have a hard boundary and drawing one implies a certainty about where the rain
# stops that a 3 km model does not have.
# Half a dBZ: one pixel of anti-aliasing at the floor and then full colour.
# The Weather Channel's outline is a line, not a haze, and Jason chose it
# against the 3 dBZ fade side by side (D-66). Where the rain stops is still
# the 15 dBZ floor; only the drawing of that line changed.
EDGE_FADE_DBZ = 0.5
# Steps of edge softness.
#
# This was 4, to save bytes, and it was the single worst-looking decision in
# this file. Four steps of alpha put hard contour rings around every faint
# echo — most visible over water, where the field changes slowly and each ring
# covers a wide area. It read as exactly the blockiness the cubic sampling was
# added to remove, and no amount of smoothing upstream could survive being
# posterised at the end.
#
# 32 costs about 350 KB a frame and is the difference between "a grid" and
# "clouds". The bytes were a false economy: the picture is the product.
EDGE_FADE_STEPS = 32
# The ramp is walked as a continuous gradient rather than 14 hard bands, in
# this many steps across its range. Hard bands drew visible contour edges once
# a national frame was zoomed to one bay — a staircase in the picture that
# corresponds to nothing in the weather.
#
# **24 was not enough, and the claim that 16 looked like 48 was wrong.** It was
# judged on a national frame, where one step spanned a few pixels. On the
# middle tier at a one-degree view each step is ~3 dBZ, and where rain varies
# slowly a single step covers a wide flat area — so the picture became
# plateaus with hard edges between them. Jason read it as pixelation, and it
# was: not the 3 km model's, ours. Verified 2026-09-02 by cropping the
# published texas-inland frame around Livingston at 8x — the gradients inside
# a step were smooth (the cubic sampling works); the edges between steps were
# staircases.
#
# 96 steps is under 0.8 dBZ apart, below what the eye separates on this ramp.
# It costs bytes only through the 256-colour palette the PNG is quantised to,
# which was already merging 24 ramp levels x 32 alpha levels; the picture is
# the product, and this is not a blur — nothing is averaged, the same value
# just maps to a nearer colour.
RAMP_STEPS = 96


def _url(day, run, minutes, suffix=""):
    """The file holding this lead time, and the label its index uses.

    Whole hours live in the hourly surface files; the half-hours live in the
    sub-hourly ones, which carry four 15-minute steps each. Both publish
    composite reflectivity, and both are indexed the same way — only the
    filename and the way the record labels its lead time differ.
    """
    if minutes % 60 == 0:
        name = f"wrfsfcf{minutes // 60:02d}"
    else:
        # e.g. 45 min lives in wrfsubhf01, alongside 15, 30 and 60.
        name = f"wrfsubhf{-(-minutes // 60):02d}"
    return f"{NOMADS}/hrrr.{day}/conus/hrrr.t{run:02d}z.{name}.grib2{suffix}"


def _lead_label(minutes):
    return f"{minutes // 60} hour fcst" if minutes % 60 == 0 else f"{minutes} min fcst"


def _get(url, byte_range=None, timeout=90):
    req = urllib.request.Request(url)
    if byte_range:
        req.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
    return urllib.request.urlopen(req, timeout=timeout).read()


def refc_byte_range(day, run, minutes):
    """Byte range of the composite-reflectivity record for this lead time.

    The .idx sidecar lists every record as
    `number:offset:d=date:field:level:lead:`, so one small text fetch turns a
    130 MB file into a 0.3 MB range request. Sub-hourly files hold four lead
    times, so the lead label has to be matched as well as the field — taking
    the first REFC would silently return :15 for every half-hour asked for.
    """
    url = _url(day, run, minutes, ".idx")
    lines = _get(url, timeout=30).decode().splitlines()
    want = _lead_label(minutes)
    for i, line in enumerate(lines):
        parts = line.split(":")
        if len(parts) > 5 and parts[3] == "REFC" and parts[5].strip() == want:
            start = int(parts[1])
            end = int(lines[i + 1].split(":")[1]) - 1 if i + 1 < len(lines) else ""
            return start, end
    raise LookupError(f"no REFC '{want}' in {url}")


def latest_complete_run(now=None):
    """Newest run whose whole forecast window has been published.

    Checked against the LAST hour we need, not the first: a run that is still
    uploading would otherwise be chosen and then come up short, which would
    silently shorten the outlook the app offers.
    """
    now = now or datetime.datetime.now(datetime.timezone.utc)
    for back in range(0, 10):
        t = now - datetime.timedelta(hours=back)
        day, run = t.strftime("%Y%m%d"), t.hour
        try:
            urllib.request.urlopen(
                urllib.request.Request(_url(day, run, max(FORECAST_MINUTES), ".idx")),
                timeout=20).read(1)
            return day, run
        except Exception:
            continue
    raise SystemExit("no complete HRRR run published in the last 10 hours")


def chosen_run(now=None):
    """The run this workflow chose, else the newest complete one.

    The forecast gate (`gate.py`, radar build 7's item 11a) picks the run once
    and hands it on through `HRRR_RUN` as `YYYYMMDDHH`, so the pictures, the
    wind and the cloud of one run come from the same run: each used to ask
    NOMADS for the newest itself, minutes apart, and a run completing in
    between gave the wind and cloud a newer run than the pictures. Run by hand
    with no `HRRR_RUN`, it asks NOMADS as before.
    """
    chosen = os.environ.get("HRRR_RUN", "").strip()
    if len(chosen) == 10 and chosen.isdigit():
        return chosen[:8], int(chosen[8:10])
    return latest_complete_run(now)


def read_refc(day, run, minutes):
    lo, hi = refc_byte_range(day, run, minutes)
    raw = _get(_url(day, run, minutes), (lo, hi))
    fd, path = tempfile.mkstemp(suffix=".grib2")
    os.write(fd, raw)
    os.close(fd)
    try:
        with open(path, "rb") as f:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                raise ValueError("range request did not contain a GRIB message")
            keys = ("Nx", "Ny", "LaDInDegrees", "LoVInDegrees", "Latin1InDegrees",
                    "Latin2InDegrees", "latitudeOfFirstGridPointInDegrees",
                    "longitudeOfFirstGridPointInDegrees", "DxInMetres", "DyInMetres",
                    "radius", "validityDate", "validityTime")
            meta = {}
            for k in keys:
                try:
                    meta[k] = eccodes.codes_get(gid, k)
                except Exception:
                    meta[k] = None
            values = eccodes.codes_get_values(gid).reshape(meta["Ny"], meta["Nx"])
            eccodes.codes_release(gid)
        return values, meta, len(raw)
    finally:
        os.unlink(path)


def colourise(dbz):
    """dBZ -> RGBA along a continuous ramp, with a soft outer edge.

    The ramp's anchor colours are the conventional radar ones — cyan and blue
    for light, green through yellow for moderate, red for heavy — so intensity
    still reads the way a boater expects. What changed is that the colour moves
    between them smoothly instead of stepping, because hard bands put a
    staircase in the picture that corresponds to nothing in the weather.
    """
    stops = np.array([t for t, _ in RAMP], dtype=float)
    colours = np.array([c for _, c in RAMP], dtype=float)
    floor, ceiling = stops[0], stops[-1]

    h, w = dbz.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    if BANDS:
        band = np.clip(np.searchsorted(stops, dbz, side="right") - 1, 0, len(stops) - 1)
        for channel in range(3):
            out[..., channel] = colours[band, channel].astype(np.uint8)
    else:
        quantised = np.clip(dbz, floor, ceiling)
        quantised = floor + np.round(
            (quantised - floor) / (ceiling - floor) * RAMP_STEPS) / RAMP_STEPS * (ceiling - floor)
        for channel in range(3):
            out[..., channel] = np.interp(quantised, stops, colours[:, channel]).astype(np.uint8)
    lit = dbz >= floor
    fade = np.clip((dbz - floor) / EDGE_FADE_DBZ, 0.0, 1.0)
    fade = np.ceil(fade * EDGE_FADE_STEPS) / EDGE_FADE_STEPS
    out[..., 3] = np.where(lit, (fade * ALPHA).astype(np.uint8), 0)
    return out


def grid_tops(values, meta):
    """(lats, lons) of every core's top on the model's own grid (`peaks.band_tops`)."""
    edges = [stop for stop, _ in RAMP if stop >= RESTORE_FLOOR_DBZ]
    rows, cols = band_tops(np.nan_to_num(values, nan=-99.0), edges)
    if rows.size == 0:
        return np.array([]), np.array([])
    lcc = CRS.from_proj4(
        f"+proj=lcc +lat_1={meta['Latin1InDegrees']} +lat_2={meta['Latin2InDegrees']} "
        f"+lat_0={meta['LaDInDegrees']} +lon_0={meta['LoVInDegrees']} "
        f"+a={meta['radius']} +b={meta['radius']} +units=m +no_defs")
    to_lcc = Transformer.from_crs("EPSG:4326", lcc, always_xy=True)
    x0, y0 = to_lcc.transform(meta["longitudeOfFirstGridPointInDegrees"],
                              meta["latitudeOfFirstGridPointInDegrees"])
    to_ll = Transformer.from_crs(lcc, "EPSG:4326", always_xy=True)
    lons, lats = to_ll.transform(x0 + cols * meta["DxInMetres"], y0 + rows * meta["DyInMetres"])
    return np.asarray(lats), np.asarray(lons)


def render(values, meta, bbox, size, paint=None, edge_cells=None, tops=None):
    """Cubic-spline sample from the model's Lambert grid into web mercator.

    `edge_cells`, for a painted layer only: its last channel (the alpha) is
    faded to nothing over that many grid cells inside the model's edge, so the
    layer ends softly where the model's coverage ends instead of in a cut
    (B-262, the cloud's staircase across the Gulf). Reflectivity never takes
    it: rain at the edge keeps its colour (`edges.py`).

    `tops`, for reflectivity: (lats, lons) of the step's cores from
    `grid_tops`, restored in this picture at their own places; the picture
    is shaped with a halo and cropped back (`peaks.halo_bbox`), so every
    picture of a step carries the same values (item 19, B-265).

    This started as nearest-neighbour, on the theory that interpolating
    invented gradients the model did not produce. That was the wrong call.
    Reflectivity is a continuous field that HRRR samples every 3 km, and
    nearest-neighbour does not display that honestly — it draws hard square
    edges, which assert a sharp boundary exactly where the model is least
    certain. A storm edge is not a 3 km square, and showing one as though it
    were is its own small false precision.

    Cubic spline rather than bilinear or a blur, and the reason is a
    false-calm one rather than a cosmetic one. Measured against a real frame
    whose grid peaked at 71.9 dBZ:

        cubic spline      72.4 dBZ   (+0.4)
        bilinear          70.1 dBZ   (-1.8)
        gaussian s=0.6    66.2 dBZ   (-5.7)
        gaussian s=1.0    60.0 dBZ   (-11.9)

    Blurring is the obvious way to make a coarse field look smooth, and it
    quietly shaves the top off every storm — an app that paints a 60 dBZ core
    where the model said 72 is understating exactly the thing a boater needs
    to see. Cubic interpolation smooths the shape while leaving peaks intact.
    """
    lcc = CRS.from_proj4(
        f"+proj=lcc +lat_1={meta['Latin1InDegrees']} +lat_2={meta['Latin2InDegrees']} "
        f"+lat_0={meta['LaDInDegrees']} +lon_0={meta['LoVInDegrees']} "
        f"+a={meta['radius']} +b={meta['radius']} +units=m +no_defs")
    to_lcc = Transformer.from_crs("EPSG:4326", lcc, always_xy=True)
    x0, y0 = to_lcc.transform(meta["longitudeOfFirstGridPointInDegrees"],
                              meta["latitudeOfFirstGridPointInDegrees"])

    # Reflectivity is reprojected with a halo and cropped back after shaping,
    # so an edge pixel is shaped like a middle one (item 19).
    halo = SHAPE_HALO_PX if paint is None else 0
    wanted, wanted_size = bbox, size
    if halo:
        bbox, size = halo_bbox(bbox, size, halo)
    west, south, east, north = bbox
    def merc_y(lat):
        return math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180) * MERCATOR_R / 180

    min_x, max_x = west * MERCATOR_R / 180, east * MERCATOR_R / 180
    min_y, max_y = merc_y(south), merc_y(north)

    width, height = size
    mx, my = np.meshgrid(np.linspace(min_x, max_x, width),
                         np.linspace(max_y, min_y, height))
    lon = mx / MERCATOR_R * 180
    lat = np.degrees(2 * np.arctan(np.exp(my / MERCATOR_R * math.pi)) - math.pi / 2)
    lx, ly = to_lcc.transform(lon, lat)

    fx = (lx - x0) / meta["DxInMetres"]
    fy = (ly - y0) / meta["DyInMetres"]

    inside = (fx >= 0) & (fx < meta["Nx"] - 1) & (fy >= 0) & (fy < meta["Ny"] - 1)
    interpolated = ndimage.map_coordinates(values, [fy, fx], order=3,
                                           mode="nearest", prefilter=True)
    sampled = np.where(inside, interpolated, -99.0)
    if paint is None:
        # Reflectivity only; cloud cover reuses the reprojection untouched.
        places = pixels_of(tops[0], tops[1], bbox, size) if tops is not None else None
        sampled, before, after = shape(sampled, meta["DxInMetres"] / 111_000.0,
                                       bbox, size, SMOOTH_CELLS, tops=places,
                                       painted=(RAMP[0][0], RAMP[0][0] + EDGE_FADE_DBZ))
        if before - after > 0.3:
            # Louder than a third of a ramp step: say so, with the frame size,
            # so a core understated by smoothing is in the run log, not hidden.
            print(f"smooth: peak {before:.1f} -> {after:.1f} dBZ on {size[0]}x{size[1]}", flush=True)
        if halo:
            sampled = sampled[halo:-halo, halo:-halo]
            fx, fy = fx[halo:-halo, halo:-halo], fy[halo:-halo, halo:-halo]
            bbox, size = wanted, wanted_size
    # `paint` so the cloud renderer can reuse this reprojection rather than
    # keeping a second copy of it. One implementation of the Lambert-to-mercator
    # sampling, which is the part that is easy to get subtly wrong.
    painted = (paint or colourise)(sampled)
    if paint is not None and edge_cells:
        weight = edge_weight(fx, fy, meta["Nx"], meta["Ny"], edge_cells)
        painted = painted.copy()
        painted[..., -1] = np.round(painted[..., -1] * weight).astype(np.uint8)
    return Image.fromarray(painted)


def valid_time(meta):
    d, t = meta.get("validityDate"), meta.get("validityTime")
    if not d:
        return None
    return datetime.datetime(d // 10000, d // 100 % 100, d % 100,
                             (t or 0) // 100, (t or 0) % 100,
                             tzinfo=datetime.timezone.utc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out", help="directory to write frames into")
    ap.add_argument("--only", default="", help="render just this cell, for checking one place")
    ap.add_argument("--limit", type=int, default=0, help="render at most this many lead times")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    day, run = chosen_run()
    run_at = datetime.datetime.strptime(f"{day}{run:02d}", "%Y%m%d%H").replace(
        tzinfo=datetime.timezone.utc)
    print(f"HRRR run {run_at.isoformat()}")

    # Regional cells at their own size, plus one national frame at the model's
    # own resolution. Carried as (name, bbox, size) so the national one can be
    # a different shape without a second code path.
    wanted_cells = [(n, b, FORECAST_CELL_SIZE)
                    for n, b in cells()
                    if not args.only or n == args.only]
    if not args.only or args.only == NATIONAL_ID:
        wanted_cells.append((NATIONAL_ID, NATIONAL_BBOX, NATIONAL_SIZE))
    leads = FORECAST_MINUTES[:args.limit] if args.limit else FORECAST_MINUTES

    frames, downloaded = [], 0
    for minutes in leads:
        try:
            values, meta, nbytes = read_refc(day, run, minutes)
            downloaded += nbytes
        except Exception as e:
            # Absent, never substituted. A missing step is a thing the app has
            # to be able to say out loud.
            print(f"  +{minutes:3d} min  SKIPPED: {e}", file=sys.stderr)
            continue

        # The model's own valid time, not run + lead. If those ever disagree
        # the file is right and the arithmetic is wrong.
        when = valid_time(meta)
        written = 0
        # In parallel across the runner's cores, for the same reason the
        # measured half is: 22 independent reprojections of one grid, run one
        # after another, took eleven minutes of a run whose output ages while
        # it works. `fork` lets each worker inherit the decoded grid rather
        # than having it pickled per cell.
        global _VALUES, _META, _OUT, _MINUTES, _TOPS
        _VALUES, _META, _OUT, _MINUTES = values, meta, args.out, minutes
        # The step's cores, once, on the model grid, before the pool forks.
        _TOPS = grid_tops(values, meta)
        with multiprocessing.Pool(processes=min(len(wanted_cells),
                                                os.cpu_count() or 2)) as pool:
            for name, ok, err in pool.imap_unordered(_render_one, wanted_cells):
                if ok:
                    frames.append({
                        "leadMinutes": minutes,
                        "cell": name,
                        "validTime": when.isoformat().replace("+00:00", "Z") if when else None,
                        "image": f"{name}-refc-{minutes:04d}.png",
                    })
                    written += 1
                else:
                    print(f"  +{minutes:3d} min {name}: SKIPPED: {err}", file=sys.stderr)

        kb = sum(os.path.getsize(os.path.join(args.out, f"{n}-refc-{minutes:04d}.png"))
                 for n, _, _ in wanted_cells
                 if os.path.exists(os.path.join(args.out, f"{n}-refc-{minutes:04d}.png"))) / 1024
        print(f"  +{minutes:3d} min  {nbytes/1e6:.2f} MB in  ->  {written} cells, {kb:5.0f} KB")

    if not frames:
        raise SystemExit("no frames rendered; leaving the previous manifest in place")

    manifest = {
        "model": "HRRR",
        "field": "composite reflectivity",
        "source": "NOAA NCEP, public domain",
        "runTime": run_at.isoformat().replace("+00:00", "Z"),
        "generatedAt": datetime.datetime.now(datetime.timezone.utc)
            .isoformat(timespec="seconds").replace("+00:00", "Z"),
        "size": {"width": FORECAST_CELL_SIZE[0], "height": FORECAST_CELL_SIZE[1]},
        "dbzFloor": RAMP[0][0],
        # Recorded so the workflow can tell when the picture's own rule changed.
        # The re-render gate fires on manifest age and on the cell list; a
        # change to how dBZ becomes colour would otherwise sit unpublished until
        # the run happened to age out, and every frame meanwhile would be drawn
        # to the old rule while the code said the new one.
        "rampSteps": RAMP_STEPS,
        "quantiser": QUANTISER,
        "palette": PALETTE,
        "smoothing": SMOOTH_CELLS,
        "alpha": ALPHA,
        "bands": BANDS,
        "cells": [{"id": name,
                   "bbox": {"west": b[0], "south": b[1], "east": b[2], "north": b[3]}}
                  for name, b, _ in wanted_cells],
        "frames": frames,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"{len(leads)} lead times x {len(wanted_cells)} cells = {len(frames)} frames, "
          f"{downloaded/1e6:.1f} MB pulled from NOMADS")


if __name__ == "__main__":
    main()
