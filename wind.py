#!/usr/bin/env python3
"""HRRR 10-metre wind, as a vector field the app animates.

Radar and cloud are scalar fields and are published as pictures. Wind is not —
a picture of wind is arrows someone else chose. What the app needs is the
vectors themselves, so it can move particles along them and let the *motion*
carry direction and speed.

**Encoded as a PNG, not JSON.** The grid is 0.05 degrees, which is 100x80 values
per cell for each of u and v. As JSON that is about 110 KB a frame; as a
two-channel PNG it is a few KB, because that is precisely what image
compression is for. R carries u, G carries v, both scaled from a range stated
in the manifest. B is unused.

10 metres, not 80: it is the height wind is quoted at for boating and the
height every threshold in the rating engine is expressed in.
"""
import argparse, datetime, json, math, os, sys, tempfile

import eccodes
import numpy as np
from PIL import Image
from pyproj import CRS, Transformer
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import (CELL_ORIGINS, CELL_SPAN, MID_ORIGINS,        # noqa: E402
                   MID_SPAN, MID_WIND_STEP_DEG, NATIONAL_BBOX,
                   NATIONAL_ID, NATIONAL_WIND_STEP_DEG)
import render                                       # noqa: E402
from edges import inside_mask                       # noqa: E402

# Hourly, not every fifteen minutes. Wind at the radar's cadence would be
# 4.6 MB a step and 166 MB a run against the radar's own 17 MB, for a layer
# nobody scrubs frame by frame. The app already interpolates between hours for
# the readings on the scrubber and does the same here.
LEAD_MINUTES = list(range(0, 6 * 60 + 1, 60))

# About 3.5 miles. Finer than HRRR's own 3 km grid buys nothing, and coarser
# makes the flow look like it is following a chessboard.
STEP_DEG = 0.05

# The range u and v are scaled into. 60 kt covers everything short of a
# hurricane, and clipping is stated in the manifest rather than hidden — a
# boater in 70 kt of wind has larger problems than this layer.
SPEED_LIMIT_KT = 60.0
MS_TO_KT = 1.943844


def field_byte_range(day, run, minutes, field, level):
    """Byte range of one record, matched on field, level AND lead time.

    The lead has to be matched as well as the name: sub-hourly files hold four
    lead times, and taking the first UGRD would silently return :15 for every
    hour asked for — the same trap the reflectivity reader documents.
    """
    url = render._url(day, run, minutes, ".idx")
    lines = render._get(url, timeout=30).decode().splitlines()
    # HRRR labels its f00 records "anl", not "0 hour fcst" — the analysis is
    # not a forecast. Asking for the label render._lead_label builds would find
    # nothing at all for the one step a boater cares most about: now.
    want = "anl" if minutes == 0 else render._lead_label(minutes)
    for i, line in enumerate(lines):
        p = line.split(":")
        if len(p) > 5 and p[3] == field and p[4] == level and p[5].strip() == want:
            start = int(p[1])
            end = int(lines[i + 1].split(":")[1]) - 1 if i + 1 < len(lines) else ""
            return start, end
    raise LookupError(f"no {field} {level} '{want}' in {url}")


def read_field(day, run, minutes, field, level):
    lo, hi = field_byte_range(day, run, minutes, field, level)
    raw = render._get(render._url(day, run, minutes), (lo, hi))
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
                    "longitudeOfFirstGridPointInDegrees", "DxInMetres",
                    "DyInMetres", "radius", "validityDate", "validityTime")
            meta = {k: _key(gid, k) for k in keys}
            values = eccodes.codes_get_values(gid).reshape(meta["Ny"], meta["Nx"])
            eccodes.codes_release(gid)
        return values, meta, len(raw)
    finally:
        os.unlink(path)


def _stamp(when):
    return when.strftime("%Y-%m-%dT%H:%M:%SZ") if when else None


def _key(gid, k):
    try:
        return eccodes.codes_get(gid, k)
    except Exception:
        return None


def sample(values, meta, west, south, east, north, step=None):
    """Sample the model's Lambert grid onto a regular lat/lon grid.

    Bilinear rather than the cubic the reflectivity renderer uses. Cubic is
    there to stop a blur shaving the top off a storm core; wind is a smooth
    field with no such peaks to protect, and the exact numbers a boater acts on
    come from the forecast rather than from this layer, which exists to show
    where the air is going.
    """
    lcc = CRS.from_proj4(
        f"+proj=lcc +lat_1={meta['Latin1InDegrees']} +lat_2={meta['Latin2InDegrees']} "
        f"+lat_0={meta['LaDInDegrees']} +lon_0={meta['LoVInDegrees']} "
        f"+a={meta['radius']} +b={meta['radius']} +units=m +no_defs")
    to_lcc = Transformer.from_crs("EPSG:4326", lcc, always_xy=True)
    x0, y0 = to_lcc.transform(meta["longitudeOfFirstGridPointInDegrees"],
                              meta["latitudeOfFirstGridPointInDegrees"])

    step = step or STEP_DEG
    nx = int(round((east - west) / step)) + 1
    ny = int(round((north - south) / step)) + 1
    lons = np.linspace(west, east, nx)
    # North-first, so index 0 is the top-left pixel the way an image reads.
    lats = np.linspace(north, south, ny)
    grid_lon, grid_lat = np.meshgrid(lons, lats)

    gx, gy = to_lcc.transform(grid_lon.ravel(), grid_lat.ravel())
    col = (gx - x0) / meta["DxInMetres"]
    row = (gy - y0) / meta["DyInMetres"]
    out = map_coordinates(values, [row, col], order=1, mode="nearest")
    # `mode="nearest"` carries the edge row on past the model's coverage, so
    # the caller is told which samples were inside the grid (B-271): the mask
    # goes into the picture's blue channel.
    inside = inside_mask(col, row, meta["Nx"], meta["Ny"]).reshape(ny, nx)
    return out.reshape(ny, nx), nx, ny, inside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="public/wind")
    args = ap.parse_args()

    day, run = render.chosen_run()
    os.makedirs(args.out, exist_ok=True)
    frames, total_in, total_out = [], 0, 0

    for minutes in LEAD_MINUTES:
        try:
            u, meta, n1 = read_field(day, run, minutes, "UGRD", "10 m above ground")
            v, _, n2 = read_field(day, run, minutes, "VGRD", "10 m above ground")
        except Exception as e:
            print(f"  +{minutes:>3} min: {e}", file=sys.stderr)
            continue
        total_in += n1 + n2

        # Regional cells, plus one national frame far coarser than they are.
        # Nine cells is 15 x 12 degrees and the country is 62 wide, so zoomed
        # right out the field stopped at a hard rectangle with bare map beyond
        # it — which reads as calm rather than as absent. Found on Jason's
        # phone 2026-09-01.
        boxes = [(n, w, s2, w + CELL_SPAN[0], s2 + CELL_SPAN[1], None)
                 for n, w, s2 in CELL_ORIGINS]
        # The middle tier: a view too wide for a close cell lands here rather
        # than falling to the national frame. Half the close cells' resolution,
        # because at that zoom the field is read for where the air is going
        # rather than for a number off a pixel.
        boxes += [(f"wide-{n}", w, s2, w + MID_SPAN[0], s2 + MID_SPAN[1],
                   MID_WIND_STEP_DEG)
                  for n, w, s2 in MID_ORIGINS]
        boxes.append((NATIONAL_ID, NATIONAL_BBOX[0], NATIONAL_BBOX[1],
                      NATIONAL_BBOX[2], NATIONAL_BBOX[3], NATIONAL_WIND_STEP_DEG))

        for name, west, south, east, north, step in boxes:
            gu, nx, ny, inside = sample(u, meta, west, south, east, north, step)
            gv, _, _, _ = sample(v, meta, west, south, east, north, step)

            # Metres per second out of HRRR; knots is what this app stores and
            # what every threshold in the rating engine is written in.
            gu = np.clip(gu * MS_TO_KT, -SPEED_LIMIT_KT, SPEED_LIMIT_KT)
            gv = np.clip(gv * MS_TO_KT, -SPEED_LIMIT_KT, SPEED_LIMIT_KT)

            def encode(a):
                return np.round((a + SPEED_LIMIT_KT) / (2 * SPEED_LIMIT_KT) * 255
                                ).astype(np.uint8)

            # B is the mask: 255 inside the model, 0 past its coverage, where the
            # sample is the edge row smeared on (B-271). Installed builds read R
            # and G only; the new build draws no field where B is 0.
            rgb = np.dstack([encode(gu), encode(gv), np.where(inside, 255, 0).astype(np.uint8)])
            image = f"{name}-wind-{minutes:04d}.png"
            path = os.path.join(args.out, image)
            Image.fromarray(rgb, mode="RGB").save(path, optimize=True)
            total_out += os.path.getsize(path)

            frames.append({"cell": name, "leadMinutes": minutes, "image": image,
                           "validTime": _stamp(render.valid_time(meta)),
                           "nx": nx, "ny": ny,
                           "west": west, "south": south, "east": east, "north": north})

    manifest = {
        "model": "HRRR", "field": "wind10m", "unit": "kt",
        "runTime": f"{day[:4]}-{day[4:6]}-{day[6:]}T{run:02d}:00:00Z",
        "generatedAt": datetime.datetime.now(datetime.timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        # How to read the pixels back: R is u, G is v, both linear across
        # [-speedLimitKt, +speedLimitKt]. Stated rather than assumed, so a
        # future change to the range cannot silently rescale every arrow.
        "speedLimitKt": SPEED_LIMIT_KT,
        "stepDegrees": STEP_DEG,
        # B is 255 where the sample lies inside the model's grid and 0 past its
        # edge, where R and G carry the edge row on (B-271).
        "insideMaskChannel": "B",
        "frames": frames,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, separators=(",", ":"))

    print(f"{len(frames)} wind frames over {len(CELL_ORIGINS)} cells + national")
    print(f"  in {total_in/1048576:.0f} MB, out {total_out/1024:.0f} KB")


if __name__ == "__main__":
    main()
