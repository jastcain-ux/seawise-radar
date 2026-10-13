"""Every storm core's top, found once per step on the model's own grid (SeaWise radar build 7, item 19; B-265).

Pure numpy and scipy, so `test_render.py` can import it without eccodes.

Until build 7 each picture found its own peaks on its own crop: one top per
connected area at or above 35 dBZ. Two things went wrong with that (B-265,
confirmed on real frames 2026-10-09 and 2026-10-12). A second core inside the
same area (a 57 dBZ core on a 45 dBZ line beside a 64) was never restored,
so it came out a band or two low; and a core cut by a picture's edge, or
sharing an area with a core outside the picture, was decided differently in
every picture and every tier, so the same storm differed between them.

Now the tops are found once per step on the whole grid and every picture
restores the same ones at the same places. "Every separated local maximum"
is read by the colour bands: for every band edge at or above the restore
floor (35, 40, 45, ...), the top of every connected area at or above that
edge. Two cores each at 45 on a 40 dBZ line are separate areas at the 45
edge, so both keep their colour; a 38 dBZ top that is only part of a bigger
core's area at 35 is the same one top today's rule restores, so nothing
today restores is lost (the start reader's BLOCKER 1: a saddle rule alone
dropped 33 to 167 small 35 to 40 dBZ tops a picture). Nine labellings of a
24.5 Mpx MRMS grid cost well under a second.
"""
import math

import numpy as np
from scipy import ndimage


def band_tops(field, edges):
    """(rows, cols) of every area's top at every band edge, each place once.

    `field` is the model grid; `edges` the band edges to use (those at or
    above the restore floor). Pixels marked outside the field must already
    hold a value below the lowest edge.
    """
    seen = set()
    rows, cols = [], []
    for edge in edges:
        areas, count = ndimage.label(field >= edge)
        if count == 0:
            continue
        for r, c in zip(*area_tops(field, areas, count)):
            if (r, c) not in seen:
                seen.add((r, c))
                rows.append(int(r))
                cols.append(int(c))
    return np.array(rows, dtype=int), np.array(cols, dtype=int)


def area_tops(field, areas, count):
    """(rows, cols) of each labelled area's top: its first highest pixel in row order.

    Decided here and not by scipy: `ndimage.maximum_position` breaks a tie
    between equal pixels differently from one release to the next (1.13 gave
    the first in row order, 1.18 another), so the same field had its top on
    one row on this Mac and the next row on the runner, and a rule read two
    ways is a rule nobody can test. One rule for the step's tops and for a
    picture restored on its own (`smoothing.restore_peaks`).
    """
    mask = areas > 0
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    lab = areas.ravel()[idx]
    val = field.ravel()[idx]
    order = np.lexsort((idx, -val, lab))
    lab_sorted = lab[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = lab_sorted[1:] != lab_sorted[:-1]
    rows, cols = np.unravel_index(idx[order][first], field.shape)
    return rows.astype(int), cols.astype(int)


def pixels_of(lats, lons, bbox, size):
    """Where (lat, lon) points land in a web-mercator picture of `bbox` at `size`.

    Returns (ys, xs) as integer pixel indices, only for the points that fall
    inside the picture; the picture's own sampling grid puts pixel 0 at the
    west and north edges and the last pixel at the east and south ones, as
    `render.render` and `observed.render` lay it out with `linspace`.
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    west, south, east, north = bbox
    width, height = size

    def merc(lat):
        return np.log(np.tan((90.0 + lat) * math.pi / 360.0))

    x = (lons - west) / (east - west) * (width - 1)
    y = (merc(north) - merc(lats)) / (merc(north) - merc(south)) * (height - 1)
    xi = np.round(x).astype(int)
    yi = np.round(y).astype(int)
    keep = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    return yi[keep], xi[keep]


def halo_bbox(bbox, size, halo):
    """The box and size that hold `bbox` at `size` plus `halo` pixels on every side.

    Each picture is shaped with this halo and then cropped back, so a pixel
    at the picture's edge is smoothed and restored exactly as it would be in
    the middle of a bigger picture (the start reader: four sigma plus the
    bump's reach is under 24 px on every tier).
    """
    west, south, east, north = bbox
    width, height = size
    dx = (east - west) / (width - 1)

    def merc(lat):
        return math.log(math.tan((90.0 + lat) * math.pi / 360.0))

    def unmerc(y):
        return math.degrees(2.0 * math.atan(math.exp(y)) - math.pi / 2.0)

    dy = (merc(north) - merc(south)) / (height - 1)
    grown = (west - halo * dx, unmerc(merc(south) - halo * dy),
             east + halo * dx, unmerc(merc(north) + halo * dy))
    return grown, (width + 2 * halo, height + 2 * halo)
