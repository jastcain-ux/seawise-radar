"""Where the radar cannot see, beside what it saw (SeaWise B-33 item 2).

Pure: numpy and Pillow, so `test_coverage_mask.py` runs without eccodes. Not `coverage.py`: that name is
PyPI's test-coverage tool, which a run under `coverage run` would shadow (the build reader's note).

MRMS writes -99 where a radar looked and found no echo, and -999 where no
radar looks at all: far offshore, in a mountain's beam shadow, round a radar
that is down. The measured pictures clamp both to the same transparent
floor (`observed.NO_ECHO`), so a phone sampling a picture cannot tell dry
water from water nobody is watching, and the verdict card's rain reading
said "dry" for both. Read on the 15:06Z frame of 2026-10-26: a third of the
mosaic was -999, 85% of the gulf-central cell and 48% of gulf-east.

So each close and wide cell gets a mask, drawn from the newest step the run
read: the same size and the same pixels as its picture, alpha 255 where no
radar sees and 0 where one does. A painted pixel there never means rain. It is a
separate picture and not a marker inside the radar picture, because the
sampler reads any alpha as rain (`RadarPixel.isLit`): an unseen marker in
the picture would draw rain over every pixel nobody watches.

A pixel is unseen when it falls outside the mosaic, or when any of the four
mosaic cells around it is -999: the edge of coverage errs a kilometre
towards "cannot see", never towards "dry".

**Named by what they show.** NOAA's no-coverage area hardly moves: on
2026-10-26 the 15:06Z frame's matched the frames 2 minutes, 10 minutes and 3
hours before it exactly, and the one an hour before in 138 of 24.5 million
mosaic cells. So a mask's name is a digest of the area it was drawn from and
of its cell's place and size (`mask_name`): while the area stands still every
run names the same files, draws nothing and uploads nothing, where a name per
step cost 63 new pictures, 6 to 15 seconds ahead of the measured publish and
about 180,000 more R2 writes a month. When the area moves, the next run draws
a new set and the manifest names it; the previous set, no longer named and
carrying no stamp, goes in that run's prune.
"""
import hashlib
import math

import numpy as np
from PIL import Image

from peaks import halo_bbox

# MRMS's no-coverage value is -999 and its no-echo value -99; anything below this is the first.
NO_COVERAGE_BELOW = -500.0
# The colour an unseen pixel carries, should a map ever draw the mask: a neutral grey.
UNSEEN_RGB = (128, 128, 128)
# The mask rule's revision, in every name: a change to what counts as unseen or how a mask is drawn renames them all.
MASK_RULE = "coverage-1"
# The grid keys a digest reads, so a mosaic laid out differently never shares a name.
GRID_KEYS = ("Ni", "Nj", "latitudeOfFirstGridPointInDegrees", "longitudeOfFirstGridPointInDegrees",
             "iDirectionIncrementInDegrees", "jDirectionIncrementInDegrees", "jScansPositively")


def grid_digest(values, meta):
    """A digest of the mosaic's no-coverage area and its layout: echo or no echo never changes it."""
    h = hashlib.sha1(MASK_RULE.encode())
    for key in GRID_KEYS:
        h.update(f"{key}={meta.get(key)};".encode())
    h.update(np.packbits(values < NO_COVERAGE_BELOW).tobytes())
    return h.hexdigest()


def mask_name(cell, grid, bbox, size, halo):
    """A mask's name: its cell, `coverage`, and ten hex characters of what it shows and where it is drawn.

    `downeast-coverage-3f9a0c41d2.png`. No stamp: `prune.py` removes it in the
    run that stops naming it.
    """
    h = hashlib.sha1(grid.encode())
    h.update(f"{tuple(bbox)};{tuple(size)};{halo}".encode())
    return f"{cell}-coverage-{h.hexdigest()[:10]}.png"


def pixel_coords(meta, bbox, size, mercator_r):
    """Each pixel's fractional column and row on the mosaic, and whether it lies inside it.

    The picture is Web Mercator, `size` pixels over `bbox` (west, south, east,
    north), centres on its edges; the mosaic is a regular latitude/longitude
    grid described by its GRIB `meta`. One function for the pictures and the
    masks, so a mask's pixel is its picture's pixel.
    """
    lon0 = meta["longitudeOfFirstGridPointInDegrees"]
    if lon0 > 180:
        lon0 -= 360
    lat0 = meta["latitudeOfFirstGridPointInDegrees"]
    di = meta["iDirectionIncrementInDegrees"]
    dj = meta["jDirectionIncrementInDegrees"]
    # MRMS scans north to south, so a row index counts *down* from lat0.
    south_up = bool(meta.get("jScansPositively"))

    west, south, east, north = bbox

    def merc_y(lat):
        return math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180) * mercator_r / 180

    min_x, max_x = west * mercator_r / 180, east * mercator_r / 180
    min_y, max_y = merc_y(south), merc_y(north)

    width, height = size
    mx, my = np.meshgrid(np.linspace(min_x, max_x, width),
                         np.linspace(max_y, min_y, height))
    lon = mx / mercator_r * 180
    lat = np.degrees(2 * np.arctan(np.exp(my / mercator_r * math.pi)) - math.pi / 2)

    fx = (lon - lon0) / di
    fy = (lat - lat0) / dj if south_up else (lat0 - lat) / dj

    inside = (fx >= 0) & (fx < meta["Ni"] - 1) & (fy >= 0) & (fy < meta["Nj"] - 1)
    return fx, fy, inside


def unseen(values, meta, bbox, size, mercator_r, halo):
    """True for each pixel of the picture of `bbox` at `size` that no radar sees.

    Laid out as `observed.render` lays out its picture, through the same halo
    and cropped back, so the two line up pixel for pixel.
    """
    grown, grown_size = halo_bbox(bbox, size, halo)
    fx, fy, inside = pixel_coords(meta, grown, grown_size, mercator_r)
    blind = values < NO_COVERAGE_BELOW
    x0 = np.clip(np.floor(fx).astype(np.int64), 0, meta["Ni"] - 2)
    y0 = np.clip(np.floor(fy).astype(np.int64), 0, meta["Nj"] - 2)
    near_blind = blind[y0, x0] | blind[y0, x0 + 1] | blind[y0 + 1, x0] | blind[y0 + 1, x0 + 1]
    out = ~inside | near_blind
    return out[halo:out.shape[0] - halo, halo:out.shape[1] - halo]


def coverage_image(mask):
    """The mask as a two-colour picture: grey at alpha 255 where unseen, clear where a radar sees.

    Indexed, as the radar pictures are written, with the clear colour marked
    transparent: on 2026-10-26's 15:06Z frame a mask was 0.7 to 3 KB this way
    against 11 to 22 KB as RGBA. A phone decodes it to the same RGBA.
    """
    image = Image.fromarray(mask.astype(np.uint8), "P")
    image.putpalette([0, 0, 0, *UNSEEN_RGB])
    image.info["transparency"] = 0
    return image
