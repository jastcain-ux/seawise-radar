"""Where the cells are — both layouts.

Lifted out of `observed.py` so the lightning renderer can import the layout
without importing a GRIB reader with it. One definition, so a cell can never
exist for radar and not for lightning — `test_cells.py` guards it for gaps.

The forecast layout moved here on 2026-09-01 for the same reason, and because
it was keeping `test_cells.py` from running anywhere numpy is not installed:
a geometry test that cannot run on a laptop is a geometry test nobody runs
before pushing a cell origin.
"""

CELL_SIZE = (2400, 1920)
CELL_SPAN = (5.0, 4.0)

# Where the cells are. Hand-placed over the water this app is for — the coasts,
# the Great Lakes and the Gulf bays — rather than tiled across the whole
# country, which would be 160 cells to cover a great deal of Nebraska.
#
# Generous overlap on purpose: a spot near a seam should find a cell holding
# its whole view rather than one that clips it at the edge.
CELL_ORIGINS = [
    # Atlantic, north to south
    ("downeast",      -71.0, 42.0), ("southern-ne",   -74.0, 39.5),
    ("mid-atlantic",  -77.0, 36.5), ("carolinas",     -80.0, 33.0),
    ("sc-georgia",    -82.0, 30.0), ("florida-ne",    -82.0, 27.0),
    ("florida-se",    -83.0, 23.5),
    # Gulf, east to west
    ("florida-sw",    -84.0, 25.5), ("florida-nw",    -87.0, 28.0),
    ("alabama-miss",  -91.0, 28.0), ("louisiana",     -94.0, 27.5),
    ("houston-galv",  -96.0, 28.0), ("texas-central", -98.0, 25.5),
    ("texas-south",  -100.0, 24.0),
    # Pacific
    ("socal",        -121.0, 31.5), ("central-ca",   -124.0, 34.0),
    ("norcal",       -125.0, 37.0), ("oregon",       -126.0, 40.0),
    ("washington",   -126.0, 43.0), ("puget",        -125.0, 46.0),
    # Great Lakes. Superior takes two cells and Erie starts a degree further
    # west than it first did: at -92/45 and -83/41 the layout missed Duluth,
    # Whitefish Bay, Toledo and the whole western basin of Erie — four pieces
    # of heavily boated water, each of which fell back to NOAA's tiles
    # without anything on screen saying why. `test_cells.py` is the guard.
    ("lake-michigan",   -89.0, 41.0),
    ("lake-superior-w", -93.0, 45.0), ("lake-superior-e", -88.0, 45.5),
    ("lake-huron",      -85.0, 43.0), ("lake-erie",       -84.0, 41.0),
    ("lake-ontario",    -79.0, 42.5),

    # The deep Gulf. Added 2026-09-01: the coastal strip stopped a little way
    # offshore, and the wind layer is what made that visible — radar paints
    # only where there is rain, so its cell edges fall on empty map, while a
    # field paints everywhere and its edge becomes the picture.
    ("gulf-west",       -98.0, 22.0), ("gulf-central",    -92.5, 23.0),
    ("gulf-east",       -89.0, 23.0),

    # Inland reservoirs. The forecast layout has reached these since lakes came
    # into scope; the measured one did not, so a boater on Lake Texoma had a
    # forecast picture and no wind field and no measured radar. Placed so every
    # lake sits at least 0.4 degrees inside a cell — a spot on a cell edge gets
    # a frame that clips its own view, which is the failure the Great Lakes
    # layout already hit at Toledo and Duluth.
    ("texas-north",     -99.5, 32.0), ("ozarks",          -96.5, 35.5),
    ("tennessee-valley",-91.0, 35.0), ("southeast-lakes", -85.5, 33.5),
    ("champlain",       -75.0, 42.0), ("minnesota",       -97.0, 44.5),
    ("dakotas",        -104.5, 44.0), ("desert-southwest",-115.5, 33.5),
    ("great-salt",     -115.5, 39.0), ("sierra",          -123.0, 37.0),
    ("inland-northwest",-120.0, 46.0), ("montana-west",   -117.0, 46.0),

    # Green Bay sat 0.10 degrees inside the Lake Michigan cell, so a boater
    # there got a frame clipped at its own northern edge. Caught by the
    # edge-margin test added with the inland cells, not by anything on screen —
    # the same silent shape as the Toledo and Duluth gaps.
    ("green-bay",       -89.0, 43.0),
]


# --- Forecast layout ---------------------------------------------------
#
FORECAST_CELL_SPAN = (10.0, 7.0)
FORECAST_CELL_SIZE = (2000, 1400)
FORECAST_CELL_ORIGINS = [
    # Atlantic
    ("northeast",       -78.0, 38.0),
    ("downeast",        -72.0, 42.0),
    ("mid-atlantic",    -84.0, 32.0),
    ("southeast",       -85.0, 26.0),
    ("florida",         -86.0, 23.0),
    # Gulf
    ("gulf-east",       -92.0, 26.0),
    ("gulf-west",      -100.0, 24.0),
    # Great Lakes
    ("great-lakes",     -93.0, 40.0),
    ("lake-superior",   -93.0, 44.0),
    ("lake-erie-ontario", -83.0, 41.0),
    # Pacific
    ("socal",          -124.0, 30.0),
    ("norcal",         -128.0, 36.0),
    ("northwest",      -128.0, 42.0),
    # Inland. SeaWise supports lakes and reservoirs, and a boater on Lake
    # Texoma or Lake Mead is as entitled to a forecast picture as one on the
    # Gulf. Without these the forecast half falls back to the wind field —
    # a flat blue wash that has twice been mistaken for the radar being
    # broken. `test_cells_cover_the_water_the_app_serves` is the guard.
    ("texas-inland",   -100.0, 29.0),
    ("mid-south",       -94.0, 33.0),
    ("southwest",      -118.0, 34.0),
    ("northern-rockies", -118.0, 41.0),
    # Added 2026-09-01 with the deep-Gulf and inland cells on the measured
    # side. These were real holes rather than tight fits: the upper-midwest
    # lakes, the Dakota reservoirs, the Idaho and Washington lakes, and the
    # offshore Gulf all fell outside every forecast cell.
    ("gulf-deep",       -97.0, 21.0),
    ("upper-midwest",   -99.0, 43.0),
    ("northern-plains",-107.0, 42.0),
    ("inland-northwest",-121.0, 43.0),
]


# --- The national frame -------------------------------------------------
#
# One low-resolution frame covering the whole model domain, published beside
# the regional cells rather than instead of them.
#
# Cells exist because a national frame is useless up close: at the zoom a
# boater reads their own bay, a 4096 px national picture gives about 50 pixels
# and the phone magnifies that into mush. The reverse is just as true. Nine
# cells is 15 x 12 degrees and the continental United States is 62 wide, so at
# continental zoom no practical number of cells covers the view — and blank map
# reads as clear sky, which is the failure this app exists to prevent.
#
# So: cells close in, this far out, chosen by zoom. The app already does
# exactly that for the measured half, where NOAA's own tiles play this role.
# HRRR has no such public tile service, which is why we publish it ourselves.
#
# **The size is the model, not a guess.** HRRR is a 3 km grid — about 37 pixels
# per degree — and this is 38.7. Going finer would cost a boater bytes to carry
# no extra information; going coarser would throw some away.
NATIONAL_ID = "national"
NATIONAL_BBOX = (-127.0, 21.0, -65.0, 51.0)
NATIONAL_SIZE = (2400, 1160)

# Wind is a smooth field and is read at continental zoom to see where the air
# is going, not to read a number off — so it is published far coarser than the
# regional cells' 0.05 degrees. At 0.2 the national frame is 311 x 151 values
# against 1241 x 601, which is a sixteenth of the bytes for a picture that
# looks the same at that scale.
NATIONAL_WIND_STEP_DEG = 0.2


# --- The middle tier ----------------------------------------------------
#
# The same weather at a third size, between the close cells and the national
# frame.
#
# The map draws one image per step and never stitches, so a view too wide for
# a close cell has to fall to whatever comes next. With only two sizes that was
# the national frame — 38.7 px/degree, which is right for the country and
# visibly blocky over a few hundred miles. This fills that band.
#
# **The geometry is the forecast layout, reused rather than reinvented.** Those
# cells are already 10 x 7 degrees and already cover the water this app serves,
# so a second hand-placed list would be a second thing to keep in step — and
# the Great Lakes layout has already taught what happens when a cell list drifts
# from the water it is meant to cover.
#
# 200 px/degree, the density the forecast half already publishes at: sharp
# enough that a regional view does not look upscaled, and no finer than the
# 3 km model can justify.
# Ids are prefixed `wide-`, not `mid-`.
#
# `mid-atlantic` is already a close cell and `mid-south` a forecast one, so a
# `mid-` prefix could not be told from a real name — and the renderer picks a
# frame's pixel size by that prefix, so a close cell called `mid-atlantic`
# would have been published at the middle tier's lower resolution. Caught
# before it rendered, by a watcher counting `mid-*` ids and finding one that
# was not a middle-tier frame at all.
MID_SPAN = FORECAST_CELL_SPAN
MID_ORIGINS = FORECAST_CELL_ORIGINS
MID_SIZE = FORECAST_CELL_SIZE

# Wind is a smooth field read for direction at this zoom, not for a number off
# a pixel, so it is published at half the close cells' resolution — a quarter
# of the samples for a picture that looks the same across a region.
MID_WIND_STEP_DEG = 0.1


# --- The regional tier (radar build 7, item 11; D-232, B-76) -----------
#
# Twenty-four-degree pictures on a twelve-degree grid, between the wide tier
# and the national picture. The app draws one picture per view (D-232), so a
# view too wide for a 10 x 7 wide cell used to fall to the national picture's
# 38.7 px a degree, drawn under the cell covering most of the view with a
# straight line between them: the "zone lines" Jason reported 2026-10-09.
# With a 24-degree picture every 12 degrees, any view up to 12 degrees wide
# centred inside the model's domain is held by one picture with 6 degrees to
# spare on each side, and the view up to about 11 degrees is drawn from one
# evenly sharp picture.
#
# The grid covers the HRRR and MRMS CONUS domains (about 134W to 61W, 21N to
# 52.6N): six columns by two rows, 12 pictures. Hawaii, Alaska and Puerto
# Rico are outside both models; `test_cells.py` keeps them out of every tier.
#
# Ids are prefixed `region-`, a prefix of its own: the installed builds'
# `MeasuredRain.Tier` reads `national` and `wide-` and calls everything else
# a close cell, so these frames go under their own manifest key, `regional`,
# which every installed decoder ignores, never into `frames` or `cells`.
#
# Density: the measured and nowcast tier at 100 px a degree, 1:1 with the
# 1 km mosaic (0.01 degrees), so nothing between samples is lost; the
# forecast tier at 80 (HRRR's own 37 px a degree, 2.2x over); cloud at 40 as
# cloud is half the radar's density everywhere; wind at 0.1 degrees like the
# wide tier. Jason's look on his phone at 80 against 100 decides the
# forecast's (the plan's "Rulings for Jason later").
REGIONAL_SPAN = (24.0, 24.0)
REGIONAL_ORIGINS = [
    (f"region-{abs(west):03d}-{south:02d}", float(west), float(south))
    for south in (21, 33)
    for west in (-134, -122, -110, -98, -86, -74)
]
REGIONAL_PX_PER_DEGREE = 100          # measured and nowcast: 1:1 with the mosaic
REGIONAL_SIZE = (int(REGIONAL_SPAN[0] * REGIONAL_PX_PER_DEGREE),
                 int(REGIONAL_SPAN[1] * REGIONAL_PX_PER_DEGREE))
FORECAST_REGIONAL_PX_PER_DEGREE = 80
FORECAST_REGIONAL_SIZE = (int(REGIONAL_SPAN[0] * FORECAST_REGIONAL_PX_PER_DEGREE),
                          int(REGIONAL_SPAN[1] * FORECAST_REGIONAL_PX_PER_DEGREE))
CLOUD_REGIONAL_SIZE = (FORECAST_REGIONAL_SIZE[0] // 2, FORECAST_REGIONAL_SIZE[1] // 2)
REGIONAL_WIND_STEP_DEG = 0.1
REGIONAL_PREFIX = "region-"
REGIONAL_KEY = "regional"


def regional_cells():
    """Every regional picture as (id, bbox), west/south/east/north."""
    return [(name, (west, south, west + REGIONAL_SPAN[0], south + REGIONAL_SPAN[1]))
            for name, west, south in REGIONAL_ORIGINS]


def is_regional(name):
    return name.startswith(REGIONAL_PREFIX)

