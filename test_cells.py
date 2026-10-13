#!/usr/bin/env python3
"""The cell layouts must cover the water SeaWise serves.

A gap here does not fail loudly, which is exactly why it needs a test. The
app falls back — the measured half to NOAA's tiles, the forecast half to the
NWS gust field — and a flat blue wash has twice been mistaken for the radar
being broken. Nothing in a rendered frame tells you a boater's own bay fell
between two cells; only this does.

Run with: python -m unittest discover tools/hrrr-radar
"""
import unittest

# Deliberately not `observed` or `render`: those import a GRIB reader, and a
# geometry test that only runs where numpy and eccodes are installed is a test
# nobody runs before moving a cell origin. `cells.py` holds both layouts for
# exactly this reason.
import cells as layout


def measured_cells():
    w, h = layout.CELL_SPAN
    return [(n, (west, south, west + w, south + h))
            for n, west, south in layout.CELL_ORIGINS]


def forecast_cells():
    w, h = layout.FORECAST_CELL_SPAN
    return [(n, (west, south, west + w, south + h))
            for n, west, south in layout.FORECAST_CELL_ORIGINS]

# Coastal water, the Great Lakes, and the inland reservoirs the app supports
# since lakes were brought into scope. Not a sample: these are the spots the
# project has actually measured against, plus the largest recreational lakes
# in each region.
SPOTS = [
    # Gulf
    ("Galveston Bay", 29.30, -94.80), ("Clear Lake", 29.55, -95.05),
    ("Corpus Christi", 27.80, -97.40), ("Pensacola Pass", 30.33, -87.31),
    ("Mobile Bay", 30.40, -88.00), ("New Orleans", 29.30, -89.40),
    ("Tampa Bay", 27.80, -82.60),
    # Atlantic
    ("Key West", 24.55, -81.80), ("Miami", 25.77, -80.13),
    ("Vero Beach", 27.64, -80.37), ("Jacksonville", 30.35, -81.40),
    ("Charleston", 32.75, -79.90), ("Cape Hatteras", 35.25, -75.50),
    ("Chesapeake Bay", 38.00, -76.20), ("Delaware Bay", 39.00, -75.20),
    ("Long Island Sound", 41.10, -72.90), ("Cape Cod", 41.70, -70.30),
    ("Portland ME", 43.65, -70.25),
    # Great Lakes
    ("Lake Erie west", 41.80, -83.30), ("Lake Erie east", 42.50, -79.50),
    ("Lake Ontario", 43.60, -77.00), ("Lake Michigan", 43.00, -87.00),
    ("Lake Michigan south", 41.80, -87.40), ("Green Bay", 44.90, -87.60),
    ("Lake Huron", 44.50, -82.50), ("Lake Superior west", 46.80, -91.50),
    ("Lake Superior east", 47.30, -86.00),
    # Pacific
    ("Puget Sound", 47.60, -122.40), ("Columbia River", 46.20, -123.80),
    ("Coos Bay", 43.40, -124.30), ("San Francisco Bay", 37.80, -122.40),
    ("Monterey", 36.60, -121.90), ("Santa Barbara", 34.40, -119.70),
    ("San Diego", 32.70, -117.20),
]

# Inland reservoirs. The forecast half must reach these — SeaWise supports
# lakes, and a boater on Lake Texoma is as entitled to a forecast picture as
# one on the Gulf. The measured half does not cover them and falls back to
# NOAA's tiles, which is a real picture rather than a wash.
LAKES = [
    ("Lake Houston", 29.95, -95.13), ("Lake Livingston", 30.75, -95.10),
    ("Sam Rayburn", 31.10, -94.10), ("Lake Texoma", 33.90, -96.60),
    ("Lake of the Ozarks", 38.10, -92.70), ("Kentucky Lake", 36.80, -88.10),
    ("Table Rock", 36.60, -93.30), ("Lake Norman", 35.50, -80.90),
    ("Lake Lanier", 34.20, -84.05), ("Lake Murray SC", 34.05, -81.25),
    ("Lake Mead", 36.10, -114.40), ("Lake Powell", 37.00, -111.20),
    ("Lake Havasu", 34.50, -114.30), ("Lake Tahoe", 39.10, -120.00),
    ("Flathead Lake", 47.90, -114.10), ("Great Salt Lake", 41.10, -112.50),
    ("Lake Cumberland", 36.90, -85.00), ("Lake Champlain", 44.50, -73.30),
    ("Winnipesaukee", 43.60, -71.40), ("Mille Lacs", 46.20, -93.60),
    ("Leech Lake", 47.15, -94.40), ("Lake Sakakawea", 47.50, -102.30),
    ("Lake Oahe", 45.00, -100.40), ("Lake Pend Oreille", 48.10, -116.40),
    ("Lake Roosevelt", 48.00, -118.50), ("Lake Shasta", 40.70, -122.30),
    ("Toledo Bend", 31.50, -93.60), ("Lake Conroe", 30.40, -95.60),
]

# Offshore Gulf. Added 2026-09-01 with the deep-water cells: the coastal strip
# ran out a little way out to sea, and on the wind layer that edge is a hard
# wall with blank map beyond it rather than the invisible seam it is on radar.
OFFSHORE = [
    ("Gulf central", 26.00, -90.00), ("Gulf deep south", 24.00, -90.00),
    ("Gulf east deep", 25.00, -85.50), ("Gulf west deep", 25.00, -94.00),
    ("Gulf southwest", 23.50, -95.00), ("Gulf mid-east", 26.50, -86.00),
    ("Loop Current", 25.00, -88.00), ("Flower Garden Banks", 27.90, -93.60),
]

# How far inside a cell a spot must sit.
#
# Being inside one is not enough: a boater on the edge of a cell gets a frame
# that clips their own view, which is the failure the Great Lakes layout hit at
# Toledo and Duluth. The map opens about 0.9 degrees wide, so this keeps a spot
# holding most of its view from a single cell.
EDGE_MARGIN = 0.4


def covering(cells, lat, lon):
    return [name for name, (w, s, e, n) in cells
            if w <= lon <= e and s <= lat <= n]


class ForecastCells(unittest.TestCase):

    def test_cells_cover_the_water_the_app_serves(self):
        cells = forecast_cells()
        for name, lat, lon in SPOTS + LAKES + OFFSHORE:
            self.assertTrue(covering(cells, lat, lon),
                            f"{name} falls between the forecast cells")

    def test_a_spot_outside_conus_has_no_cell(self):
        """HRRR does not cover Hawaii, and the app says so differently from
        'no frame'. A cell reaching out there would make that a lie."""
        cells = forecast_cells()
        self.assertFalse(covering(cells, 21.30, -157.85), "Honolulu")
        self.assertFalse(covering(cells, 61.20, -149.90), "Anchorage")

    def test_cells_are_no_finer_than_the_model(self):
        """HRRR is a 3 km grid — roughly 37 px per degree. Past about 200
        px/degree the extra pixels carry no extra information and cost bytes
        on a boater's connection."""
        px_per_degree = layout.FORECAST_CELL_SIZE[1] / layout.FORECAST_CELL_SPAN[1]
        self.assertLessEqual(px_per_degree, 220)
        self.assertGreaterEqual(px_per_degree, 100, "too coarse to smooth cleanly")


class MeasuredCells(unittest.TestCase):

    def test_cells_cover_the_coast_the_great_lakes_and_inland(self):
        """Inland used to be exempt here, and that exemption ended on
        2026-09-01.

        The measured layout is also the wind layout, so a lake with no cell had
        a forecast picture, no wind field and no measured radar — and on the
        wind layer a missing cell is not a quiet fallback but a hard-edged hole
        in the map, which reads as calm rather than as absent.
        """
        cells = measured_cells()
        for name, lat, lon in SPOTS + LAKES + OFFSHORE:
            self.assertTrue(covering(cells, lat, lon),
                            f"{name} falls between the measured cells")

    def test_no_spot_sits_on_a_cell_edge(self):
        """Inside a cell is not the same as held by one.

        A spot on the boundary gets a frame that clips its own view — the
        Toledo and Duluth failure, which the layout fixed by moving origins
        rather than by widening cells.
        """
        cells = measured_cells()
        for name, lat, lon in SPOTS + LAKES + OFFSHORE:
            margins = [min(lon - w, e - lon, lat - s, n - lat)
                       for _, (w, s, e, n) in cells
                       if w <= lon <= e and s <= lat <= n]
            self.assertTrue(margins, f"{name} has no cell at all")
            self.assertGreaterEqual(
                max(margins), EDGE_MARGIN,
                f"{name} sits {max(margins):.2f} deg inside its best cell")

    def test_cells_are_finer_than_the_forecast_ones(self):
        """MRMS is a 1 km mosaic against HRRR's 3 km model, so the measured
        half earns the extra pixels and the forecast half does not."""
        measured = layout.CELL_SIZE[1] / layout.CELL_SPAN[1]
        forecast = layout.FORECAST_CELL_SIZE[1] / layout.FORECAST_CELL_SPAN[1]
        self.assertGreater(measured, forecast * 2)


class RegionalCells(unittest.TestCase):
    """The regional tier (radar build 7, item 11): one picture holds any view up to 12 degrees wide at every spot."""

    DOMAIN = (-134.0, 21.0, -61.0, 52.6)       # HRRR CONUS, from the GRIB (the start reader, 2026-10-12)

    def test_twelve_pictures_on_a_twelve_degree_grid(self):
        cells = layout.regional_cells()
        self.assertEqual(len(cells), 12)
        for name, (w, s, e, n) in cells:
            self.assertTrue(name.startswith(layout.REGIONAL_PREFIX), name)
            self.assertFalse(name.startswith("wide-"))
            self.assertEqual((e - w, n - s), layout.REGIONAL_SPAN)
        wests = sorted({w for _, (w, _, _, _) in cells}); souths = sorted({s for _, (_, s, _, _) in cells})
        self.assertEqual([b - a for a, b in zip(wests, wests[1:])], [12.0] * 5)
        self.assertEqual(souths, [21.0, 33.0])

    def test_every_spot_sits_six_degrees_inside_some_picture_or_at_the_domains_edge(self):
        """An 11-degree view anywhere the app serves is drawn from one picture: the view's half-width plus
        the chooser's margin is under 6 degrees. Near the model's own edge the picture reaches the edge, no further."""
        cells = layout.regional_cells()
        dw, ds, de, dn = self.DOMAIN
        for name, lat, lon in SPOTS + LAKES + OFFSHORE:
            best = max(min(lon - w, e - lon, lat - s, n - lat)
                       for _, (w, s, e, n) in cells if w <= lon <= e and s <= lat <= n)
            to_edge = min(lon - dw, de - lon, lat - ds, dn - lat)
            self.assertGreaterEqual(best, min(6.0, to_edge) - 1e-9,
                                    f"{name} sits only {best:.1f} degrees inside its best regional picture")

    def test_the_grid_covers_the_model_domain(self):
        cells = layout.regional_cells()
        dw, ds, de, dn = self.DOMAIN
        self.assertLessEqual(min(w for _, (w, _, _, _) in cells), dw)
        self.assertGreaterEqual(max(e for _, (_, _, e, _) in cells), de)
        self.assertLessEqual(min(s for _, (_, s, _, _) in cells), ds)
        self.assertGreaterEqual(max(n for _, (_, _, _, n) in cells), dn)

    def test_the_tier_ids_never_collide_with_the_other_tiers_or_the_national(self):
        # Every writer keeps the tier out of the `cells` and `frames` lists the installed builds read by
        # `is_regional` on the id alone, so no close, wide or national id may ever read as a tier one, and every
        # tier id must (build 7's renderer reader: nothing pinned the split that keeps the installed builds safe).
        tier = [name for name, _ in layout.regional_cells()]
        self.assertEqual(len(tier), 12)
        self.assertTrue(all(layout.is_regional(name) for name in tier))
        others = ([name for name, _w, _s in layout.CELL_ORIGINS]
                  + [name for name, _w, _s in layout.FORECAST_CELL_ORIGINS]
                  + [f"wide-{name}" for name, _w, _s in layout.MID_ORIGINS]
                  + [layout.NATIONAL_ID])
        self.assertFalse([name for name in others if layout.is_regional(name)])
        self.assertFalse(set(tier) & set(others))

    def test_a_spot_outside_conus_has_no_regional_picture(self):
        cells = layout.regional_cells()
        self.assertFalse(covering(cells, 21.30, -157.85), "Honolulu")
        self.assertFalse(covering(cells, 61.20, -149.90), "Anchorage")
        self.assertFalse(covering(cells, 18.47, -66.10), "San Juan")

    def test_the_densities_sit_between_the_wide_tier_and_the_national(self):
        national = layout.NATIONAL_SIZE[0] / (layout.NATIONAL_BBOX[2] - layout.NATIONAL_BBOX[0])
        wide = layout.MID_SIZE[0] / layout.MID_SPAN[0]
        for px in (layout.REGIONAL_PX_PER_DEGREE, layout.FORECAST_REGIONAL_PX_PER_DEGREE):
            self.assertGreater(px, national); self.assertLess(px, wide)
        self.assertEqual(layout.REGIONAL_SIZE, (2400, 2400))
        self.assertEqual(layout.FORECAST_REGIONAL_SIZE, (1920, 1920))


if __name__ == "__main__":
    unittest.main()
