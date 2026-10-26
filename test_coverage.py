"""Where the radar cannot see, pinned (SeaWise B-33 item 2). No network: a small synthetic mosaic stands in for MRMS.

The pure half (`coverage.py`) needs numpy and Pillow; the run's half (`observed.write_coverage`) imports observed,
which needs eccodes, as the runner and the pins venv have.
"""
import datetime
import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from coverage import NO_COVERAGE_BELOW, coverage_image, coverage_name, pixel_coords, unseen
from naming import stamp_in
from prune import named_by, to_remove

HERE = os.path.dirname(os.path.abspath(__file__))
R = 6378137.0        # the Mercator radius the pictures use; any positive value lays the grid out the same way

# A 2 x 2 degree mosaic at 0.01 degrees, north to south as MRMS scans: rows count down from 30.0 N, columns east
# from 90.0 W. Its west half is watched and dry (-99); its east half nobody watches (-999).
META = {"Ni": 200, "Nj": 200, "latitudeOfFirstGridPointInDegrees": 30.0,
        "longitudeOfFirstGridPointInDegrees": 270.0, "iDirectionIncrementInDegrees": 0.01,
        "jDirectionIncrementInDegrees": 0.01, "jScansPositively": 0}
VALUES = np.full((200, 200), -99.0)
VALUES[:, 100:] = -999.0
BBOX = (-89.8, 28.2, -88.2, 29.8)      # inside the mosaic, the dry/blind line through its middle at 89.0 W
SIZE = (160, 160)


class PixelGridTests(unittest.TestCase):

    def test_a_pixel_lands_on_the_mosaic_cell_under_it(self):
        fx, fy, inside = pixel_coords(META, BBOX, SIZE, R)
        self.assertTrue(inside.all())
        self.assertAlmostEqual(fx[0, 0], 20.0, places=6)        # 89.8 W is 0.2 degrees east of 90.0 W
        self.assertAlmostEqual(fx[0, -1], 180.0, places=6)
        self.assertAlmostEqual(fy[0, 0], 20.0, places=6)        # the top row is 29.8 N, 0.2 below 30.0 N
        self.assertAlmostEqual(fy[-1, 0], 180.0, places=6)

    def test_a_pixel_off_the_mosaic_is_outside(self):
        _, _, inside = pixel_coords(META, (-90.5, 28.2, -89.5, 29.8), SIZE, R)
        self.assertFalse(inside[:, 0].any())
        self.assertTrue(inside[:, -1].all())


class UnseenTests(unittest.TestCase):

    def mask(self, values=VALUES, bbox=BBOX, size=SIZE, halo=24):
        return unseen(values, META, bbox, size, R, halo)

    def test_the_mask_is_the_pictures_size(self):
        self.assertEqual(self.mask().shape, (SIZE[1], SIZE[0]))
        self.assertEqual(self.mask(halo=0).shape, (SIZE[1], SIZE[0]))

    def test_no_echo_is_seen_and_no_coverage_is_not(self):
        m = self.mask()
        self.assertFalse(m[:, :70].any(), "watched dry water read as unseen")
        self.assertTrue(m[:, 90:].all(), "water nobody watches read as seen")

    def test_the_edge_errs_towards_unseen(self):
        # 160 pixels over 160 mosaic cells, one pixel a cell; the dry/blind line is at column 100, about pixel 80.
        # A pixel between columns 99 and 100 touches a blind cell, so it reads unseen though column 99 is watched.
        fx, _, _ = pixel_coords(META, BBOX, SIZE, R)
        m = self.mask(halo=0)
        for x in range(SIZE[0]):
            col = fx[0, x]
            if col + 1 < 100:
                self.assertFalse(m[0, x], f"pixel {x} at column {col:.2f} is a full cell from the line")
            if col >= 99:
                self.assertTrue(m[0, x], f"pixel {x} at column {col:.2f} touches a blind cell")

    def test_a_storm_over_watched_water_is_seen(self):
        values = VALUES.copy()
        values[40:60, 20:60] = 45.0
        self.assertFalse(self.mask(values)[:, :70].any())

    def test_past_the_mosaic_is_unseen(self):
        values = np.full((200, 200), -99.0)
        m = unseen(values, META, (-90.5, 28.2, -89.5, 29.8), SIZE, R, 24)
        self.assertTrue(m[:, :70].all())
        self.assertFalse(m[:, 90:].any())

    def test_only_the_no_coverage_value_is_blind(self):
        self.assertLess(-999.0, NO_COVERAGE_BELOW)
        self.assertGreater(-99.0, NO_COVERAGE_BELOW)
        self.assertGreater(-30.0, NO_COVERAGE_BELOW)


class ImageTests(unittest.TestCase):

    def test_unseen_is_opaque_and_seen_is_clear_after_a_round_trip(self):
        m = unseen(VALUES, META, BBOX, SIZE, R, 24)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.png")
            coverage_image(m).save(path, optimize=True)
            with Image.open(path) as back:
                self.assertEqual(back.mode, "P")
                rgba = np.asarray(back.convert("RGBA"))
        self.assertTrue((rgba[..., 3][m] == 255).all())
        self.assertTrue((rgba[..., 3][~m] == 0).all())
        self.assertEqual({tuple(p) for p in rgba[m][:, :3]}, {(128, 128, 128)})

    def test_a_mask_name_carries_its_steps_stamp(self):
        name = coverage_name("downeast", "20261026T1500Z")
        self.assertEqual(name, "downeast-20261026T1500Z-coverage.png")
        self.assertEqual(stamp_in(name), ("20261026", "1500"))


class PruneTests(unittest.TestCase):

    def test_the_manifests_masks_are_named_and_kept(self):
        manifest = {"frames": [{"image": "a.png"}],
                    "coverage": {"observedTime": "2026-10-26T15:00Z", "frames": [{"cell": "a", "image": "m.png"}]}}
        self.assertEqual(named_by(manifest), {"a.png", "m.png"})
        now = datetime.datetime(2026, 10, 26, 20, 0, tzinfo=datetime.timezone.utc)
        old = coverage_name("a", "20261026T1500Z")       # 300 minutes old, named: kept
        manifest["coverage"]["frames"] = [{"cell": "a", "image": old}]
        self.assertEqual(to_remove([old], named_by(manifest), now, 180), [])

    def test_an_unnamed_mask_goes_with_the_frames_of_its_age(self):
        now = datetime.datetime(2026, 10, 26, 20, 0, tzinfo=datetime.timezone.utc)
        young, old = coverage_name("a", "20261026T1800Z"), coverage_name("a", "20261026T1500Z")
        self.assertEqual(to_remove([young, old], set(), now, 180), [old])


class WriteCoverageTests(unittest.TestCase):
    """The run's half: observed.write_coverage, which imports eccodes through observed."""

    WHEN = datetime.datetime(2026, 10, 26, 15, 0, tzinfo=datetime.timezone.utc)

    def setUp(self):
        # test_nowcast puts stand-ins for observed and render in sys.modules when it loads (it runs without a GRIB
        # reader); these tests need the real ones, and put the stand-ins back after.
        self.stand_ins = {name: sys.modules.pop(name) for name in ("observed", "render")
                          if name in sys.modules and getattr(sys.modules[name], "__file__", None) is None}
        import observed
        self.observed = observed
        self.dir = tempfile.TemporaryDirectory()
        self.out = self.dir.name
        self.reads = []
        self.saved_read = observed.read_frame

        def fake_read(key):
            self.reads.append(key)
            return VALUES, META, 0
        observed.read_frame = fake_read
        self.cells = [("gulf-a", BBOX), ("wide-gulf-a", BBOX)]

    def tearDown(self):
        self.observed.read_frame = self.saved_read
        self.dir.cleanup()
        sys.modules.update(self.stand_ins)

    def write(self, held):
        return self.observed.write_coverage(self.out, self.WHEN, "the-key", self.cells, SIZE, held)

    def test_the_held_step_is_drawn_without_a_second_download(self):
        listed = self.write((self.WHEN, VALUES, META))
        self.assertEqual(self.reads, [])
        self.assertEqual([e["image"] for e in listed],
                         ["gulf-a-20261026T1500Z-coverage.png", "wide-gulf-a-20261026T1500Z-coverage.png"])

    def test_a_step_not_held_is_read_once(self):
        self.write(None)
        self.assertEqual(self.reads, ["the-key"])
        self.write(None)
        self.assertEqual(self.reads, ["the-key"], "masks on disk were drawn again")

    def test_each_mask_is_its_tiers_size(self):
        self.write((self.WHEN, VALUES, META))
        with Image.open(os.path.join(self.out, "gulf-a-20261026T1500Z-coverage.png")) as close:
            self.assertEqual(close.size, SIZE)
        with Image.open(os.path.join(self.out, "wide-gulf-a-20261026T1500Z-coverage.png")) as wide:
            self.assertEqual(wide.size, self.observed.MID_SIZE)

    def test_a_failed_read_lists_only_the_masks_on_disk(self):
        def broken(key):
            raise OSError("NOAA down")
        self.observed.read_frame = broken
        self.assertEqual(self.write(None), [])

    def test_the_masks_are_for_the_close_and_wide_tiers_alone(self):
        names = [n for n, _ in self.observed.coverage_cells(self.observed.cells() + self.observed.regional_cells())]
        self.assertNotIn(self.observed.NATIONAL_ID, names)
        self.assertFalse(any(self.observed.is_regional(n) for n in names))
        self.assertTrue(any(n.startswith("wide-") for n in names))
        self.assertEqual(len(names), len([c for c in self.observed.cells() if c[0] != self.observed.NATIONAL_ID]))

    def test_the_manifest_names_the_masks_after_they_are_drawn(self):
        with open(os.path.join(HERE, "observed.py")) as f:
            src = f.read()
        main = src[src.index("def main():"):]
        self.assertLess(main.index("write_coverage("), main.index('json.dump(manifest'))
        self.assertIn('if coverage_frames:\n        manifest["coverage"] = {', main)
        self.assertIn("fx, fy, inside = pixel_coords(meta, bbox, size, MERCATOR_R)", src)


if __name__ == "__main__":
    unittest.main()
