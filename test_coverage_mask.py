"""Where the radar cannot see, pinned (SeaWise B-33 item 2). No network: a small synthetic mosaic stands in for MRMS.

The pure half (`coverage_mask.py`) needs numpy and Pillow; the run's half (`observed.write_coverage`) imports observed,
which needs eccodes, as the runner and the pins venv have.
"""
import datetime
import hashlib
import json
import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from coverage_mask import (MASK_RULE, NO_COVERAGE_BELOW, blind_cells, coverage_image, mask_name, mosaic_window,
                           pixel_coords, unseen)
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
# The same mosaic stored south to north, as a GRIB may be: row 0 is the southern edge.
META_UP = dict(META, latitudeOfFirstGridPointInDegrees=28.01, jScansPositively=1)
VALUES_UP = VALUES[::-1].copy()
# A small box in the mosaic's north-west corner, all watched: its mask reads only the cells near it.
SMALL, SMALL_SIZE = (-89.95, 29.6, -89.6, 29.95), (35, 35)


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

    # The drawing, pinned to the mask rule's revision: masks on disk keep their names until MASK_RULE changes, so a
    # change to what is drawn (unseen, pixel_coords, halo_bbox, coverage_image) must come with a new revision.
    PINNED = {"coverage-1": "c64e7a32c996413a9a29c68d6f7a15fe8fbc9695"}

    def test_the_drawing_is_pinned_to_the_mask_rule(self):
        rgba = np.asarray(coverage_image(unseen(VALUES, META, BBOX, SIZE, R, 24)).convert("RGBA"))
        self.assertEqual(self.PINNED.get(MASK_RULE), hashlib.sha1(rgba.tobytes()).hexdigest(),
                         "the mask's drawing changed: bump MASK_RULE in coverage_mask.py and pin the new pixels here")


class NameTests(unittest.TestCase):
    """Masks named by what they show, cell by cell, so a still coverage names the same files every run."""

    def name(self, values=VALUES, meta=META, bbox=BBOX, size=SIZE, halo=24, cell="downeast"):
        return mask_name(cell, blind_cells(values), meta, bbox, size, halo)

    def test_echo_never_changes_the_name(self):
        rain = VALUES.copy()
        rain[40:60, 20:60] = 45.0
        self.assertEqual(self.name(rain), self.name())

    def test_coverage_moving_under_the_cell_renames_it_and_elsewhere_does_not(self):
        here, there = VALUES.copy(), VALUES.copy()
        here[10, 10] = -999.0                  # inside SMALL's window
        there[150, 50] = -999.0                # far south of it
        small = dict(bbox=SMALL, size=SMALL_SIZE)
        self.assertNotEqual(self.name(here, **small), self.name(**small))
        self.assertEqual(self.name(there, **small), self.name(**small))

    def test_the_window_holds_every_cell_the_mask_reads(self):
        # Flip every mosaic cell outside the window between watched and blind: the mask must not change, or the
        # name would miss a change that moves it. Both scan orders.
        for values, meta in ((VALUES, META), (VALUES_UP, META_UP)):
            r0, r1, c0, c1 = mosaic_window(meta, SMALL)
            self.assertTrue(0 <= r0 < r1 <= meta["Nj"] and 0 <= c0 < c1 <= meta["Ni"])
            outside = np.ones(values.shape, dtype=bool)
            outside[r0:r1, c0:c1] = False
            flipped = values.copy()
            flipped[outside] = np.where(values[outside] < NO_COVERAGE_BELOW, -99.0, -999.0)
            np.testing.assert_array_equal(unseen(flipped, meta, SMALL, SMALL_SIZE, R, 24),
                                          unseen(values, meta, SMALL, SMALL_SIZE, R, 24))

    def test_both_scan_orders_draw_the_same_mask(self):
        np.testing.assert_array_equal(unseen(VALUES_UP, META_UP, BBOX, SIZE, R, 24), unseen(VALUES, META, BBOX, SIZE, R, 24))

    def test_the_name_follows_the_grid_the_cell_and_its_drawing(self):
        name = self.name()
        self.assertRegex(name, r"^downeast-coverage-[0-9a-f]{10}\.png$")
        self.assertEqual(name, self.name())
        self.assertNotEqual(name, self.name(meta=dict(META, latitudeOfFirstGridPointInDegrees=30.01)))
        self.assertNotEqual(name, self.name(bbox=(-89.8, 28.2, -88.2, 29.9)))
        self.assertNotEqual(name, self.name(size=(161, 160)))
        self.assertNotEqual(name, self.name(halo=0))
        self.assertEqual(self.name(cell="gulf-a")[len("gulf-a"):], name[len("downeast"):])

    def test_a_mask_name_carries_no_stamp(self):
        # Kept while a manifest names it, as a mask or a replaced one; never by a stamp.
        self.assertIsNone(stamp_in(self.name()))


class PruneTests(unittest.TestCase):

    NOW = datetime.datetime(2026, 10, 26, 20, 0, tzinfo=datetime.timezone.utc)

    def test_the_masks_and_the_replaced_masks_are_named_and_kept(self):
        manifest = {"frames": [{"image": "a.png"}],
                    "coverage": {"observedTime": "2026-10-26T15:00:00Z", "frames": [{"cell": "a", "image": "m.png"}],
                                 "replaced": [{"cell": "a", "image": "old.png", "replacedAt": "2026-10-26T14:40:00Z"}]}}
        self.assertEqual(named_by(manifest), {"a.png", "m.png", "old.png"})
        self.assertEqual(to_remove(["m.png", "old.png"], named_by(manifest), self.NOW, 180), [])

    def test_a_mask_no_manifest_names_goes(self):
        manifest = {"coverage": {"frames": [{"cell": "a", "image": "m.png"}]}}
        self.assertEqual(to_remove(["m.png", "gone.png"], named_by(manifest), self.NOW, 180), ["gone.png"])


class WriteCoverageTests(unittest.TestCase):
    """The run's half: observed.write_coverage, which imports eccodes through observed."""

    WHEN = datetime.datetime(2026, 10, 26, 15, 0, tzinfo=datetime.timezone.utc)
    MOVED = VALUES.copy()
    MOVED[:, 90:100] = -999.0

    def setUp(self):
        # test_nowcast puts stand-ins for observed and render in sys.modules when it loads (it runs without a GRIB
        # reader); these tests need the real ones, and put the stand-ins back after.
        self.stand_ins = {name: sys.modules.pop(name) for name in ("observed", "render")
                          if name in sys.modules and getattr(sys.modules[name], "__file__", None) is None}
        import observed
        self.observed = observed
        self.dir = tempfile.TemporaryDirectory()
        self.out = self.dir.name
        self.cells = [("gulf-a", BBOX), ("wide-gulf-a", BBOX)]
        self.drawn_count = []
        self.saved_unseen = observed.unseen
        observed.unseen = lambda *a, **k: self.drawn_count.append(1) or self.saved_unseen(*a, **k)

    def tearDown(self):
        self.observed.unseen = self.saved_unseen
        self.dir.cleanup()
        sys.modules.update(self.stand_ins)

    def write(self, held, previous=None):
        self.drawn_count.clear()
        return self.observed.write_coverage(self.out, self.cells, SIZE, held, previous)

    def at(self, minutes):
        return self.WHEN + datetime.timedelta(minutes=minutes)

    def on_disk(self):
        return sorted(n for n in os.listdir(self.out) if n.endswith(".png"))

    def test_the_held_step_is_drawn_and_named_with_its_time(self):
        block = self.write((self.WHEN, VALUES, META))
        self.assertEqual(block["observedTime"], "2026-10-26T15:00:00Z")
        self.assertEqual([e["cell"] for e in block["frames"]], ["gulf-a", "wide-gulf-a"])
        self.assertEqual(sorted(e["image"] for e in block["frames"]), self.on_disk())
        self.assertNotIn("replaced", block)

    def test_a_still_coverage_draws_nothing_the_next_run(self):
        first = self.write((self.WHEN, VALUES, META))
        rain = VALUES.copy()
        rain[40:60, 20:60] = 45.0
        second = self.write((self.at(15), rain, META), previous=first)
        self.assertEqual(self.drawn_count, [], "a mask on disk was drawn again")
        self.assertEqual(second["frames"], first["frames"])
        self.assertEqual(second["observedTime"], "2026-10-26T15:15:00Z")
        self.assertNotIn("replaced", second)

    def test_a_replaced_mask_stays_named_seventy_minutes_and_a_flip_back_finds_it(self):
        a = self.write((self.WHEN, VALUES, META))
        b = self.write((self.at(15), self.MOVED, META), previous=a)
        self.assertEqual(len(self.drawn_count), 2)
        self.assertEqual([e["image"] for e in b["replaced"]], [e["image"] for e in a["frames"]])
        self.assertEqual({e["replacedAt"] for e in b["replaced"]}, {"2026-10-26T15:15:00Z"})
        self.assertEqual(to_remove(self.on_disk(), named_by({"coverage": b}), self.at(15), 180), [],
                         "the run that replaces a mask removed it while a phone may still ask for it")
        # Back to A within the hour (12:06, 14:06, 15:06 on 2026-10-26): A is on disk, nothing drawn, nothing new.
        back = self.write((self.at(30), VALUES, META), previous=b)
        self.assertEqual(self.drawn_count, [])
        self.assertEqual(back["frames"], a["frames"])
        self.assertEqual([e["image"] for e in back["replaced"]], [e["image"] for e in b["frames"]])
        # Seventy minutes after B was replaced, it leaves the block, and the next prune takes it.
        later = self.write((self.at(30 + 70), VALUES, META), previous=back)
        self.assertNotIn("replaced", later)
        self.assertEqual(sorted(to_remove(self.on_disk(), named_by({"coverage": later}), self.at(100), 180)),
                         sorted(e["image"] for e in b["frames"]))

    def test_a_replaced_mask_gone_from_disk_is_not_named(self):
        a = self.write((self.WHEN, VALUES, META))
        for e in a["frames"]:
            os.unlink(os.path.join(self.out, e["image"]))
        b = self.write((self.at(15), self.MOVED, META), previous=a)
        self.assertNotIn("replaced", b)

    def test_an_older_step_read_never_turns_the_coverage_back(self):
        # The re-check reader's SF2: a run that re-reads only 14:50 (a cell had failed, MRMS had nothing newer)
        # must not name 14:50's coverage over 15:00's.
        newer = self.write((self.WHEN, self.MOVED, META))
        older = self.write((self.at(-10), VALUES, META), previous=newer)
        self.assertEqual(older, newer)
        self.assertEqual(self.drawn_count, [])
        # With 15:00's masks gone, the older step is all there is.
        for e in newer["frames"]:
            os.unlink(os.path.join(self.out, e["image"]))
        fallback = self.write((self.at(-10), VALUES, META), previous=newer)
        self.assertEqual(fallback["observedTime"], "2026-10-26T14:50:00Z")

    def test_each_mask_is_its_tiers_size(self):
        block = self.write((self.WHEN, VALUES, META))
        close, wide = (os.path.join(self.out, e["image"]) for e in block["frames"])
        with Image.open(close) as im:
            self.assertEqual(im.size, SIZE)
        with Image.open(wide) as im:
            self.assertEqual(im.size, self.observed.MID_SIZE)

    def test_with_no_step_read_the_last_block_stands_while_its_masks_are_on_disk(self):
        first = self.write((self.WHEN, VALUES, META))
        self.assertEqual(self.write(None, previous=first), first)
        os.unlink(os.path.join(self.out, first["frames"][0]["image"]))
        self.assertIsNone(self.write(None, previous=first))
        self.assertIsNone(self.write(None, previous=None))

    def test_a_mask_that_fails_is_left_out(self):
        def fail_close(values, meta, bbox, size, *rest):
            if size == SIZE:
                raise ValueError("planted")
            return self.saved_unseen(values, meta, bbox, size, *rest)
        self.observed.unseen = fail_close
        block = self.write((self.WHEN, VALUES, META))
        self.assertEqual([e["cell"] for e in block["frames"]], ["wide-gulf-a"])
        self.assertEqual(len(self.on_disk()), 1)

    def test_a_save_that_fails_part_way_leaves_nothing_under_the_masks_name(self):
        # The re-check reader's SF3: without the write aside, a half-written mask would be named every run while the
        # coverage stood still.
        draw = self.observed.coverage_image

        class HalfWritten:
            def __init__(self, image):
                self.image = image

            def save(self, path, **kw):
                with open(path, "wb") as f:
                    f.write(b"\x89PNG half")
                raise OSError("planted: the disk filled")
        self.observed.coverage_image = lambda mask: HalfWritten(draw(mask))
        try:
            block = self.write((self.WHEN, VALUES, META))
        finally:
            self.observed.coverage_image = draw
        self.assertIsNone(block)
        self.assertEqual(self.on_disk(), [])
        retry = self.write((self.WHEN, VALUES, META))
        self.assertEqual(len(retry["frames"]), 2, "the next run draws them whole")

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
        self.assertIn('write_coverage(args.out, coverage_cells(wanted_cells), size, held, published.get("coverage"))', main)
        self.assertIn('if coverage:\n        manifest["coverage"] = coverage', main)
        self.assertIn("held = (when, values, meta)", main)
        self.assertIn('multiprocessing.get_context("fork").Pool(', main)
        self.assertIn("fx, fy, inside = pixel_coords(meta, bbox, size, MERCATOR_R)", src)


class MainTests(unittest.TestCase):
    """observed.main end to end, NOAA stubbed, on a coarse mosaic laid over the one cell it renders.

    The batch 9 build reader drove main through seven cases by hand and asked for one in the suite; the re-check
    reader's going-back-in-time case is the last test.
    """

    T1 = datetime.datetime(2026, 10, 26, 14, 50, tzinfo=datetime.timezone.utc)
    T2 = datetime.datetime(2026, 10, 26, 15, 0, tzinfo=datetime.timezone.utc)
    T3 = datetime.datetime(2026, 10, 26, 15, 10, tzinfo=datetime.timezone.utc)

    def setUp(self):
        self.stand_ins = {name: sys.modules.pop(name) for name in ("observed", "render")
                          if name in sys.modules and getattr(sys.modules[name], "__file__", None) is None}
        import observed
        self.observed = observed
        self.saved = (observed.read_frame, observed.wanted_keys)
        self.dir = tempfile.TemporaryDirectory()
        self.reads, self.grids = [], {}
        self.cell, (west, south, east, north) = next(
            (name, bbox) for name, bbox in observed.cells()
            if name != observed.NATIONAL_ID and not name.startswith("wide-"))
        # 0.05 degrees a cell over the cell's box and a degree round it; its west half watched, its east half not.
        ni, nj = int(round((east - west + 2) / 0.05)) + 1, int(round((north - south + 2) / 0.05)) + 1
        self.meta = {"Ni": ni, "Nj": nj, "latitudeOfFirstGridPointInDegrees": north + 1,
                     "longitudeOfFirstGridPointInDegrees": west - 1 + 360, "iDirectionIncrementInDegrees": 0.05,
                     "jDirectionIncrementInDegrees": 0.05, "jScansPositively": 0}
        self.a = np.full((nj, ni), -99.0)
        self.a[:, ni // 2:] = -999.0
        self.b = self.a.copy()
        self.b[:, ni // 2 - 10:ni // 2] = -999.0
        observed.read_frame = lambda key: (self.reads.append(key), (self.grids[key], self.meta, 0))[1]

    def tearDown(self):
        self.observed.read_frame, self.observed.wanted_keys = self.saved
        self.dir.cleanup()
        sys.modules.update(self.stand_ins)

    def run_main(self, steps):
        self.reads.clear()
        self.observed.wanted_keys = lambda now=None: steps
        argv = sys.argv
        sys.argv = ["observed.py", "--out", self.dir.name, "--only", self.cell, "--width", "240"]
        try:
            self.observed.main()
        finally:
            sys.argv = argv
        with open(os.path.join(self.dir.name, "manifest.json")) as f:
            return json.load(f)

    def test_three_runs_name_reuse_and_redraw_as_the_coverage_moves(self):
        self.grids = {"k1": self.a, "k2": self.a}
        first = self.run_main([(self.T1, "k1"), (self.T2, "k2")])
        self.assertEqual(self.reads, ["k1", "k2"])
        coverage = first["coverage"]
        self.assertEqual(coverage["observedTime"], "2026-10-26T15:00:00Z")
        self.assertEqual([f["cell"] for f in coverage["frames"]], [self.cell])
        mask = coverage["frames"][0]["image"]
        with Image.open(os.path.join(self.dir.name, mask)) as im:
            self.assertEqual(im.size, (240, 192))
            alpha = np.asarray(im.convert("RGBA"))[..., 3]
        self.assertTrue(0 < (alpha == 255).mean() < 1, "the mask should hold both seen and unseen water")

        # Nothing new from NOAA: no read, the frames and the block as they were.
        second = self.run_main([(self.T1, "k1"), (self.T2, "k2")])
        self.assertEqual(self.reads, [])
        self.assertEqual(second["coverage"], coverage)

        # A new step whose no-coverage area moved under the cell: read once, a new mask, the old one replaced.
        self.grids["k3"] = self.b
        third = self.run_main([(self.T1, "k1"), (self.T2, "k2"), (self.T3, "k3")])
        self.assertEqual(self.reads, ["k3"])
        self.assertEqual(third["coverage"]["observedTime"], "2026-10-26T15:10:00Z")
        self.assertNotEqual(third["coverage"]["frames"][0]["image"], mask)
        self.assertEqual([e["image"] for e in third["coverage"]["replaced"]], [mask])
        self.assertIn(mask, named_by(third), "a phone on the last manifest would find its mask gone")

    def test_a_new_step_over_a_still_coverage_draws_nothing(self):
        self.grids = {"k1": self.a, "k2": self.a}
        first = self.run_main([(self.T1, "k1"), (self.T2, "k2")])
        rain = self.a.copy()
        rain[5:15, 5:15] = 45.0
        self.grids["k3"] = rain
        drawn, saved = [], self.observed.unseen
        self.observed.unseen = lambda *a, **k: drawn.append(1) or saved(*a, **k)
        try:
            third = self.run_main([(self.T1, "k1"), (self.T2, "k2"), (self.T3, "k3")])
        finally:
            self.observed.unseen = saved
        self.assertEqual(drawn, [])
        self.assertEqual(third["coverage"]["frames"], first["coverage"]["frames"])
        self.assertEqual(third["coverage"]["observedTime"], "2026-10-26T15:10:00Z")

    def test_a_run_that_mends_an_older_frame_never_turns_the_coverage_back(self):
        self.grids = {"k1": self.a, "k2": self.b}
        first = self.run_main([(self.T1, "k1"), (self.T2, "k2")])
        self.assertEqual(first["coverage"]["observedTime"], "2026-10-26T15:00:00Z")
        older = next(f["image"] for f in first["frames"] if f["observedTime"].startswith("2026-10-26T14:50"))
        os.unlink(os.path.join(self.dir.name, older))          # a cell of 14:50 failed; this run mends it
        second = self.run_main([(self.T1, "k1"), (self.T2, "k2")])
        self.assertEqual(self.reads, ["k1"])
        self.assertEqual(second["coverage"], first["coverage"])


if __name__ == "__main__":
    unittest.main()
