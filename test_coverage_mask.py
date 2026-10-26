"""Where the radar cannot see, pinned (SeaWise B-33 item 2). No network: a small synthetic mosaic stands in for MRMS.

The pure half (`coverage_mask.py`) needs numpy and Pillow; the run's half (`observed.write_coverage`) imports observed,
which needs eccodes, as the runner and the pins venv have.
"""
import datetime
import json
import multiprocessing
import os
import sys
import tempfile
import types
import unittest

import numpy as np
from PIL import Image

from coverage_mask import NO_COVERAGE_BELOW, coverage_image, grid_digest, mask_name, pixel_coords, unseen
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


class NameTests(unittest.TestCase):
    """Masks named by what they show, so a still coverage names the same files every run."""

    def test_echo_never_changes_the_digest_and_coverage_always_does(self):
        rain = VALUES.copy()
        rain[40:60, 20:60] = 45.0
        self.assertEqual(grid_digest(rain, META), grid_digest(VALUES, META))
        moved = VALUES.copy()
        moved[0, 99] = -999.0
        self.assertNotEqual(grid_digest(moved, META), grid_digest(VALUES, META))

    def test_a_grid_laid_out_differently_never_shares_a_digest(self):
        shifted = dict(META, latitudeOfFirstGridPointInDegrees=30.01)
        self.assertNotEqual(grid_digest(VALUES, shifted), grid_digest(VALUES, META))

    def test_the_name_follows_the_area_the_cell_and_its_drawing(self):
        grid = grid_digest(VALUES, META)
        name = mask_name("downeast", grid, BBOX, SIZE, 24)
        self.assertRegex(name, r"^downeast-coverage-[0-9a-f]{10}\.png$")
        self.assertEqual(name, mask_name("downeast", grid, BBOX, SIZE, 24))
        self.assertNotEqual(name, mask_name("downeast", grid_digest(VALUES, dict(META, Ni=201)), BBOX, SIZE, 24))
        self.assertNotEqual(name, mask_name("downeast", grid, (-89.8, 28.2, -88.2, 29.9), SIZE, 24))
        self.assertNotEqual(name, mask_name("downeast", grid, BBOX, (161, 160), 24))
        self.assertNotEqual(name, mask_name("downeast", grid, BBOX, SIZE, 0))

    def test_a_mask_name_carries_no_stamp(self):
        # So the run that stops naming a set removes it, rather than keeping it three hours by a stamp.
        self.assertIsNone(stamp_in(mask_name("downeast", grid_digest(VALUES, META), BBOX, SIZE, 24)))


class PruneTests(unittest.TestCase):

    NOW = datetime.datetime(2026, 10, 26, 20, 0, tzinfo=datetime.timezone.utc)

    def test_the_manifests_masks_are_named_and_kept(self):
        manifest = {"frames": [{"image": "a.png"}],
                    "coverage": {"observedTime": "2026-10-26T15:00Z", "frames": [{"cell": "a", "image": "m.png"}]}}
        self.assertEqual(named_by(manifest), {"a.png", "m.png"})
        kept = mask_name("a", grid_digest(VALUES, META), BBOX, SIZE, 24)
        manifest["coverage"]["frames"] = [{"cell": "a", "image": kept}]
        self.assertEqual(to_remove([kept], named_by(manifest), self.NOW, 180), [])

    def test_a_set_no_longer_named_goes_in_that_runs_prune(self):
        moved = VALUES.copy()
        moved[0, 99] = -999.0
        current = mask_name("a", grid_digest(moved, META), BBOX, SIZE, 24)
        previous = mask_name("a", grid_digest(VALUES, META), BBOX, SIZE, 24)
        manifest = {"coverage": {"frames": [{"cell": "a", "image": current}]}}
        self.assertEqual(to_remove([current, previous], named_by(manifest), self.NOW, 180), [previous])


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

    def write(self, held, previous=None):
        return self.observed.write_coverage(self.out, self.cells, SIZE, held, previous)

    def drawn(self):
        return sorted(n for n in os.listdir(self.out) if n.endswith(".png"))

    def test_the_held_step_is_drawn_and_named_with_its_time(self):
        block = self.write((self.WHEN, VALUES, META))
        self.assertEqual(self.reads, [], "the step the run read is never downloaded again")
        self.assertEqual(block["observedTime"], "2026-10-26T15:00:00Z")
        self.assertEqual([e["cell"] for e in block["frames"]], ["gulf-a", "wide-gulf-a"])
        self.assertEqual(sorted(e["image"] for e in block["frames"]), self.drawn())

    def test_a_still_coverage_draws_nothing_the_next_run(self):
        first = self.write((self.WHEN, VALUES, META))
        drawn = []
        saved = self.observed.unseen
        self.observed.unseen = lambda *a, **k: drawn.append(1) or saved(*a, **k)
        try:
            rain = VALUES.copy()
            rain[40:60, 20:60] = 45.0
            later = self.WHEN + datetime.timedelta(minutes=15)
            second = self.write((later, rain, META))
        finally:
            self.observed.unseen = saved
        self.assertEqual(drawn, [], "a mask on disk was drawn again")
        self.assertEqual(second["frames"], first["frames"])
        self.assertEqual(second["observedTime"], "2026-10-26T15:15:00Z")

    def test_a_moved_coverage_draws_and_names_a_new_set(self):
        first = self.write((self.WHEN, VALUES, META))
        moved = VALUES.copy()
        moved[:, 90:100] = -999.0
        second = self.write((self.WHEN, moved, META))
        self.assertTrue(set(e["image"] for e in first["frames"]).isdisjoint(e["image"] for e in second["frames"]))
        self.assertEqual(len(self.drawn()), 4)

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
        self.assertEqual(self.reads, [], "a run that read no step downloads nothing for the masks")

    def test_a_mask_that_fails_is_left_out_and_leaves_no_file(self):
        saved = self.observed.unseen

        def fail_close(values, meta, bbox, size, *rest):
            if size == SIZE:
                raise ValueError("planted")
            return saved(values, meta, bbox, size, *rest)
        self.observed.unseen = fail_close
        try:
            block = self.write((self.WHEN, VALUES, META))
        finally:
            self.observed.unseen = saved
        self.assertEqual([e["cell"] for e in block["frames"]], ["wide-gulf-a"])
        self.assertEqual(len(self.drawn()), 1)

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
        self.assertIn("fx, fy, inside = pixel_coords(meta, bbox, size, MERCATOR_R)", src)


class MainTests(unittest.TestCase):
    """observed.main end to end on the synthetic mosaic, NOAA stubbed: what each run names, reads and draws.

    The batch 9 build reader drove main through seven cases by hand and asked for one in the suite.
    """

    T1 = datetime.datetime(2026, 10, 26, 14, 50, tzinfo=datetime.timezone.utc)
    T2 = datetime.datetime(2026, 10, 26, 15, 0, tzinfo=datetime.timezone.utc)
    T3 = datetime.datetime(2026, 10, 26, 15, 10, tzinfo=datetime.timezone.utc)

    def setUp(self):
        self.stand_ins = {name: sys.modules.pop(name) for name in ("observed", "render")
                          if name in sys.modules and getattr(sys.modules[name], "__file__", None) is None}
        import observed
        self.observed = observed
        self.saved = (observed.read_frame, observed.wanted_keys, observed.multiprocessing)
        self.dir = tempfile.TemporaryDirectory()
        self.reads, self.grids = [], {}
        observed.read_frame = lambda key: (self.reads.append(key), (self.grids[key], META, 0))[1]
        # The runner forks; the Mac spawns, and a spawned worker has none of the grid the pool was set up with.
        observed.multiprocessing = types.SimpleNamespace(Pool=multiprocessing.get_context("fork").Pool)
        self.cell = next(name for name, _ in observed.cells()
                         if name != observed.NATIONAL_ID and not name.startswith("wide-"))

    def tearDown(self):
        self.observed.read_frame, self.observed.wanted_keys, self.observed.multiprocessing = self.saved
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
        self.grids = {"k1": VALUES, "k2": VALUES}
        first = self.run_main([(self.T1, "k1"), (self.T2, "k2")])
        self.assertEqual(self.reads, ["k1", "k2"])
        coverage = first["coverage"]
        self.assertEqual(coverage["observedTime"], "2026-10-26T15:00:00Z")
        self.assertEqual([f["cell"] for f in coverage["frames"]], [self.cell])
        mask = coverage["frames"][0]["image"]
        with Image.open(os.path.join(self.dir.name, mask)) as im:
            self.assertEqual(im.size, (240, 192))

        # Nothing new from NOAA: no read, the frames and the block as they were.
        second = self.run_main([(self.T1, "k1"), (self.T2, "k2")])
        self.assertEqual(self.reads, [])
        self.assertEqual(second["coverage"], coverage)

        # A new step whose no-coverage area moved: read once, a new mask named with the new step's time.
        moved = VALUES.copy()
        moved[:, 90:100] = -999.0
        self.grids["k3"] = moved
        third = self.run_main([(self.T1, "k1"), (self.T2, "k2"), (self.T3, "k3")])
        self.assertEqual(self.reads, ["k3"])
        self.assertEqual(third["coverage"]["observedTime"], "2026-10-26T15:10:00Z")
        self.assertNotEqual(third["coverage"]["frames"][0]["image"], mask)
        self.assertEqual(named_by(third) & {mask}, set(), "the replaced mask is still named")

    def test_a_new_step_over_a_still_coverage_draws_nothing(self):
        self.grids = {"k1": VALUES, "k2": VALUES}
        first = self.run_main([(self.T1, "k1"), (self.T2, "k2")])
        rain = VALUES.copy()
        rain[40:60, 20:60] = 45.0
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


if __name__ == "__main__":
    unittest.main()
