"""The smoothing helper, pinned. Pure numpy; runs without eccodes."""
import unittest
import numpy as np
from smoothing import smooth, shape, restore_peaks, sigma_px, OUTSIDE
from edges import edge_weight, inside_mask
from peaks import band_tops, pixels_of, halo_bbox
from scipy import ndimage

BBOX = (-100.0, 25.0, -90.0, 32.0)
SIZE = (2000, 1400)          # 200 px per degree, the forecast close tier
HRRR = 0.027                 # degrees per 3 km cell


class SmoothingTests(unittest.TestCase):

    def test_sigma_is_a_fraction_of_one_model_cell_in_output_pixels(self):
        self.assertAlmostEqual(sigma_px(HRRR, BBOX, SIZE, 0.3), 0.3 * HRRR * 200, places=6)

    def test_zero_strength_changes_nothing(self):
        f = np.random.default_rng(1).uniform(-10, 60, (80, 80))
        out, before, after = smooth(f, HRRR, BBOX, SIZE, 0.0)
        np.testing.assert_array_equal(out, f)
        self.assertEqual(before, after)

    def test_a_broad_core_keeps_its_peak(self):
        f = np.full((200, 200), 20.0); f[80:120, 80:120] = 60.0
        out, before, after = smooth(f, HRRR, BBOX, SIZE, 0.3)
        self.assertEqual(before, 60.0)
        self.assertAlmostEqual(after, 60.0, places=5)
        self.assertAlmostEqual(out.max(), 60.0, places=5)

    def test_outside_the_grid_stays_outside_and_never_bleeds_in(self):
        f = np.full((100, 100), 40.0); f[:, 50:] = OUTSIDE
        out, _, _ = smooth(f, HRRR, BBOX, SIZE, 0.3)
        np.testing.assert_array_equal(out[:, 50:], OUTSIDE)
        self.assertGreaterEqual(out[:, :50].min(), 40.0 - 1e-6,
                                "a -99 sentinel next to real data used to ring a dark rim round every cell")

    def test_the_cost_on_a_single_model_cell_is_reported_not_hidden(self):
        # One 3 km cell at 70 dBZ over 20: the case the measured half refuses
        # to smooth (strength 0). The helper must report the loss so the run
        # log shows it rather than a frame quietly understating a core.
        f = np.full((60, 60), 20.0); f[27:33, 27:33] = 70.0
        _, before, after = smooth(f, HRRR, BBOX, SIZE, 0.3)
        self.assertEqual(before, 70.0)
        self.assertLess(after, 70.0)
        self.assertGreater(after, 55.0)


    def test_shape_puts_the_peak_back_on_a_single_model_cell(self):
        f = np.full((60, 60), 20.0); f[27:33, 27:33] = 70.0
        _, _, lost = smooth(f, HRRR, BBOX, SIZE, 0.7)
        out, before, after = shape(f, HRRR, BBOX, SIZE, 0.7)
        self.assertLess(lost, 66.0, "a plain 0.7-cell smooth takes several dBZ off one cell")
        self.assertGreaterEqual(after, before - 0.3, "with the peaks restored the core keeps its colour")
        self.assertLessEqual(out.max(), before + 0.5, "and nothing is invented above it")

    def test_restoring_peaks_adds_no_plateaus(self):
        # A textured field: the restoration must not paint flat squares. Count
        # 3x3 neighbourhoods of the added bump that are exactly flat but non-zero
        # — a maximum filter makes thousands; point bumps make none.
        rng = np.random.default_rng(5)
        f = ndimage.gaussian_filter(rng.uniform(10, 60, (120, 120)), 1.5)
        sm = ndimage.gaussian_filter(f, sigma_px(HRRR, BBOX, SIZE, 0.7), mode="nearest")
        bump = restore_peaks(f, sm, HRRR * 200) - sm
        flat = 0
        for y in range(1, 119):
            for x in range(1, 119):
                w = bump[y-1:y+2, x-1:x+2]
                if w.max() > 0.05 and (w.max() - w.min()) < 1e-9:
                    flat += 1
        self.assertEqual(flat, 0)

    def test_restoring_peaks_invents_nothing_above_the_field(self):
        rng = np.random.default_rng(7)
        f = ndimage.gaussian_filter(rng.uniform(10, 65, (120, 120)), 1.0)
        sm = ndimage.gaussian_filter(f, sigma_px(HRRR, BBOX, SIZE, 0.7), mode="nearest")
        out = restore_peaks(f, sm, HRRR * 200)
        self.assertTrue(np.all(out <= np.maximum(f, sm) + 1e-9), "a restored pixel never exceeds the field")
        self.assertTrue(np.all(out >= sm - 1e-9), "and never falls below the smooth")

    def test_a_small_yellow_core_stays_yellow(self):
        # 38 dBZ on one model cell over 20: a plain smooth takes it under 35,
        # the first yellow band, and it paints green. The restore must keep it.
        f = np.full((60, 60), 20.0); f[27:33, 27:33] = 38.0
        _, _, plain = smooth(f, HRRR, BBOX, SIZE, 0.7)
        out, _, after = shape(f, HRRR, BBOX, SIZE, 0.7)
        self.assertLess(plain, 35.0, "the plain smooth does drop it out of yellow")
        self.assertGreaterEqual(after, 38.0 - 0.3, "the restore keeps the core in yellow")

    def test_restoring_peaks_never_lowers_a_pixel(self):
        rng = np.random.default_rng(3)
        f = rng.uniform(10, 60, (80, 80)); sm = shape(f, HRRR, BBOX, SIZE, 0.7)[0]
        from scipy import ndimage
        plain = ndimage.gaussian_filter(f, sigma_px(HRRR, BBOX, SIZE, 0.7), mode="nearest")
        self.assertTrue(np.all(sm >= plain - 1e-6))

    def test_shape_keeps_the_outside_outside(self):
        f = np.full((100, 100), 40.0); f[:, 50:] = OUTSIDE
        out, _, _ = shape(f, HRRR, BBOX, SIZE, 0.7)
        np.testing.assert_array_equal(out[:, 50:], OUTSIDE)
        self.assertGreaterEqual(out[:, :50].min(), 40.0 - 1e-6)


class PeakTests(unittest.TestCase):
    """Every core's top, found once per step (radar build 7, item 19; B-265); D-244's clamp (B-270)."""

    EDGES = [35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0]

    def test_two_cores_in_one_area_both_keep_their_tops(self):
        # Two 45 dBZ cores on one 40 dBZ line: one area at 35 and at 40, two
        # at 45. Today's rule restored one top; both are found now.
        f = np.full((60, 60), 20.0); f[28:32, 10:50] = 41.0; f[29:31, 15:17] = 45.0; f[29:31, 43:45] = 45.5
        rows, cols = band_tops(f, self.EDGES)
        places = set(zip(rows.tolist(), cols.tolist()))
        self.assertIn((29, 43), places, "the higher core is the area's top")
        self.assertTrue(any(r in (29, 30) and c in (15, 16) for r, c in places), "the second core is found too")

    def test_nothing_the_old_rule_restored_is_lost(self):
        # A 38 dBZ top that is part of a bigger core's area at 35: the old
        # rule gave that area one top, the big core's; so does this one, and
        # the small top is not separately restored, exactly as before.
        rng = np.random.default_rng(11)
        f = ndimage.gaussian_filter(rng.uniform(10, 60, (150, 150)), 2.0)
        areas, n = ndimage.label(f >= 35.0)
        old = set(tuple(map(int, t)) for t in ndimage.maximum_position(f, areas, index=list(range(1, n + 1))))
        rows, cols = band_tops(f, self.EDGES)
        new = set(zip(rows.tolist(), cols.tolist()))
        self.assertTrue(old <= new, f"{len(old - new)} of the old rule's tops lost")
        self.assertGreater(n, 5)

    def test_restoring_at_given_places_matches_the_picture_rule_for_one_core(self):
        f = np.full((60, 60), 20.0); f[27:33, 27:33] = 70.0
        sm = ndimage.gaussian_filter(f, sigma_px(HRRR, BBOX, SIZE, 0.7), mode="nearest")
        own = restore_peaks(f, sm, HRRR * 200)
        given = restore_peaks(f, sm, HRRR * 200, tops=band_tops(f, self.EDGES))
        np.testing.assert_allclose(given, own, atol=1e-9)
        np.testing.assert_array_equal(restore_peaks(f, sm, HRRR * 200, tops=(np.array([]), np.array([]))), sm,
                                      "no tops given: nothing restored")

    def test_the_clamp_keeps_a_light_shower_painted(self):
        # A 25 dBZ shower one model cell across: the smooth alone takes it
        # under the paint floor (B-270: 85% of such showers drawn as dry).
        f = np.full((60, 60), -30.0); f[29:31, 29:31] = 25.0
        _, _, plain = smooth(f, HRRR, BBOX, SIZE, 0.7)
        bare = shape(f, HRRR, BBOX, SIZE, 0.7)[0]
        held = shape(f, HRRR, BBOX, SIZE, 0.7, painted=(15.0, 15.5))[0]
        self.assertLess(bare[29:31, 29:31].max(), 15.0, "the plain shaping does drop it below the floor")
        self.assertGreaterEqual(held[29:31, 29:31].min(), 15.5, "the clamp keeps every painted pixel painted")
        self.assertTrue(np.all(held[29:31, 29:31] <= 25.0 + 1e-9), "and never above the field")
        outside = np.ones_like(f, dtype=bool); outside[29:31, 29:31] = False
        np.testing.assert_array_equal(held[outside] >= 15.5, bare[outside] >= 15.5,
                                      "a pixel the field did not paint is not painted by the clamp")

    def test_the_clamp_never_lifts_an_unpainted_pixel(self):
        f = np.full((40, 40), 14.0); f[20, 20] = 14.9
        held = shape(f, HRRR, BBOX, SIZE, 0.7, painted=(15.0, 15.5))[0]
        self.assertLess(held.max(), 15.0)

    def test_tops_land_on_the_pictures_own_grid(self):
        bbox = (-100.0, 25.0, -90.0, 32.0); size = (2000, 1400)
        ys, xs = pixels_of([32.0, 25.0, 28.5, 40.0], [-100.0, -90.0, -95.0, -95.0], bbox, size)
        self.assertEqual(xs.tolist(), [0, 1999, 1000], "west edge, east edge, the middle; the 40N point is off the picture")
        self.assertEqual(ys[0], 0); self.assertEqual(ys[1], 1399)
        self.assertTrue(0 < ys[2] < 1399)

    def test_the_halo_grows_the_box_by_whole_pixels(self):
        bbox = (-100.0, 25.0, -90.0, 32.0); size = (2000, 1400)
        grown, grown_size = halo_bbox(bbox, size, 24)
        self.assertEqual(grown_size, (2048, 1448))
        dx = 10.0 / 1999
        self.assertAlmostEqual(grown[0], -100.0 - 24 * dx, places=9)
        self.assertAlmostEqual(grown[2], -90.0 + 24 * dx, places=9)
        self.assertLess(grown[1], 25.0); self.assertGreater(grown[3], 32.0)
        # The original box's corners land exactly 24 px in on the grown picture.
        ys, xs = pixels_of([32.0, 25.0], [-100.0, -90.0], grown, grown_size)
        self.assertEqual(xs.tolist(), [24, 2023]); self.assertEqual(ys.tolist(), [24, 1423])


class EdgeTests(unittest.TestCase):
    """The model's edge (radar build 7, item 14; B-262, B-271)."""

    def test_the_weight_is_one_inside_and_zero_outside(self):
        fx = np.array([-1.0, 0.0, 5.0, 98.0, 99.0, 120.0])
        fy = np.full(6, 50.0)
        np.testing.assert_array_equal(edge_weight(fx, fy, 100, 100, None), [0, 1, 1, 1, 0, 0])

    def test_the_weight_fades_over_the_last_cells_on_every_side(self):
        # Three cells from the west edge: 0, 1/3, 2/3, 1; the same from the south.
        w = edge_weight(np.array([0.0, 1.0, 2.0, 3.0, 50.0]), np.full(5, 50.0), 100, 100, 3)
        np.testing.assert_allclose(w, [0, 1 / 3, 2 / 3, 1, 1])
        w = edge_weight(np.full(4, 50.0), np.array([98.0, 97.0, 96.0, 95.0]), 100, 100, 3)
        np.testing.assert_allclose(w, [1 / 3, 2 / 3, 1, 1])

    def test_outside_the_grid_weighs_nothing_whatever_the_fade(self):
        self.assertEqual(edge_weight(np.array([-0.5]), np.array([10.0]), 100, 100, 3)[0], 0.0)
        self.assertEqual(edge_weight(np.array([10.0]), np.array([99.5]), 100, 100, 3)[0], 0.0)

    def test_the_inside_mask_marks_the_edge_row_smear(self):
        col = np.array([-3.0, 0.0, 10.0, 99.0, 99.5])
        row = np.full(5, 10.0)
        np.testing.assert_array_equal(inside_mask(col, row, 100, 100), [False, True, True, True, False])
        np.testing.assert_array_equal(inside_mask(np.full(3, 10.0), np.array([-1.0, 50.0, 100.0]), 100, 100),
                                      [False, True, False])


if __name__ == "__main__":
    unittest.main()
