"""The smoothing helper, pinned. Pure numpy; runs without eccodes."""
import unittest
import numpy as np
from smoothing import smooth, shape, restore_peaks, sigma_px, OUTSIDE
from edges import edge_weight, inside_mask
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
