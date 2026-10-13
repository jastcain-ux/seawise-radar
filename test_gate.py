"""The forecast gate, pinned (radar build 7, item 11a; B-263). Pure; no network."""
import unittest
from gate import decide, run_key, latest_key

CELLS = {"gulf-west", "gulf-east", "national"}
STYLE = {"rampSteps": 96, "quantiser": "rgba", "palette": "twc-2-bands-r3",
         "smoothing": 0.7, "alpha": 255}
EXPECTED = 36 * 3


def manifest(run="2026-10-13T00:00:00Z", frames=EXPECTED, cells=CELLS, **style):
    m = {"runTime": run, "cells": [{"id": c} for c in cells],
         "frames": [{"i": i} for i in range(frames)]}
    m.update(STYLE)
    m.update(style)
    return m


class GateTests(unittest.TestCase):

    def test_run_keys_compare_as_strings(self):
        self.assertEqual(run_key("2026-10-13T00:00:00Z"), "2026101300")
        self.assertEqual(latest_key("20261013", 1), "2026101301")
        self.assertLess(run_key("2026-10-13T00:00:00Z"), latest_key("20261013", 1))
        self.assertLess(run_key("2026-10-12T23:00:00Z"), latest_key("20261013", 0))

    def test_nothing_published_renders(self):
        self.assertTrue(decide(None, CELLS, STYLE, "2026101301", EXPECTED)[0])

    def test_the_same_run_whole_skips(self):
        # The case the old age gate got wrong: the newest complete run is
        # always over 90 minutes old, so it re-rendered the run already live.
        render, reason = decide(manifest(), CELLS, STYLE, "2026101300", EXPECTED)
        self.assertFalse(render, reason)

    def test_a_newer_run_renders(self):
        render, reason = decide(manifest(), CELLS, STYLE, "2026101301", EXPECTED)
        self.assertTrue(render)
        self.assertIn("newer run", reason)

    def test_a_short_set_renders_so_a_skipped_lead_is_retried(self):
        render, reason = decide(manifest(frames=EXPECTED - 3), CELLS, STYLE, "2026101300", EXPECTED)
        self.assertTrue(render)
        self.assertIn("short", reason)

    def test_a_changed_layout_renders_whatever_the_run(self):
        self.assertTrue(decide(manifest(cells={"gulf-west", "national"}), CELLS, STYLE,
                               "2026101300", EXPECTED)[0])

    def test_a_changed_colour_rule_renders_whatever_the_run(self):
        for key, value in [("palette", "twc-2-bands-r4"), ("rampSteps", 48),
                           ("quantiser", "rgb"), ("smoothing", 0.5), ("alpha", 200)]:
            render, reason = decide(manifest(**{key: value}), CELLS, STYLE, "2026101300", EXPECTED)
            self.assertTrue(render, key)
            self.assertIn(key, reason)

    def test_nomads_down_skips_rather_than_re_rendering(self):
        render, reason = decide(manifest(), CELLS, STYLE, None, EXPECTED)
        self.assertFalse(render)
        self.assertIn("NOMADS", reason)

    def test_nomads_down_with_nothing_published_still_renders(self):
        # render.py then fails on its own and leaves no manifest, as today.
        self.assertTrue(decide(None, CELLS, STYLE, None, EXPECTED)[0])

    def test_a_published_run_newer_than_nomads_answer_skips(self):
        # NOMADS listing an older run than the one live (a directory gone
        # missing) must not roll the picture back.
        render, _ = decide(manifest(run="2026-10-13T02:00:00Z"), CELLS, STYLE, "2026101301", EXPECTED)
        self.assertFalse(render)


if __name__ == "__main__":
    unittest.main()
