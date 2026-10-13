"""Names and the prune's grace, pinned (radar build 7, item 21; B-258, B-272). Pure; no network."""
import datetime
import unittest

from naming import publication_stamp, style_revision, stamp_in, stamp_age_minutes
from prune import to_remove

NOW = datetime.datetime(2026, 10, 13, 1, 50, tzinfo=datetime.timezone.utc)


class NamingTests(unittest.TestCase):

    def test_the_publication_stamp(self):
        self.assertEqual(publication_stamp(NOW), "20261013T0150Z")

    def test_the_style_revision_changes_with_the_rule_and_not_with_key_order(self):
        a = style_revision({"palette": "twc-2-bands-r4", "alpha": 255})
        b = style_revision({"alpha": 255, "palette": "twc-2-bands-r4"})
        c = style_revision({"palette": "twc-2-bands-r3", "alpha": 255})
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len(a), 6)

    def test_stamps_are_read_from_every_products_names(self):
        self.assertEqual(stamp_in("gulf-east-refc-0015-20261013T0120Z.png"), ("20261013", "0120"))
        self.assertEqual(stamp_in("downeast-20261012T2320Z-a1b2c3.png"), ("20261012", "2320"))
        self.assertEqual(stamp_in("national-20261013T0120Z-015.png"), ("20261013", "0120"))
        self.assertIsNone(stamp_in("gulf-east-refc-0015.png"))
        self.assertIsNone(stamp_in("manifest.json"))

    def test_the_age_counts_from_the_stamp(self):
        self.assertAlmostEqual(stamp_age_minutes("x-20261013T0120Z.png", NOW), 30.0)
        self.assertIsNone(stamp_age_minutes("x-refc-0015.png", NOW))


class PruneTests(unittest.TestCase):

    KEEP = {"manifest.json", "a-refc-0015-20261013T0137Z.png"}

    def test_what_the_manifest_names_stays(self):
        self.assertEqual(to_remove(["a-refc-0015-20261013T0137Z.png", "manifest.json"], self.KEEP, NOW, 60), [])

    def test_the_previous_set_stays_an_hour(self):
        young = "a-refc-0015-20261013T0120Z.png"      # 30 minutes old
        old = "a-refc-0015-20261013T0020Z.png"        # 90 minutes old
        self.assertEqual(to_remove([young, old], self.KEEP, NOW, 60), [old])

    def test_with_no_grace_everything_unnamed_goes_as_before(self):
        young = "a-refc-0015-20261013T0120Z.png"
        self.assertEqual(to_remove([young], self.KEEP, NOW, 0), [young])

    def test_a_name_with_no_stamp_goes_when_unnamed(self):
        self.assertEqual(to_remove(["a-refc-0015.png"], self.KEEP, NOW, 60), ["a-refc-0015.png"])

    def test_measured_frames_keep_three_hours_of_stamps(self):
        # The observed stamps are the frames' own times; a phone's manifest can name one up to about an hour
        # past the two-hour window, so the measured prune runs with a three-hour grace.
        names = ["downeast-20261012T2300Z-abc123.png",   # 170 minutes old: kept
                 "downeast-20261012T2240Z-abc123.png"]   # 190 minutes old: gone
        self.assertEqual(to_remove(names, self.KEEP, NOW, 180), ["downeast-20261012T2240Z-abc123.png"])


if __name__ == "__main__":
    unittest.main()
