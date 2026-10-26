"""Names and the prune's grace, pinned (radar build 7, item 21; B-258, B-272, B-282). Pure; no network."""
import datetime
import os
import re
import unittest

from naming import field_image, publication_stamp, style_revision, stamp_in, stamp_age_minutes
from prune import named_by, to_remove

HERE = os.path.dirname(os.path.abspath(__file__))

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



class FieldNameTests(unittest.TestCase):
    """Cloud and wind pictures named by their publication (B-282)."""

    def test_a_wind_name_carries_cell_lead_and_publication(self):
        name = field_image("national", "wind", 60, "20261014T0519Z")
        self.assertEqual(name, "national-wind-0060-20261014T0519Z.png")
        self.assertEqual(stamp_in(name), ("20261014", "0519"))

    def test_a_cloud_name_too_and_lead_zero_keeps_four_digits(self):
        name = field_image("gulf-east", "cloud", 0, "20261014T0519Z")
        self.assertEqual(name, "gulf-east-cloud-0000-20261014T0519Z.png")
        self.assertEqual(stamp_in(name), ("20261014", "0519"))

    def test_the_stamp_is_the_publication_stamp(self):
        published = datetime.datetime(2026, 10, 14, 5, 19, 42, tzinfo=datetime.timezone.utc)
        self.assertEqual(field_image("wide-gulf-east", "wind", 360, publication_stamp(published)),
                         "wide-gulf-east-wind-0360-20261014T0519Z.png")


class NamedByTests(unittest.TestCase):

    def test_the_top_level_and_the_regional_tier_are_both_named(self):
        manifest = {"frames": [{"image": "a.png"}], "regional": {"cells": [], "frames": [{"image": "r.png"}]}}
        self.assertEqual(named_by(manifest), {"a.png", "r.png"})

    def test_without_a_regional_tier_the_frames_alone(self):
        self.assertEqual(named_by({"frames": [{"image": "a.png"}]}), {"a.png"})
        self.assertEqual(named_by({}), set())

    def test_a_named_regional_picture_older_than_the_window_stays(self):
        # NOMADS down for hours: the manifest still names its regional pictures, so they must not go.
        regional = "regional-25n-90w-wind-0060-20261012T2200Z.png"     # 230 minutes old
        manifest = {"frames": [{"image": "national-wind-0060-20261012T2200Z.png"}],
                    "regional": {"frames": [{"image": regional}]}}
        keep = named_by(manifest) | {"manifest.json"}
        self.assertEqual(to_remove([regional, "national-wind-0060-20261012T2200Z.png", "manifest.json"],
                                   keep, NOW, 135, from_replacement=True, unstamped_as_old_as_stamped=True), [])
        self.assertEqual(to_remove([regional], keep, NOW, 70), [])


class CloudWindPruneTests(unittest.TestCase):
    """The cloud and wind prune: kept from replacement, and the changeover (B-282; the plan reader's S1)."""

    CURRENT = "national-wind-0060-20261013T0120Z.png"      # 30 minutes old, named
    KEEP = {"manifest.json", CURRENT}
    REPLACED = "national-wind-0060-20261012T2320Z.png"     # 150 minutes old, replaced 30 minutes ago
    UNSTAMPED = "national-wind-0060.png"

    def prune(self, names, **flags):
        return to_remove(names + ["manifest.json"], self.KEEP, NOW, 135, **flags)

    def test_a_long_gap_never_deletes_the_set_just_replaced(self):
        # Published 150 minutes apart: the previous set is past the window by its own stamp, but it was
        # replaced only now, and a phone that fetched its manifest a moment ago still asks for it.
        current = "national-wind-0060-20261013T0150Z.png"     # just published
        previous = "national-wind-0060-20261012T2320Z.png"    # 150 minutes before it
        older = "national-wind-0060-20261012T2120Z.png"       # replaced by `previous` 150 minutes ago
        names = [current, previous, older, "manifest.json"]
        keep = {"manifest.json", current}
        self.assertEqual(to_remove(names, keep, NOW, 135, from_replacement=True), [older])
        # Timed from its own stamp, it would go in the run that replaced it.
        self.assertEqual(to_remove(names, keep, NOW, 135), [previous, older])

    def test_a_set_leaves_135_minutes_after_it_was_replaced(self):
        replaced_136 = "national-wind-0060-20261012T2200Z.png"   # replaced by the one below
        replacer = "national-wind-0060-20261012T2334Z.png"       # 136 minutes old, itself replaced at 0120
        self.assertEqual(self.prune([self.CURRENT, replacer, replaced_136], from_replacement=True),
                         [replaced_136])

    def test_the_newest_unnamed_set_with_nothing_newer_stays(self):
        # A step that died before its manifest leaves a newer set nothing names; nothing has replaced it.
        orphan = "national-wind-0060-20261012T2000Z.png"
        self.assertEqual(to_remove([orphan], {"manifest.json"}, NOW, 135, from_replacement=True), [])

    def test_the_unstamped_names_take_the_oldest_stamped_age(self):
        # The plan's case, timed from the stamp: the 150-minute set and the unstamped names go.
        self.assertEqual(self.prune([self.CURRENT, self.REPLACED, self.UNSTAMPED], unstamped_as_old_as_stamped=True),
                         [self.REPLACED, self.UNSTAMPED])
        # As the workflow runs it, from replacement: the 150-minute set was replaced 30 minutes ago and
        # stays; the unstamped names, replaced by the first stamped set 150 minutes ago, go.
        self.assertEqual(self.prune([self.CURRENT, self.REPLACED, self.UNSTAMPED],
                                    from_replacement=True, unstamped_as_old_as_stamped=True),
                         [self.UNSTAMPED])

    def test_the_unstamped_names_stay_while_the_first_stamped_set_is_young(self):
        self.assertEqual(self.prune([self.CURRENT, self.UNSTAMPED],
                                    from_replacement=True, unstamped_as_old_as_stamped=True), [])

    def test_the_unstamped_names_stay_with_no_stamped_name_at_all(self):
        self.assertEqual(to_remove([self.UNSTAMPED, "manifest.json"], {"manifest.json"}, NOW, 135,
                                   from_replacement=True, unstamped_as_old_as_stamped=True), [])

    def test_without_the_flag_an_unstamped_unnamed_name_goes_at_once(self):
        self.assertEqual(self.prune([self.CURRENT, self.UNSTAMPED]), [self.UNSTAMPED])


class ForecastPruneTests(unittest.TestCase):
    """The radar's own forecast prunes, timed from replacement too (B-293)."""

    def read(self, path):
        with open(os.path.join(HERE, path)) as f:
            return f.read()

    def test_a_long_gap_never_deletes_the_radar_set_just_replaced(self):
        # NOMADS down: the forecast published 150 minutes after the last. A phone that fetched the previous
        # manifest a moment ago still asks for its pictures; they stay, and the set before them goes.
        current = "gulf-east-refc-0015-20261013T0150Z.png"
        previous = "gulf-east-refc-0015-20261012T2320Z.png"
        older = "gulf-east-refc-0015-20261012T2220Z.png"      # replaced by `previous` 150 minutes ago
        national = "national-refc-0015-20261013T0150Z.png"
        names = [current, national, previous, older, "manifest.json"]
        keep = {"manifest.json", current, national}
        self.assertEqual(to_remove(names, keep, NOW, 135, from_replacement=True), [older])
        # Timed from its own stamp, as before B-293, the set just replaced went in the same run.
        self.assertEqual(to_remove(names, keep, NOW, 135), [previous, older])

    def test_the_regional_tier_keeps_its_replaced_set_after_a_long_gap(self):
        current = "regional-25n-90w-refc-0060-20261013T0150Z.png"
        previous = "regional-25n-90w-refc-0060-20261012T2320Z.png"
        names = [current, previous, "manifest.json"]
        self.assertEqual(to_remove(names, {"manifest.json", current}, NOW, 135, from_replacement=True), [])

    def test_an_unnamed_unstamped_name_in_the_forecast_root_still_goes(self):
        # Only the cloud and wind changeover keeps unstamped names; the radar root never needed it.
        current = "gulf-east-refc-0015-20261013T0150Z.png"
        self.assertEqual(to_remove([current, "gulf-east-refc-0015.png", "manifest.json"],
                                   {"manifest.json", current}, NOW, 135, from_replacement=True),
                         ["gulf-east-refc-0015.png"])

    def test_both_forecast_workflows_prune_from_replacement(self):
        radar = self.read(os.path.join(".github", "workflows", "radar.yml"))
        self.assertIn("python prune.py --dir public --keep-newer-than 135 --from-replacement\n", radar)
        regional = self.read(os.path.join(".github", "workflows", "forecast-regional.yml"))
        self.assertIn("python prune.py --dir site/forecast-regional --keep-newer-than 135 --from-replacement\n", regional)

    def test_a_failed_forecast_render_carries_no_partial_set(self):
        # The edge B-282's build named: a partial newer set would count as the replacer of the set before it.
        # The radar root and the regional tier never carry one: their render steps stop the job, and the
        # carry step after them has no `if:` that would run it anyway.
        radar = self.read(os.path.join(".github", "workflows", "radar.yml"))
        render = radar.index("- name: Render forecast frames when NOMADS has a newer run")
        carry = radar.index("- name: Carry the frames to the next run\n")
        self.assertLess(render, carry)
        self.assertNotIn("continue-on-error", radar[render:radar.index("- name: Start the regional tier")])
        self.assertNotIn("if:", radar[carry:radar.index("- name: Publish everything")])
        regional = self.read(os.path.join(".github", "workflows", "forecast-regional.yml"))
        step = regional[regional.index("- name: Render the regional tier"):regional.index("- name: Prune pictures")]
        self.assertNotIn("continue-on-error", step)
        carry = regional[regional.index("- name: Carry the tier"):regional.index("- name: Publish the tier")]
        self.assertNotIn("if:", carry)


class FieldSourceTests(unittest.TestCase):
    """cloud.py and wind.py import eccodes, so their naming is checked in their source (B-282)."""

    def read(self, path):
        with open(os.path.join(HERE, path)) as f:
            return f.read()

    def check_field_script(self, path, kind):
        src = self.read(path)
        self.assertIn("published_at = datetime.datetime.now(datetime.timezone.utc)", src)
        self.assertIn("stamp = publication_stamp(published_at)", src)
        self.assertIn(f'field_image(name, "{kind}", minutes, stamp)', src)
        self.assertIn('"generatedAt": published_at.strftime(', src)
        self.assertIsNone(re.search(r'f"\{name\}-' + kind + r'-\{minutes:04d\}\.png"', src),
                          f"{path} writes an unstamped name")
        self.assertEqual(src.count("datetime.datetime.now("), 1, f"{path} reads the clock once")

    def test_cloud_names_its_pictures_by_the_publication(self):
        self.check_field_script("cloud.py", "cloud")

    def test_wind_names_its_pictures_by_the_publication(self):
        self.check_field_script("wind.py", "wind")

    def test_the_workflow_prunes_cloud_and_wind_from_replacement(self):
        src = self.read(os.path.join(".github", "workflows", "radar.yml"))
        for d in ("public/cloud", "public/wind"):
            self.assertIn(f"python prune.py --dir {d} --keep-newer-than 135 --from-replacement "
                          f"--unstamped-as-old-as-stamped", src)


if __name__ == "__main__":
    unittest.main()
