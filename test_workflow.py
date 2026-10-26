"""The render workflow's jobs, pinned (SeaWise B-20: the nowcast in a job of its own). Pure; no network, no YAML library.

The runner has no YAML parser installed, so the jobs are read off the file's own layout: a job is a two-space
key under `jobs:`, its `needs:` and `if:` are four-space keys, its steps are six-space list items.
"""
import os
import re
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


def jobs_of(path):
    """{job: {"needs": [..], "if": str or None, "text": its lines, "steps": [step names or uses]}}."""
    with open(os.path.join(HERE, path)) as f:
        lines = f.read().split("\n")
    start = lines.index("jobs:")
    jobs, current = {}, None
    for line in lines[start + 1:]:
        m = re.match(r"^  ([a-z0-9-]+):$", line)
        if m:
            current = jobs.setdefault(m.group(1), {"needs": [], "if": None, "text": [], "steps": []})
            continue
        if current is None:
            continue
        current["text"].append(line)
        m = re.match(r"^    needs: (.+)$", line)
        if m:
            current["needs"] = [n.strip() for n in m.group(1).strip("[]").split(",")]
        m = re.match(r"^    if: (.+)$", line)
        if m:
            current["if"] = m.group(1)
        m = re.match(r"^      - (?:name|uses): (.+)$", line)
        if m:
            current["steps"].append(m.group(1))
    for job in jobs.values():
        job["text"] = "\n".join(job["text"])
    return jobs


class NowcastJobTests(unittest.TestCase):

    def setUp(self):
        self.jobs = jobs_of(os.path.join(".github", "workflows", "radar.yml"))

    def test_the_measured_half_publishes_without_waiting_for_the_nowcast(self):
        measured = self.jobs["measured"]
        self.assertNotIn("nowcast.py", measured["text"])
        self.assertNotIn("public/nowcast", measured["text"])
        self.assertIn("python prune.py --dir public/observed --keep-newer-than 180", measured["text"])
        self.assertEqual(self.jobs["deploy-measured"]["needs"], ["measured"])
        self.assertEqual(self.jobs["r2-measured"]["needs"], ["measured"])

    def test_the_nowcast_job_draws_from_the_measured_files_and_prunes_after(self):
        nowcast = self.jobs["nowcast"]
        self.assertEqual(nowcast["needs"], ["measured"])
        steps = nowcast["steps"]
        order = ["Take the files from the measured job",
                 "Render the nowcast from the last two measured frames",
                 "Prune projected frames the timeline can no longer reach, an hour after",
                 "Carry the frames to the next run (nowcast)",
                 "Hand the files to the forecast job",
                 "Publish the measured half with the nowcast"]
        self.assertEqual([s for s in steps if s in order], order)
        self.assertIn("name: public-after-measured", nowcast["text"])
        self.assertIn("python nowcast.py --out public/nowcast", nowcast["text"])
        self.assertIn("python prune.py --dir public/nowcast --keep-newer-than 70", nowcast["text"])
        self.assertIn("-nowcast\n", nowcast["text"] + "\n")
        self.assertIn("name: public-after-nowcast", nowcast["text"])
        self.assertIn("name: pages-nowcast", nowcast["text"])

    def test_a_failed_nowcast_render_does_not_stop_the_job(self):
        text = self.jobs["nowcast"]["text"]
        render = text[text.index("- name: Render the nowcast"):text.index("- name: Prune projected")]
        self.assertIn("continue-on-error: true", render)

    def test_the_nowcast_job_holds_no_pages_or_identity_token(self):
        text = self.jobs["nowcast"]["text"]
        self.assertIn("permissions:\n      contents: read", text)
        self.assertNotIn("pages: write", text)
        self.assertNotIn("id-token", text)

    def test_the_forecast_takes_the_nowcast_files_or_the_measured_ones_when_the_nowcast_job_failed(self):
        forecast = self.jobs["forecast"]
        self.assertEqual(forecast["needs"], ["measured", "nowcast"])
        self.assertIn("!cancelled()", forecast["if"])
        self.assertIn("needs.measured.result == 'success'", forecast["if"])
        text = forecast["text"]
        nowcast = text[text.index("- name: Take the files from the nowcast job"):
                       text.index("- name: Take the files from the measured job")]
        measured = text[text.index("- name: Take the files from the measured job"):
                        text.index("- name: Render forecast frames")]
        self.assertIn("if: needs.nowcast.result == 'success'", nowcast)
        self.assertIn("name: public-after-nowcast", nowcast)
        self.assertIn("if: needs.nowcast.result != 'success'", measured)
        self.assertIn("name: public-after-measured", measured)

    def test_pages_never_deploys_the_nowcast_stage(self):
        deployed = re.findall(r"artifact_name: (\S+)", "\n".join(j["text"] for j in self.jobs.values()))
        self.assertEqual(sorted(deployed), ["pages-full", "pages-measured"])

    def test_the_full_pages_deploy_survives_a_failed_nowcast_job(self):
        full = self.jobs["deploy-full"]
        self.assertEqual(full["needs"], ["forecast", "deploy-measured"])
        self.assertIsNotNone(full["if"], "with no if, a failed nowcast job would skip this deploy")
        for part in ("!cancelled()", "needs.forecast.result == 'success'",
                     "needs.deploy-measured.result == 'success'"):
            self.assertIn(part, full["if"])

    def test_r2_gets_the_nowcast_between_the_measured_and_the_full_copies(self):
        r2 = self.jobs["r2-nowcast"]
        self.assertEqual(r2["needs"], ["nowcast", "r2-measured"])
        self.assertIn("!cancelled()", r2["if"])
        self.assertIn("needs.nowcast.result == 'success'", r2["if"])
        self.assertIn("name: pages-nowcast", r2["text"])
        self.assertIn("timeout-minutes: 4", r2["text"])
        self.assertIn("continue-on-error: true", r2["text"])
        self.assertEqual(self.jobs["r2-full"]["needs"], ["forecast", "r2-nowcast"])
        self.assertIn("!cancelled()", self.jobs["r2-full"]["if"])

    def test_every_r2_publish_leaves_the_regional_tier_alone(self):
        for name in ("r2-measured", "r2-nowcast", "r2-full"):
            self.assertIn("python publish_r2.py --dir site --ignore-prefix forecast-regional/",
                          self.jobs[name]["text"], name)

    def test_every_job_runs_on_the_pinned_image(self):
        for name, job in self.jobs.items():
            self.assertIn("runs-on: ubuntu-24.04", job["text"], name)


if __name__ == "__main__":
    unittest.main()
