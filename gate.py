"""Whether the forecast job renders this tick (SeaWise radar build 7, item 11a; B-263).

Exit 0 means render, exit 1 means skip; the workflow's step is an if/then/else
on this. The decision itself is `decide`, pure and pinned by `test_gate.py`.

Until build 7 the gate re-rendered whenever the published run was over 90
minutes old. The newest complete HRRR run is always 60 to 110 minutes behind
the clock, so three ticks in four re-rendered the run already live (36 of 47
runs on 2026-10-12, 8.5 to 15.7 minutes each against 4.6 to 5.3 for a skip):
the runner minutes the fourth tier needs, spent on identical pictures. Now the
forecast renders when NOMADS has a run newer than the one published, when the
published set is short (a lead skipped by a NOMADS hiccup is retried on the
next tick, which the age gate used to do by accident), or when the layout or
the colour rule changed. A NOMADS outage skips rather than re-renders.

The run chosen here is handed to `render.py`, `wind.py` and `cloud.py` through
`HRRR_RUN` (written to `hrrr-run.txt` for this step and to `GITHUB_ENV` for the
next ones), so the three products come from one run: each used to ask NOMADS
for the newest run itself, minutes apart, and a run completing in between gave
the wind and cloud a newer run than the pictures.
"""
import argparse
import json
import os
import sys


def run_key(iso):
    """'2026-10-13T00:00:00Z' -> '2026101300', so runs compare as strings."""
    return iso[0:4] + iso[5:7] + iso[8:10] + iso[11:13]


def latest_key(day, run):
    return f"{day}{run:02d}"


def decide(published, want_cells, style, latest, expected_frames):
    """(render, reason).

    `published` is the live manifest or None; `want_cells` the ids the code
    lays out; `style` the manifest keys that describe the colour rule, with
    the values the code holds; `latest` the newest complete run's key or None
    when NOMADS did not answer; `expected_frames` the count a whole render
    writes.
    """
    if published is None:
        return True, "nothing published yet"
    if {c["id"] for c in published.get("cells", [])} != set(want_cells):
        return True, "cell layout changed"
    for key, value in style.items():
        if published.get(key) != value:
            return True, f"{key} changed"
    if latest is None:
        return False, "NOMADS did not answer; the published run stays"
    have = run_key(published.get("runTime", "0000-00-00T00"))
    if have < latest:
        return True, f"newer run {latest} (published {have})"
    if len(published.get("frames", [])) < expected_frames:
        return True, (f"published set short: {len(published.get('frames', []))} "
                      f"of {expected_frames} frames")
    return False, f"run {have} is current and whole"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="public/manifest.json")
    ap.add_argument("--run-file", default="hrrr-run.txt",
                    help="where the chosen run is written for this step")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import cells as layout
    import render

    published = json.load(open(args.manifest)) if os.path.exists(args.manifest) else None
    want = {n for n, _, _ in layout.FORECAST_CELL_ORIGINS} | {layout.NATIONAL_ID}
    style = {"rampSteps": render.RAMP_STEPS, "quantiser": render.QUANTISER,
             "palette": render.PALETTE, "smoothing": render.SMOOTH_CELLS,
             "alpha": render.ALPHA}
    try:
        day, run = render.latest_complete_run()
        latest = latest_key(day, run)
    except (SystemExit, Exception) as e:          # NOMADS down or no run: never a crash
        print(f"gate: no complete run from NOMADS ({e})")
        latest = None
    expected = len(render.FORECAST_MINUTES) * len(want)

    render_now, reason = decide(published, want, style, latest, expected)
    print(f"gate: {'render' if render_now else 'skip'}: {reason}")
    if render_now and latest is not None:
        with open(args.run_file, "w") as f:
            f.write(latest)
        github_env = os.environ.get("GITHUB_ENV")
        if github_env:
            with open(github_env, "a") as f:
                f.write(f"HRRR_RUN={latest}\n")
    sys.exit(0 if render_now else 1)


if __name__ == "__main__":
    main()
