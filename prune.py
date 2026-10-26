#!/usr/bin/env python3
"""Drop frames the timeline can no longer reach, an hour after they left it.

The published directory is carried between runs, so nothing removes old
frames on its own. Left alone it would grow by ~400 KB every ten minutes for
ever — about 58 MB a day for the measured half alone.

Anything not named in the manifest goes, once it is old enough. The manifest
is written last by `observed.py` and `render.py` and lists exactly the frames
the app can ask for; but a phone holds the manifest it fetched for up to an
hour (the app asks again when the measured one is 30 minutes old, and a
paused screen keeps its own), and a picture it names that is gone answers
404 and the step is passed over (B-258: measured live, 7 of the previous
nowcast run's 8 national pictures gone within ten minutes of a publish). So
a frame whose stamp is under `--keep-newer-than` minutes old is kept even
when the manifest no longer names it (radar build 7, item 21). The stamp is
read from the name (`naming.stamp_in`), never from the file's time, which a
cache may not keep. A file with no stamp that the manifest does not name
goes as before.

The forecast names carried no stamp until build 7, so the first publish
after it removes the old names at once; the app's item 9 (a 404 asks the
manifests again) carries a phone over that one step.

Run it over the forecast directory too: the forecast frames sit in the root
of `public/`, and the moment the cell list changes every frame belonging to a
retired cell is orphaned and would be carried for ever.

What the manifest names is every picture under `frames`, under
`regional.frames`, under `coverage.frames` and under `coverage.replaced`
(`named_by`; the measured manifest's masks of where no radar sees, and those it
replaced in the last 70 minutes, B-33 item 2), the regional tier included: the nowcast's,
the measured's, cloud's and wind's regional pictures sit in the same
directory, and keeping only the top level let a named regional picture go
while its manifest still named it (B-282's plan, step 4).

The cloud and wind directories (`public/cloud`, `public/wind`) are pruned with
two more rules (B-282), and the radar's own forecast (`public`, and the
regional tier's `forecast-regional`) with the first of them (B-293):

- `--from-replacement`: a set's keep runs from when it was **replaced**, not
  from its own stamp. Their stamp is the publication, and a forecast publish
  can come 135 minutes or more after the last (a failed step, NOMADS down), so
  timing from the stamp would delete the set a phone fetched a moment ago in
  the very run that replaced it. A set is replaced at the stamp of the next
  newer set in the directory; the newest unnamed set, replaced by nothing yet,
  stays. A set stamped newer than everything the manifest names was never
  live, and goes at once: cloud and wind run on after a failure, so a step
  that dies part-way leaves such a set, and kept, it would count as the
  replacement of the set the manifest still names, which would then go up to
  135 minutes early (batch 9's build reader). Where the manifest names only
  unstamped pictures (the changeover), every stamped set is newer than them.
  Where it names nothing present, nothing is judged never live.
- `--unstamped-as-old-as-stamped`: the changeover. Until B-282 these names
  carried no stamp and were overwritten each run; after it they are never
  written again, and their bytes stay right for a phone still on a pre-change
  manifest. Every one of them was replaced by the first stamped set, so an
  unstamped name is given the age of the oldest stamped name present, and is
  kept while no stamped name is present at all. Without the flag an unnamed,
  unstamped name goes at once, as before.
"""
import argparse, datetime, json, os

from naming import stamp_age_minutes


def named_by(manifest):
    """Every picture the manifest names: its `frames`, its regional tier's and its coverage masks (absent, none)."""
    named = {frame["image"] for frame in manifest.get("frames", [])}
    named |= {frame["image"] for frame in (manifest.get("regional") or {}).get("frames", [])}
    # The coverage block is read tolerantly: a malformed one must never stop the measured job's prune.
    coverage = manifest.get("coverage")
    if isinstance(coverage, dict):
        for key in ("frames", "replaced"):
            entries = coverage.get(key)
            if isinstance(entries, list):
                named |= {e["image"] for e in entries if isinstance(e, dict) and isinstance(e.get("image"), str)}
    return named


def to_remove(names, keep, now, grace_minutes, from_replacement=False, unstamped_as_old_as_stamped=False):
    """Of `names`, those not in `keep` whose stamp is missing or older than the grace.

    With `from_replacement` a stamped name's age is counted from the next newer
    stamp among `names` (when its set was replaced), and with none newer it
    stays; a stamped name newer than every picture `keep` names goes at once,
    and is no set's replacement. With `unstamped_as_old_as_stamped` a name with
    no stamp is as old as the oldest stamped name among `names`, and stays when
    there is none.
    """
    ages = {name: stamp_age_minutes(name, now) for name in names}
    never_live = set()
    if from_replacement:
        named = [name for name in names if name in keep and name != "manifest.json"]
        if named:
            named_ages = [ages[name] for name in named if ages[name] is not None]
            youngest_named = min(named_ages) if named_ages else float("inf")
            never_live = {name for name in names
                          if name not in keep and ages[name] is not None and ages[name] < youngest_named}
    stamped = sorted({ages[name] for name in names
                      if ages[name] is not None and name not in never_live})   # youngest first
    out = []
    for name in names:
        if name in keep:
            continue
        if name in never_live:
            out.append(name)
            continue
        age = ages[name]
        if age is None and unstamped_as_old_as_stamped:
            if not stamped:
                continue
            age = stamped[-1]
        elif age is not None and from_replacement:
            newer = [a for a in stamped if a < age]
            if not newer:
                continue
            age = newer[-1]
        if age is not None and age < grace_minutes:
            continue
        out.append(name)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--keep-newer-than", type=float, default=0.0, metavar="MINUTES",
                    help="keep a frame the manifest no longer names while its stamp is younger than this")
    ap.add_argument("--from-replacement", action="store_true",
                    help="count a stamped frame's age from the next newer stamp, when it was replaced (B-282)")
    ap.add_argument("--unstamped-as-old-as-stamped", action="store_true",
                    help="give a name with no stamp the age of the oldest stamped name; keep it while none (B-282)")
    args = ap.parse_args()

    manifest_path = os.path.join(args.dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print("no manifest; pruning nothing")
        return
    with open(manifest_path) as f:
        keep = named_by(json.load(f))
    keep.add("manifest.json")

    now = datetime.datetime.now(datetime.timezone.utc)
    # The forecast frames sit in the root of `public/` and the measured ones
    # in `public/observed`, so pruning the root must step over directories.
    names = [n for n in os.listdir(args.dir) if not os.path.isdir(os.path.join(args.dir, n))]
    removed = to_remove(names, keep, now, args.keep_newer_than,
                        from_replacement=args.from_replacement,
                        unstamped_as_old_as_stamped=args.unstamped_as_old_as_stamped)
    for name in removed:
        os.unlink(os.path.join(args.dir, name))
    print(f"kept {len(names) - len(removed)} files ({len(keep) - 1} named by the manifest), removed {len(removed)}")


if __name__ == "__main__":
    main()
