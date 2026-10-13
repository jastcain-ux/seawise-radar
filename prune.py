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
"""
import argparse, datetime, json, os

from naming import stamp_age_minutes


def to_remove(names, keep, now, grace_minutes):
    """Of `names`, those not in `keep` whose stamp is missing or older than the grace."""
    out = []
    for name in names:
        if name in keep:
            continue
        age = stamp_age_minutes(name, now)
        if age is not None and age < grace_minutes:
            continue
        out.append(name)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--keep-newer-than", type=float, default=0.0, metavar="MINUTES",
                    help="keep a frame the manifest no longer names while its stamp is younger than this")
    args = ap.parse_args()

    manifest_path = os.path.join(args.dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print("no manifest; pruning nothing")
        return
    with open(manifest_path) as f:
        keep = {frame["image"] for frame in json.load(f).get("frames", [])}
    keep.add("manifest.json")

    now = datetime.datetime.now(datetime.timezone.utc)
    # The forecast frames sit in the root of `public/` and the measured ones
    # in `public/observed`, so pruning the root must step over directories.
    names = [n for n in os.listdir(args.dir) if not os.path.isdir(os.path.join(args.dir, n))]
    removed = to_remove(names, keep, now, args.keep_newer_than)
    for name in removed:
        os.unlink(os.path.join(args.dir, name))
    print(f"kept {len(names) - len(removed)} files ({len(keep) - 1} named by the manifest), removed {len(removed)}")


if __name__ == "__main__":
    main()
