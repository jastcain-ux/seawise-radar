"""Picture names that never change meaning under a phone (SeaWise radar build 7, item 21; B-258, B-272).

Pure Python, so `test_prune.py` runs without eccodes.

Until build 7 a forecast picture was named by its cell and lead alone, so a
new run overwrote it in place: a phone still on the previous manifest that
asked for a picture it did not hold got the new run's picture under the old
time, about an hour ahead (B-258); and a restyle redrew the measured names
with new bytes, so an open Radar tab kept old pictures beside new ones
(B-272). Now a forecast name ends in the stamp of the publication that made
it, a measured name in the revision of the drawing rule, and `prune.py`
keeps the previous set an hour before removing it. The manifests still
switch last (`publish_r2.py`).

The cloud and wind pictures were the last named by cell and lead alone: each
forecast run overwrote them in place, so a phone on the previous manifest got
the next run's picture under its old hour (B-282). They now end in their
publication's stamp too, through `field_image`, the one rule both scripts use
(they import eccodes, so the rule lives here where the pure tests reach it).
"""
import hashlib
import json
import re

STAMP = re.compile(r"(\d{8}T\d{4})Z")


def publication_stamp(now):
    """A UTC instant as the token a name carries: 20261013T0120Z."""
    return now.strftime("%Y%m%dT%H%MZ")


def field_image(cell, kind, minutes, stamp):
    """A cloud or wind picture's name: cell, kind, lead and publication (B-282).

    `field_image("national", "wind", 60, "20261014T0519Z")` is
    `national-wind-0060-20261014T0519Z.png`, the forecast radar's shape.
    """
    return f"{cell}-{kind}-{minutes:04d}-{stamp}.png"


def style_revision(style):
    """Six hex characters of the drawing rule, so a restyle changes the name, not the bytes."""
    return hashlib.sha1(json.dumps(style, sort_keys=True).encode()).hexdigest()[:6]


def stamp_in(name):
    """The first stamp token in a name as (YYYYMMDD, HHMM), or None."""
    m = STAMP.search(name)
    if not m:
        return None
    token = m.group(1)
    return token[:8], token[9:13]


def stamp_age_minutes(name, now):
    """Minutes from the name's first stamp to `now` (UTC), or None when it has none."""
    found = stamp_in(name)
    if found is None:
        return None
    day, hhmm = found
    import datetime
    then = datetime.datetime(int(day[:4]), int(day[4:6]), int(day[6:8]), int(hhmm[:2]), int(hhmm[2:]),
                             tzinfo=datetime.timezone.utc)
    return (now - then).total_seconds() / 60.0
