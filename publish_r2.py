#!/usr/bin/env python3
"""Copy the published site into Cloudflare R2, behind radar.seawiseweather.com (SeaWise D-199).

GitHub Pages stays the app's host until every build that reads it has lapsed;
this publishes the same files to R2 beside it, in jobs of its own that no
deploy waits for.

**It must stay short.** The workflow runs one run at a time, so however long
these jobs take, the next run's Pages deploys wait that long, and a measured
frame past 30 minutes is one the app throws away. So every call has a short
timeout and one retry, the whole publish has a deadline (DEADLINE_S), and the
jobs a timeout of their own. A slow or down R2 costs a run about a minute or
two, never the radar on Pages.

The order is the point (the R2 plan's BLOCKER). A manifest names the frames the
app will ask for, and Cloudflare holds a 404 at the edge for minutes, so a
manifest must never land before the frames it names:

  1. every frame (anything that is not a pointer) that differs from R2's copy;
  2. only if all of them landed, inside the deadline, the pointers: each
     `manifest.json` and `lightning/index.json`;
  3. last, and only if those landed, whatever R2 holds that the site no longer does.

The first failure stops the run, so a failed or late frame never lets a pointer
or a delete through, and the next run carries on from what landed. "Differs" is
decided by content, never by size or time: R2's ETag for a single-part upload is
the file's MD5 (checked against radar.seawiseweather.com on 2026-10-04), and
every upload here is single part. A check at the end lists the bucket again and
fails unless it holds this run's files exactly.

Without its three secrets it says so and does nothing, so the workflow can carry
this before the bucket exists.
"""
import argparse
import hashlib
import mimetypes
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Pointers are read every minute; frames are named by time or carry the run's version.
POINTER_CACHE = "public, max-age=60"
FRAME_CACHE = "public, max-age=600"

# Seconds from the start of a publish after which it stops before its next
# stage. Normal runs put a few hundred small files in seconds; the first run,
# into an empty bucket, may need two or three runs to fill it, which is fine.
DEADLINE_S = 120

# Without these the directory is not a whole site.
REQUIRED = ("manifest.json", "observed/manifest.json")

# A bucket this size or more is guarded against a directory that is plainly not
# the site: under half as many files as the bucket holds. Never a share of
# deletes: after a gap of a few hours every measured and nowcast name changes,
# nearly half the bucket, and that publish must go through. And never a missing
# layer folder: a layer whose step failed is missing from the site Pages gets
# too, and R2 drops it the same way rather than keep an old wind or cloud picture
# (the cloud layer has no age check in the app, SeaWise B-123) or refuse the
# whole publish and let the lightning index age.
GUARD_FROM = 100


def is_pointer(key):
    return key.rsplit("/", 1)[-1] == "manifest.json" or key == "lightning/index.json"


def content_type(key):
    if key.endswith(".png"):
        return "image/png"
    if key.endswith(".json"):
        return "application/json"
    return mimetypes.guess_type(key)[0] or "application/octet-stream"


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def local_files(root):
    """Every file under root, as key -> MD5. Keys use '/' and never start with it."""
    out = {}
    for d, _, names in os.walk(root):
        for n in names:
            path = os.path.join(d, n)
            key = os.path.relpath(path, root).replace(os.sep, "/")
            out[key] = md5_of(path)
    return out


def remote_files(client, bucket):
    """Every object in the bucket, as key -> ETag without its quotes."""
    out = {}
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            out[obj["Key"]] = obj["ETag"].strip('"')
    return out


def plan(local, remote):
    """(frames to put, pointers to put, keys to delete), each sorted."""
    changed = [k for k in local if remote.get(k) != local[k]]
    frames = sorted(k for k in changed if not is_pointer(k))
    pointers = sorted(k for k in changed if is_pointer(k))
    deletes = sorted(k for k in remote if k not in local)
    return frames, pointers, deletes


def check_site(local, remote):
    """Refuse a directory that is not a whole site. Returns a reason, or None."""
    missing = [k for k in REQUIRED if k not in local]
    if missing:
        return f"the site lacks {', '.join(missing)}"
    if len(remote) >= GUARD_FROM and len(local) < len(remote) / 2:
        return f"it holds {len(local)} files against the bucket's {len(remote)}"
    return None


def put(client, bucket, root, key):
    with open(os.path.join(root, key), "rb") as f:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=f.read(),
            ContentType=content_type(key),
            CacheControl=POINTER_CACHE if is_pointer(key) else FRAME_CACHE,
        )


def delete(client, bucket, root, key):
    # One DeleteObject per key, never DeleteObjects: R2 does not bill deletes,
    # and DeleteObjects needs a body checksum whose kind R2 may not accept.
    client.delete_object(Bucket=bucket, Key=key)


def run_all(action, client, bucket, root, keys, workers, deadline, what):
    """Do action for every key; the first failure, or the deadline, stops the rest and raises."""
    if not keys:
        return
    if time.monotonic() > deadline:
        raise TimeoutError(f"past the {DEADLINE_S} s deadline before {what}; the next run carries on")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(action, client, bucket, root, k): k for k in keys}
        try:
            for fut in as_completed(futures):
                err = fut.exception()
                if err is not None:
                    raise RuntimeError(f"{what} {futures[fut]} failed: {err}") from err
                if time.monotonic() > deadline:
                    raise TimeoutError(f"past the {DEADLINE_S} s deadline while {what}; the next run carries on")
        except BaseException:
            for other in futures:
                other.cancel()
            raise


def publish(client, bucket, root, workers=16, log=print, deadline_s=DEADLINE_S):
    """Publish root to the bucket in the safe order, then check it. Returns 0 or 1."""
    deadline = time.monotonic() + deadline_s
    local = local_files(root)
    remote = remote_files(client, bucket)
    reason = check_site(local, remote)
    if reason:
        log(f"::error::R2: refusing to publish {root}: {reason}; nothing changed")
        return 1
    frames, pointers, deletes = plan(local, remote)
    log(f"R2: {len(local)} files here, {len(remote)} in the bucket; "
        f"{len(frames)} frames and {len(pointers)} pointers to put, {len(deletes)} to delete")
    run_all(put, client, bucket, root, frames, workers, deadline, "putting frames")
    run_all(put, client, bucket, root, pointers, workers, deadline, "putting pointers")
    run_all(delete, client, bucket, root, deletes, workers, deadline, "deleting")

    after = remote_files(client, bucket)
    wrong = sorted(k for k in local if after.get(k) != local[k])
    extra = sorted(k for k in after if k not in local)
    if wrong or extra:
        log(f"::error::R2 check: {len(wrong)} files differ or are missing "
            f"(first {wrong[:3]}), {len(extra)} extra (first {extra[:3]})")
        return 1
    log(f"R2 check: the bucket holds this run's {len(local)} files exactly")
    return 0


def client_from_env():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(
            # Some SDK versions send checksum headers R2 rejects; this script
            # compares MD5 ETags, so it asks for checksums only where required.
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
            # Short and few, so a stalled R2 costs seconds, not minutes (see above):
            # two tries in all, the first and one retry.
            retries={"total_max_attempts": 2, "mode": "standard"},
            connect_timeout=5,
            read_timeout=10,
            max_pool_connections=32,
        ),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--bucket", default=os.environ.get("R2_BUCKET", "seawise-radar"))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    names = ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
    absent = [n for n in names if not os.environ.get(n)]
    if absent:
        print(f"::warning::R2 not configured ({', '.join(absent)} unset); nothing published to R2")
        return 0
    try:
        return publish(client_from_env(), args.bucket, args.dir, args.workers)
    except Exception as err:  # one line in the run's summary, then a failed step
        print(f"::error::R2 publish stopped: {err}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
