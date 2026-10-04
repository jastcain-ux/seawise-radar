"""The R2 publisher's order and safety, against an in-memory bucket (no network, no boto3)."""
import contextlib
import hashlib
import io
import os
import tempfile
import time
import unittest

import publish_r2 as p


class FakeBucket:
    """Enough of boto3's S3 client for publish(): list, put, delete, in call order."""

    def __init__(self, objects=None, fail_on=None, put_delay=0):
        self.objects = dict(objects or {})  # key -> bytes
        self.calls = []                     # ("put" | "delete", key), in order
        self.fail_on = fail_on              # a key whose put raises
        self.put_delay = put_delay          # seconds each put takes

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        bucket = self

        class Pager:
            def paginate(self, Bucket):
                keys = sorted(bucket.objects)
                for i in range(0, len(keys), 2):  # small pages, so paging is exercised
                    yield {"Contents": [{"Key": k, "ETag": '"%s"' % hashlib.md5(bucket.objects[k]).hexdigest()}
                                        for k in keys[i:i + 2]]}
        return Pager()

    def put_object(self, Bucket, Key, Body, ContentType, CacheControl):
        time.sleep(self.put_delay)
        if Key == self.fail_on:
            raise IOError("boom")
        self.calls.append(("put", Key, ContentType, CacheControl))
        self.objects[Key] = Body

    def delete_object(self, Bucket, Key):
        self.calls.append(("delete", Key))
        self.objects.pop(Key, None)


def site(files):
    root = tempfile.mkdtemp()
    for key, body in files.items():
        path = os.path.join(root, *key.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(body)
    return root


BASE = {
    "manifest.json": b'{"frames":["a-refc-0015.png"]}',
    "a-refc-0015.png": b"forecast",
    "observed/manifest.json": b'{"frames":["n-1250.png"]}',
    "observed/n-1250.png": b"measured",
    "lightning/index.json": b'{"cells":[]}',
    "lightning/a.json": b"[]",
}


def quiet(*_):
    pass


class PublishOrder(unittest.TestCase):
    def test_every_frame_lands_before_any_pointer_and_deletes_come_last(self):
        bucket = FakeBucket({"observed/n-1240.png": b"old"})
        self.assertEqual(p.publish(bucket, "b", site(BASE), workers=4, log=quiet), 0)
        kinds = [(c[0], p.is_pointer(c[1])) for c in bucket.calls]
        first_pointer = kinds.index(("put", True))
        self.assertTrue(all(k == ("put", False) for k in kinds[:first_pointer]))
        self.assertTrue(all(k == ("put", True) for k in kinds[first_pointer:-1]))
        self.assertEqual(bucket.calls[-1], ("delete", "observed/n-1240.png"))
        self.assertEqual(set(bucket.objects), set(BASE))

    def test_a_failed_frame_stops_pointers_and_deletes(self):
        bucket = FakeBucket({"observed/n-1240.png": b"old"}, fail_on="observed/n-1250.png")
        with self.assertRaises(RuntimeError):
            p.publish(bucket, "b", site(BASE), workers=1, log=quiet)
        self.assertFalse(any(p.is_pointer(c[1]) for c in bucket.calls))
        self.assertFalse(any(c[0] == "delete" for c in bucket.calls))
        self.assertIn("observed/n-1240.png", bucket.objects)

    def test_only_changed_files_are_put(self):
        bucket = FakeBucket(BASE)
        files = dict(BASE, **{"observed/n-1300.png": b"new", "observed/manifest.json": b'{"frames":["n-1300.png"]}'})
        self.assertEqual(p.publish(bucket, "b", site(files), log=quiet), 0)
        self.assertEqual(sorted(c[1] for c in bucket.calls), ["observed/manifest.json", "observed/n-1300.png"])

    def test_same_size_new_content_is_put(self):
        bucket = FakeBucket(BASE)
        files = dict(BASE, **{"a-refc-0015.png": b"FORECAST"})  # same length, new bytes
        self.assertEqual(p.publish(bucket, "b", site(files), log=quiet), 0)
        self.assertEqual([c[1] for c in bucket.calls], ["a-refc-0015.png"])

    def test_headers(self):
        bucket = FakeBucket()
        p.publish(bucket, "b", site(BASE), log=quiet)
        got = {c[1]: (c[2], c[3]) for c in bucket.calls}
        self.assertEqual(got["manifest.json"], ("application/json", "public, max-age=60"))
        self.assertEqual(got["lightning/index.json"], ("application/json", "public, max-age=60"))
        self.assertEqual(got["lightning/a.json"], ("application/json", "public, max-age=600"))
        self.assertEqual(got["observed/n-1250.png"], ("image/png", "public, max-age=600"))


class PublishSafety(unittest.TestCase):
    def test_a_site_without_its_manifests_publishes_nothing(self):
        bucket = FakeBucket(BASE)
        files = {k: v for k, v in BASE.items() if k != "observed/manifest.json"}
        self.assertEqual(p.publish(bucket, "b", site(files), log=quiet), 1)
        self.assertEqual(bucket.calls, [])

    def test_a_site_far_smaller_than_the_bucket_is_refused(self):
        many = {f"observed/x-{i:04d}.png": b"x" for i in range(200)}
        bucket = FakeBucket(dict(BASE, **many))
        self.assertEqual(p.publish(bucket, "b", site(BASE), log=quiet), 1)
        self.assertEqual(bucket.calls, [])

    def test_a_layer_missing_from_the_site_is_dropped_as_pages_drops_it(self):
        wind = {f"wind/c-wind-{i:04d}.png": b"w" for i in range(150)}
        bucket = FakeBucket(dict(BASE, **wind))
        files = dict(BASE, **{f"observed/n-{i:04d}.png": b"m" for i in range(150)})
        self.assertEqual(p.publish(bucket, "b", site(files), log=quiet), 0)
        self.assertFalse(any(k.startswith("wind/") for k in bucket.objects))
        self.assertEqual(set(bucket.objects), set(files))

    def test_after_a_gap_every_measured_name_changes_and_it_still_publishes(self):
        old = {f"observed/n-0{i:03d}.png": b"old%d" % i for i in range(200)}
        new = {f"observed/n-1{i:03d}.png": b"new%d" % i for i in range(200)}
        bucket = FakeBucket(dict(BASE, **old))
        self.assertEqual(p.publish(bucket, "b", site(dict(BASE, **new)), log=quiet), 0)
        self.assertEqual(sum(1 for c in bucket.calls if c[0] == "delete"), 200)
        self.assertEqual(set(bucket.objects), set(BASE) | set(new))

    def test_past_the_deadline_no_pointer_or_delete_is_sent(self):
        frames = {f"observed/n-{i:04d}.png": b"f%d" % i for i in range(40)}
        bucket = FakeBucket({"observed/old.png": b"o"}, put_delay=0.02)
        with self.assertRaises(TimeoutError):
            p.publish(bucket, "b", site(dict(BASE, **frames)), workers=1, log=quiet, deadline_s=0.1)
        self.assertFalse(any(p.is_pointer(c[1]) for c in bucket.calls))
        self.assertFalse(any(c[0] == "delete" for c in bucket.calls))
        self.assertLess(len(bucket.calls), 40)

    def test_a_nearly_empty_bucket_is_not_guarded(self):
        bucket = FakeBucket({"r2-check.txt": b"t", "r2-check.png": b"p"})
        self.assertEqual(p.publish(bucket, "b", site(BASE), log=quiet), 0)
        self.assertEqual(set(bucket.objects), set(BASE))

    def test_the_check_fails_when_the_bucket_does_not_match(self):
        class Forgetful(FakeBucket):
            def put_object(self, **kw):
                super().put_object(**kw)
                if kw["Key"] == "lightning/a.json":
                    self.objects[kw["Key"]] = b"something else"
        self.assertEqual(p.publish(Forgetful(), "b", site(BASE), log=quiet), 1)

    def test_without_secrets_it_does_nothing_and_succeeds(self):
        saved = {n: os.environ.pop(n, None) for n in ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")}
        try:
            import sys
            argv, sys.argv = sys.argv, ["publish_r2.py", "--dir", site(BASE)]
            out = io.StringIO()  # held back, or the workflow reads the warning as its own
            try:
                with contextlib.redirect_stdout(out):
                    self.assertEqual(p.main(), 0)
            finally:
                sys.argv = argv
            self.assertIn("R2 not configured", out.getvalue())
        finally:
            for n, v in saved.items():
                if v is not None:
                    os.environ[n] = v


class Secrets(unittest.TestCase):
    def test_line_breaks_and_spaces_pasted_with_a_secret_are_removed(self):
        got = p.settings_from_env({
            "R2_ENDPOINT": " https://acct.r2.cloudflarestorage.com\n",
            "R2_ACCESS_KEY_ID": "abc123\n",
            "R2_SECRET_ACCESS_KEY": "\tsecret \r\n",
        })
        self.assertEqual(got, {
            "R2_ENDPOINT": "https://acct.r2.cloudflarestorage.com",
            "R2_ACCESS_KEY_ID": "abc123",
            "R2_SECRET_ACCESS_KEY": "secret",
        })

    def test_a_blank_secret_counts_as_unset(self):
        got = p.settings_from_env({"R2_ENDPOINT": "\n", "R2_ACCESS_KEY_ID": "x"})
        self.assertEqual(got["R2_ENDPOINT"], "")
        self.assertEqual(got["R2_SECRET_ACCESS_KEY"], "")


if __name__ == "__main__":
    unittest.main()
