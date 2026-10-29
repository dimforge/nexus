"""Bucket permutation integrity, unsolved colors, statistics and memory guards."""

import json, os
from pathlib import Path
import sys, unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay import HERE
import bucket_replay as bucket


class BucketTests(unittest.TestCase):
    def fixture(self, op="bucket_build", name="n513-b3-s130-g2"):
        root = (
            Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
            / op
            / "c/fixtures"
        )
        l = json.loads((root / "layout.json").read_text())
        c = next(
            c
            for c in json.loads((root / op / "suite.json").read_text())["cases"]
            if c["name"] == name
        )
        d = root / op / name
        b = {n: np.fromfile(d / f"{n}.bin", dtype=np.uint32) for n in bucket.BUFFERS}
        k = json.loads((HERE / f"contracts/{op}.json").read_text())
        e = {
            n: np.fromfile(d / f"{n}.expected.bin", dtype=np.uint32)
            if n in k["outputs"]
            else a.copy()
            for n, a in b.items()
        }
        return l, c, b, e, k

    def test_every_required_case(self):
        for op in bucket.OPERATIONS:
            contract = json.loads((HERE / f"contracts/{op}.json").read_text())
            for name in contract["required_cases"]:
                with self.subTest(op=op, name=name):
                    l, c, b, e, k = self.fixture(op, name)
                    bucket.check_outputs(l, c, b, e, e, k)

    def test_legal_atomic_link_permutation_is_accepted(self):
        for op in ("bucket_scatter", "bucket_build"):
            l, c, b, e, k = self.fixture(op)
            _, rows, _, _, _ = bucket.validate_case(l, c, b)
            row = next(row for row in rows if len(row) > 1)
            a = {n: v.copy() for n, v in e.items()}
            i, j = row[:2]
            x, y = map(int, a["Indices"][[i, j]])
            links = a["SortedLinks"].reshape(-1, l["link_words"])
            links[[x, y]] = links[[y, x]]
            a["Indices"][[i, j]] = a["Indices"][[j, i]]
            bucket.check_outputs(l, c, b, a, e, k)

    def test_corrupt_copy_inverse_index_statistics_and_guards_fail(self):
        for name, index in (
            ("SortedLinks", 0),
            ("Indices", 1),
            ("Buckets", 1),
            ("Stats", 0),
            ("Indirect", 0),
            ("Indirect", -1),
            ("SortedLinks", -1),
            ("Aux0", -1),
            ("BatchIds", 0),
        ):
            l, c, b, e, k = self.fixture()
            a = {n: v.copy() for n, v in e.items()}
            a[name][index] ^= np.uint32(1)
            with (
                self.subTest(name=name, index=index),
                self.assertRaises(AssertionError),
            ):
                bucket.check_outputs(l, c, b, a, e, k)

    def test_unsolved_colors_and_zero_length_gaps(self):
        l, c, b, e, k = self.fixture()
        _, rows, _, _, _ = bucket.validate_case(l, c, b)
        links = b["Links"].reshape(-1, l["link_words"])
        for i in range(c["count"]):
            length = int(links[i, l["link"]["len"]])
            color = int(b["Colors"][i])
            if not length:
                self.assertFalse(any(i in row for row in rows))
                self.assertEqual(e["Indices"][i], b["Indices"][i])
            elif color == 0 or color >= c["color_stride"] - 1:
                self.assertTrue(any(i in row for row in rows[-c["batches"] :]))
        self.assertEqual(links[0, l["link"]["solver_body_a"]], 0xFFFFFFFF)
        links[0, l["link"]["len"]] = 1
        with self.assertRaisesRegex(ValueError, "active link body"):
            bucket.validate_case(l, c, b)

    def test_dispatch_clips_large_buckets_and_counts_high_colors(self):
        l, c, b, e, k = self.fixture("bucket_dispatch", "n33-b2-s130-g1-clip")
        bucket.check_outputs(l, c, b, e, e, k)
        self.assertEqual(e["Indirect"][0], 256)
        self.assertEqual(e["Stats"][l["stats"]["max_bucket"]], 20000)
        l, c, b, e, k = self.fixture("bucket_dispatch")
        self.assertGreater(e["Stats"][l["stats"]["highest_color"]], 63)

    def test_invalid_reset_prefix_dimensions_and_uniforms_are_rejected(self):
        for op in ("bucket_count", "bucket_scatter"):
            l, c, b, _, _ = self.fixture(op)
            b["Buckets"][0] += 1
            with self.assertRaises(ValueError):
                bucket.validate_case(l, c, b)
        for mutation in ("stride", "batches", "bound", "scan", "storage"):
            l, c, b, _, _ = self.fixture()
            if mutation == "stride":
                c["color_stride"] = 2
            if mutation == "batches":
                c["batches"] = 0
            if mutation == "bound":
                b["Plan"][l["plan"]["bound"]] += 1
            if mutation == "scan":
                b["Stride0"][0] += 1
            if mutation == "storage":
                b["Indices"] = b["Indices"][:-1]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                bucket.validate_case(l, c, b)
