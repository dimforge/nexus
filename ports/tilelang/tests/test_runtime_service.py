"""Real compiled-C control protocol and immutable specialization regressions."""

import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_compile import load_variant
from runtime_service import serve
from test_frame_cpu import variant


def write_variant(directory, value):
    directory.mkdir()
    for name, source in value["sources"].items():
        (directory / name).write_text(source)
    for name, words in value["initial"].items():
        words.tofile(directory / f"{name}.bin")
    (directory / "frame.json").write_text(json.dumps(value["manifest"]))
    for name in ("case", "layout"):
        (directory / f"{name}.json").write_text("{}")
    (directory / "specialization.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "files": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in directory.iterdir()
                },
            }
        )
    )


class RuntimeServiceTests(unittest.TestCase):
    def test_binary_readback_keeps_following_control_reply_aligned(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "variant"
            value = variant(8)
            write_variant(directory, value)
            raw = io.BytesIO()
            output = io.TextIOWrapper(raw, encoding="utf-8", write_through=True)
            requests = [
                {"command": "read_binary", "buffer": "State"},
                {"command": "close"},
            ]
            serve(directory, io.StringIO("\n".join(map(json.dumps, requests))), output)
            raw.seek(0)
            self.assertEqual(json.loads(raw.readline()), {"ready": True})
            count = json.loads(raw.readline())["ok"]["binary_words"]
            np.testing.assert_array_equal(
                np.frombuffer(raw.read(count * 4), dtype="<u4"),
                value["initial"]["State"],
            )
            self.assertEqual(json.loads(raw.readline()), {"ok": None})
            self.assertEqual(raw.read(), b"")

    def test_specialization_rejects_a_different_device_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "variant"
            write_variant(directory, variant(8))
            with self.assertRaisesRegex(ValueError, "target differs"):
                load_variant(directory, expected_target="cuda")

    def test_checkpoint_restore_and_reset_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_variant(root / "variant", variant(8))
            checkpoint = {
                "schema": 1,
                "steps": 1,
                "capacities": {"pairs": 8},
                "regions": {"state": [100, 200]},
            }
            requests = [
                {"command": "step"},
                {"command": "restore", "checkpoint": checkpoint},
                {"command": "checkpoint"},
                {"command": "step"},
                {"command": "read", "buffer": "State"},
                {"command": "reset"},
                {"command": "checkpoint"},
            ]
            out = io.StringIO()
            serve(
                root / "variant", io.StringIO("\n".join(map(json.dumps, requests))), out
            )
            rows = list(map(json.loads, out.getvalue().splitlines()))
            self.assertEqual(rows[3]["ok"], checkpoint)
            self.assertEqual(rows[4]["ok"]["step"], 2)
            self.assertEqual(rows[5]["ok"][:2], [101, 200])
            self.assertEqual(rows[7]["ok"]["steps"], 0)
            self.assertEqual(rows[7]["ok"]["regions"], {"state": [42, 17]})

    def test_capacity_return_extend_step_and_explicit_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = variant(1), variant(8, 3)
            for v in (a, b):
                v["manifest"]["max_steps"] = None
            write_variant(root / "small", a)
            write_variant(root / "large", b)
            requests = [
                {"command": "step"},
                {"command": "read", "buffer": "State"},
                {"command": "extend", "directory": str(root / "large")},
                {"command": "step"},
                {"command": "read", "buffer": "State"},
                {"command": "unknown"},
                {"command": "close"},
            ]
            out = io.StringIO()
            serve(
                root / "small", io.StringIO("\n".join(map(json.dumps, requests))), out
            )
            rows = list(map(json.loads, out.getvalue().splitlines()))
            self.assertEqual(rows[0], {"ready": True})
            self.assertEqual((rows[1]["kind"], rows[1]["required"]), ("capacity", 5))
            self.assertEqual(rows[2]["ok"][:2], [42, 17])
            self.assertEqual(rows[4]["ok"]["step"], 1)
            self.assertEqual(rows[4]["ok"]["retries"], 1)
            self.assertEqual(rows[5]["ok"][3:5], [43, 17])
            self.assertEqual(rows[6]["kind"], "runtime")
            self.assertEqual(rows[7], {"ok": None})

    def test_write_protocol_checks_words_and_preserves_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_variant(root / "small", variant(1))
            write_variant(root / "large", variant(8, 3))
            requests = [
                {
                    "command": "write",
                    "region": "state",
                    "word_offset": 0,
                    "words": [100, 200],
                },
                {
                    "command": "write",
                    "region": "state",
                    "word_offset": 0,
                    "words": [-1],
                },
                {
                    "command": "write",
                    "region": "state",
                    "word_offset": 0,
                    "words": [True],
                },
                {
                    "command": "write",
                    "region": "state",
                    "word_offset": 1,
                    "words": [9, 10],
                },
                {"command": "extend", "directory": str(root / "large")},
                {"command": "step"},
                {"command": "read", "buffer": "State"},
            ]
            out = io.StringIO()
            serve(
                root / "small", io.StringIO("\n".join(map(json.dumps, requests))), out
            )
            rows = list(map(json.loads, out.getvalue().splitlines()))
            self.assertEqual(rows[1], {"ok": None})
            for row in rows[2:5]:
                self.assertEqual(row["kind"], "runtime")
            self.assertEqual(rows[-1]["ok"][3:5], [101, 200])

    def test_library_inventory_requires_known_sources_and_verified_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "variant"
            write_variant(directory, variant(8))
            path = directory / "specialization.json"
            original = json.loads(path.read_text())
            for libraries in (
                {"count.c": "../outside"},
                {"unknown.c": "State.bin"},
                {"count.c": "missing.native"},
            ):
                metadata = dict(original, libraries=libraries)
                path.write_text(json.dumps(metadata))
                with self.assertRaises(ValueError):
                    load_variant(directory)

    def test_modified_or_missing_artifact_inventory_rejects_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "variant"
            write_variant(directory, variant(8))
            self.assertIsNotNone(load_variant(directory))
            original = (directory / "State.bin").read_bytes()
            (directory / "State.bin").write_bytes(bytes(len(original)))
            with self.assertRaisesRegex(ValueError, "changed"):
                load_variant(directory)
            (directory / "State.bin").write_bytes(original)
            path = directory / "specialization.json"
            metadata = json.loads(path.read_text())
            metadata["files"].pop("count.c")
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "omits"):
                load_variant(directory)


if __name__ == "__main__":
    unittest.main()
