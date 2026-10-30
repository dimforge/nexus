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
