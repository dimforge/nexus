"""A handoff includes every resident specialization and validates migration ABI."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import test_frame_manifest as manifests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handoff import kernel_artifacts


class FrameBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.contract = {"execution": "resident_frames", "inputs": ["State", "Count"]}
        self.write(
            "bindings.json",
            {
                "schema": 3,
                "execution": "resident_frames",
                "max_steps": 2,
                "variants": ["variants/0", "variants/1"],
                "storage_buffers": ["State", "Count"],
                "buffer_bytes": {"State": 16, "Count": 16},
            },
        )
        self.write("variants/variants.json", ["0", "1"])
        for i in range(2):
            frame = manifests.frame()
            frame["gates"][0]["capacity"] = 1 + 3 * i
            for stage in frame["program"]["stages"]:
                stage["source"] = stage["source"].replace(".metal", ".ascend")
                self.write_bytes(f"variants/{i}/{stage['source']}", b"source-only test")
            self.write(f"variants/{i}/frame.json", frame)
            for name in ("State", "Count"):
                self.write_bytes(f"variants/{i}/{name}.bin", bytes(16))

    def write_bytes(self, name, value):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(value)

    def write(self, name, value):
        self.write_bytes(name, json.dumps(value).encode())

    def edit(self, name, edit):
        d = json.loads((self.root / name).read_text())
        edit(d)
        self.write(name, d)

    def test_every_variant_source_and_upload_is_required(self):
        required = kernel_artifacts(self.root, self.contract)
        self.assertEqual(len(required), 12)
        for i in range(2):
            self.assertTrue(
                {
                    f"variants/{i}/{n}"
                    for n in (
                        "frame.json",
                        "count.ascend",
                        "commit.ascend",
                        "State.bin",
                        "Count.bin",
                    )
                }
                <= required
            )
        (self.root / "variants/1/count.ascend").unlink()
        with self.assertRaisesRegex(ValueError, "Missing artifact"):
            kernel_artifacts(self.root, self.contract)

    def test_incompatible_growth_or_persistent_copy_is_rejected(self):
        original = (self.root / "variants/1/frame.json").read_bytes()
        for edit in (
            lambda d: d["gates"][0].update(capacity=1),
            lambda d: d["gates"][0]["counters"][0].update(word=1),
            lambda d: d["persistent"]["pose"].update(word_len=1),
            lambda d: d.update(max_steps=3),
            lambda d: d["program"]["stages"][0].update(source="count.metal"),
        ):
            self.write_bytes("variants/1/frame.json", original)
            self.edit("variants/1/frame.json", edit)
            with self.assertRaises(ValueError):
                kernel_artifacts(self.root, self.contract)

    def test_ladder_paths_return_storage_and_upload_sizes_are_checked(self):
        original = (self.root / "bindings.json").read_bytes()
        for edit in (
            lambda d: d.update(variants=["../0"]),
            lambda d: d.update(max_steps=True),
            lambda d: d["buffer_bytes"].update(State=8),
        ):
            self.write_bytes("bindings.json", original)
            self.edit("bindings.json", edit)
            with self.assertRaises(ValueError):
                kernel_artifacts(self.root, self.contract)
        self.write_bytes("bindings.json", original)
        self.write_bytes("variants/0/State.bin", bytes(12))
        with self.assertRaisesRegex(ValueError, "storage size"):
            kernel_artifacts(self.root, self.contract)
