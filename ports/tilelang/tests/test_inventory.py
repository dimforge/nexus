"""Negative checks for the Rust-mirrored shader inventory (stdlib only)."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inventory import audit


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rs = self.root / "src_rbd_shaders/dynamics/body.rs"
        self.py = self.root / "ports/tilelang/shaders/dynamics/body.py"
        self.rs.parent.mkdir(parents=True)
        self.py.parent.mkdir(parents=True)
        self.rs.write_text("fn integrate_linearized() {}")
        self.py.write_text("def integrate_linearized(): pass")
        self.path = self.root / "ports/tilelang/inventory.json"
        self.inventory = {
            "schema": 1,
            "modules": [
                {
                    "reference": "src_rbd_shaders/dynamics/body.rs",
                    "implementation": "ports/tilelang/shaders/dynamics/body.py",
                    "coverage": "partial",
                    "operations": ["integrate"],
                    "symbols": ["integrate_linearized"],
                    "remaining": "Other body functions",
                }
            ],
            "capabilities": [
                {
                    "id": "body",
                    "coverage": "partial",
                    "operations": ["integrate"],
                    "remaining": "Other body operations",
                }
            ],
        }
        self.config = {
            "inventory": "ports/tilelang/inventory.json",
            "features": [{"id": "integrate"}],
        }

    def check(self):
        self.path.write_text(json.dumps(self.inventory))
        return audit(self.root, self.config)["errors"]

    def test_mapped_subset_passes(self):
        self.assertEqual(self.check(), [])

    def test_deferred_scope_requires_explicit_reason_and_keeps_coverage(self):
        capability = self.inventory["capabilities"][0]
        capability["required_for_completion"] = False
        self.assertIn("Invalid completion scope", ";".join(self.check()))
        capability["scope_reason"] = "User deferred this capability"
        self.assertEqual(self.check(), [])
        self.assertEqual(capability["coverage"], "partial")
        capability["required_for_completion"] = "false"
        self.assertIn("Invalid completion scope", ";".join(self.check()))

    def test_new_and_removed_rust_modules_need_review(self):
        new = self.rs.with_name("new.rs")
        new.write_text("fn new_feature() {}")
        self.assertIn("New Rust module", ";".join(self.check()))
        new.unlink()
        self.rs.unlink()
        self.assertIn("Removed Rust module", ";".join(self.check()))

    def test_mirrored_path_is_required(self):
        self.inventory["modules"][0]["implementation"] = "ports/tilelang/monolith.py"
        self.assertIn("mirror", ";".join(self.check()))

    def test_missing_symbol_and_unmapped_shader_are_rejected(self):
        self.py.write_text("def renamed(): pass")
        self.assertIn("Missing mapped", ";".join(self.check()))
        self.py.with_name("new.py").write_text("def added(): pass")
        self.assertIn("Unmapped TileLang", ";".join(self.check()))

    def test_new_operation_needs_module_and_capability_mapping(self):
        self.config["features"].append({"id": "new"})
        errors = ";".join(self.check())
        self.assertIn("without shader", errors)
        self.assertIn("without capability", errors)

    def test_unported_file_cannot_certify_an_operation(self):
        self.inventory["modules"][0]["coverage"] = "unported"
        self.assertIn("Unported module cannot certify", ";".join(self.check()))

    def test_duplicate_dispositions_are_rejected(self):
        self.inventory["modules"].append(self.inventory["modules"][0].copy())
        self.assertIn("Duplicate Rust", ";".join(self.check()))


if __name__ == "__main__":
    unittest.main()
