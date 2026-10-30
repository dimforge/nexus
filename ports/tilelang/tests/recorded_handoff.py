"""Shared immutable source-bundle / recorded Metal-return test setup."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import handoff
from replay import HERE, ROOT


class RecordedHandoff(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / "bundle"
        self.returned = self.root / "returned"
        self.bundle.mkdir()
        self.returned.mkdir()
        self.snapshot = Path(os.environ.get(self.root_environment, ROOT))
        results = Path(os.environ.get(self.results_environment, HERE / "results"))
        self.output = self.root / "comparison.json"
        self.manifest = {
            "schema": 1,
            "qualification": "Metal-return protocol regression; not Ascend execution",
            "inputs": handoff.evidence_inputs(self.snapshot),
            "files": {},
            "cpu_oracles": {},
            "cases": [],
        }

        def copy(source, relative):
            target = self.bundle / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            self.manifest["files"][relative] = handoff.digest(target)

        for op in self.operations:
            source = results / op / "ascend"
            report = json.loads((source / "report.json").read_text())
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["inputs"], self.manifest["inputs"])
            metal_report = json.loads((results / op / "metal/report.json").read_text())
            self.assertEqual(metal_report["status"], "passed")
            self.assertEqual(metal_report["inputs"], report["inputs"])
            contract_path = self.snapshot / f"ports/tilelang/contracts/{op}.json"
            contract = json.loads(contract_path.read_text())
            copy(contract_path, f"contracts/{op}.json")
            copy(source / "fixtures/layout.json", f"layouts/{op}.json")
            copy(source / "report.json", f"reports/{op}.json")
            copy(source / "cpu-oracle-noalias.ll", f"reports/{op}.ll")
            self.manifest["cpu_oracles"][contract["feature"]] = {
                "report": f"reports/{op}.json",
                "llvm_ir": f"reports/{op}.ll",
            }
            for row in report["cases"]:
                case = row["fixture"]
                folder = source / "fixtures" / op / case["name"]
                relative = f"cases/{op}/{case['name']}"
                required = handoff.kernel_artifacts(folder, contract)
                required |= {f"{n}.bin" for n in contract["inputs"]}
                required |= {f"{n}.expected.bin" for n in contract["outputs"]}
                for name in required:
                    self.assertEqual(handoff.digest(folder / name), row["files"][name])
                    copy(folder / name, f"{relative}/{name}")
                self.manifest["cases"].append(
                    {
                        "feature": contract["feature"],
                        "case": case,
                        "directory": relative,
                        "layout": f"layouts/{op}.json",
                        "contract": f"contracts/{op}.json",
                    }
                )
                actual = self.returned / relative
                actual.mkdir(parents=True)
                recorded = results / op / "metal/fixtures" / op / case["name"]
                returned_files = [f"{n}.actual.bin" for n in contract["inputs"]]
                if contract.get("execution") == "resident_frames":
                    returned_files.append("stats.json")
                for name in returned_files:
                    shutil.copyfile(recorded / name, actual / name)
        self.save_manifest()

    def save_manifest(self):
        handoff.write_json(self.bundle / "manifest.json", self.manifest)
        (self.bundle / "manifest.sha256").write_text(
            handoff.digest(self.bundle / "manifest.json") + "\n"
        )
        handoff.write_json(
            self.returned / "session.json",
            {
                "schema": 1,
                "bundle_sha256": handoff.digest(self.bundle / "manifest.json"),
                "environment": {
                    "device_model": "Apple M4 Max: recorded Metal outputs for protocol test",
                    "device_id": "local Metal regression",
                    "cann_version": "not used; no Ascend execution",
                    "driver_version": "recorded Metal regression",
                    "compiler_version": "recorded Metal regression",
                },
            },
        )

    def compare(self):
        result = subprocess.run(
            [
                sys.executable,
                str(self.snapshot / "ports/tilelang/handoff.py"),
                "verify-results",
                str(self.bundle),
                str(self.returned),
                "--output",
                str(self.output),
            ],
            capture_output=True,
            text=True,
        )
        self.assertTrue(self.output.is_file(), result.stderr)
        return result.returncode != 0
