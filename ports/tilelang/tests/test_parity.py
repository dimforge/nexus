"""Negative tests for silent drift and misleading validation claims."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import parity
import cpu_oracle


class ParityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.port = self.root / parity.PORT
        (self.root / "src").mkdir()
        (self.root / "src/solver.rs").write_text("fn solve() { normal_impulse(); }\n")
        self.port.mkdir(parents=True)
        (self.port / "reference").mkdir()
        (self.port / "reference/noalias_probe.rs").write_text("// test probe\n")
        (self.port / "kernel.py").write_text("# Implementation\n")
        (self.port / "requirements.txt").write_text("tilelang==0.1.15\n")
        parity.write_json(self.port / "contract.json", {"required_cases": ["case"]})
        parity.write_json(
            self.port / "parity.json",
            {
                "schema": 1,
                "reference_scope": ["src/**/*.rs"],
                "implementation_scope": [
                    "ports/tilelang/*.py",
                    "ports/tilelang/contract.json",
                    "ports/tilelang/parity.json",
                    "ports/tilelang/requirements.txt",
                ],
                "features": [{"id": "solve"}],
                "targets": ["c", "metal", "ascend"],
                "required_reconciliation_targets": ["c"],
            },
        )
        self.change = self.port / "changes/initial.json"
        parity.write_json(
            self.change,
            {
                "id": "initial",
                "summary": "Port solver",
                "features": ["solve"],
                "kinds": ["port"],
                "disposition": "implementation_updated",
                "regressions": ["case"],
                "differences": ["No intended behavior change"],
            },
        )
        self.evidence = self.root / "report.json"
        self.report = {
            "schema": 1,
            "feature": "solve",
            "target": "c",
            "status": "passed",
            "level": "executed",
            "suite": "portable",
            "environment": {
                "tilelang": "0.1.15",
                "rustc": "rustc test",
                "rustflags": list(cpu_oracle.FLAGS),
                "cpu_oracle": {
                    "profile": cpu_oracle.PROFILE,
                    "rustc": "rustc test",
                    "rustflags": list(cpu_oracle.FLAGS),
                    "rustc_bootstrap": "1",
                    "mutable_reference_noalias": False,
                    "shared_reference_noalias": True,
                    "probe_source_sha256": parity.digest(
                        self.port / "reference/noalias_probe.rs"
                    ),
                    "llvm_ir_sha256": "a" * 64,
                },
            },
            "inputs": parity.evidence_inputs(self.root),
            "cases": [
                {
                    "name": "case",
                    "status": "passed",
                    "max_absolute_errors": {"Tiles": 0.0},
                }
            ],
        }
        parity.write_json(self.evidence, self.report)

    def test_smoke_selection_never_satisfies_full_qualification(self):
        self.report.update(suite="representative_smoke", selected_cases=["case"])
        inputs = parity.evidence_inputs(self.root)
        self.assertEqual(
            parity.smoke_evidence_state(self.root, self.report, inputs, ["case"]),
            "executed",
        )
        self.assertEqual(
            parity.evidence_state(self.root, self.report, inputs), "diagnostic_only"
        )
        for selection in ([], ["typo"], ["case", "case"]):
            self.assertNotEqual(
                parity.smoke_evidence_state(self.root, self.report, inputs, selection),
                "executed",
            )
        parity.write_json(self.evidence, self.report)
        with self.assertRaises(ValueError):
            self.accept()

    def accept(self):
        with patch.object(parity.subprocess, "run"):
            return parity.reconcile(self.root, self.change, [self.evidence])

    def test_duplicate_overrides_cannot_hide_a_failed_report(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parity.status(self.root, [self.evidence, self.evidence])

    def test_success_is_scoped_and_ascend_is_unvalidated(self):
        result = self.accept()
        self.assertEqual(result["source_reconciliation"], "current")
        self.assertEqual(result["backends"]["c"]["state"], "executed")
        self.assertEqual(result["backends"]["ascend"]["state"], "unvalidated")

    def test_rust_semantic_change_invalidates_evidence(self):
        self.accept()
        (self.root / "src/solver.rs").write_text(
            "fn solve() { fixed_normal_impulse(); }\n"
        )
        result = parity.status(self.root)
        self.assertEqual(result["source_reconciliation"], "required")
        self.assertEqual(result["changes"]["reference"]["modified"], ["src/solver.rs"])
        self.assertEqual(result["affected_features"], ["solve"])
        self.assertEqual(result["backends"]["c"]["state"], "stale")

    def test_new_rust_feature_cannot_be_silently_unmapped(self):
        self.accept()
        (self.root / "src/joint.rs").write_text("fn new_joint() {}\n")
        result = parity.status(self.root)
        self.assertEqual(result["unmapped_reference_files"], ["src/joint.rs"])
        self.assertEqual(result["source_reconciliation"], "required")

    def test_deleted_dependency_invalidates_evidence(self):
        self.accept()
        (self.root / "src/solver.rs").unlink()
        self.assertEqual(
            parity.status(self.root)["changes"]["reference"]["removed"],
            ["src/solver.rs"],
        )

    def test_new_implementation_file_invalidates_evidence(self):
        self.accept()
        (self.port / "helper.py").write_text("# new helper\n")
        self.assertEqual(parity.status(self.root)["backends"]["c"]["state"], "stale")

    def test_tolerance_change_invalidates_evidence(self):
        self.accept()
        parity.write_json(
            self.port / "contract.json", {"required_cases": ["case"], "atol": 100}
        )
        self.assertEqual(parity.status(self.root)["backends"]["c"]["state"], "stale")

    def test_baseline_hash_bump_does_not_refresh_evidence(self):
        self.accept()
        (self.root / "src/solver.rs").write_text("fn wrong_solve() {}\n")
        path = self.port / "baseline.json"
        baseline = parity.read_json(path)
        baseline["inputs"] = parity.evidence_inputs(self.root)
        parity.write_json(path, baseline)
        self.assertEqual(parity.status(self.root)["backends"]["c"]["state"], "stale")

    def test_reconciliation_requires_execution_and_change_record(self):
        with self.assertRaisesRegex(ValueError, "execution evidence"):
            parity.reconcile(self.root, self.change, [])
        self.report["level"] = "source_lowered"
        parity.write_json(self.evidence, self.report)
        with self.assertRaisesRegex(ValueError, "execution evidence"):
            parity.reconcile(self.root, self.change, [self.evidence])
        self.report["level"] = "executed"
        parity.write_json(self.evidence, self.report)
        parity.write_json(self.change, {"summary": "just update hashes"})
        with self.assertRaisesRegex(ValueError, "nonempty fields"):
            parity.reconcile(self.root, self.change, [self.evidence])

    def test_failed_incomplete_or_wrong_toolchain_report_is_rejected(self):
        for mutation in [
            {"status": "failed"},
            {"cases": []},
            {"suite": "boundary-stress"},
            {
                "cases": [
                    {
                        "name": "other_case",
                        "status": "passed",
                        "max_absolute_errors": {},
                    }
                ]
            },
            {"environment": {"tilelang": "unreviewed"}},
        ]:
            with self.subTest(mutation=mutation):
                parity.write_json(self.evidence, {**self.report, **mutation})
                with self.assertRaises(ValueError):
                    parity.reconcile(self.root, self.change, [self.evidence])

    def test_missing_or_unsafe_cpu_oracle_is_rejected(self):
        for mutation in (
            {},
            {"mutable_reference_noalias": True},
            {"shared_reference_noalias": False},
            {"rustflags": []},
            {"rustc": "different"},
            {"rustc_bootstrap": "0"},
            {"probe_source_sha256": "b" * 64},
            {"llvm_ir_sha256": "bad"},
        ):
            with self.subTest(mutation=mutation):
                env = dict(self.report["environment"])
                env["cpu_oracle"] = (
                    {**env["cpu_oracle"], **mutation} if mutation else {}
                )
                report = {**self.report, "environment": env}
                for level in ("executed", "source_lowered"):
                    self.assertEqual(
                        parity.evidence_state(
                            self.root,
                            {**report, "level": level},
                            parity.evidence_inputs(self.root),
                        ),
                        "oracle_configuration_mismatch",
                    )

    def test_modifying_accepted_evidence_or_change_record_is_visible(self):
        self.accept()
        (self.port / "evidence/c.json").write_text("{}")
        self.assertEqual(
            parity.status(self.root)["backends"]["c"]["state"],
            "missing_or_modified_evidence",
        )
        self.change.write_text("{}")
        self.assertEqual(parity.status(self.root)["source_reconciliation"], "required")

    def test_release_gate_rejects_explicit_known_gap(self):
        contract = self.port / "contract.json"
        parity.write_json(
            contract,
            {
                "required_cases": ["case"],
                "known_gaps": [
                    {
                        "id": "numeric-boundary",
                        "description": "Unresolved boundary",
                        "blocks_release": True,
                    }
                ],
            },
        )
        self.report["inputs"] = parity.evidence_inputs(self.root)
        parity.write_json(self.evidence, self.report)
        self.accept()
        with (
            patch.object(parity, "ROOT", self.root),
            patch.object(sys, "argv", ["parity.py", "check", "--release"]),
        ):
            self.assertEqual(parity.main(), 1)

    def test_reconciling_cpu_keeps_old_metal_evidence_stale(self):
        metal = self.root / "metal.json"
        parity.write_json(metal, {**self.report, "target": "metal"})
        with patch.object(parity.subprocess, "run"):
            parity.reconcile(self.root, self.change, [self.evidence, metal])
        (self.port / "kernel.py").write_text("# changed implementation\n")
        self.report["inputs"] = parity.evidence_inputs(self.root)
        parity.write_json(self.evidence, self.report)
        with self.assertRaisesRegex(ValueError, "new change record"):
            self.accept()
        record = parity.read_json(self.change)
        self.change = self.port / "changes/followup.json"
        parity.write_json(self.change, {**record, "id": "followup"})
        result = self.accept()
        self.assertEqual(result["backends"]["c"]["state"], "executed")
        self.assertEqual(result["backends"]["metal"]["state"], "stale")


class MultiFeatureTests(unittest.TestCase):
    setUp = ParityTests.setUp
    accept = ParityTests.accept

    def add_feature(self):
        path = self.port / "parity.json"
        config = parity.read_json(path)
        config["features"].append(
            {"id": "integrate", "contract": "ports/tilelang/integrate.json"}
        )
        parity.write_json(path, config)
        parity.write_json(
            self.port / "integrate.json", {"required_cases": ["integration-case"]}
        )
        record = parity.read_json(self.change)
        record["features"].append("integrate")
        parity.write_json(self.change, record)
        self.report["inputs"] = parity.evidence_inputs(self.root)
        parity.write_json(self.evidence, self.report)

    def test_one_operation_cannot_certify_another(self):
        self.add_feature()
        with self.assertRaisesRegex(ValueError, "integrate"):
            self.accept()
        result = parity.status(self.root, [self.evidence])
        self.assertEqual(result["feature_backends"]["solve"]["c"]["state"], "executed")
        self.assertEqual(
            result["feature_backends"]["integrate"]["c"]["state"], "unvalidated"
        )

    def test_contract_is_selected_by_feature(self):
        self.add_feature()
        wrong = {**self.report, "feature": "integrate"}
        self.assertEqual(
            parity.evidence_state(self.root, wrong, parity.evidence_inputs(self.root)),
            "incomplete",
        )
        wrong["cases"] = [
            {
                "name": "integration-case",
                "status": "passed",
                "max_absolute_errors": {"Poses": 0.0},
            }
        ]
        second = self.root / "integrate-report.json"
        parity.write_json(second, wrong)
        with patch.object(parity.subprocess, "run"):
            result = parity.reconcile(self.root, self.change, [self.evidence, second])
        self.assertEqual(
            result["feature_backends"]["integrate"]["c"]["state"], "executed"
        )
        self.assertEqual(result["feature_backends"]["solve"]["c"]["state"], "executed")
        self.assertEqual(
            result["feature_backends"]["integrate"]["ascend"]["state"], "unvalidated"
        )


if __name__ == "__main__":
    unittest.main()
