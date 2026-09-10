"""Weight-free tests for reference validation, not SAM model parity."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(ROOT / "scripts"))
from preflight import check_artifact, check_source, confined_file, preflight, read_json
from check_parity import compare_array, compare_files, validate_rules


def float_rule(name="projection", **changes):
    return {"name": name, "mode": "float", "max_abs": 1e-5,
            "relative_l2": 1e-5, "zero_reference_floor": 1e-12} | changes


class ParityTests(unittest.TestCase):
    def test_identical(self):
        x = np.arange(128, dtype=np.float32).reshape(4, 32)
        r = compare_array(x, x.copy(), float_rule())
        self.assertTrue(r["pass"])
        self.assertEqual(r["max_abs"], 0)
        self.assertEqual(r["relative_l2"], 0)

    def test_both_metrics_required(self):
        # Large reference can hide a bad individual element in relative L2.
        x = np.full(10000, 1000, dtype=np.float32)
        y = x.copy(); y[57] += 0.1
        r = compare_array(x, y, float_rule())
        self.assertFalse(r["pass"])
        self.assertLess(r["relative_l2"], 1e-5)
        self.assertEqual(r["worst_flat_index"], 57)
        # Small values can pass absolute error while badly failing relative L2.
        x = np.full(10, 1e-8, dtype=np.float32)
        r = compare_array(x, x * 2, float_rule())
        self.assertLess(r["max_abs"], 1e-5)
        self.assertFalse(r["pass"])

    def test_shape_and_dtype_are_not_silently_normalized(self):
        x = np.arange(6, dtype=np.float32)
        for y in [x.reshape(2, 3), x.astype(np.float16)]:
            self.assertFalse(compare_array(x, y, float_rule())["pass"])

    def test_no_integer_to_float_precision_loss(self):
        x = np.array([2**60, 2**60 + 1], dtype=np.int64)
        y = x.copy(); y[1] += 1
        r = compare_array(x, y, {"name": "indices", "mode": "exact"})
        self.assertFalse(r["pass"])
        self.assertEqual(r["mismatched_elements"], 1)
        self.assertFalse(compare_array(x, x, float_rule())["pass"])

    def test_nonfinite_and_empty_never_pass(self):
        for value in [np.nan, np.inf, -np.inf]:
            x = np.array([value], dtype=np.float32)
            for mode in ["float", "exact"]:
                self.assertFalse(compare_array(x, x, float_rule(mode=mode))["pass"])
        self.assertFalse(compare_array(np.array([], dtype=np.float32),
                                       np.array([], dtype=np.float32), float_rule())["pass"])

    def test_zero_reference_and_chunked_reduction(self):
        x = np.zeros(131073, dtype=np.float32)
        self.assertTrue(compare_array(x, x, float_rule())["pass"])
        y = x.copy(); y[-1] = 1
        r = compare_array(x, y, float_rule())
        self.assertEqual(r["max_abs"], 1)
        self.assertEqual(r["relative_l2"], 1e12)
        self.assertEqual(r["worst_flat_index"], len(x) - 1)

    def test_rules_require_explicit_finite_thresholds(self):
        good = {"schema_version": 1, "boundary": "synthetic test only",
                "tensors": [float_rule()]}
        for field, value in [("max_abs", -1), ("max_abs", float("inf")),
                             ("relative_l2", float("nan")),
                             ("relative_l2", True), ("zero_reference_floor", 0)]:
            bad = copy.deepcopy(good); bad["tensors"][0][field] = value
            with self.assertRaises(ValueError): validate_rules(bad)
        bad = copy.deepcopy(good); del bad["tensors"][0]["max_abs"]
        with self.assertRaises(ValueError): validate_rules(bad)
        for bad in [good | {"tensors": []}, good | {"boundary": ""},
                    good | {"tensors": [float_rule(), float_rule()]}]:
            with self.assertRaises(ValueError): validate_rules(bad)

    def test_real_safetensors_order_and_missing_taps(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref, got = Path(tmp)/"ref.safetensors", Path(tmp)/"got.safetensors"
            x = np.array([1, 2], dtype=np.float32)
            save_file({"z_first": x, "a_second": x}, ref)
            save_file({"z_first": x + 1, "a_second": x + 2}, got)
            rules = {"schema_version": 1, "boundary": "synthetic projections",
                     "tensors": [float_rule("z_first"), float_rule("a_second")]}
            report = compare_files(ref, got, rules)
            self.assertFalse(report["pass"])
            self.assertEqual(report["first_failed_boundary"], "z_first")
            self.assertEqual(report["failed_count"], 2)
            save_file({"z_first": x}, got)
            with self.assertRaisesRegex(ValueError, "tensor set mismatch"):
                compare_files(ref, got, rules)
            save_file({"z_first": x, "a_second": x, "unexpected": x}, got)
            with self.assertRaisesRegex(ValueError, "unexpected"):
                compare_files(ref, got, rules)
            save_file({"z_first": x, "a_second": x}, got)
            self.assertTrue(compare_files(ref, got, rules)["pass"])

    def test_cli_failure_report_and_exit_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            rule = {"schema_version": 1, "boundary": "synthetic CLI",
                    "tensors": [float_rule()]}
            (p/"rules.json").write_text(json.dumps(rule))
            save_file({"projection": np.zeros(1, dtype=np.float32)}, p/"ref.safetensors")
            save_file({"projection": np.ones(1, dtype=np.float32)}, p/"got.safetensors")
            run = subprocess.run([sys.executable, str(ROOT/"scripts/check_parity.py"),
                                  "--reference", str(p/"ref.safetensors"),
                                  "--candidate", str(p/"got.safetensors"),
                                  "--rules", str(p/"rules.json"),
                                  "--report", str(p/"report.json")],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertFalse(json.loads((p/"report.json").read_text())["pass"])


class PreflightTests(unittest.TestCase):
    def test_bytes_hash_size_and_path_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            content = b"opaque bytes, not deserialized"
            (p/"model.ckpt").write_bytes(content)
            entry = {"path": "model.ckpt", "bytes": len(content),
                     "sha256": hashlib.sha256(content).hexdigest()}
            self.assertTrue(check_artifact(p, entry)["verified_local_bytes"])
            for bad in [entry | {"sha256": "0"*64}, entry | {"bytes": 0},
                        entry | {"bytes": len(content)+1}, entry | {"bytes": True},
                        entry | {"sha256": "not-a-hash"}]:
                with self.assertRaises(ValueError): check_artifact(p, bad)
            for rel in ["../model.ckpt", "/etc/passwd", "", "..\\model.ckpt"]:
                with self.assertRaises(ValueError): confined_file(p, rel)
            (p/"outside").symlink_to("/etc/passwd")
            with self.assertRaises(ValueError): confined_file(p, "outside")

    def test_json_rejects_duplicates_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"input.json"
            for bad in ['{"a":1,"a":2}', '{"a":NaN}']:
                p.write_text(bad)
                with self.assertRaises(ValueError): read_json(p)

    def test_source_and_incomplete_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            source = p/"source"; source.mkdir()
            def git(*args):
                return subprocess.check_output(["git", "-c", "core.hooksPath=/dev/null",
                                                "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                                                "-c", "commit.gpgsign=false", "-C", str(source), *args],
                                               stderr=subprocess.PIPE, text=True).strip()
            git("init", "--quiet")
            (source/"source.txt").write_text("official-ish synthetic input\n")
            git("add", "source.txt"); git("commit", "--quiet", "-m", "test input")
            revision = git("rev-parse", "HEAD")
            self.assertTrue(check_source(source, revision)["clean"])
            manifest = {"schema_version": 1, "dependency_closure_complete": False,
                        "source_revisions": [{"role": "official_body_oracle", "revision": revision,
                                               "repository": "https://example.invalid/source"}],
                        "model_repositories": [{"repository": "facebook/sam-3d-body-dinov3",
                                                "revision": "1"*40,
                                                "selected_files": [{"path": "model.ckpt", "bytes": 1,
                                                                    "sha256": "0"*64}]}]}
            result = preflight(manifest, "body", source, None)
            self.assertTrue(result["pass"])
            self.assertTrue(result["source_only"])
            self.assertFalse(result["reference_ready"])
            result = preflight(manifest, "body", source, p)
            self.assertFalse(result["pass"])
            self.assertFalse(result["reference_ready"])
            with self.assertRaises(ValueError): check_source(source, "0"*40)
            (source/"untracked.txt").write_text("unexpected")
            with self.assertRaises(ValueError): check_source(source, revision)


if __name__ == "__main__":
    unittest.main()
