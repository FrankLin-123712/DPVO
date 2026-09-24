#!/usr/bin/env python3
"""Regression tests for statistic_dpvo/sweep_dpvo.py."""

from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sweep_dpvo


class SweepDpvoTests(unittest.TestCase):
    def test_default_sweeps_start_from_default_yaml_values(self) -> None:
        base = {
            "PATCHES_PER_FRAME": 96,
            "PATCH_LIFETIME": 13,
            "REMOVAL_WINDOW": 22,
            "OPTIMIZATION_WINDOW": 10,
            "BA_ITERATIONS": 2,
        }
        sweeps = sweep_dpvo.default_sweeps(base)
        self.assertEqual(sweeps["PATCHES_PER_FRAME"][0], 96)
        self.assertEqual(sweeps["PATCH_LIFETIME"][0], 13)
        self.assertEqual(sweeps["REMOVAL_WINDOW"][0], 22)
        self.assertEqual(sweeps["OPTIMIZATION_WINDOW"][0], 10)
        self.assertEqual(sweeps["BA_ITERATIONS"], [20, 18, 16, 14, 12, 10, 8, 6, 4, 2])
        self.assertEqual(sweeps["IMAGE_SIZE"][0], (480, 640))

    def test_write_candidate_config_only_changes_requested_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "default.yaml"
            output = Path(directory) / "candidate.yaml"
            source.write_text(
                "PATCHES_PER_FRAME: 96\n"
                "PATCH_LIFETIME: 13\n"
                "CENTROID_SEL_STRAT: 'RANDOM'\n"
            )
            sweep_dpvo.write_candidate_config(source, output, {"PATCHES_PER_FRAME": 64})
            rendered = output.read_text()
        self.assertIn("PATCHES_PER_FRAME: 64", rendered)
        self.assertIn("PATCH_LIFETIME: 13", rendered)
        self.assertIn("CENTROID_SEL_STRAT: 'RANDOM'", rendered)

    def test_candidate_id_is_stable_for_scalar_and_image_size(self) -> None:
        self.assertEqual(sweep_dpvo.candidate_id("PATCHES_PER_FRAME", 64), "patches_per_frame_064")
        self.assertEqual(sweep_dpvo.candidate_id("IMAGE_SIZE", (384, 512)), "image_size_384x512")

    def test_report_preserves_custom_run_settings_and_partial_ate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = sweep_dpvo.parse_args([
                "--config", str(root / "fast config.yaml"),
                "--result-dir", str(root / "results"),
                "--sweep-file", str(root / "custom.json"),
                "--parameters", "BA_ITERATIONS", "--trials", "1",
                "--scenes", "MH_01_easy", "--reuse-eval", "--no-keyframe",
            ])
            rows = [dict(sweep_parameter="BA_ITERATIONS", sweep_value=str(value),
                         total_ops=value * 10, total_memory_bytes=value * 20, ate_m=ate)
                    for value, ate in ((2, ""), (6, 0.1))]
            with patch.object(sweep_dpvo, "TOOLS_DIR", root):
                sweep_dpvo.write_report(args, rows)
            report = (root / "one_at_a_time_sweep_report.md").read_text()
        self.assertIn("--trials 1", report)
        self.assertIn("--reuse-eval", report)
        self.assertIn("--no-keyframe", report)
        self.assertNotIn("--run-eval", report)
        self.assertIn("fast config.yaml", report)
        self.assertIn("--sweep-file", report)
        self.assertIn("`BA_ITERATIONS`: 2, 6", report)
        self.assertIn("1/2 candidates", report)
        self.assertIn("-200.0%", report)
        self.assertNotIn("V2_03_difficult", report)
        self.assertNotIn("`PATCHES_PER_FRAME`:", report)


if __name__ == "__main__":
    unittest.main()
