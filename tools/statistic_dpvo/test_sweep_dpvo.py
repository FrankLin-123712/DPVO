#!/usr/bin/env python3
"""Regression tests for statistic_dpvo/sweep_dpvo.py."""

from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
