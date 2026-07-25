#!/usr/bin/env python3
"""Regression tests for statistic_dpvo/statistic_dpvo.py."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import statistic_dpvo as statistic


class StatisticDpvoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.args = statistic.parse_args(["--format", "json"])
        cls.pa, cls.graph, cls.rows = statistic.build_rows(cls.args)

    def test_default_is_a_mature_cpp_factor_snapshot(self) -> None:
        self.assertEqual(self.graph.frame_count, statistic.DEFAULT_ACTIVE_FRAMES)
        self.assertEqual(self.graph.edge_count, 47_712)
        self.assertEqual(self.graph.post_prune_edge_count, 45_312)
        self.assertEqual(self.graph.new_edge_count, 2_400)
        self.assertEqual(self.graph.unique_patches, 2_208)
        self.assertEqual(self.graph.edge_groups, 497)
        self.assertEqual(self.graph.free_pose_count, 10)

    def test_checked_in_onnx_node_boundaries_are_preserved(self) -> None:
        feature = [row for row in self.rows if row.source == "ONNX:feature_extractor.onnx"]
        update = [row for row in self.rows if row.source == "ONNX:update_block.onnx"]
        self.assertEqual(len(feature), 154)
        self.assertEqual(len(update), 311)

    def test_dynamic_scatter_shapes_use_factor_graph_groups(self) -> None:
        kk = next(row for row in self.rows if row.layer == "/agg_kk/Add")
        ij = next(row for row in self.rows if row.layer == "/agg_ij/Add")
        self.assertIn(f"1x{self.graph.unique_patches}x384", kk.shape)
        self.assertIn(f"1x{self.graph.edge_groups}x384", ij.shape)

    def test_shape_gather_is_not_labeled_factor_graph_access(self) -> None:
        row = next(row for row in self.rows if row.layer == "/fnet/Gather")
        self.assertEqual(row.access_pattern, "regular metadata")

    def test_execution_order_places_postprocess_after_update_onnx(self) -> None:
        weight = next(index for index, row in enumerate(self.rows) if row.layer == "/w/w.3/Sigmoid")
        target = next(index for index, row in enumerate(self.rows) if row.layer == "TargetAndWeight")
        linearize = next(index for index, row in enumerate(self.rows) if row.layer == "LinearizePatchCenters")
        self.assertLess(weight, target)
        self.assertLess(target, linearize)

    def test_correlation_includes_bilinear_and_dot_product(self) -> None:
        row = next(row for row in self.rows if row.layer == "BuildCorrelationVolume")
        samples = self.graph.edge_count * 3 * 3 * 2 * 49
        self.assertEqual(row.fp32_ops, samples * (19 * 128 + 3))

    def test_reproject_reads_complete_cpp_patch_tensor(self) -> None:
        row = next(row for row in self.rows if row.layer == "ReprojectPatchGrid")
        active_patches = self.graph.frame_count * self.pa.patches_per_frame
        expected = (
            statistic.tensor_bytes((self.graph.referenced_frame_count, 7))
            + statistic.tensor_bytes((active_patches, 3, 3, 3))
            + statistic.tensor_bytes((self.graph.referenced_frame_count, 4))
            + statistic.tensor_bytes((self.graph.edge_count, 3), "int64")
        )
        self.assertEqual(row.mem_read_bytes, expected)

    def test_pa_override_scales_patch_multiplicity(self) -> None:
        args = statistic.parse_args(["--pa", "PATCHES_PER_FRAME=48"])
        pa, graph, _ = statistic.build_rows(args)
        self.assertEqual(pa.patches_per_frame, 48)
        self.assertEqual(graph.edge_count, self.graph.edge_count // 2)
        self.assertEqual(graph.unique_patches, self.graph.unique_patches // 2)
        self.assertEqual(graph.edge_groups, self.graph.edge_groups)

    def test_startup_snapshot_is_rejected_as_nonsteady(self) -> None:
        args = statistic.parse_args(["--active-frames", "23"])
        with self.assertRaisesRegex(ValueError, "steady-frame"):
            statistic.build_rows(args)

    def test_invalid_manual_graph_overrides_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "edges must be positive"):
            statistic.build_rows(statistic.parse_args(["--edges", "0"]))
        with self.assertRaisesRegex(ValueError, "fixed pose"):
            statistic.build_rows(
                statistic.parse_args(["--active-frames", "35", "--free-poses", "35"])
            )

    def test_unknown_pa_key_is_rejected_instead_of_silently_ignored(self) -> None:
        args = statistic.parse_args(["--pa", "PATCH_LIFETME=9"])
        with self.assertRaisesRegex(ValueError, "unsupported operation-count P_a key"):
            statistic.build_rows(args)

    def test_total_and_json_preserve_exact_byte_invariants(self) -> None:
        total = statistic.sum_rows(self.rows)
        self.assertEqual(total.total_ops, sum(row.total_ops for row in self.rows))
        self.assertEqual(
            total.total_memory_bytes,
            sum(row.mem_read_bytes + row.mem_write_bytes for row in self.rows),
        )
        payload = {
            "rows": [row.raw_dict() for row in self.rows],
            "total": total.raw_dict(),
        }
        decoded = json.loads(json.dumps(payload, allow_nan=False))
        self.assertTrue(
            all(
                row["total_memory_bytes"]
                == row["mem_read_bytes"] + row["mem_write_bytes"]
                for row in decoded["rows"]
            )
        )

    def test_csv_contains_requested_columns_and_total(self) -> None:
        total = statistic.sum_rows(self.rows)
        rendered = statistic.render_csv([*self.rows, total])
        header = rendered.splitlines()[0]
        for column in (
            "Module",
            "Layer",
            "Shape",
            "FP32 Ops",
            "FP64 Ops",
            "Total Ops",
            "Mem Read (Bytes)",
            "Mem Write (Bytes)",
            "Total Memory (Bytes)",
            "Op Intensity (Ops/Byte)",
            "Memory Access Pattern",
        ):
            self.assertIn(column, header)
        self.assertIn("TOTAL,TOTAL", rendered)
        self.assertIn(statistic.format_si_number(total.total_ops), rendered)
        self.assertIn(statistic.format_si_number(total.total_memory_bytes), rendered)

    def test_decimal_si_number_format(self) -> None:
        self.assertEqual(statistic.format_si_number(0), "0")
        self.assertEqual(statistic.format_si_number(999), "999")
        self.assertEqual(statistic.format_si_number(1_000), "1K")
        self.assertEqual(statistic.format_si_number(38_592), "38.592K")
        self.assertEqual(statistic.format_si_number(614_400), "614.4K")
        self.assertEqual(statistic.format_si_number(12_419_481_600), "12.419G")

    def test_new_granularity_flags_are_mutually_exclusive(self) -> None:
        self.assertEqual(statistic.parse_args(["--per-module"]).detail, "module")
        self.assertEqual(statistic.parse_args(["--per-layer"]).detail, "layer")
        self.assertEqual(statistic.parse_args([]).detail, "layer")
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            statistic.parse_args(["--per-module", "--per-layer"])

    def test_per_module_json_aggregates_rows_without_changing_total(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "module.json"
            result = statistic.main(
                ["--per-module", "--format", "json", "--output", str(output)]
            )
            self.assertEqual(result, 0)
            payload = json.loads(output.read_text())
        expected_modules = list(dict.fromkeys(row.module for row in self.rows))
        self.assertEqual(payload["workload"]["output_granularity"], "per-module")
        self.assertEqual(len(payload["rows"]), len(expected_modules))
        self.assertTrue(all(row["layer"] == "MODULE TOTAL" for row in payload["rows"]))
        self.assertEqual(payload["total"]["total_ops"], statistic.sum_rows(self.rows).total_ops)
        self.assertIsInstance(payload["total"]["total_ops"], int)


if __name__ == "__main__":
    unittest.main()
