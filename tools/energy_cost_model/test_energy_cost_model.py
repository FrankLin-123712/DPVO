from dataclasses import replace
from pathlib import Path
import unittest

from actions import estimate_dense_on_gemmini, lower_memory_bytes
from model import estimate_energy
from parameters import AlgorithmParams, EnergyTable, HardwareParams, MappingParams
from workloads import DenseOp, build_update_dense


class ParameterAndWorkloadTests(unittest.TestCase):
    def test_steady_edges_account_for_immature_recent_sources(self):
        params = AlgorithmParams()
        self.assertEqual(params.active_source_frames, 23)
        self.assertEqual(params.active_edges, 47_712)
        self.assertEqual(params.active_unique_patches, 2_208)

    def test_new_frame_graph_statistics(self):
        params = replace(AlgorithmParams(), edge_mode="new-frame")
        self.assertEqual(params.active_edges, 2_400)
        self.assertEqual(params.active_unique_patches, 1_248)
        self.assertEqual(params.active_unique_frame_pairs, 25)

    def test_short_removal_window_only_reintroduces_one_edge_per_old_patch(self):
        params = AlgorithmParams(
            patches_per_frame=10,
            patch_lifetime=5,
            removal_window=2,
        )
        self.assertEqual(params.active_source_frames, 5)
        self.assertEqual(params.active_edges, 200)
        self.assertEqual(params.active_unique_patches, 50)
        new_params = replace(params, edge_mode="new-frame")
        self.assertEqual(new_params.active_edges, 90)
        self.assertEqual(new_params.active_unique_patches, 50)

    def test_softagg_h_uses_unique_groups(self):
        params = AlgorithmParams(unique_patches=123, unique_frame_pairs=17)
        ops = {op.name: op for op in build_update_dense(params).dense_ops}
        self.assertEqual(ops["update.agg_kk.f"].m, params.active_edges)
        self.assertEqual(ops["update.agg_kk.h"].m, 123)
        self.assertEqual(ops["update.agg_ij.h"].m, 17)

    def test_invalid_hit_rate_is_rejected(self):
        with self.assertRaises(ValueError):
            replace(HardwareParams.fp16_default(), random_l1_hit_rate=1.2)

    def test_hardware_precision_and_storage_width_must_match(self):
        with self.assertRaises(ValueError):
            replace(HardwareParams.fp16_default(), input_bytes=1)

    def test_unimplemented_accelerator_mapping_is_rejected(self):
        with self.assertRaises(ValueError):
            MappingParams(ba="gemmini")

    def test_negative_unit_energy_is_rejected(self):
        with self.assertRaises(ValueError):
            EnergyTable({"gemmini.mac": -1.0})

    def test_unique_group_override_cannot_exceed_edges(self):
        with self.assertRaises(ValueError):
            AlgorithmParams(edges=10, unique_patches=11)

    def test_missing_generated_header_is_not_silent(self):
        with self.assertRaises(FileNotFoundError):
            HardwareParams.from_gemmini_header(Path("definitely-missing-gemmini-header.h"))


class ActionModelTests(unittest.TestCase):
    def test_spad_reads_are_tile_level_not_mac_level(self):
        hardware = HardwareParams.fp16_default()
        op = DenseOp("tiny", m=4, n=4, k=4, input_bytes=2, weight_bytes=2, output_bytes=2)
        estimate = estimate_dense_on_gemmini(op, hardware, True, "WS")
        self.assertEqual(estimate.executed_macs, 64)
        self.assertEqual(estimate.actions.counts["spad.read_byte"], 64)

    def test_spad_capacity_changes_reload_traffic(self):
        op = DenseOp("capacity", m=16, n=16, k=4, input_bytes=2, weight_bytes=2, output_bytes=2)
        large = replace(HardwareParams.fp16_default(), sp_capacity_kib=0.25)
        small = replace(HardwareParams.fp16_default(), sp_capacity_kib=0.125)
        large_estimate = estimate_dense_on_gemmini(op, large, True, "WS")
        small_estimate = estimate_dense_on_gemmini(op, small, True, "WS")
        self.assertGreater(small_estimate.dram_read_bytes, large_estimate.dram_read_bytes)

    def test_capacity_blocking_avoids_full_a_reload_per_n_tile(self):
        hardware = replace(
            HardwareParams.fp16_default(),
            sp_capacity_kib=1,
            acc_capacity_kib=1,
        )
        op = DenseOp("blocked", m=64, n=64, k=16, input_bytes=2, weight_bytes=2, output_bytes=2)
        estimate = estimate_dense_on_gemmini(op, hardware, True, "WS")
        old_all_or_nothing_bytes = (op.m * op.k * op.input_bytes) * 16 + (
            op.k * op.n * op.weight_bytes
        )
        self.assertEqual(estimate.dram_read_bytes, 16_384)
        self.assertLess(estimate.dram_read_bytes, old_all_or_nothing_bytes)

    def test_precision_mismatch_requires_explicit_mapping(self):
        hardware = HardwareParams.default_int8()
        op = DenseOp("fp16", m=4, n=4, k=4, input_bytes=2, weight_bytes=2, output_bytes=2)
        with self.assertRaises(ValueError):
            estimate_dense_on_gemmini(op, hardware, True, "WS")

    def test_random_reads_include_cache_line_amplification(self):
        hardware = HardwareParams.fp16_default()
        l2_bytes, dram_bytes = lower_memory_bytes(
            16,
            "random",
            hardware,
            random_line_utilization=0.25,
        )
        self.assertAlmostEqual(l2_bytes, 64 * (1 - hardware.random_l1_hit_rate))
        self.assertAlmostEqual(
            dram_bytes,
            l2_bytes * (1 - hardware.random_l2_hit_rate),
        )


class TrendTests(unittest.TestCase):
    def test_fewer_patches_reduce_energy_on_fixed_hardware(self):
        hardware = HardwareParams.fp16_default()
        mapping = MappingParams()
        baseline = estimate_energy(AlgorithmParams(patches_per_frame=96), hardware, mapping)
        reduced = estimate_energy(AlgorithmParams(patches_per_frame=64), hardware, mapping)
        self.assertLess(reduced.total_energy_pj, baseline.total_energy_pj)

    def test_target_rate_power_is_energy_times_rate(self):
        report = estimate_energy(
            AlgorithmParams(patches_per_frame=16, edge_mode="new-frame"),
            HardwareParams.fp16_default(),
            MappingParams(),
            target_fps=30,
        )
        self.assertAlmostEqual(
            report.target_dynamic_power_w,
            report.total_energy_pj * 1e-12 * 30,
        )


if __name__ == "__main__":
    unittest.main()
