"""CPU-only checks for replay serialization and interception (no torch import)."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from generate_correlation_replay_testdata import (
    CorrelationRecorder, ReplayWriter, capture_inputs, is_feature, parse_args,
)


class HostTensor:
    """Minimal tensor adapter to exercise the recorder without CUDA extensions."""
    def __init__(self, array):
        self.array = np.asarray(array)

    def __getitem__(self, item):
        return HostTensor(self.array[item])

    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return self.array


class Tracker:
    def __init__(self):
        self.M, self.pmem, self.mem = 2, 3, 2
        # Includes wrapped patch/frame indices, duplicate targets, and non-sorted edges.
        self.pg = SimpleNamespace(kk=HostTensor(np.array([7, 0, 6], dtype=np.int64)),
                                  jj=HostTensor(np.array([3, 0, 3], dtype=np.int64)))
        self.gmap_ = HostTensor(np.arange(3 * 2 * 128 * 9, dtype=np.float32).reshape(3, 2, 128, 3, 3))
        self.pyramid = (HostTensor(np.ones((1, 2, 128, 8, 8), dtype=np.float32)),
                        HostTensor(np.ones((1, 2, 128, 2, 2), dtype=np.float32)))
        self.coords = HostTensor(np.zeros((1, 3, 2, 3, 3), dtype=np.float32))
        self.result = HostTensor(np.arange(3 * 882, dtype=np.float32).reshape(1, 3, 882))
        self.n, self.counter, self.is_initialized = 2, 3, False
        self.raise_error = False

    def corr(self, coords, indicies=None):
        if self.raise_error:
            raise RuntimeError("original correlation failed")
        return self.result

    def update(self):
        return self.corr(self.coords)

    def motion_probe(self):
        return self.corr(self.coords, indicies=(self.pg.kk, self.pg.jj))


class ReplayTests(unittest.TestCase):
    def test_wrap_mapping_and_owned_snapshots(self):
        tracker = Tracker()
        arrays = capture_inputs(tracker, tracker.coords)
        np.testing.assert_array_equal(arrays["kk"], [7, 0, 6])
        np.testing.assert_array_equal(arrays["gmap_slot_ids"], [0])
        np.testing.assert_array_equal(arrays["fmap_slot_ids"], [0, 1])
        # Resolving each exported slot exactly reconstructs Python's modulo lookup.
        for k in arrays["kk"]:
            exported = arrays[f"gmap_slot_{(k // tracker.M) % tracker.pmem:03d}"][k % tracker.M]
            original = tracker.gmap_.array.reshape(-1, 128, 3, 3)[k % (tracker.M * tracker.pmem)]
            np.testing.assert_array_equal(exported, original)
        tracker.gmap_.array.fill(-1)
        tracker.coords.array.fill(-2)
        self.assertEqual(arrays["gmap_slot_000"][0, 0, 0, 0], 0)
        self.assertEqual(arrays["coords"][0, 0, 0, 0], 0)

    def test_probe_indices_override_graph_and_empty_edges(self):
        tracker = Tracker()
        indices = (HostTensor(np.array([5], dtype=np.int64)), HostTensor(np.array([4], dtype=np.int64)))
        arrays = capture_inputs(tracker, HostTensor(np.zeros((1, 1, 2, 3, 3), dtype=np.float32)), indices)
        np.testing.assert_array_equal(arrays["gmap_slot_ids"], [2])
        np.testing.assert_array_equal(arrays["fmap_slot_ids"], [0])
        empty = HostTensor(np.array([], dtype=np.int64))
        arrays = capture_inputs(tracker, HostTensor(np.zeros((1, 0, 2, 3, 3), dtype=np.float32)), (empty, empty))
        self.assertEqual(arrays["coords"].shape, (0, 2, 3, 3))
        self.assertFalse(any(is_feature(k) for k in arrays))

    def test_reject_half_and_nonfinite_input(self):
        tracker = Tracker()
        tracker.pyramid[0].array = tracker.pyramid[0].array.astype(np.float16)
        with self.assertRaisesRegex(ValueError, "float32"):
            capture_inputs(tracker, tracker.coords)
        tracker = Tracker()
        tracker.coords.array.fill(np.nan)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            capture_inputs(tracker, tracker.coords)

    def test_all_stages_restore_and_versioned_binary_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tracker = Tracker()
            recorder = CorrelationRecorder(tracker, ReplayWriter(root))
            with recorder.installed():
                recorder.start_frame(0, Path("frame0.png"))
                self.assertIs(tracker.motion_probe(), tracker.result)
                tracker.update()
                tracker.update()
                tracker.is_initialized = True
                recorder.start_frame(1, Path("frame1.png"))
                tracker.update()
                # Simulate a slot being overwritten or moved during keyframe removal.
                tracker.pyramid[0].array[0, 1].fill(5)
                recorder.terminating = True
                recorder.next_iteration = 0
                tracker.update()
            self.assertNotIn("corr", tracker.__dict__)
            self.assertNotIn("update", tracker.__dict__)
            self.assertNotIn("motion_probe", tracker.__dict__)
            records = recorder.writer.records
            self.assertEqual([r["stage"] for r in records], ["motion_probe", "initialization", "initialization", "update", "terminate"])
            self.assertEqual([r["iteration"] for r in records], [None, 0, 1, 0, 0])
            feature = "fmap1_slot_001"
            self.assertEqual(records[0]["feature_files"][feature], records[3]["feature_files"][feature])
            self.assertNotEqual(records[0]["feature_files"][feature], records[4]["feature_files"][feature])
            self.assertEqual(len(list((root / "features").glob("*.bin"))), 4)
            self.assertNotIn("gmap_slot_ids", records[0]["feature_files"])
            self.assertTrue(np.all(np.fromfile(root / "case_0000" / f"{feature}.bin", dtype="<f4") == 1))
            self.assertTrue(np.all(np.fromfile(root / "case_0004" / f"{feature}.bin", dtype="<f4") == 5))
            for case in (root / "cases.txt").read_text().splitlines():
                case_dir = root / case
                for line in (case_dir / "manifest.txt").read_text().splitlines():
                    if line.startswith("#"):
                        continue
                    name, dtype, *shape = line.split()
                    expected_bytes = np.prod([int(d) for d in shape]) * np.dtype(dtype).itemsize
                    self.assertEqual((case_dir / f"{name}.bin").stat().st_size, expected_bytes)
                golden = np.fromfile(case_dir / "golden_corr.bin", dtype="<f4").reshape(1, 3, 882)
                np.testing.assert_array_equal(golden, tracker.result.array)
                self.assertEqual(json.loads((case_dir / "metadata.json").read_text())["edges"], 3)

    def test_restore_after_original_raises(self):
        with tempfile.TemporaryDirectory() as temp:
            tracker = Tracker()
            tracker.raise_error = True
            recorder = CorrelationRecorder(tracker, ReplayWriter(Path(temp)))
            with self.assertRaisesRegex(RuntimeError, "original correlation failed"):
                with recorder.installed():
                    tracker.update()
            self.assertNotIn("corr", tracker.__dict__)
            self.assertEqual(recorder.stage, "unknown")
            self.assertEqual(recorder.writer.records, [])

    def test_yaml_cannot_enable_half_precision(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "config.yaml"
            config.write_text("MIXED_PRECISION: True\nPATCHES_PER_FRAME: 32\nLOOP_CLOSURE: false\n")
            args = parse_args(["--config-yaml", str(config), "--no-mixed-precision"])
            self.assertFalse(args.mixed_precision)
            self.assertEqual(args.patches_per_frame, 32)
            self.assertTrue(args.no_undistort)
            self.assertFalse(args.skip_terminate_updates)


if __name__ == "__main__":
    unittest.main()
