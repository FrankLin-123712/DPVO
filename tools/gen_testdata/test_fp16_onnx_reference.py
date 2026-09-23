"""CPU-only tests: python -m unittest discover -s tools/gen_testdata -p test_fp16_onnx_reference.py"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T

from fp16_onnx_reference import MixedGraph, MixedReference, POLICY, scatter_max, scatter_sum
from testdata_precision import prepare_reference


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def graph(self, nodes, inputs, outputs, initializers=(), policy=True):
        model = h.make_model(h.make_graph(nodes, "test", inputs, outputs, list(initializers)),
                             opset_imports=[h.make_opsetid("", 11), h.make_opsetid("dpvo", 1)], ir_version=7)
        if policy:
            h.set_model_props(model, {"dpvo_precision_policy": POLICY})
        path = self.root / "test.onnx"
        onnx.save(model, path)
        return MixedGraph(path)

    def test_half_rounding_happens_before_fp32_add(self):
        # 2048 + .75 -> 2048 in half. A following FP32 + .25 must survive.
        # Whole-network half would lose .25; FP32-only graph gives 2049.
        graph = self.graph([
            h.make_node("Cast", ["X"], ["A"], to=T.FLOAT16),
            h.make_node("MatMul", ["A", "W"], ["H"]),
            h.make_node("Cast", ["H"], ["F"], to=T.FLOAT),
            h.make_node("Add", ["F", "B"], ["Y"])],
            [h.make_tensor_value_info("X", T.FLOAT, [1, 2])],
            [h.make_tensor_value_info("Y", T.FLOAT, [1, 1])],
            [nh.from_array(np.ones((2, 1), np.float16), "W"),
             nh.from_array(np.array(.25, np.float32), "B")])
        y = graph.run({"X": np.array([[2048, .75]], np.float32)})["Y"]
        self.assertEqual(y.dtype, np.float32)
        self.assertEqual(y.item(), 2048.25)

    def test_cpu_region_rejects_half_arithmetic(self):
        graph = self.graph([h.make_node("Add", ["A", "A"], ["Y"])],
            [h.make_tensor_value_info("A", T.FLOAT16, [1])],
            [h.make_tensor_value_info("Y", T.FLOAT16, [1])])
        with self.assertRaisesRegex(TypeError, "CPU region"):
            graph.run({"A": np.ones(1, np.float16)})

    def test_reference_accumulates_in_fp32_not_half_pe_order(self):
        graph = self.graph([h.make_node("MatMul", ["A", "B"], ["Y"])],
            [h.make_tensor_value_info("A", T.FLOAT16, [1, 3]),
             h.make_tensor_value_info("B", T.FLOAT16, [3, 1])],
            [h.make_tensor_value_info("Y", T.FLOAT16, [1, 1])])
        y = graph.run({"A": np.array([[2048, 1, -2048]], np.float16),
                       "B": np.ones((3, 1), np.float16)})["Y"]
        self.assertEqual(y.item(), 1)  # Sequential half accumulation would give 0.

    def test_conv_bias_is_half_operand_and_output_rounds(self):
        graph = self.graph([h.make_node("Conv", ["X", "W", "B"], ["Y"])],
            [h.make_tensor_value_info("X", T.FLOAT16, [1, 1, 1, 1])],
            [h.make_tensor_value_info("Y", T.FLOAT16, [1, 1, 1, 1])],
            [nh.from_array(np.ones((1, 1, 1, 1), np.float16), "W"),
             nh.from_array(np.array([1], np.float16), "B")])
        self.assertEqual(graph.run({"X": np.array([[[[2048]]]], np.float16)})["Y"].item(), 2048)

    def test_input_dtype_and_shape_are_checked(self):
        graph = self.graph([h.make_node("Identity", ["A"], ["Y"])],
            [h.make_tensor_value_info("A", T.FLOAT, [2])],
            [h.make_tensor_value_info("Y", T.FLOAT, [2])])
        with self.assertRaises(TypeError):
            graph.run({"A": np.ones(2, np.float16)})
        with self.assertRaises(ValueError):
            graph.run({"A": np.ones(3, np.float32)})
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            graph.run({"A": np.array([0, np.nan], np.float32)})

    def test_policy_required(self):
        with self.assertRaisesRegex(ValueError, "precision policy"):
            self.graph([], [], [], policy=False)

    def test_unknown_custom_op_rejected(self):
        with self.assertRaises((NotImplementedError, RuntimeError)):
            self.graph([h.make_node("missing", ["A"], ["Y"], domain="dpvo")],
                       [h.make_tensor_value_info("A", T.FLOAT, [1])],
                       [h.make_tensor_value_info("Y", T.FLOAT, [1])])

    def test_scatter_sparse_groups_ties_negative_indices(self):
        x = np.array([[[2, -1], [3, -4], [3, 5], [99, 99]]], np.float32)
        ids = np.array([0, 2, 2, -1], np.int64)
        # Direct methods exercise the independent implementations, no ORT code.
        maximum, argmax = scatter_max._run(None, x, ids, dim=1)
        summed, = scatter_sum._run(None, x, ids, dim=1)
        np.testing.assert_array_equal(maximum[:, [0, 2]], [[[2, -1], [3, 5]]])
        np.testing.assert_array_equal(argmax[:, 2], [[1, 2]])
        self.assertTrue(np.all(np.isneginf(maximum[:, 1])))
        np.testing.assert_array_equal(summed, [[[2, -1], [0, 0], [6, 1]]])
        expanded = np.broadcast_to(ids[None, :, None], x.shape)
        np.testing.assert_array_equal(scatter_sum._run(None, x, expanded)[0], summed)
        self.assertEqual(scatter_sum._run(None, x[:, :0], ids[:0])[0].shape, (1, 0, 2))

    def test_output_precision_collision(self):
        (self.root / "metadata.json").write_text('{"nn_reference": {"backend": "test"}}')
        args = argparse.Namespace(nn_precision="fp32", onnx_model_dir=None, output_root=self.root)
        with self.assertRaisesRegex(ValueError, "different NN precision"):
            prepare_reference(args)

    def test_cli_defaults_and_fp16_config(self):
        import generate_dpvo_python_testdata as tracker
        import generate_dpvo_runner_parity_testdata as component
        for module in (tracker, component):
            with patch.object(sys, "argv", ["generator"]):
                args = module.parse_args()
                self.assertEqual(args.nn_precision, "fp32")
                self.assertFalse(str(args.output_root).endswith("_fp16"))
            with patch.object(sys, "argv", ["generator", "--nn-precision=fp16", "--onnx-model-dir", str(self.root)]):
                args = module.parse_args()
                self.assertTrue(str(args.output_root).endswith("_fp16"))
                if module is tracker:
                    config = tracker.build_tracker_config(args)
                    self.assertTrue(config.MIXED_PRECISION)
                    self.assertFalse(config.NN_FP16_WEIGHTS)

    def test_shell_forwards_precision_without_overwriting_fp32_path(self):
        log = self.root / "calls"
        fake = self.root / "python"
        fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@" >> "$CALL_LOG"\n')
        fake.chmod(0o755)
        env = dict(os.environ, PYTHON_BIN=str(fake), CALL_LOG=str(log),
                   NN_PRECISION="fp16", ONNX_MODEL_DIR="/models with spaces", TESTDATA_ROOT=str(self.root))
        script = Path(__file__).with_name("gen_testdata.sh")
        subprocess.run(["bash", str(script), "0"], env=env, check=True, capture_output=True)
        calls = log.read_text()
        self.assertEqual(calls.count("--nn-precision"), 2)
        self.assertIn("/models with spaces", calls)
        self.assertIn("dpvo_runner_parity_small_fp16", calls)
        self.assertIn("dpvo_python_fast_p16_fp16", calls)


class DeployedGraphTests(unittest.TestCase):
    """Opt-in local artifacts; unit tests above do not require the checkout."""
    @classmethod
    def setUpClass(cls):
        value = os.environ.get("DPVO_RUNNER_ROOT")
        if not value:
            raise unittest.SkipTest("Set DPVO_RUNNER_ROOT to test deployed graphs/fixtures")
        cls.runner = Path(value)

    def test_four_exact_ort_fixtures(self):
        root = self.runner / "testdata/fp16_kernels"
        paths = list(root.glob("*.onnx"))
        self.assertEqual(len(paths), 4)
        for path in paths:
            graph = MixedGraph(path, require_policy=False)
            feeds = {}
            for info in graph.model.graph.input:
                shape = [d.dim_value for d in info.type.tensor_type.shape.dim]
                feeds[info.name] = np.fromfile(root / f"{path.stem}.{info.name}.bin", dtype="<f2").reshape(shape)
            actual = graph.run(feeds)["Y"]
            expected = np.fromfile(root / f"{path.stem}.Y.bin", dtype="<f2").reshape(actual.shape)
            np.testing.assert_array_equal(actual, expected, err_msg=path.stem)

    def test_real_models_dtype_shapes_dynamic_edges(self):
        reference = MixedReference(self.runner / "models/fp16")
        image = np.linspace(-.5, 1.5, 3*32*32, dtype=np.float32).reshape(1, 1, 3, 32, 32)
        fmap, imap = reference.feature(image)
        self.assertEqual(fmap.shape, (1, 1, 128, 8, 8))
        self.assertEqual(imap.dtype, np.float16)
        for edges in (1, 4):
            idx = np.arange(edges, dtype=np.int64)
            missing = np.full(edges, -1, np.int64)
            result = reference.update(np.zeros((1, edges, 384), np.float16),
                np.ones((1, edges, 384), np.float16), np.zeros((1, edges, 882), np.float16),
                idx, idx+1, idx, missing, missing)
            self.assertEqual([v.dtype for v in result], [np.float16, np.float32, np.float32])
            self.assertTrue(all(np.all(np.isfinite(v)) for v in result))


if __name__ == "__main__":
    unittest.main()
