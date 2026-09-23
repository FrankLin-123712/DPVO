"""Independent graph reference: half operands/results, float32 reductions.

Uses ONNX ReferenceEvaluator operators, NOT ORT/Systolic kernels. No torch/CUDA
dependency here; the tracker adapter is in fp16_tracker_adapter.py.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun

POLICY = "compute_half_cpu_float_v1"
ACCELERATED = {"Conv", "MatMul", "Gemm"}
CPU_OPS = {"Constant", "Relu", "Unsqueeze", "Shape", "Gather", "InstanceNormalization",
           "Add", "Concat", "Reshape", "Mul", "ReduceMean", "Sub", "Div",
           "ConstantOfShape", "Equal", "Where", "Expand", "Pow", "Sqrt",
           "GatherElements", "Sigmoid", "Less", "Not", "Unique", "Exp", "Identity"}


def _float_operands(*values):
    for value in values:
        if value is not None and value.dtype != np.float16:
            raise TypeError("Accelerated operands must be float16, including bias")
    return [None if v is None else v.astype(np.float32) for v in values]


class MatMul(OpRun):
    op_domain = ""

    def _run(self, a, b):
        a, b = _float_operands(a, b)
        return (np.matmul(a, b).astype(np.float16),)


class Gemm(OpRun):
    op_domain = ""

    def _run(self, a, b, c=None, alpha=1.0, beta=1.0, transA=0, transB=0):
        a, b, c = _float_operands(a, b, c)
        y = np.float32(alpha) * np.matmul(a.T if transA else a, b.T if transB else b)
        if c is not None and beta != 0:
            y = y + np.float32(beta) * c
        return (y.astype(np.float16),)


class Conv(OpRun):
    op_domain = ""

    def _run(self, X, W, B=None, auto_pad="NOTSET", dilations=None,
             group=1, kernel_shape=None, pads=None, strides=None):
        x, w, bias = _float_operands(X, W, B)
        if x.ndim != 4 or w.ndim != 4:
            raise ValueError("Reference Conv supports NCHW 2D convolution only")
        group = group or 1
        if group < 1 or x.shape[1] != w.shape[1] * group or w.shape[0] % group:
            raise ValueError("Invalid Conv groups/channels")
        kh, kw = w.shape[2:]
        if kernel_shape and list(kernel_shape) != [kh, kw]:
            raise ValueError("Conv kernel_shape differs from weights")
        dh, dw = dilations or (1, 1)
        sh, sw = strides or (1, 1)
        if min(dh, dw, sh, sw) < 1:
            raise ValueError("Conv strides/dilations must be positive")
        eh, ew = (kh - 1) * dh + 1, (kw - 1) * dw + 1
        mode = auto_pad.decode() if isinstance(auto_pad, bytes) else auto_pad
        pt, pl, pb, pr = pads or (0, 0, 0, 0)
        if mode in ("SAME_UPPER", "SAME_LOWER"):
            h, width = x.shape[2:]
            ph = max(0, ((h + sh - 1) // sh - 1) * sh + eh - h)
            pw = max(0, ((width + sw - 1) // sw - 1) * sw + ew - width)
            pt, pl = ph // 2, pw // 2
            if mode == "SAME_LOWER":
                pt, pl = ph - pt, pw - pl
            pb, pr = ph - pt, pw - pl
        elif mode == "VALID":
            pt = pl = pb = pr = 0
        elif mode not in (None, "", "NOTSET"):
            raise ValueError(f"Unsupported auto_pad: {mode}")
        x = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)))
        windows = np.lib.stride_tricks.sliding_window_view(x, (eh, ew), axis=(2, 3))
        windows = windows[:, :, ::sh, ::sw, ::dh, ::dw]
        n, _, oh, ow, _, _ = windows.shape
        cin, cout = w.shape[1], w.shape[0] // group
        y = np.empty((n, w.shape[0], oh, ow), dtype=np.float32)
        for g in range(group):
            a = windows[:, g*cin:(g+1)*cin].transpose(0, 2, 3, 1, 4, 5)
            a = np.ascontiguousarray(a).reshape(n*oh*ow, -1)
            b = w[g*cout:(g+1)*cout].reshape(cout, -1)
            y[:, g*cout:(g+1)*cout] = (a @ b.T).reshape(n, oh, ow, cout).transpose(0, 3, 1, 2)
        if bias is not None:
            if bias.shape != (w.shape[0],):
                raise ValueError("Invalid Conv bias shape")
            y += bias[None, :, None, None]
        return (y.astype(np.float16),)


def _scatter_inputs(src, index, dim):
    if src.dtype != np.float32 or src.ndim != 3 or dim not in (1, -2):
        raise ValueError("dpvo scatter expects float32 [B,E,C], dim=1")
    if index.dtype != np.int64:
        raise TypeError("dpvo scatter indices must be int64")
    if index.ndim == 1:
        if index.size != src.shape[1]:
            raise ValueError("scatter edge count mismatch")
        ids = index
    else:
        expanded = np.broadcast_to(index, src.shape)
        if not src.shape[1]:
            ids = np.empty(0, dtype=np.int64)
        elif not src.shape[0] or not src.shape[2]:
            raise ValueError("Expanded scatter index needs nonempty batch/channel dimensions")
        else:
            ids = expanded[0, :, 0]
            if not np.array_equal(expanded, np.broadcast_to(ids[None, :, None], src.shape)):
                raise ValueError("dpvo scatter requires one group index per edge")
    groups = max(0, int(ids.max()) + 1) if ids.size else 0
    return ids, (src.shape[0], groups, src.shape[2])


class scatter_sum(OpRun):
    op_domain = "dpvo"

    def _run(self, src, index, dim=1):
        ids, shape = _scatter_inputs(src, index, dim)
        out = np.zeros(shape, dtype=np.float32)
        for edge, group in enumerate(ids):
            if group >= 0:
                out[:, group] += src[:, edge]
        return (out,)


class scatter_max(OpRun):
    op_domain = "dpvo"

    def _run(self, src, index, dim=1):
        ids, shape = _scatter_inputs(src, index, dim)
        out = np.full(shape, -np.inf, dtype=np.float32)
        argmax = np.full(shape, -1, dtype=np.int64)
        for edge, group in enumerate(ids):
            if group >= 0:
                greater = src[:, edge] > out[:, group]
                out[:, group] = np.where(greater, src[:, edge], out[:, group])
                argmax[:, group] = np.where(greater, edge, argmax[:, group])
        return out, argmax


class MixedGraph:
    """Checked execution of the deployed graph without graph optimization.

    The rt_nodes_/rt_inits_ access follows ONNX 1.16.2 ReferenceEvaluator.run.
    Keeping the node loop here makes unexpected float16/float64 CPU math fail
    explicitly instead of accepting an accidental precision-policy change.
    """

    def __init__(self, path: Path, require_policy=True):
        self.path = Path(path)
        self.model = onnx.load(str(self.path), load_external_data=False)
        props = {p.key: p.value for p in self.model.metadata_props}
        if require_policy and props.get("dpvo_precision_policy") != POLICY:
            raise ValueError(f"{path}: expected precision policy {POLICY}")
        if any(t.data_location == onnx.TensorProto.EXTERNAL for t in self.model.graph.initializer):
            raise ValueError("External-data models must be consolidated before reference generation")
        versions = {v.domain: v.version for v in self.model.opset_import}
        if versions.get("", versions.get("ai.onnx")) != 11:
            raise ValueError("Reference policy currently supports ONNX opset 11 only")
        for node in self.model.graph.node:
            if node.domain in ("", "ai.onnx") and node.op_type in CPU_OPS | ACCELERATED | {"Cast"}:
                continue
            if node.domain == "dpvo" and node.op_type in {"scatter_sum", "scatter_max"}:
                continue
            raise NotImplementedError(f"Unsupported reference node: {node.domain}::{node.op_type}")
        self.evaluator = ReferenceEvaluator(
            self.model, new_ops=[Conv, MatMul, Gemm, scatter_sum, scatter_max], optimized=False)

    @staticmethod
    def _check_tensor(info, value, symbols):
        typ = info.type.tensor_type
        expected = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(typ.elem_type))
        if value.dtype != expected:
            raise TypeError(f"{info.name}: expected {expected}, got {value.dtype}")
        dims = typ.shape.dim
        if value.ndim != len(dims):
            raise ValueError(f"{info.name}: rank mismatch")
        for dim, size in zip(dims, value.shape):
            if dim.HasField("dim_value") and dim.dim_value != size:
                raise ValueError(f"{info.name}: shape mismatch")
            if dim.dim_param:
                if symbols.setdefault(dim.dim_param, size) != size:
                    raise ValueError(f"{info.name}: inconsistent dimension {dim.dim_param}")

    def run(self, feeds):
        inputs = self.model.graph.input
        if set(feeds) != {v.name for v in inputs}:
            raise ValueError(f"Expected graph inputs {[v.name for v in inputs]}")
        symbols = {}
        for info in inputs:
            self._check_tensor(info, feeds[info.name], symbols)
        values = {"": None, **self.evaluator.rt_inits_, **feeds}
        for proto, node in zip(self.model.graph.node, self.evaluator.rt_nodes_):
            args = [values[name] for name in node.input]
            accelerated = proto.domain in ("", "ai.onnx") and proto.op_type in ACCELERATED
            if proto.op_type != "Cast" and not accelerated:
                if any(isinstance(v, np.ndarray) and v.dtype in (np.float16, np.float64) for v in args):
                    raise TypeError(f"{proto.name or proto.op_type}: CPU region must not consume half/double")
            outputs = node.run(*args, **({"context": values} if node.need_context() else {}))
            for name, value in zip(node.output, outputs):
                if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
                    if value.dtype == np.float64 or (value.dtype == np.float16 and not accelerated and proto.op_type != "Cast"):
                        raise TypeError(f"{proto.name or proto.op_type}: unexpected output dtype {value.dtype}")
                values[name] = value
        outputs = {info.name: values[info.name] for info in self.model.graph.output}
        for info in self.model.graph.output:
            self._check_tensor(info, outputs[info.name], symbols)
            if not np.all(np.isfinite(outputs[info.name])):
                raise ValueError(f"{info.name}: nonfinite reference output")
        return outputs

    def metadata(self):
        from collections import Counter
        def types(values):
            return {v.name: onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type) for v in values}
        return {"path": str(self.path.resolve()),
                "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                "operators": dict(Counter(f"{n.domain or 'ai.onnx'}::{n.op_type}" for n in self.model.graph.node)),
                "initializer_dtypes": dict(Counter(onnx.TensorProto.DataType.Name(v.data_type) for v in self.model.graph.initializer)),
                "inputs": types(self.model.graph.input), "outputs": types(self.model.graph.output)}


class MixedReference:
    def __init__(self, model_dir):
        root = Path(model_dir)
        self.feature_graph = MixedGraph(root / "feature_extractor_opset11.onnx")
        self.update_graph = MixedGraph(root / "update_block_opset11.onnx")

    def feature(self, images):
        out = self.feature_graph.run({"images": np.asarray(images, dtype=np.float32)})
        return out["fmap"], out["imap"]

    def update(self, net, ctx, corr, ii, jj, kk, ix, jx):
        out = self.update_graph.run({
            "net": np.asarray(net, dtype=np.float16), "ctx": np.asarray(ctx, dtype=np.float16),
            "corr": np.asarray(corr, dtype=np.float16),
            **{name: np.asarray(v, dtype=np.int64) for name, v in
               zip(("ii", "jj", "kk", "ix", "jx"), (ii, jj, kk, ix, jx))}})
        return out["net_out"], out["delta"], out["weight"]

    def metadata(self):
        import os
        import platform
        return {"backend": "onnx_reference_numpy", "policy": POLICY,
                "nn_source": "onnx_initializers_not_checkpoint", "accumulation": "float32_reference_not_gemmini_pe",
                "golden_float_storage": "float32", "numpy": np.__version__, "onnx": onnx.__version__,
                "python": platform.python_version(), "platform": platform.platform(),
                "thread_environment": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                "feature": self.feature_graph.metadata(), "update": self.update_graph.metadata()}


def main():
    """Small real-network cases runnable without torch/CUDA or a checkpoint."""
    import argparse
    import json
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--onnx-model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    from testdata_precision import prepare_reference
    args.nn_precision = "fp16"
    reference = prepare_reference(args)
    # Exact dyadic input values; this is a synthetic NN case, not a trajectory.
    image = ((np.arange(3*32*32) % 17 - 8) / 16).astype(np.float32).reshape(1, 1, 3, 32, 32)
    fmap, imap = reference.feature(image)
    def save(name, tensors):
        root = args.output_root / name
        root.mkdir(parents=True, exist_ok=True)
        lines = ["# dpvo_runner_parity_case_v1"]
        for key, value in tensors.items():
            value = np.asarray(value)
            # Existing parity readers expect float32 files. Widening half is exact.
            if np.issubdtype(value.dtype, np.floating):
                value = value.astype(np.float32)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{key}: nonfinite golden")
            value = value.astype(value.dtype.newbyteorder("<"), copy=False)
            np.ascontiguousarray(value).tofile(root / f"{key}.bin")
            lines.append(f"{key} {value.dtype.name} " + " ".join(map(str, value.shape)))
        (root / "manifest.txt").write_text("\n".join(lines) + "\n")
    # Integer centers make the patch extraction oracle a direct array slice.
    centers = np.array([[2, 2], [5, 5]], dtype=np.float32)
    scaled_fmap = (fmap.astype(np.float32) * np.float32(.25)).astype(np.float16).astype(np.float32)
    scaled_imap = (imap.astype(np.float32) * np.float32(.25)).astype(np.float16).astype(np.float32)
    patches = np.empty((2, 3, 3, 3), np.float32)
    colors = []
    for i, (cx, cy) in enumerate(centers.astype(int)):
        yy, xx = np.meshgrid(np.arange(cy-1, cy+2), np.arange(cx-1, cx+2), indexing="ij")
        patches[i] = np.stack([xx, yy, np.ones_like(xx)])
        colors.append(image[0, 0, :, 4*cy+2, 4*cx+2])
    save("patchify_small", {
        "image": image, "centers": centers, "golden_fmap": scaled_fmap,
        "golden_imap": np.stack([scaled_imap[0, 0, :, y:y+1, x:x+1] for x, y in centers.astype(int)]),
        "golden_gmap": np.stack([scaled_fmap[0, 0, :, y-1:y+2, x-1:x+2] for x, y in centers.astype(int)]),
        "golden_patches": patches, "golden_colors": np.stack(colors)})
    edges = 4
    net = np.zeros((1, edges, 384), np.float16)
    ctx = scaled_imap[0, 0, :, 2:4, 2:4].reshape(384, edges).T[None].astype(np.float16)
    corr = ((np.arange(edges*882) % 13 - 6) / 32).astype(np.float16).reshape(1, edges, 882)
    ii = np.array([0, 0, 0, 1], np.int64)
    jj = np.array([1, 2, 3, 2], np.int64)
    kk = np.array([0, 0, 0, 1], np.int64)
    ix, jx = np.array([-1, 0, 1, -1], np.int64), np.array([1, 2, -1, -1], np.int64)
    net_out, delta, weight = reference.update(net, ctx, corr, ii, jj, kk, ix, jx)
    save("update_small", dict(net=net, ctx=ctx, corr=corr, ii=ii, jj=jj, kk=kk,
                              golden_net=net_out, golden_delta=delta, golden_weight=weight))
    (args.output_root / "metadata.json").write_text(json.dumps({
        "format": "dpvo_mixed_nn_smoke_v1", "nn_reference": reference.metadata(),
        "input": "deterministic synthetic 32x32 image and 4 update edges",
        "scope": "patchify and update only; no correlation or BA golden"}, indent=2) + "\n")
    print(f"Wrote graph-reference NN cases to {args.output_root}")


if __name__ == "__main__":
    main()
