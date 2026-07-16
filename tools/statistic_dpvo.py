#!/usr/bin/env python3
"""Layer-boundary operation and logical-memory statistics for DPVO.

The script deliberately models *algorithmic* work, not a cache, scratchpad,
tiling policy, or Gemmini dataflow.  Every reported row is a canonical layer:
all of its logical input/parameter/metadata tensors are read once from DRAM and
all of its logical output tensors are written once to DRAM.  Reuse inside a row
is free; reuse across rows is forbidden.

The neural-network rows are read directly from the exported ONNX graphs.  A
small dependency-free protobuf reader is included so the tool also works in the
minimal dpvo_runner deployment environment.  The non-ONNX rows follow the C++
implementation in systolic_runner/dpvo_runner.

Counting convention:
  * one scalar multiply and one scalar accumulate are two useful operations;
  * comparisons, conversions, divides, square roots, and nonlinear functions
    each count as one scalar operation;
  * tensor views/copies/indexed gathers have zero arithmetic operations but do
    have logical memory traffic;
  * address-generation and implementation-specific container/sort overhead are
    excluded;
  * BA source-level formulas assume the valid-residual path.  The factor graph
    is constructed exactly for the selected steady-frame parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import sys
from dataclasses import asdict, dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"
DEFAULT_ONNX_DIR = REPO_ROOT / "exported_models"
DEFAULT_ACTIVE_FRAMES = 64

FP32 = "fp32"
FP16 = "fp16"
FP64 = "fp64"
INT = "int_bool"

DTYPE_BYTES = {
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
    "bool": 1,
}

# ONNX TensorProto.DataType values used by the checked-in graphs.
ONNX_DTYPE = {
    1: "float32",
    2: "uint8",
    3: "int8",
    4: "uint16",
    5: "int16",
    6: "int32",
    7: "int64",
    9: "bool",
    10: "float16",
    11: "float64",
    12: "uint32",
    13: "uint64",
}

COMPUTE_ONNX_OPS = {
    "Add",
    "Cast",
    "Conv",
    "Div",
    "Equal",
    "Exp",
    "GreaterOrEqual",
    "InstanceNormalization",
    "MatMul",
    "Mul",
    "Pow",
    "ReduceMean",
    "Relu",
    "scatter_max",
    "scatter_sum",
    "Sigmoid",
    "Sqrt",
    "Sub",
    "Where",
}

SUPPORTED_PA_KEYS = {
    "PATCHES_PER_FRAME",
    "REMOVAL_WINDOW",
    "OPTIMIZATION_WINDOW",
    "PATCH_LIFETIME",
    "BA_ITERATIONS",
    "CENTROID_SEL_STRAT",
}


@dataclass(frozen=True)
class TensorInfo:
    shape: tuple[int, ...]
    dtype: str
    values: tuple[int | float | bool, ...] | None = None

    @property
    def elements(self) -> int:
        return product(self.shape)

    @property
    def bytes(self) -> int:
        return self.elements * DTYPE_BYTES[self.dtype]


@dataclass
class OnnxNode:
    name: str
    op_type: str
    inputs: list[str]
    outputs: list[str]
    attributes: dict[str, Any]


@dataclass
class OnnxGraph:
    inputs: dict[str, TensorInfo]
    initializers: dict[str, TensorInfo]
    nodes: list[OnnxNode]


@dataclass
class LayerRow:
    module: str
    layer: str
    shape: str
    invocations: int
    fp16_ops: int = 0
    fp32_ops: int = 0
    fp64_ops: int = 0
    int_bool_ops: int = 0
    mem_read_bytes: int = 0
    mem_write_bytes: int = 0
    access_pattern: str = "regular contiguous"
    source: str = ""

    @property
    def total_ops(self) -> int:
        return self.fp16_ops + self.fp32_ops + self.fp64_ops + self.int_bool_ops

    @property
    def total_memory_bytes(self) -> int:
        return self.mem_read_bytes + self.mem_write_bytes

    @property
    def operation_intensity(self) -> float:
        if self.total_memory_bytes == 0:
            return math.inf if self.total_ops else 0.0
        return self.total_ops / self.total_memory_bytes

    def raw_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["total_ops"] = self.total_ops
        result["total_memory_bytes"] = self.total_memory_bytes
        result["operation_intensity_ops_per_byte"] = self.operation_intensity
        return result


@dataclass
class PaConfig:
    patches_per_frame: int = 96
    removal_window: int = 22
    optimization_window: int = 10
    patch_lifetime: int = 13
    ba_iterations: int = 2
    centroid_sel_strat: str = "RANDOM"


@dataclass
class GraphStats:
    frame_count: int
    edge_count: int
    post_prune_edge_count: int
    new_edge_count: int
    unique_patches: int
    edge_groups: int
    source_frame_count: int
    target_frame_count: int
    referenced_frame_count: int
    free_pose_count: int
    free_endpoint_activations: int
    free_pose_block_activations: int
    pairs: list[tuple[int, int]] = field(repr=False)


def product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def tensor_bytes(shape: Sequence[int], dtype: str = "float32") -> int:
    return product(shape) * DTYPE_BYTES[dtype]


def parse_scalar(text: str) -> Any:
    value = text.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return float(value) if any(ch in value for ch in ".eE") else int(value)
    except ValueError:
        return value


def load_simple_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if not path.exists():
        raise FileNotFoundError(f"DPVO config does not exist: {path}")
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        if value.strip():
            data[key.strip().upper()] = parse_scalar(value)
    return data


def decode_varint(data: bytes, offset: int = 0) -> tuple[int, int]:
    shift = 0
    value = 0
    while True:
        if offset >= len(data):
            raise ValueError("truncated protobuf varint")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
        if shift >= 70:
            raise ValueError("invalid protobuf varint")


def protobuf_fields(data: bytes) -> Iterator[tuple[int, int, Any]]:
    offset = 0
    while offset < len(data):
        key, offset = decode_varint(data, offset)
        field_number, wire_type = key >> 3, key & 7
        if wire_type == 0:
            value, offset = decode_varint(data, offset)
        elif wire_type == 1:
            value = data[offset : offset + 8]
            offset += 8
        elif wire_type == 2:
            size, offset = decode_varint(data, offset)
            value = data[offset : offset + size]
            offset += size
        elif wire_type == 5:
            value = data[offset : offset + 4]
            offset += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type}")
        yield field_number, wire_type, value


def protobuf_text(value: Any) -> str:
    return bytes(value).decode("utf-8", errors="replace")


def signed_int64(value: int) -> int:
    return value - (1 << 64) if value >= (1 << 63) else value


def packed_varints(data: bytes, *, signed: bool = False) -> list[int]:
    values: list[int] = []
    offset = 0
    while offset < len(data):
        value, offset = decode_varint(data, offset)
        values.append(signed_int64(value) if signed else value)
    return values


def repeated_ints(
    fields: Sequence[tuple[int, int, Any]], field_number: int, *, signed: bool = True
) -> list[int]:
    result: list[int] = []
    for number, wire_type, value in fields:
        if number != field_number:
            continue
        if wire_type == 0:
            integer = int(value)
            result.append(signed_int64(integer) if signed else integer)
        elif wire_type == 2:
            result.extend(packed_varints(bytes(value), signed=signed))
    return result


def tensor_values(
    fields: Sequence[tuple[int, int, Any]], dtype: str, elements: int
) -> tuple[int | float | bool, ...] | None:
    # Large tensors are weights.  Their shape/type is enough for accounting.
    if elements > 64:
        return None
    raw = next(
        (bytes(value) for number, wire_type, value in fields if number == 9 and wire_type == 2),
        None,
    )
    formats = {
        "float32": "f",
        "float64": "d",
        "int8": "b",
        "uint8": "B",
        "int16": "h",
        "uint16": "H",
        "int32": "i",
        "uint32": "I",
        "int64": "q",
        "uint64": "Q",
        "bool": "?",
    }
    if raw is not None and dtype in formats:
        item_size = DTYPE_BYTES[dtype]
        if len(raw) >= elements * item_size:
            return tuple(struct.unpack("<" + formats[dtype] * elements, raw[: elements * item_size]))

    if dtype == "float32":
        values: list[float] = []
        for number, wire_type, value in fields:
            if number != 4:
                continue
            if wire_type == 5:
                values.append(struct.unpack("<f", bytes(value))[0])
            elif wire_type == 2:
                payload = bytes(value)
                values.extend(struct.unpack("<" + "f" * (len(payload) // 4), payload))
        return tuple(values) if len(values) == elements else None
    if dtype in {"int32", "int64", "uint32", "uint64", "bool"}:
        field_number = 7 if dtype in {"int64", "uint64"} else 5
        values = repeated_ints(fields, field_number, signed=dtype in {"int32", "int64"})
        if dtype == "bool":
            return tuple(bool(value) for value in values) if len(values) == elements else None
        return tuple(values) if len(values) == elements else None
    return None


def parse_tensor_proto(data: bytes) -> tuple[str, TensorInfo]:
    fields = list(protobuf_fields(data))
    dims = repeated_ints(fields, 1, signed=False)
    dtype_number = next(
        (int(value) for number, wire_type, value in fields if number == 2 and wire_type == 0),
        None,
    )
    name = next(
        (protobuf_text(value) for number, wire_type, value in fields if number == 8 and wire_type == 2),
        "",
    )
    if dtype_number not in ONNX_DTYPE:
        raise ValueError(f"unsupported ONNX tensor dtype {dtype_number} for {name!r}")
    dtype = ONNX_DTYPE[dtype_number]
    shape = tuple(int(value) for value in dims)
    return name, TensorInfo(shape, dtype, tensor_values(fields, dtype, product(shape)))


def parse_value_info(data: bytes) -> tuple[str, str | None, tuple[int | str, ...] | None]:
    fields = list(protobuf_fields(data))
    name = next(
        (protobuf_text(value) for number, wire_type, value in fields if number == 1 and wire_type == 2),
        "",
    )
    type_proto = next(
        (bytes(value) for number, wire_type, value in fields if number == 2 and wire_type == 2),
        None,
    )
    if type_proto is None:
        return name, None, None
    tensor_type = next(
        (bytes(value) for number, wire_type, value in protobuf_fields(type_proto) if number == 1 and wire_type == 2),
        None,
    )
    if tensor_type is None:
        return name, None, None
    tensor_fields = list(protobuf_fields(tensor_type))
    dtype_number = next(
        (int(value) for number, wire_type, value in tensor_fields if number == 1 and wire_type == 0),
        None,
    )
    shape_proto = next(
        (bytes(value) for number, wire_type, value in tensor_fields if number == 2 and wire_type == 2),
        None,
    )
    dims: list[int | str] = []
    if shape_proto is not None:
        for number, wire_type, value in protobuf_fields(shape_proto):
            if number != 1 or wire_type != 2:
                continue
            dim_fields = list(protobuf_fields(bytes(value)))
            dim_value = next(
                (int(item) for num, wire, item in dim_fields if num == 1 and wire == 0),
                None,
            )
            dim_param = next(
                (protobuf_text(item) for num, wire, item in dim_fields if num == 2 and wire == 2),
                None,
            )
            dims.append(dim_value if dim_value is not None else (dim_param or "?"))
    return name, ONNX_DTYPE.get(dtype_number) if dtype_number is not None else None, tuple(dims)


def parse_attribute(data: bytes) -> tuple[str, Any]:
    fields = list(protobuf_fields(data))
    name = next(
        (protobuf_text(value) for number, wire_type, value in fields if number == 1 and wire_type == 2),
        "",
    )
    tensor = next(
        (bytes(value) for number, wire_type, value in fields if number == 5 and wire_type == 2),
        None,
    )
    if tensor is not None:
        return name, parse_tensor_proto(tensor)[1]
    ints = repeated_ints(fields, 8)
    if ints:
        return name, ints
    integer = next(
        (signed_int64(int(value)) for number, wire_type, value in fields if number == 3 and wire_type == 0),
        None,
    )
    if integer is not None:
        return name, integer
    float_value = next(
        (struct.unpack("<f", bytes(value))[0] for number, wire_type, value in fields if number == 2 and wire_type == 5),
        None,
    )
    if float_value is not None:
        return name, float_value
    string_value = next(
        (bytes(value) for number, wire_type, value in fields if number == 4 and wire_type == 2),
        None,
    )
    return name, string_value


def parse_onnx_model(path: Path) -> OnnxGraph:
    if not path.exists():
        raise FileNotFoundError(f"ONNX model does not exist: {path}")
    graph_data = next(
        (bytes(value) for number, wire_type, value in protobuf_fields(path.read_bytes()) if number == 7 and wire_type == 2),
        None,
    )
    if graph_data is None:
        raise ValueError(f"ONNX model has no graph: {path}")
    graph_fields = list(protobuf_fields(graph_data))
    initializers: dict[str, TensorInfo] = {}
    for number, wire_type, value in graph_fields:
        if number == 5 and wire_type == 2:
            name, info = parse_tensor_proto(bytes(value))
            initializers[name] = info

    inputs: dict[str, TensorInfo] = {}
    for number, wire_type, value in graph_fields:
        if number != 11 or wire_type != 2:
            continue
        name, dtype, shape = parse_value_info(bytes(value))
        if name in initializers or dtype is None or shape is None:
            continue
        # Symbolic dimensions are replaced by the caller before inference.
        resolved = tuple(int(dim) if isinstance(dim, int) else -1 for dim in shape)
        inputs[name] = TensorInfo(resolved, dtype)

    nodes: list[OnnxNode] = []
    for number, wire_type, value in graph_fields:
        if number != 1 or wire_type != 2:
            continue
        fields = list(protobuf_fields(bytes(value)))
        node_inputs = [
            protobuf_text(item) for num, wire, item in fields if num == 1 and wire == 2
        ]
        outputs = [
            protobuf_text(item) for num, wire, item in fields if num == 2 and wire == 2
        ]
        name = next(
            (protobuf_text(item) for num, wire, item in fields if num == 3 and wire == 2),
            "",
        )
        op_type = next(
            (protobuf_text(item) for num, wire, item in fields if num == 4 and wire == 2),
            "",
        )
        attributes = dict(
            parse_attribute(bytes(item))
            for num, wire, item in fields
            if num == 5 and wire == 2
        )
        nodes.append(OnnxNode(name, op_type, node_inputs, outputs, attributes))
    return OnnxGraph(inputs, initializers, nodes)


def broadcast_shape(lhs: Sequence[int], rhs: Sequence[int]) -> tuple[int, ...]:
    result: list[int] = []
    for left, right in zip(reversed(lhs), reversed(rhs)):
        if left == right or left == 1 or right == 1:
            result.append(max(left, right))
        else:
            raise ValueError(f"cannot broadcast shapes {tuple(lhs)} and {tuple(rhs)}")
    longer = lhs if len(lhs) > len(rhs) else rhs
    result.extend(reversed(longer[: abs(len(lhs) - len(rhs))]))
    return tuple(reversed(result))


def normalize_axis(axis: int, rank: int) -> int:
    return axis + rank if axis < 0 else axis


def scalar_or_equal_values(
    op: str, lhs: TensorInfo, rhs: TensorInfo
) -> tuple[int | float | bool, ...] | None:
    if lhs.values is None or rhs.values is None:
        return None
    count = max(len(lhs.values), len(rhs.values))
    if len(lhs.values) not in {1, count} or len(rhs.values) not in {1, count}:
        return None
    left = lhs.values * count if len(lhs.values) == 1 else lhs.values
    right = rhs.values * count if len(rhs.values) == 1 else rhs.values
    operations = {
        "Add": lambda a, b: a + b,
        "Sub": lambda a, b: a - b,
        "Mul": lambda a, b: a * b,
        "Div": lambda a, b: a / b,
        "Pow": lambda a, b: a**b,
        "Equal": lambda a, b: a == b,
        "GreaterOrEqual": lambda a, b: a >= b,
    }
    function = operations.get(op)
    if function is None:
        return None
    return tuple(function(a, b) for a, b in zip(left, right))


def reshape_target(input_shape: Sequence[int], values: Sequence[int | float | bool]) -> tuple[int, ...]:
    target = [int(value) for value in values]
    unknown = [index for index, value in enumerate(target) if value == -1]
    if len(unknown) > 1:
        raise ValueError(f"invalid reshape target {target}")
    for index, value in enumerate(target):
        if value == 0:
            target[index] = int(input_shape[index])
    if unknown:
        known = product(value for value in target if value != -1)
        target[unknown[0]] = product(input_shape) // known
    return tuple(target)


def infer_onnx_node(
    node: OnnxNode,
    tensors: dict[str, TensorInfo],
    unique_patches: int,
    edge_groups: int,
) -> list[TensorInfo]:
    inputs = [tensors[name] for name in node.inputs if name]
    op = node.op_type
    if op == "Constant":
        value = node.attributes.get("value")
        if not isinstance(value, TensorInfo):
            raise ValueError(f"Constant node {node.name!r} has no tensor value")
        return [value]
    if not inputs:
        raise ValueError(f"node {node.name!r} ({op}) has no inferred inputs")

    first = inputs[0]
    if op in {"Identity", "Relu", "Sigmoid", "Sqrt", "Exp"}:
        return [TensorInfo(first.shape, first.dtype)]
    if op == "Shape":
        return [TensorInfo((len(first.shape),), "int64", tuple(first.shape))]
    if op == "Cast":
        dtype_number = int(node.attributes["to"])
        dtype = ONNX_DTYPE[dtype_number]
        values = tuple(int(value) for value in first.values) if first.values is not None else None
        return [TensorInfo(first.shape, dtype, values)]
    if op in {"Add", "Sub", "Mul", "Div", "Pow", "Equal", "GreaterOrEqual"}:
        shape = broadcast_shape(inputs[0].shape, inputs[1].shape)
        dtype = "bool" if op in {"Equal", "GreaterOrEqual"} else inputs[0].dtype
        values = scalar_or_equal_values(op, inputs[0], inputs[1])
        return [TensorInfo(shape, dtype, values)]
    if op == "Where":
        shape = broadcast_shape(inputs[0].shape, broadcast_shape(inputs[1].shape, inputs[2].shape))
        values = None
        if all(info.values is not None for info in inputs[:3]):
            condition, yes, no = (info.values for info in inputs[:3])
            assert condition is not None and yes is not None and no is not None
            count = max(len(condition), len(yes), len(no))
            if all(len(values_) in {1, count} for values_ in (condition, yes, no)):
                cvals = condition * count if len(condition) == 1 else condition
                yvals = yes * count if len(yes) == 1 else yes
                nvals = no * count if len(no) == 1 else no
                values = tuple(y if bool(c) else n for c, y, n in zip(cvals, yvals, nvals))
        return [TensorInfo(shape, inputs[1].dtype, values)]
    if op == "MatMul":
        lhs, rhs = inputs[:2]
        if len(lhs.shape) < 2 or len(rhs.shape) < 2 or lhs.shape[-1] != rhs.shape[-2]:
            raise ValueError(f"invalid MatMul shapes {lhs.shape} and {rhs.shape} at {node.name}")
        batch = broadcast_shape(lhs.shape[:-2], rhs.shape[:-2])
        return [TensorInfo(batch + (lhs.shape[-2], rhs.shape[-1]), lhs.dtype)]
    if op == "Conv":
        data, weights = inputs[:2]
        strides = node.attributes.get("strides", [1, 1])
        pads = node.attributes.get("pads", [0, 0, 0, 0])
        dilations = node.attributes.get("dilations", [1, 1])
        out_h = (
            data.shape[2]
            + int(pads[0])
            + int(pads[2])
            - int(dilations[0]) * (weights.shape[2] - 1)
            - 1
        ) // int(strides[0]) + 1
        out_w = (
            data.shape[3]
            + int(pads[1])
            + int(pads[3])
            - int(dilations[1]) * (weights.shape[3] - 1)
            - 1
        ) // int(strides[1]) + 1
        return [TensorInfo((data.shape[0], weights.shape[0], out_h, out_w), data.dtype)]
    if op == "InstanceNormalization":
        return [TensorInfo(first.shape, first.dtype)]
    if op == "ReduceMean":
        axes_value = node.attributes.get("axes", list(range(len(first.shape))))
        axes = [normalize_axis(int(axis), len(first.shape)) for axis in axes_value]
        keepdims = int(node.attributes.get("keepdims", 1))
        shape = list(first.shape)
        if keepdims:
            for axis in axes:
                shape[axis] = 1
        else:
            shape = [dim for index, dim in enumerate(shape) if index not in axes]
        return [TensorInfo(tuple(shape), first.dtype)]
    if op == "Gather":
        data, indices = inputs[:2]
        axis = normalize_axis(int(node.attributes.get("axis", 0)), len(data.shape))
        shape = data.shape[:axis] + indices.shape + data.shape[axis + 1 :]
        values = None
        if data.values is not None and indices.values is not None and len(data.shape) == 1:
            values = tuple(data.values[int(index)] for index in indices.values)
        return [TensorInfo(shape, data.dtype, values)]
    if op == "GatherElements":
        return [TensorInfo(inputs[1].shape, first.dtype)]
    if op == "Unsqueeze":
        axes_value = inputs[1].values if len(inputs) > 1 else node.attributes.get("axes")
        if axes_value is None:
            raise ValueError(f"Unsqueeze axes are unknown at {node.name}")
        rank = len(first.shape) + len(axes_value)
        axes = sorted(normalize_axis(int(axis), rank) for axis in axes_value)
        shape = list(first.shape)
        for axis in axes:
            shape.insert(axis, 1)
        return [TensorInfo(tuple(shape), first.dtype, first.values)]
    if op == "Concat":
        axis = normalize_axis(int(node.attributes.get("axis", 0)), len(first.shape))
        shape = list(first.shape)
        shape[axis] = sum(info.shape[axis] for info in inputs)
        values = None
        if axis == 0 and all(info.values is not None for info in inputs):
            values = tuple(value for info in inputs for value in (info.values or ()))
        return [TensorInfo(tuple(shape), first.dtype, values)]
    if op == "Reshape":
        target = inputs[1].values
        if target is None:
            raise ValueError(f"Reshape target is unknown at {node.name}")
        shape = reshape_target(first.shape, target)
        return [TensorInfo(shape, first.dtype, first.values)]
    if op == "Expand":
        target = inputs[1].values
        if target is None:
            raise ValueError(f"Expand target is unknown at {node.name}")
        return [TensorInfo(tuple(int(value) for value in target), first.dtype)]
    if op == "ConstantOfShape":
        if first.values is None:
            raise ValueError(f"ConstantOfShape target is unknown at {node.name}")
        shape = tuple(int(value) for value in first.values)
        value = node.attributes.get("value")
        dtype = value.dtype if isinstance(value, TensorInfo) else "float32"
        fill = value.values[0] if isinstance(value, TensorInfo) and value.values else 0
        values = tuple([fill] * product(shape)) if product(shape) <= 64 else None
        return [TensorInfo(shape, dtype, values)]
    if op == "Unique":
        groups = unique_patches if node.name.startswith("/agg_kk/") else edge_groups
        edge_count = first.elements
        return [
            TensorInfo((groups,), first.dtype),
            TensorInfo((groups,), "int64"),
            TensorInfo((edge_count,), "int64"),
            TensorInfo((groups,), "int64"),
        ]
    if op == "scatter_max":
        groups = unique_patches if node.name.startswith("/agg_kk/") else edge_groups
        shape = (first.shape[0], groups, first.shape[2])
        return [TensorInfo(shape, first.dtype), TensorInfo(shape, "int64")]
    if op == "scatter_sum":
        groups = unique_patches if node.name.startswith("/agg_kk/") else edge_groups
        return [TensorInfo((first.shape[0], groups, first.shape[2]), first.dtype)]
    raise ValueError(f"unsupported ONNX op {op!r} at node {node.name!r}")


def operation_precision(dtype: str) -> str:
    if dtype == "float16":
        return FP16
    if dtype == "float32":
        return FP32
    if dtype == "float64":
        return FP64
    return INT


def node_operation_count(
    node: OnnxNode, inputs: list[TensorInfo], outputs: list[TensorInfo]
) -> tuple[str, int]:
    op = node.op_type
    out_elements = outputs[0].elements if outputs else 0
    precision = operation_precision(
        outputs[0].dtype if outputs and outputs[0].dtype != "bool" else inputs[0].dtype if inputs else "bool"
    )
    if op == "MatMul":
        reduction = inputs[0].shape[-1]
        return precision, 2 * out_elements * reduction
    if op == "Conv":
        weights = inputs[1]
        reduction = weights.shape[1] * weights.shape[2] * weights.shape[3]
        # ONNX Conv bias is one extra add per output value.
        bias_ops = out_elements if len(inputs) >= 3 else 0
        # TensorProto already stores C_in/group in weights.shape[1].
        return precision, 2 * out_elements * reduction + bias_ops
    if op in {"Add", "Sub", "Mul", "Div", "Pow", "Relu", "Sqrt", "Exp", "Sigmoid"}:
        return precision, out_elements
    if op in {"Equal", "GreaterOrEqual", "Where", "Cast"}:
        return INT, out_elements
    if op == "ReduceMean":
        return precision, inputs[0].elements
    if op == "InstanceNormalization":
        channels = inputs[0].shape[0] * inputs[0].shape[1]
        return precision, 8 * inputs[0].elements + channels
    if op in {"scatter_max", "scatter_sum"}:
        return precision, inputs[0].elements
    return INT, 0


def format_tensor(info: TensorInfo) -> str:
    dims = "x".join(str(dim) for dim in info.shape) if info.shape else "scalar"
    return f"{dims}:{info.dtype}"


def node_module(model_name: str, node: OnnxNode) -> str:
    pieces = [piece for piece in node.name.split("/") if piece]
    if not pieces:
        return model_name
    root = pieces[0]
    if model_name == "feature_extractor" and root in {"fnet", "inet"}:
        return f"{model_name}.{root}"
    if model_name == "update_block" and root in {"corr", "norm", "c1", "c2", "agg_kk", "agg_ij", "gru", "d", "w"}:
        return f"{model_name}.{root}"
    return model_name


def node_access_pattern(node: OnnxNode) -> str:
    if node.op_type in {"scatter_max", "scatter_sum", "GatherElements", "Unique"}:
        return "factor-graph indexed"
    if node.op_type == "Gather":
        if node.inputs and "Shape" in node.inputs[0]:
            return "regular metadata"
        return "neighbor/factor index policy"
    if node.op_type in {"Shape", "Constant", "ConstantOfShape", "Reshape", "Unsqueeze", "Concat", "Expand"}:
        return "regular metadata"
    return "regular contiguous"


def onnx_rows(
    path: Path,
    model_name: str,
    input_overrides: dict[str, TensorInfo],
    unique_patches: int,
    edge_groups: int,
    invocations: int,
    include_all_nodes: bool,
) -> list[LayerRow]:
    graph = parse_onnx_model(path)
    tensors = dict(graph.initializers)
    tensors.update(graph.inputs)
    tensors.update(input_overrides)
    rows: list[LayerRow] = []
    unnamed_count = 0
    for node in graph.nodes:
        try:
            inputs = [tensors[name] for name in node.inputs if name]
        except KeyError as exc:
            raise ValueError(f"missing ONNX tensor {exc.args[0]!r} before node {node.name!r}") from exc
        outputs = infer_onnx_node(node, tensors, unique_patches, edge_groups)
        if len(outputs) != len(node.outputs):
            raise ValueError(
                f"shape inference returned {len(outputs)} outputs for {node.name!r}, expected {len(node.outputs)}"
            )
        for name, info in zip(node.outputs, outputs):
            tensors[name] = info

        if include_all_nodes or node.op_type in COMPUTE_ONNX_OPS:
            precision, operations = node_operation_count(node, inputs, outputs)
            counts = {FP16: 0, FP32: 0, FP64: 0, INT: 0}
            counts[precision] = operations * invocations
            read_bytes = sum(info.bytes for info in inputs)
            # Constant attributes and regular initializers are parameter tensors
            # fetched for every invocation under the layer-boundary model.
            if node.op_type == "Constant":
                read_bytes = sum(info.bytes for info in outputs)
            shape = (
                ",".join(format_tensor(info) for info in inputs)
                + " -> "
                + ",".join(format_tensor(info) for info in outputs)
            )
            unnamed_count += int(not node.name)
            layer_name = node.name or f"{node.op_type}_{unnamed_count}"
            rows.append(
                LayerRow(
                    module=node_module(model_name, node),
                    layer=layer_name,
                    shape=shape,
                    invocations=invocations,
                    fp16_ops=counts[FP16],
                    fp32_ops=counts[FP32],
                    fp64_ops=counts[FP64],
                    int_bool_ops=counts[INT],
                    mem_read_bytes=read_bytes * invocations,
                    mem_write_bytes=sum(info.bytes for info in outputs) * invocations,
                    access_pattern=node_access_pattern(node),
                    source=f"ONNX:{path.name}",
                )
            )
    return rows


def build_factor_pairs(frame_count: int, patch_lifetime: int, removal_window: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    """Construct the C++ update-time graph, then its post-Keyframe graph.

    For a mature graph with R >= L, this simulation reduces to:

      E_post / M = R(2L-1) - L(L-1)/2
      E_update    = E_post + M(2L-1)
      K_update    = M(R+1)

    Simulation is retained because it also handles R < L and startup edges.
    """
    pairs: list[tuple[int, int]] = []
    new_count = 0
    snapshot: list[tuple[int, int]] = []
    for current in range(frame_count):
        graph_frames = current + 1
        start = max(graph_frames - patch_lifetime, 0)
        new_pairs = [(source, current) for source in range(start, current)]
        new_pairs.extend((current, target) for target in range(start, graph_frames))
        pairs.extend(new_pairs)
        if graph_frames == frame_count:
            snapshot = list(pairs)
            new_count = len(new_pairs)
        # Keyframe() prunes after Update().  Initialization starts at frame 8;
        # by a valid steady frame this produces the same retained graph as
        # pruning on every prior frame.
        if graph_frames >= 8:
            oldest = max(graph_frames - removal_window, 0)
            pairs = [pair for pair in pairs if pair[0] >= oldest]
    return snapshot, pairs, new_count


def derive_graph_stats(
    pa: PaConfig,
    frame_count: int,
    edges_override: int | None,
    unique_patches_override: int | None,
    edge_groups_override: int | None,
    free_poses_override: int | None,
) -> GraphStats:
    pairs, post_pairs, new_pair_count = build_factor_pairs(
        frame_count, pa.patch_lifetime, pa.removal_window
    )
    source_frames = {source for source, _ in pairs}
    target_frames = {target for _, target in pairs}
    referenced_frames = source_frames | target_frames
    edge_count_derived = len(pairs) * pa.patches_per_frame
    edge_count = edges_override if edges_override is not None else edge_count_derived
    scale = edge_count / edge_count_derived if edge_count_derived else 0.0
    unique_patches = (
        unique_patches_override
        if unique_patches_override is not None
        else round(len(source_frames) * pa.patches_per_frame * scale)
    )
    edge_groups = edge_groups_override if edge_groups_override is not None else round(len(pairs) * scale)
    free_pose_count = (
        free_poses_override
        if free_poses_override is not None
        else min(pa.optimization_window, max(frame_count - 1, 0))
    )
    fixed_pose_count = max(frame_count - free_pose_count, 1)
    endpoint_activations = 0
    block_activations = 0
    for source, target in pairs:
        free_endpoints = int(source >= fixed_pose_count) + int(target >= fixed_pose_count)
        endpoint_activations += free_endpoints * pa.patches_per_frame
        block_activations += free_endpoints * free_endpoints * pa.patches_per_frame
    endpoint_activations = round(endpoint_activations * scale)
    block_activations = round(block_activations * scale)
    new_edges = round(new_pair_count * pa.patches_per_frame * scale)
    post_prune_edges = round(len(post_pairs) * pa.patches_per_frame * scale)
    return GraphStats(
        frame_count=frame_count,
        edge_count=edge_count,
        post_prune_edge_count=post_prune_edges,
        new_edge_count=new_edges,
        unique_patches=unique_patches,
        edge_groups=edge_groups,
        source_frame_count=len(source_frames),
        target_frame_count=len(target_frames),
        referenced_frame_count=len(referenced_frames),
        free_pose_count=free_pose_count,
        free_endpoint_activations=endpoint_activations,
        free_pose_block_activations=block_activations,
        pairs=pairs,
    )


def make_custom_row(
    module: str,
    layer: str,
    shape: str,
    invocations: int,
    *,
    fp16: int = 0,
    fp32: int = 0,
    fp64: int = 0,
    int_bool: int = 0,
    read: int = 0,
    write: int = 0,
    pattern: str,
    source: str,
) -> LayerRow:
    return LayerRow(
        module,
        layer,
        shape,
        invocations,
        fp16 * invocations,
        fp32 * invocations,
        fp64 * invocations,
        int_bool * invocations,
        read * invocations,
        write * invocations,
        pattern,
        source,
    )


def patchify_rows(pa: PaConfig, height: int, width: int) -> list[LayerRow]:
    rows: list[LayerRow] = []
    m = pa.patches_per_frame
    fh, fw = height // 4, width // 4
    image_values = 3 * height * width
    fmap_values = 128 * fh * fw
    imap_values = 384 * fh * fw
    centers_bytes = tensor_bytes((m, 2))
    strategy = pa.centroid_sel_strat.upper()
    if strategy == "GRADIENT_BIAS":
        pixels = (height - 1) * (width - 1)
        candidates = 3 * m
        fp32_ops = 34 * pixels + math.ceil(candidates * math.log2(max(candidates, 1)))
        int_ops = 2 * pixels + 2 * candidates
        read = tensor_bytes((1, 1, 3, height, width))
        pattern = "gradient-ranked random index policy"
    elif strategy == "RANDOM":
        fp32_ops = 0
        int_ops = 2 * m
        read = 0
        pattern = "random index policy"
    else:
        # A centers manifest is deterministic but still an external index policy.
        fp32_ops = 0
        int_ops = 0
        read = centers_bytes
        pattern = "manifest-provided index policy"
    rows.append(
        make_custom_row(
            "patchify",
            "SelectPatchCenters",
            f"image=1x1x3x{height}x{width} -> centers={m}x2",
            1,
            fp32=fp32_ops,
            int_bool=int_ops,
            read=read,
            write=centers_bytes,
            pattern=pattern,
            source="C++:kernels.cpp/SelectPatchCenters",
        )
    )
    rows.append(
        make_custom_row(
            "patchify",
            "ScaleFeatureMaps",
            f"fmap=128x{fh}x{fw}, imap=384x{fh}x{fw}",
            1,
            fp32=fmap_values + imap_values,
            read=4 * (fmap_values + imap_values),
            write=4 * (fmap_values + imap_values),
            pattern="regular contiguous",
            source="C++:kernels.cpp/PatchifyFromCenters",
        )
    )
    for name, input_shape, output_shape in (
        ("PatchifyImap", (384, fh, fw), (m, 384, 1, 1)),
        ("PatchifyGmap", (128, fh, fw), (m, 128, 3, 3)),
        ("PatchifyColor", (3, height, width), (m, 3, 1, 1)),
    ):
        samples = product(output_shape)
        extra_ops = 4 * m if name == "PatchifyColor" else 0
        rows.append(
            make_custom_row(
                "patchify",
                name,
                f"{'x'.join(map(str, input_shape))} + {m}x2 -> {'x'.join(map(str, output_shape))}",
                1,
                # Two coordinate adds plus the 17 scalar operations in
                # BilinearSample for every output value.
                fp32=19 * samples + extra_ops,
                read=tensor_bytes(input_shape) + centers_bytes,
                write=tensor_bytes(output_shape),
                pattern="centroid-indexed bilinear",
                source="C++:kernels.cpp/PatchifySingle",
            )
        )
    rows.append(
        make_custom_row(
            "patchify",
            "BuildGridTensor",
            f"height={fh},width={fw} -> 3x{fh}x{fw}",
            1,
            int_bool=2 * fh * fw,
            write=tensor_bytes((3, fh, fw)),
            pattern="regular affine grid",
            source="C++:kernels.cpp/BuildGridTensor",
        )
    )
    rows.append(
        make_custom_row(
            "patchify",
            "PatchifyGrid",
            f"3x{fh}x{fw} + {m}x2 -> {m}x3x3x3",
            1,
            fp32=19 * m * 3 * 3 * 3,
            read=tensor_bytes((3, fh, fw)) + centers_bytes,
            write=tensor_bytes((m, 3, 3, 3)),
            pattern="centroid-indexed bilinear",
            source="C++:kernels.cpp/PatchifySingle",
        )
    )
    l2h, l2w = fh // 4, fw // 4
    pooled_values = 128 * l2h * l2w
    rows.append(
        make_custom_row(
            "feature_pyramid",
            "AveragePool2d(level=4)",
            f"128x{fh}x{fw} -> 128x{l2h}x{l2w}",
            1,
            fp32=17 * pooled_values,
            read=tensor_bytes((128, fh, fw)),
            write=tensor_bytes((128, l2h, l2w)),
            pattern="regular window",
            source="C++:kernels.cpp/AveragePool2d",
        )
    )
    return rows


def tracker_rows(
    pa: PaConfig,
    graph: GraphStats,
    height: int,
    width: int,
    updates_per_frame: int,
    include_keyframe: bool,
) -> tuple[list[LayerRow], list[LayerRow], list[LayerRow]]:
    """Return rows before ONNX update, BA rows, and rows after BA."""
    e = graph.edge_count
    k = graph.unique_patches
    n = graph.frame_count
    d = 6 * graph.free_pose_count
    fh, fw = height // 4, width // 4
    l2h, l2w = fh // 4, fw // 4
    refs = graph.referenced_frame_count
    pre: list[LayerRow] = []
    pre.append(
        make_custom_row(
            "factor_graph",
            "AppendForwardBackwardFactors",
            f"E_old={max(e - graph.new_edge_count, 0)}, E_new={graph.new_edge_count} -> E={e}",
            1,
            int_bool=graph.new_edge_count,
            read=(max(e - graph.new_edge_count, 0) * (384 * 4 + 3 * 8) + graph.new_edge_count * 2 * 8),
            write=e * (384 * 4 + 3 * 8),
            pattern="factor-graph append policy",
            source="C++:dpvo.cpp/AppendFactors",
        )
    )
    # DPVOTracker::Update first calls CopyPatchTensorFromGraph(graph_), so the
    # canonical C++ input is the complete active patch tensor, not only unique
    # kk entries. Correlation/GatherContext below use compact referenced slots.
    active_patch_count = n * pa.patches_per_frame
    reproject_read = (
        tensor_bytes((refs, 7))
        + tensor_bytes((active_patch_count, 3, 3, 3))
        + tensor_bytes((refs, 4))
        + tensor_bytes((e, 3), "int64")
    )
    pre.append(
        make_custom_row(
            "projective_ops",
            "ReprojectPatchGrid",
            f"poses={refs}x7, patches={active_patch_count}x3x3x3, edges={e} -> {e}x2x3x3",
            updates_per_frame,
            # RelativePose plus nine projected points per edge, retaining the
            # explicit float/double arithmetic in the C++ helpers.
            fp32=242 * e,
            fp64=720 * e,
            read=reproject_read,
            write=tensor_bytes((e, 2, 3, 3)),
            pattern="factor-graph indexed",
            source="C++:projective_ops.cpp/ReprojectPatchGrid",
        )
    )
    corr_samples = e * 3 * 3 * 2 * 49
    corr_read = (
        tensor_bytes((k, 128, 3, 3))
        + tensor_bytes((graph.target_frame_count, 128, fh, fw))
        + tensor_bytes((graph.target_frame_count, 128, l2h, l2w))
        + tensor_bytes((e, 2, 3, 3))
        + tensor_bytes((e, 2), "int64")
    )
    pre.append(
        make_custom_row(
            "correlation",
            "BuildCorrelationVolume",
            f"E={e}, P=3, levels=2, radius=3, C=128 -> 1x{e}x882",
            updates_per_frame,
            # For every level/sample/channel: a 17-op bilinear interpolation
            # and a 2-op multiply-accumulate. Coordinate formation contributes
            # three operations per level/sample (six across the two levels).
            fp32=corr_samples * (19 * 128 + 3),
            read=corr_read,
            write=tensor_bytes((1, e, 882)),
            pattern="factor-graph + coordinate-indexed bilinear",
            source="C++:correlation.cpp/BuildCorrelationVolumeImpl",
        )
    )
    pre.append(
        make_custom_row(
            "factor_graph",
            "GatherContext",
            f"imap={k}x384, kk={e} -> 1x{e}x384",
            updates_per_frame,
            read=tensor_bytes((k, 384)) + tensor_bytes((e,), "int64"),
            write=tensor_bytes((1, e, 384)),
            pattern="factor-graph indexed",
            source="C++:dpvo.cpp/GatherContext",
        )
    )

    ba: list[LayerRow] = []
    ba.append(
        make_custom_row(
            "update_postprocess",
            "TargetAndWeight",
            f"coords={e}x2x3x3, delta/weight=1x{e}x2 -> target/weight=1x{e}x2",
            updates_per_frame,
            fp32=4 * e,
            read=tensor_bytes((e, 2, 3, 3)) + 2 * tensor_bytes((1, e, 2)),
            write=2 * tensor_bytes((1, e, 2)),
            pattern="regular contiguous",
            source="C++:dpvo.cpp/Update",
        )
    )
    ba_invocations = updates_per_frame * pa.ba_iterations
    linearization_bytes = e * (2 * 4 + 1 + 26 * 8)
    ba.append(
        make_custom_row(
            "bundle_adjustment",
            "LinearizePatchCenters",
            f"E={e}, K={k}, poses={refs} -> coord/Ji/Jj/Jz[{e}]",
            ba_invocations,
            # Common valid-depth path: projection, JP*JA, Adjoint, pose
            # Jacobians, and depth Jacobian.
            fp32=405 * e,
            fp64=269 * e,
            read=reproject_read,
            write=linearization_bytes,
            pattern="factor-graph indexed",
            source="C++:projective_ops.cpp/LinearizePatchCenters",
        )
    )
    # Exact arithmetic in the C++ loop bodies on the all-valid path:
    # residual/depth scalar terms = 18 per edge; cross+RHS = 84 per free
    # endpoint; one 6x6 weighted pose block = 252 operations.
    assembly_ops = (
        18 * e
        + 84 * graph.free_endpoint_activations
        + 252 * graph.free_pose_block_activations
    )
    b_bytes = tensor_bytes((d, d))
    emat_bytes = tensor_bytes((d, k))
    c_bytes = tensor_bytes((k,))
    v_bytes = tensor_bytes((d,))
    w_bytes = tensor_bytes((k,))
    ba.append(
        make_custom_row(
            "bundle_adjustment",
            "NormalEquationAssembly",
            f"E={e}, pose_dim={d}, K={k} -> B={d}x{d}, E={d}x{k}, C/v/w",
            ba_invocations,
            fp32=assembly_ops,
            int_bool=6 * e,
            read=linearization_bytes + 2 * tensor_bytes((1, e, 2)) + tensor_bytes((e, 3), "int64"),
            write=b_bytes + emat_bytes + 2 * c_bytes + v_bytes,
            pattern="factor-graph sparse block accumulation",
            source="C++:kernels.cpp/BundleAdjustOneIteration",
        )
    )
    schur_ops = 2 * k + 3 * d * d * k + d * d + 3 * d * k + 4 * d
    ba.append(
        make_custom_row(
            "bundle_adjustment",
            "SchurComplement",
            f"B={d}x{d}, E={d}x{k}, C={k} -> S={d}x{d}, y={d}, Q={k}",
            ba_invocations,
            fp32=schur_ops,
            read=b_bytes + emat_bytes + 2 * c_bytes + v_bytes,
            write=b_bytes + v_bytes + c_bytes,
            pattern="dense-by-sparse reduction",
            source="C++:kernels.cpp/BundleAdjustOneIteration",
        )
    )
    decomposition_inner = d * (d - 1) * (d + 1) // 6
    cholesky_ops = 2 * decomposition_inner + d * (d + 1) // 2 + 2 * d * (d - 1) + 2 * d
    ba.append(
        make_custom_row(
            "bundle_adjustment",
            "CholeskySolveFloat32",
            f"S={d}x{d}, y={d} -> dX={d}",
            ba_invocations,
            fp32=cholesky_ops,
            read=b_bytes + v_bytes,
            write=v_bytes,
            pattern="regular dense triangular",
            source="C++:kernels.cpp/CholeskySolveFloat32",
        )
    )
    ba.append(
        make_custom_row(
            "bundle_adjustment",
            "DepthBackSubstitution",
            f"E_matrix={d}x{k}, dX={d}, Q/w={k} -> dZ={k}",
            ba_invocations,
            fp32=k * (2 * d + 2),
            read=emat_bytes + v_bytes + 2 * c_bytes,
            write=c_bytes,
            pattern="regular dense reduction",
            source="C++:kernels.cpp/BundleAdjustOneIteration",
        )
    )
    ba.append(
        make_custom_row(
            "bundle_adjustment",
            "ApplyStateIncrement",
            f"dX={d}, dZ={k} -> poses={graph.free_pose_count}x7, patch_depth={k}x3x3",
            ba_invocations,
            fp32=3 * k + 76 * graph.free_pose_count,
            fp64=291 * graph.free_pose_count,
            read=v_bytes + c_bytes + tensor_bytes((graph.free_pose_count, 7)) + tensor_bytes((k, 3, 3)),
            write=tensor_bytes((graph.free_pose_count, 7)) + tensor_bytes((k, 3, 3)),
            pattern="factor-graph state indexed",
            source="C++:kernels.cpp/BundleAdjustOneIteration",
        )
    )

    post: list[LayerRow] = []
    point_count = n * pa.patches_per_frame
    post.append(
        make_custom_row(
            "projective_ops",
            "PatchCenterPointCloud",
            f"poses={n}x7, patches={point_count}x3x3x3 -> points={point_count}x3",
            updates_per_frame,
            fp32=49 * point_count,
            fp64=117 * point_count,
            read=tensor_bytes((n, 7)) + tensor_bytes((point_count, 3, 3, 3)) + tensor_bytes((n, 4)) + tensor_bytes((point_count,), "int64"),
            write=tensor_bytes((point_count, 3)),
            pattern="patch-to-frame indexed",
            source="C++:projective_ops.cpp/PatchCenterPointCloud",
        )
    )
    if include_keyframe:
        flow_edges = 2 * pa.patches_per_frame
        # MotionMagnitude is invoked in both directions. Each invocation copies
        # the complete graph patch tensor and active pose/intrinsics arrays
        # before its three internal reprojections.
        keyframe_state_read = 2 * (
            tensor_bytes((n, 7))
            + tensor_bytes((active_patch_count, 3, 3, 3))
            + tensor_bytes((n, 4))
        )
        post.append(
            make_custom_row(
                "keyframe",
                "BidirectionalMotionMagnitude",
                f"two directions x {pa.patches_per_frame} edges -> scalar motion",
                1,
                # Two full and one translation-only reproject per edge, then
                # 145 float operations for the two weighted flow norms.
                fp32=754 * flow_edges,
                fp64=1674 * flow_edges,
                read=tensor_bytes((flow_edges, 3), "int64") + keyframe_state_read,
                write=tensor_bytes((2,)),
                pattern="factor-graph pair filter",
                source="C++:dpvo.cpp/MotionMagnitude + projective_ops.cpp/FlowMagnitude",
            )
        )
        post.append(
            make_custom_row(
                "factor_graph",
                "RemoveOldFactors",
                f"E={e} -> E_kept={graph.post_prune_edge_count}",
                1,
                int_bool=e,
                read=e * (384 * 4 + 4 * 4 + 3 * 8),
                write=graph.post_prune_edge_count * (384 * 4 + 4 * 4 + 3 * 8),
                pattern="factor-graph age mask",
                source="C++:dpvo.cpp/RemoveFactors",
            )
        )
    return pre, ba, post


def sum_rows(rows: Iterable[LayerRow], module: str = "TOTAL", layer: str = "TOTAL") -> LayerRow:
    items = list(rows)
    patterns = sorted({row.access_pattern for row in items})
    return LayerRow(
        module=module,
        layer=layer,
        shape=f"{len(items)} layer(s)",
        invocations=0,
        fp16_ops=sum(row.fp16_ops for row in items),
        fp32_ops=sum(row.fp32_ops for row in items),
        fp64_ops=sum(row.fp64_ops for row in items),
        int_bool_ops=sum(row.int_bool_ops for row in items),
        mem_read_bytes=sum(row.mem_read_bytes for row in items),
        mem_write_bytes=sum(row.mem_write_bytes for row in items),
        access_pattern="; ".join(patterns),
        source="aggregated",
    )


def aggregate_modules(rows: Sequence[LayerRow]) -> list[LayerRow]:
    order: list[str] = []
    grouped: dict[str, list[LayerRow]] = {}
    for row in rows:
        if row.module not in grouped:
            grouped[row.module] = []
            order.append(row.module)
        grouped[row.module].append(row)
    return [sum_rows(grouped[module], module, "MODULE TOTAL") for module in order]


TABLE_HEADERS = [
    "Module",
    "Layer",
    "Shape",
    "Invocations",
    "FP16 Ops",
    "FP32 Ops",
    "FP64 Ops",
    "INT/Bool Ops",
    "Total Ops",
    "Mem Read (Bytes)",
    "Mem Write (Bytes)",
    "Total Memory (Bytes)",
    "Op Intensity (Ops/Byte)",
    "Memory Access Pattern",
]


def display_row(row: LayerRow, human: bool) -> dict[str, str | int]:
    number = (lambda value: f"{value:,}") if human else (lambda value: value)
    invocations: str | int = "-" if row.invocations == 0 else row.invocations
    return {
        "Module": row.module,
        "Layer": row.layer,
        "Shape": row.shape,
        "Invocations": invocations,
        "FP16 Ops": number(row.fp16_ops),
        "FP32 Ops": number(row.fp32_ops),
        "FP64 Ops": number(row.fp64_ops),
        "INT/Bool Ops": number(row.int_bool_ops),
        "Total Ops": number(row.total_ops),
        "Mem Read (Bytes)": number(row.mem_read_bytes),
        "Mem Write (Bytes)": number(row.mem_write_bytes),
        "Total Memory (Bytes)": number(row.total_memory_bytes),
        "Op Intensity (Ops/Byte)": f"{row.operation_intensity:.6f}",
        "Memory Access Pattern": row.access_pattern,
    }


def render_markdown(rows: Sequence[LayerRow]) -> str:
    displayed = [display_row(row, human=True) for row in rows]
    widths = {
        header: max(len(header), *(len(str(row[header])) for row in displayed))
        for header in TABLE_HEADERS
    }

    def format_one(row: dict[str, str | int]) -> str:
        return "| " + " | ".join(str(row[header]).ljust(widths[header]) for header in TABLE_HEADERS) + " |"

    lines = [
        format_one({header: header for header in TABLE_HEADERS}),
        "| " + " | ".join("-" * widths[header] for header in TABLE_HEADERS) + " |",
    ]
    lines.extend(format_one(row) for row in displayed)
    return "\n".join(lines)


def render_csv(rows: Sequence[LayerRow]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=TABLE_HEADERS)
    writer.writeheader()
    for row in rows:
        writer.writerow(display_row(row, human=False))
    return buffer.getvalue()


def parse_pa_assignments(assignments: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(f"invalid --pa {assignment!r}; expected KEY=VALUE")
        key, value = assignment.split("=", 1)
        normalized = key.strip().upper()
        if normalized not in SUPPORTED_PA_KEYS:
            choices = ", ".join(sorted(SUPPORTED_PA_KEYS))
            raise ValueError(
                f"unsupported operation-count P_a key {normalized!r}; supported keys: {choices}"
            )
        result[normalized] = parse_scalar(value)
    return result


def resolve_pa(args: argparse.Namespace) -> PaConfig:
    values = load_simple_yaml(args.config)
    values.update(parse_pa_assignments(args.pa))
    explicit = {
        "PATCHES_PER_FRAME": args.patches_per_frame,
        "REMOVAL_WINDOW": args.removal_window,
        "OPTIMIZATION_WINDOW": args.optimization_window,
        "PATCH_LIFETIME": args.patch_lifetime,
        "BA_ITERATIONS": args.ba_iterations,
        "CENTROID_SEL_STRAT": args.centroid_sel_strat,
    }
    values.update({key: value for key, value in explicit.items() if value is not None})
    return PaConfig(
        patches_per_frame=int(values.get("PATCHES_PER_FRAME", 96)),
        removal_window=int(values.get("REMOVAL_WINDOW", 22)),
        optimization_window=int(values.get("OPTIMIZATION_WINDOW", 10)),
        patch_lifetime=int(values.get("PATCH_LIFETIME", 13)),
        ba_iterations=int(values.get("BA_ITERATIONS", 2)),
        centroid_sel_strat=str(values.get("CENTROID_SEL_STRAT", "RANDOM")).upper(),
    )


def validate_inputs(args: argparse.Namespace, pa: PaConfig, frame_count: int) -> None:
    positive = {
        "height": args.height,
        "width": args.width,
        "patches_per_frame": pa.patches_per_frame,
        "removal_window": pa.removal_window,
        "optimization_window": pa.optimization_window,
        "patch_lifetime": pa.patch_lifetime,
        "ba_iterations": pa.ba_iterations,
        "active_frames": frame_count,
        "updates_per_frame": args.updates_per_frame,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    for name in ("edges", "unique_patches", "edge_groups", "free_poses"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if args.free_poses is not None and args.free_poses >= frame_count:
        raise ValueError(
            f"free_poses must leave at least one fixed pose; got {args.free_poses} for {frame_count} frames"
        )
    if pa.centroid_sel_strat not in {"RANDOM", "GRADIENT_BIAS", "MANIFEST"}:
        raise ValueError(
            "CENTROID_SEL_STRAT must be RANDOM, GRADIENT_BIAS, or MANIFEST; "
            f"got {pa.centroid_sel_strat!r}"
        )
    if args.height % 16 or args.width % 16:
        raise ValueError("height and width must be divisible by 16 for the C++ two-level feature pyramid")
    minimum_steady_frames = max(
        pa.removal_window + pa.patch_lifetime,
        pa.optimization_window + 1,
        8,
    )
    if frame_count < minimum_steady_frames:
        raise ValueError(
            "active_frames is too small for steady-frame accounting; use at least "
            f"{minimum_steady_frames}"
        )


def build_rows(args: argparse.Namespace) -> tuple[PaConfig, GraphStats, list[LayerRow]]:
    pa = resolve_pa(args)
    # Keep this workload horizon fixed across P_a candidates. R + L is only a
    # candidate-specific maturity constraint and must not become the default
    # horizon, otherwise point-cloud/state traffic would be compared at
    # different sequence positions.
    frame_count = args.active_frames or DEFAULT_ACTIVE_FRAMES
    validate_inputs(args, pa, frame_count)
    graph = derive_graph_stats(
        pa,
        frame_count,
        args.edges,
        args.unique_patches,
        args.edge_groups,
        args.free_poses,
    )
    feature_path = args.onnx_dir / "feature_extractor.onnx"
    update_path = args.onnx_dir / "update_block.onnx"
    include_all = args.onnx_nodes == "all"
    rows: list[LayerRow] = []
    rows.extend(
        onnx_rows(
            feature_path,
            "feature_extractor",
            {"images": TensorInfo((1, 1, 3, args.height, args.width), "float32")},
            graph.unique_patches,
            graph.edge_groups,
            1,
            include_all,
        )
    )
    rows.extend(patchify_rows(pa, args.height, args.width))
    pre, ba, post = tracker_rows(
        pa,
        graph,
        args.height,
        args.width,
        args.updates_per_frame,
        not args.no_keyframe,
    )
    rows.extend(pre)
    e = graph.edge_count
    update_inputs = {
        "net": TensorInfo((1, e, 384), "float32"),
        "ctx": TensorInfo((1, e, 384), "float32"),
        "corr": TensorInfo((1, e, 882), "float32"),
        "ii": TensorInfo((e,), "int64"),
        "jj": TensorInfo((e,), "int64"),
        "kk": TensorInfo((e,), "int64"),
        "ix": TensorInfo((e,), "int64"),
        "jx": TensorInfo((e,), "int64"),
    }
    rows.extend(
        onnx_rows(
            update_path,
            "update_block",
            update_inputs,
            graph.unique_patches,
            graph.edge_groups,
            args.updates_per_frame,
            include_all,
        )
    )
    rows.extend(ba)
    rows.extend(post)
    return pa, graph, rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count DPVO useful operations and layer-boundary logical DRAM bytes."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--pa",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a P_a value; repeatable (e.g. --pa PATCHES_PER_FRAME=48).",
    )
    parser.add_argument("--patches-per-frame", type=int)
    parser.add_argument("--removal-window", type=int)
    parser.add_argument("--optimization-window", type=int)
    parser.add_argument("--patch-lifetime", type=int)
    parser.add_argument("--ba-iterations", type=int)
    parser.add_argument(
        "--centroid-sel-strat",
        choices=("RANDOM", "GRADIENT_BIAS", "MANIFEST"),
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument(
        "--active-frames",
        type=int,
        help=f"Common reference graph frame count. Default: {DEFAULT_ACTIVE_FRAMES}; use the same value for every P_a candidate.",
    )
    parser.add_argument("--updates-per-frame", type=int, default=1)
    parser.add_argument("--edges", type=int, help="Override derived update-time factor count E.")
    parser.add_argument("--unique-patches", type=int, help="Override derived unique BA/correlation patch count K.")
    parser.add_argument("--edge-groups", type=int, help="Override unique (ii,jj) group count used by SoftAgg.")
    parser.add_argument("--free-poses", type=int, help="Override free BA pose count.")
    parser.add_argument("--no-keyframe", action="store_true", help="Exclude keyframe motion test and factor pruning.")
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR)
    parser.add_argument(
        "--onnx-nodes",
        choices=("all", "compute"),
        default="all",
        help="all preserves ONNX node boundaries; compute hides zero-arithmetic metadata/view nodes.",
    )
    parser.add_argument("--detail", choices=("layer", "module"), default="layer")
    parser.add_argument("--format", choices=("markdown", "csv", "json"), default="markdown")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        pa, graph, layer_rows = build_rows(args)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    visible_rows = aggregate_modules(layer_rows) if args.detail == "module" else list(layer_rows)
    total = sum_rows(layer_rows)
    output_rows = visible_rows + [total]
    metadata = {
        "scope": "one accepted steady-state frame",
        "accounting_model": "layer-boundary DRAM materialization",
        "pa": asdict(pa),
        "workload": {
            "height": args.height,
            "width": args.width,
            "active_frames": graph.frame_count,
            "updates_per_frame": args.updates_per_frame,
            "keyframe_test_included": not args.no_keyframe,
            "onnx_node_filter": args.onnx_nodes,
        },
        "derived": {key: value for key, value in asdict(graph).items() if key != "pairs"},
        "counting_convention": {
            "mac": "2 operations (multiply + accumulate)",
            "layer_memory": "read every input/weight/metadata tensor once; write every output tensor once",
            "cross_layer_reuse": "forbidden",
            "within_layer_reuse": "allowed",
            "onnx_precision": "from exported TensorProto (checked-in models are float32)",
            "ba_path": "source-level valid-residual path; implementation container/address overhead excluded",
            "data_dependent_control": "initialized accepted-frame path; motion-probe/bootstrap and optional loop closure excluded; keyframe frame-removal outcome represented by active_frames",
        },
    }
    if args.format == "markdown":
        lines = [
            f"Scope: {metadata['scope']}",
            "",
            "P_a: " + ", ".join(f"{key}={value}" for key, value in asdict(pa).items()),
            "",
            "Workload: "
            + ", ".join(f"{key}={value}" for key, value in metadata["workload"].items()),
            "",
            "Derived: "
            + ", ".join(
                f"{key}={value}"
                for key, value in metadata["derived"].items()
                if key
                in {
                    "frame_count",
                    "edge_count",
                    "post_prune_edge_count",
                    "new_edge_count",
                    "unique_patches",
                    "edge_groups",
                    "free_pose_count",
                }
            ),
            "",
            "Counting: MAC=2 ops; every layer rereads input/weight/metadata and writes its complete output.",
            "",
            render_markdown(output_rows),
            "",
        ]
        text = "\n".join(lines)
    elif args.format == "csv":
        text = render_csv(output_rows)
    else:
        payload = dict(metadata)
        payload["rows"] = [row.raw_dict() for row in visible_rows]
        payload["total"] = total.raw_dict()
        text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
