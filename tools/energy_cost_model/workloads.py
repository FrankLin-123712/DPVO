"""DPVO workload construction.

This layer maps P_a to module-level workload descriptors. It intentionally
keeps energy and timing out of the formulas; those are added by model.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    from .parameters import AlgorithmParams
except ImportError:  # pragma: no cover - script execution fallback
    from parameters import AlgorithmParams


DPVO_DIM = 384
FNET_DIM = 128
GMAP_DIM = 128
ENCODER_BASE_DIM = 32


@dataclass(frozen=True)
class ConvLayer:
    name: str
    cin: int
    cout: int
    kernel: int
    stride: int
    padding: int
    hin: int
    win: int

    @property
    def hout(self) -> int:
        return conv_out(self.hin, self.kernel, self.stride, self.padding)

    @property
    def wout(self) -> int:
        return conv_out(self.win, self.kernel, self.stride, self.padding)

    def as_dense(self, dtype_bytes: int) -> "DenseOp":
        return DenseOp(
            name=self.name,
            m=self.hout * self.wout,
            n=self.cout,
            k=self.cin * self.kernel * self.kernel,
            input_bytes=dtype_bytes,
            weight_bytes=dtype_bytes,
            output_bytes=dtype_bytes,
            kind="conv2d-as-gemm",
        )


@dataclass(frozen=True)
class DenseOp:
    name: str
    m: int
    n: int
    k: int
    input_bytes: int
    weight_bytes: int
    output_bytes: int
    kind: str = "gemm"

    @property
    def macs(self) -> int:
        return self.m * self.n * self.k

    @property
    def minimum_tensor_bytes(self) -> int:
        return (
            self.m * self.k * self.input_bytes
            + self.k * self.n * self.weight_bytes
            + self.m * self.n * self.output_bytes
        )


@dataclass(frozen=True)
class IrregularWorkload:
    name: str
    kind: str
    elements: int = 0
    read_bytes: int = 0
    write_bytes: int = 0
    metadata_bytes: int = 0
    macs: int = 0
    alu_ops: int = 0
    atomics: int = 0
    branches: int = 0
    syncs: int = 0
    access_pattern: str = "random"
    notes: str = ""


@dataclass(frozen=True)
class ModuleWorkload:
    name: str
    preferred_mapping: str
    dense_ops: tuple[DenseOp, ...] = field(default_factory=tuple)
    irregular: tuple[IrregularWorkload, ...] = field(default_factory=tuple)
    notes: str = ""

    @property
    def useful_macs(self) -> int:
        return sum(op.macs for op in self.dense_ops) + sum(work.macs for work in self.irregular)

    @property
    def minimum_bytes(self) -> int:
        return sum(op.minimum_tensor_bytes for op in self.dense_ops) + sum(
            work.read_bytes + work.write_bytes + work.metadata_bytes for work in self.irregular
        )


def conv_out(size: int, kernel: int, stride: int, padding: int, dilation: int = 1) -> int:
    return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def build_basic_encoder4_layers(height: int, width: int, output_dim: int, prefix: str) -> list[ConvLayer]:
    layers: list[ConvLayer] = []

    layers.append(ConvLayer(f"{prefix}.conv1", 3, 32, 7, 2, 3, height, width))
    h1, w1 = layers[-1].hout, layers[-1].wout

    for block in range(2):
        layers.append(ConvLayer(f"{prefix}.layer1.{block}.conv1", 32, 32, 3, 1, 1, h1, w1))
        layers.append(ConvLayer(f"{prefix}.layer1.{block}.conv2", 32, 32, 3, 1, 1, h1, w1))

    layers.append(ConvLayer("{}.layer2.0.conv1".format(prefix), 32, 64, 3, 2, 1, h1, w1))
    h2, w2 = layers[-1].hout, layers[-1].wout
    layers.append(ConvLayer(f"{prefix}.layer2.0.conv2", 64, 64, 3, 1, 1, h2, w2))
    layers.append(ConvLayer(f"{prefix}.layer2.0.downsample", 32, 64, 1, 2, 0, h1, w1))

    layers.append(ConvLayer(f"{prefix}.layer2.1.conv1", 64, 64, 3, 1, 1, h2, w2))
    layers.append(ConvLayer(f"{prefix}.layer2.1.conv2", 64, 64, 3, 1, 1, h2, w2))
    layers.append(ConvLayer(f"{prefix}.conv2", 64, output_dim, 1, 1, 0, h2, w2))
    return layers


def dense_linear(name: str, samples: int, in_dim: int, out_dim: int, dtype_bytes: int) -> DenseOp:
    return DenseOp(
        name=name,
        m=samples,
        n=out_dim,
        k=in_dim,
        input_bytes=dtype_bytes,
        weight_bytes=dtype_bytes,
        output_bytes=dtype_bytes,
        kind="linear",
    )


def build_feature_encoder(params: AlgorithmParams) -> ModuleWorkload:
    fnet = build_basic_encoder4_layers(params.height, params.width, FNET_DIM, "fnet")
    inet = build_basic_encoder4_layers(params.height, params.width, DPVO_DIM, "inet")
    dense_ops = tuple(layer.as_dense(params.nn_dtype_bytes) for layer in fnet + inet)
    out_h = fnet[-1].hout
    out_w = fnet[-1].wout
    return ModuleWorkload(
        name="feature_context_encoder",
        preferred_mapping="gemmini",
        dense_ops=dense_ops,
        notes=f"BasicEncoder4 fnet+inet, output feature grid {out_h}x{out_w}",
    )


def build_patch_extraction(params: AlgorithmParams) -> ModuleWorkload:
    patches = params.patches_per_frame
    p = params.patch_size
    dtype_bytes = params.nn_dtype_bytes

    imap_samples = patches * DPVO_DIM
    gmap_samples = patches * p * p * GMAP_DIM
    grid_samples = patches * p * p * 3
    color_samples = patches * 3
    total_samples = imap_samples + gmap_samples + grid_samples + color_samples

    # Bilinear patchify reads four neighboring elements per sampled value.
    read_bytes = total_samples * 4 * dtype_bytes
    write_bytes = total_samples * dtype_bytes
    alu_ops = total_samples * 8
    branches = patches * 12
    metadata = patches * 2 * 4

    notes = "altcorr.patchify for imap, gmap, grid patches, and optional color"
    if params.centroid_selection == "GRADIENT_BIAS":
        gradient_pixels = params.height * params.width
        read_bytes += gradient_pixels
        write_bytes += params.feature_height * params.feature_width * 4
        alu_ops += gradient_pixels * 6 + patches * 3 * 20
        metadata += patches * 3 * 8
        notes += "; includes image-gradient centroid candidate scoring"

    return ModuleWorkload(
        name="patch_extraction",
        preferred_mapping="cpu",
        irregular=(
            IrregularWorkload(
                name="patchify",
                kind="bilinear-gather",
                elements=total_samples,
                read_bytes=read_bytes,
                write_bytes=write_bytes,
                metadata_bytes=metadata,
                alu_ops=alu_ops,
                branches=branches,
                access_pattern="random",
                notes=notes,
            ),
        ),
    )


def build_correlation(params: AlgorithmParams) -> ModuleWorkload:
    edges = params.active_edges
    p = params.patch_size
    r = params.corr_radius
    levels = params.corr_levels
    dtype_bytes = params.nn_dtype_bytes
    output_diameter = 2 * r + 1
    # Keep this aligned with the existing workload analyzer and DPVO corr tensor.
    dot_diameter = 2 * r + 2

    macs = levels * edges * p * p * dot_diameter * dot_diameter * GMAP_DIM
    feature_vector_reads = levels * edges * p * p * dot_diameter * dot_diameter * 2 * GMAP_DIM
    output_values = levels * edges * p * p * output_diameter * output_diameter
    coord_metadata = edges * p * p * 2 * 4 + edges * 2 * 8

    cpu_work = IrregularWorkload(
        name="local_correlation_lookup",
        kind="dot-bilinear-sample",
        elements=output_values,
        read_bytes=feature_vector_reads * dtype_bytes,
        write_bytes=output_values * dtype_bytes,
        metadata_bytes=coord_metadata,
        macs=macs,
        alu_ops=output_values * 8,
        branches=edges * levels * p * p,
        access_pattern="random",
        notes="default model treats local correlation as semi-irregular gather + dot",
    )

    # Alternative Gemmini-friendly representation if the caller selects
    # --corr-mapping gemmini: build each edge/patch sample as a row and each
    # local candidate as a column. The gather/staging traffic is still charged.
    dense = DenseOp(
        name="correlation.batched_dot",
        m=edges * p * p,
        n=levels * output_diameter * output_diameter,
        k=GMAP_DIM,
        input_bytes=dtype_bytes,
        weight_bytes=dtype_bytes,
        output_bytes=dtype_bytes,
        kind="batched-correlation-gemm",
    )

    staging = IrregularWorkload(
        name="correlation_gemmini_staging",
        kind="gather-stage",
        elements=feature_vector_reads,
        read_bytes=feature_vector_reads * dtype_bytes,
        write_bytes=feature_vector_reads * dtype_bytes,
        metadata_bytes=coord_metadata,
        alu_ops=output_values * 4,
        branches=edges * levels,
        access_pattern="random",
        notes="cost to gather local correlation operands before Gemmini GEMM",
    )

    return ModuleWorkload(
        name="correlation_lookup",
        preferred_mapping="cpu",
        dense_ops=(dense,),
        irregular=(cpu_work, staging),
        notes=f"E={edges}, P={p}, R={r}, levels={levels}, C={GMAP_DIM}",
    )


def build_update_dense(params: AlgorithmParams) -> ModuleWorkload:
    edges = params.active_edges
    dtype_bytes = params.nn_dtype_bytes
    p = params.patch_size
    corr_dim = params.corr_levels * (2 * params.corr_radius + 1) ** 2 * p * p
    ops: list[DenseOp] = []

    for idx, (in_dim, out_dim) in enumerate(
        ((corr_dim, DPVO_DIM), (DPVO_DIM, DPVO_DIM), (DPVO_DIM, DPVO_DIM))
    ):
        ops.append(dense_linear(f"update.corr.{idx}", edges, in_dim, out_dim, dtype_bytes))

    for branch in ("c1", "c2"):
        ops.append(dense_linear(f"update.{branch}.0", edges, DPVO_DIM, DPVO_DIM, dtype_bytes))
        ops.append(dense_linear(f"update.{branch}.2", edges, DPVO_DIM, DPVO_DIM, dtype_bytes))

    for agg in ("agg_kk", "agg_ij"):
        for proj in ("f", "g", "h"):
            ops.append(dense_linear(f"update.{agg}.{proj}", edges, DPVO_DIM, DPVO_DIM, dtype_bytes))

    for block in range(2):
        ops.append(dense_linear(f"update.gru.{block}.gate", edges, DPVO_DIM, DPVO_DIM, dtype_bytes))
        ops.append(dense_linear(f"update.gru.{block}.res0", edges, DPVO_DIM, DPVO_DIM, dtype_bytes))
        ops.append(dense_linear(f"update.gru.{block}.res2", edges, DPVO_DIM, DPVO_DIM, dtype_bytes))

    ops.append(dense_linear("update.delta_head", edges, DPVO_DIM, 2, dtype_bytes))
    ops.append(dense_linear("update.weight_head", edges, DPVO_DIM, 2, dtype_bytes))

    return ModuleWorkload(
        name="update_dense_linears",
        preferred_mapping="gemmini",
        dense_ops=tuple(ops),
        notes=f"Update MLP/GRU/SoftAgg projection linears, E={edges}, DIM={DPVO_DIM}",
    )


def build_soft_aggregation(params: AlgorithmParams) -> ModuleWorkload:
    edges = params.active_edges
    dtype_bytes = params.nn_dtype_bytes
    feature_bytes = edges * DPVO_DIM * dtype_bytes
    metadata = edges * 3 * 8
    atomics = edges * DPVO_DIM * 2
    alu_ops = edges * DPVO_DIM * 8

    return ModuleWorkload(
        name="soft_aggregation_scatter",
        preferred_mapping="cpu",
        irregular=(
            IrregularWorkload(
                name="agg_kk_scatter",
                kind="scatter-softmax-sum",
                elements=edges * DPVO_DIM,
                read_bytes=feature_bytes * 3,
                write_bytes=feature_bytes * 2,
                metadata_bytes=metadata,
                alu_ops=alu_ops,
                atomics=atomics,
                branches=edges,
                access_pattern="random",
                notes="torch.unique + scatter_softmax + scatter_sum over kk groups",
            ),
            IrregularWorkload(
                name="agg_ij_scatter",
                kind="scatter-softmax-sum",
                elements=edges * DPVO_DIM,
                read_bytes=feature_bytes * 3,
                write_bytes=feature_bytes * 2,
                metadata_bytes=metadata,
                alu_ops=alu_ops,
                atomics=atomics,
                branches=edges,
                access_pattern="random",
                notes="torch.unique + scatter_softmax + scatter_sum over ii/jj groups",
            ),
        ),
    )


def build_ba(params: AlgorithmParams) -> ModuleWorkload:
    edges = params.active_edges
    iters = params.ba_iterations
    dtype_bytes = params.ba_dtype_bytes
    unique_patches = params.active_unique_patches
    poses = params.optimization_window
    state_dim = 6 * poses

    input_bytes_per_edge = (2 * 7 + 3 + 2 + 2 + 4) * 4 + 3 * 8
    block_atomic_bytes = (
        2 * 4 * 36 * 2 * dtype_bytes
        + 2 * 2 * 6 * 2 * dtype_bytes
        + 2 * 2 * 6 * 2 * dtype_bytes
        + 2 * 3 * 2 * dtype_bytes
    )
    jacobian_bytes_per_edge = input_bytes_per_edge + block_atomic_bytes

    jacobian = IrregularWorkload(
        name="ba_jacobian_hessian_assembly",
        kind="se3-projection-jacobian",
        elements=edges * iters,
        read_bytes=edges * iters * input_bytes_per_edge,
        write_bytes=edges * iters * block_atomic_bytes,
        metadata_bytes=edges * iters * 3 * 8,
        macs=edges * iters * params.ba_macs_per_edge,
        alu_ops=edges * iters * params.ba_macs_per_edge,
        atomics=edges * iters * 96,
        branches=edges * iters * 8,
        access_pattern="random",
        notes=f"local fastba.BA residual/Jacobian/Hessian assembly, {jacobian_bytes_per_edge} B/edge/iter",
    )

    eqet = state_dim * unique_patches * state_dim
    equ = state_dim * unique_patches
    etdx = unique_patches * state_dim
    chol = int(state_dim**3 / 3)
    solve = state_dim * state_dim
    schur_macs = iters * (eqet + equ + etdx + chol + solve)
    schur_values = (
        state_dim * state_dim
        + state_dim * unique_patches
        + unique_patches
        + state_dim
        + unique_patches
        + state_dim * state_dim
        + state_dim
        + unique_patches
    )

    schur = IrregularWorkload(
        name="schur_complement_solve",
        kind="sparse-block-reduce-dense-solve",
        elements=schur_values,
        read_bytes=iters * schur_values * dtype_bytes,
        write_bytes=iters * (state_dim * state_dim + state_dim + unique_patches) * dtype_bytes,
        metadata_bytes=unique_patches * 8 + poses * 8,
        macs=schur_macs,
        alu_ops=schur_macs,
        branches=iters * (unique_patches + poses),
        access_pattern="sequential",
        notes=f"state_dim={state_dim}, unique_patches={unique_patches}",
    )

    return ModuleWorkload(
        name="bundle_adjustment",
        preferred_mapping="cpu",
        irregular=(jacobian, schur),
        notes=f"local BA, poses={poses}, E={edges}, K={unique_patches}, iters={iters}",
    )


def build_graph_management(params: AlgorithmParams) -> ModuleWorkload:
    new_edges = params.new_edges_per_frame
    active_edges = params.active_edges
    dtype_bytes = params.nn_dtype_bytes
    metadata = (new_edges * 3 + active_edges) * 8
    state_bytes = (
        params.patches_per_frame * (DPVO_DIM + GMAP_DIM * params.patch_size * params.patch_size) * dtype_bytes
        + params.patches_per_frame * 3 * params.patch_size * params.patch_size * 4
    )

    return ModuleWorkload(
        name="graph_management",
        preferred_mapping="cpu",
        irregular=(
            IrregularWorkload(
                name="append_remove_keyframe_state",
                kind="control-dynamic-graph",
                elements=new_edges + active_edges,
                read_bytes=metadata + state_bytes,
                write_bytes=metadata + state_bytes,
                metadata_bytes=metadata,
                alu_ops=(new_edges + active_edges) * 12,
                branches=(new_edges + active_edges) * 4,
                syncs=2,
                access_pattern="random",
                notes="append forward/back factors, keyframe removal, state ring-buffer updates",
            ),
        ),
    )


def build_dpvo_workloads(params: AlgorithmParams) -> list[ModuleWorkload]:
    workloads = [
        build_feature_encoder(params),
        build_patch_extraction(params),
        build_correlation(params),
        build_update_dense(params),
        build_soft_aggregation(params),
        build_ba(params),
        build_graph_management(params),
    ]

    if params.update_iterations <= 1:
        return workloads

    scaled: list[ModuleWorkload] = []
    repeated_names = {"correlation_lookup", "update_dense_linears", "soft_aggregation_scatter", "bundle_adjustment"}
    for workload in workloads:
        if workload.name not in repeated_names:
            scaled.append(workload)
            continue
        scaled.append(scale_workload(workload, params.update_iterations))
    return scaled


def scale_workload(workload: ModuleWorkload, factor: int) -> ModuleWorkload:
    dense_ops = tuple(
        DenseOp(
            name=f"{op.name}.iter{itr}",
            m=op.m,
            n=op.n,
            k=op.k,
            input_bytes=op.input_bytes,
            weight_bytes=op.weight_bytes,
            output_bytes=op.output_bytes,
            kind=op.kind,
        )
        for itr in range(factor)
        for op in workload.dense_ops
    )
    irregular = tuple(
        IrregularWorkload(
            name=work.name,
            kind=work.kind,
            elements=work.elements * factor,
            read_bytes=work.read_bytes * factor,
            write_bytes=work.write_bytes * factor,
            metadata_bytes=work.metadata_bytes * factor,
            macs=work.macs * factor,
            alu_ops=work.alu_ops * factor,
            atomics=work.atomics * factor,
            branches=work.branches * factor,
            syncs=work.syncs * factor,
            access_pattern=work.access_pattern,
            notes=work.notes,
        )
        for work in workload.irregular
    )
    return ModuleWorkload(
        name=workload.name,
        preferred_mapping=workload.preferred_mapping,
        dense_ops=dense_ops,
        irregular=irregular,
        notes=f"{workload.notes}; scaled by update_iterations={factor}",
    )
