# Energy cost model = f(P_a, H, P_h): DPVO-on-Gemmini 建模觀念教學

這份文件的目標不是只整理每篇論文講了什麼，而是把它們共同的建模精神濃縮成一個可以拿來做 DPVO deployment on Gemmini SoC 的方法論。

你現在想要的模型可以寫成：

```text
Energy cost model = f(P_a, H, P_h)
```

其中：

- `P_a`: 演算法可調變參數，例如影像解析度、patch 數量、keyframe/window 大小、correlation search radius、feature dimension、iteration 次數、precision、哪些 module 要近似或簡化。
- `H`: 硬體架構拓樸，也就是有哪些 compute unit、memory hierarchy、interconnect，以及它們怎麼連。例如 CPU + RoCC + Gemmini + scratchpad + accumulator + L1/L2/DRAM。
- `P_h`: 硬體可調參數，也就是把 `H` 這張拓樸圖具體化的數值。例如 Gemmini array dimension、PE 數量、scratchpad size/banks、accumulator size、data precision、DMA bandwidth、L2 size、DRAM bandwidth、frequency、voltage、是否支援 gating/skipping/sparsity。

核心結論先講在前面：

```text
P_a 不會直接決定 energy。
P_a 先決定 workload shape、資料統計與資料流。
H 與 P_h 決定這些 workload 能怎麼 map 到硬體。
mapping / dataflow / tiling 決定每一層 memory 與 compute component 的 action counts。
energy = action counts * per-action energy。
power = energy / execution time。
```

所以比較完整的形式其實是：

```text
E_dynamic(P_a, H, P_h)
  = sum over modules k
      sum over hardware components c
        sum over action types a
          N(k, c, a | P_a, H, P_h, M_k) * e(c, a | P_h, technology)

T(P_a, H, P_h)
  = performance_model(P_a, H, P_h, M_k)

P_dynamic
  = E_dynamic / T
```

其中 `M_k` 是第 `k` 個 DPVO module 的 mapping choice，例如跑在 CPU 還是 Gemmini、用哪種 dataflow、tile size、batch size、data layout、是否融合 kernel、是否用 sparse/irregular 專用處理。這個 `M_k` 是 energy model 的隱含變數，也是多篇論文一再強調不能忽略的東西。

如果要估總 power，還要加上 static/leakage：

```text
E_total = E_dynamic + sum over states s P_static(s) * T_s
P_total = E_total / T
```

但你在 `Energy_cost_model.md` 定義的 output 是 dynamic power，所以主要模型可以先以 `E_dynamic / T` 為主，並在報告中把 static/leakage 分開列為 optional term。

---

## 1. 閱讀來源與共同脈絡

本文件整理自 `papers/4_hardware_modeling/4.1_energy_cost_model/` 中的論文：

- [1_Energy-aware_Pruning_2016.pdf](1_Energy-aware_Pruning_2016.pdf)
- [2_Eyeriss_2016.pdf](2_Eyeriss_2016.pdf)
- [3_EyerissV2_2018.pdf](3_EyerissV2_2018.pdf)
- [4_MAESTRO_2018.pdf](4_MAESTRO_2018.pdf)
- [5_ScaleSim_2018.pdf](5_ScaleSim_2018.pdf)
- [6_Timeloop_2019.pdf](6_Timeloop_2019.pdf)
- [7_Accelergy_2019.pdf](7_Accelergy_2019.pdf)
- [8_ZigZag_2020.pdf](8_ZigZag_2020.pdf)
- [9_SparseLoop_2021.pdf](9_SparseLoop_2021.pdf)
- [10_Ruby_Improving_Hardware_Efficiency_for_Tensor_Algebra_Accelerators_Through_Imperfect_Factorization_2022.pdf](10_Ruby_Improving_Hardware_Efficiency_for_Tensor_Algebra_Accelerators_Through_Imperfect_Factorization_2022.pdf)
- [11_ScaleSimV3_2025.pdf](11_ScaleSimV3_2025.pdf)

並對齊你的背景筆記：

- [Energy_cost_model.md](../../Energy_cost_model.md)
- [Our_DPVO_Deployment_on_Gemmini-SoC_case.md](../../Our_DPVO_Deployment_on_Gemmini-SoC_case.md)
- [DPVO paper](../../1_algo/Hybrid_VO/10_DPVO_2023.pdf)
- [Gemmini paper](<../../2_hw_accelerate/2.4_Gemmini_soc/Gemmini%20-%20Enabling%20Systematic%20Deep-Learning%20Architecture.pdf>)

這些論文的共同精神可以壓成五句話：

1. `#MACs` 與 `#parameters` 不是 energy 的好代理。
   Memory access，尤其是 DRAM 和高層 memory hierarchy 的 data movement，常常比 compute 更貴。
2. Energy model 的主體是 action count model。
   你要先知道每個 component 做了幾次 read、write、MAC、NoC transfer、DMA、idle、gated、random access、repeated access，才能乘上 unit energy。
3. Mapping/dataflow/tiling 是 energy 的核心變數。
   同一個 workload、同一個硬體、同樣的 MAC 數，因為 dataflow 和 tiling 不同，可能有完全不同的 memory traffic、utilization、latency 與 energy。
4. Hardware model 不是只有 PE 數量。
   Memory hierarchy、buffer size、banking、bandwidth、NoC、DRAM controller、request queue、data layout、precision、clock gating 都會改變 energy/power。
5. 對 DPVO 這種有 dense tensor、gather/scatter、graph aggregation、sparse block solver、CPU control 的系統，不能只用 CNN accelerator 的簡化模型。
   Dense module 可以用 Timeloop/MAESTRO/ScaleSim 類方法，irregular/sparse module 要用 SparseLoop/Accelergy 的精神另外建 action taxonomy。

---

## 2. 為什麼不能只看 operations 或 execution time

你在 `Our_DPVO_Deployment_on_Gemmini-SoC_case.md` 裡已經指出兩個很重要的批判：

- 只用 operations 和 memory amount 太粗淺，因為 DPVO 有 data movement、irregular access、atomic add、synchronization。
- execution time 是 compute、data movement、synchronization 綜合後的結果，不能倒過來當成 operation 數量的 proxy。

這和 energy-cost-model 論文的核心一致。

Energy-aware pruning 一開始就提醒：減少 weights 或 MAC 不一定降低 energy。因為：

- 有些 layer 的 feature map movement 比 weight movement 更重要。
- FC layer 減參數很有效，但 convolution layer 的 feature activation 可能主導 energy。
- 兩個模型即使有相同 MAC 數，memory hierarchy access count 可能完全不同。

Eyeriss 進一步把這件事變成硬體設計原則：energy 是整個系統的 energy，不只是 PE 陣列的 energy。DRAM、global buffer、NoC、PE scratchpad 的每一次 data movement 都要算，而且要用 dataflow 讓高 reuse 的資料停在低能耗 memory level。

Timeloop 則把這件事推到 mapping search：同一個 architecture 和 layer，可以有很多 mapping performance 很接近，但 energy 差很多。反過來，只有最少 DRAM access 也不夠，因為 on-chip buffer 和 NoC access 仍可能差很多。

對 DPVO-on-Gemmini 來說，這代表：

```text
不能問：DPVO 有幾個 MAC，所以 power 多少？

要問：
1. 每個 DPVO module 產生什麼 tensor/graph/sparse workload？
2. 這些 workload 被 map 到 CPU/Gemmini/custom unit 的哪裡？
3. 對每個 mapping，資料在哪些 memory level 之間移動？
4. 每個 memory/component 的 read/write/compute/idle/gated/random/repeated action 有幾次？
5. 每種 action 的 energy 是多少？
6. 在 bandwidth、bank conflict、DMA、CPU synchronization 下，總時間是多少？
7. dynamic power = dynamic energy / time。
```

---

## 3. `H` 與 `P_h` 要分開想

你在問題中把 `H` 定義為硬體架構，把 `P_h` 定義為硬體可調參數。這個區分非常好，因為它可以避免把「架構拓樸」和「參數數值」混在一起。

### 3.1 `H`: 硬體拓樸

`H` 是硬體的 graph：

```text
CPU cores
  |
L1 I/D cache
  |
L2 cache / system bus
  |
RoCC interface
  |
Gemmini controller
  |
Gemmini DMA <-> scratchpad <-> systolic array <-> accumulator
  |
DRAM
```

更細可以包含：

- Compute units: CPU scalar/vector unit、Gemmini systolic array、可能的 BA accelerator、可能的 correlation accelerator。
- Local memories: Gemmini scratchpad、accumulator、PE local registers。
- Shared memories: L1、L2、LLC、DRAM。
- Interconnect: RoCC interface、TileLink/system bus、DMA path、NoC/crossbar。
- Supported dataflow: output-stationary、weight-stationary、input-stationary 或 Gemmini 實際支援的 execution mode。
- Control structure: CPU 發 command、Gemmini queue、DMA mvin/mvout、sync mechanism。

### 3.2 `P_h`: 拓樸上的數值

`P_h` 是把 `H` 裡每個 node/edge 填上參數：

- Gemmini array dimension: `DIM x DIM`
- PE 數量與 MAC precision
- scratchpad capacity、bank count、ports、row width
- accumulator capacity、precision
- DMA bandwidth、burst size、request queue size
- L1/L2 cache size、line size、associativity
- DRAM bandwidth、latency、row buffer behavior
- frequency、voltage、technology node
- data type: int8/int16/fp16/fp32/mixed precision
- 是否支援 zero gating、sparsity skipping、compression
- CPU 數量與頻率

一個簡潔的理解是：

```text
H: 這台機器長什麼樣子。
P_h: 這台機器每個零件有多大、多快、多耗能。
```

---

## 4. Energy cost model 的正確分層

完整建模可以分成七層。每一層都對應到某些論文的貢獻。

### Layer 1: Workload model

輸入 `P_a`，輸出每個 module 的 workload shape 與資料統計。

例如 DPVO：

```text
P_a = {
  image_resolution,
  feature_channels,
  patch_count,
  patch_size,
  correlation_radius,
  active_frame_window,
  graph_edge_count,
  update_iterations,
  BA_iterations,
  precision
}
```

Workload model 要能回答：

- encoder conv 有多少 input/output feature map？
- patch extraction 會讀幾個 feature point？每個 point 要幾次 bilinear interpolation？
- correlation lookup 有多少 patch pair / edge？每個 edge 有多少 local candidates？
- update network / factor head 的 MLP 或 GRU tensor shape 是什麼？
- aggregation graph 有多少 edges？degree distribution 如何？
- BA 的 block sparse matrix size、nonzero block 數、iteration 次數是多少？

這一層的輸出不是 energy，而是 workload descriptor：

```text
W_k(P_a) = {
  operator_type,
  tensor_shapes,
  graph_shapes,
  sparsity/density,
  access_pattern,
  precision,
  batch_size,
  dependency
}
```

### Layer 2: Mapping / implementation model

輸入 workload descriptor、`H`、`P_h`，輸出每個 module 的 mapping。

例如：

```text
Feature encoder -> Gemmini conv/GEMM
Patch extraction -> CPU gather + interpolation
Correlation lookup -> either CPU irregular loop or batched Gemmini GEMM
Soft aggregation -> CPU sparse/graph loop
Factor head -> Gemmini if batch large enough, otherwise CPU
BA -> CPU sparse block solver or future BA accelerator
```

對 dense tensor module，mapping 包含：

- dataflow: output stationary / weight stationary / input stationary
- loop order
- tile size
- spatial unrolling
- temporal blocking
- tensor layout
- batch size
- padding or imperfect factorization strategy

對 irregular/sparse module，mapping 包含：

- sparse format: CSR/CSC/COO/block sparse/bitmask
- edge traversal order
- gather/scatter strategy
- whether atomics are used
- cache blocking
- whether to sort/reorder graph edges
- CPU/Gemmini/custom accelerator partition

這一層是 Timeloop、MAESTRO、ZigZag、Ruby、ScaleSim 強調的核心：mapping 會決定 access count、utilization、latency 和 energy。

### Layer 3: Action count model

這是 energy model 的主體。

Action count model 要輸出：

```text
N(k, c, a)
```

意思是第 `k` 個 module，在 component `c` 上，action `a` 發生幾次。

Component examples：

- CPU ALU / FPU / vector unit
- Gemmini MAC array
- PE local register / accumulator
- scratchpad
- L1 cache
- L2 cache
- DRAM
- DMA / TileLink bus
- NoC / interconnect
- control queue / command buffer

Action examples：

- MAC
- add / multiply / compare / branch
- register read/write
- scratchpad read/write
- accumulator read/write
- L1/L2 read/write hit/miss
- DRAM read/write
- DMA transfer
- NoC unicast/multicast/reduction
- random read/write
- repeated/sequential read/write
- metadata read/write
- zero-gated MAC
- skipped MAC
- idle cycle
- synchronization / atomic add

Accelergy 與 ScaleSimV3 特別提醒：action type 不能太粗。`read` 不是只有一種 read。Random read、repeated read、constant data write、gated MAC、zero-gated MAC、bypassed read 的 energy 可能差很多。

### Layer 4: Per-action energy model

這層提供每一種 action 的 unit energy：

```text
e(c, a | P_h, technology)
```

來源可以是：

- CACTI: SRAM/cache/DRAM-like memory access energy
- Accelergy: component/action energy table
- Aladdin / RTL / synthesis / post-layout numbers
- published pJ/access numbers
- FireSim/RTL activity trace + power model
- board measurement 校正後的估計

常見趨勢：

- DRAM access energy 遠高於 on-chip SRAM。
- 大 SRAM 比小 SRAM 每次 access 更貴。
- multiplier energy 通常隨 bitwidth 近似平方成長。
- adder/memory/wire energy 常與 bitwidth 更接近線性關係。
- random access 通常比 repeated/sequential access 更貴。
- idle/gated action 仍可能有 static/leakage，但 dynamic 較低。

### Layer 5: Timing model

Power 需要時間：

```text
P_dynamic = E_dynamic / T
```

所以除了 energy，也要估 runtime。

最簡化可以寫成：

```text
T_k = max(T_compute, T_spad, T_dma, T_l2, T_dram, T_sync) + T_overhead
```

如果沒有 overlap，則要改成 sum。如果有 double buffering 或 pipeline，才可以用 max 近似。

ScaleSim/ScaleSimV3 的提醒很重要：

- systolic array cycle 不等於 end-to-end latency。
- DRAM request queue、row buffer hit/miss、bank conflict、controller delay 都可能造成 stall。
- data layout 會造成 on-chip bank conflict。
- array 越大不一定越省 energy，因為 utilization 和 leakage/idle 可能變差。

### Layer 6: Power model

Dynamic power：

```text
P_dynamic = E_dynamic / T
```

Total power：

```text
P_total = (E_dynamic + E_static) / T
```

若你目前只想估 dynamic power，可以先報：

```text
E_dynamic_per_frame
T_per_frame
P_dynamic = E_dynamic_per_frame / T_per_frame
```

但建議同時保留：

```text
E_static = P_leak_active * T_active
         + P_leak_idle * T_idle
         + P_power_gated * T_power_gated
```

因為 ScaleSimV3 顯示 idle、clock gated、power gated state 對 energy/EDP 排序會有影響。

### Layer 7: Calibration / validation

Analytical model 一定要校正。

你可以分階段：

1. Paper-level analytical model: 先用公開 pJ/action、CACTI、Accelergy。
2. Gemmini simulator / Spike / verilator / trace: 取得 mvin/mvout、compute、flush、stall、cycle。
3. FireSim: 取得更接近 SoC 的 cache/memory contention、DMA、CPU interaction。
4. RTL/VCD/power estimation: 校正 unit energy。
5. Board measurement: 校正整體 power offset。

每一次校正都不是推翻模型，而是更新：

```text
N(c, a) 是否估錯？
e(c, a) 是否估錯？
T 是否估錯？
mapping 是否和實際 compiler/runtime 不一致？
```

---

## 5. 各篇論文放進模型的位置

| 論文                 | 對 energy model 的核心貢獻                                                                                         | 對 DPVO-on-Gemmini 的啟發                                                                                             |
| -------------------- | ------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| Energy-aware Pruning | `E = compute energy + data movement energy`，不能只看 weights/MACs。                                             | DPVO 調小 patch/feature/channel 不一定等比例省電，要看 feature map、graph edge、correlation data movement。           |
| Eyeriss              | Row-stationary dataflow 與 memory hierarchy energy model，將 DRAM/GLB/NoC/RF 都納入。                              | Gemmini 上 feature encoder/correlation 若能提高 reuse，才可能省 energy。                                              |
| EyerissV2            | Compact/sparse DNN shape 變化大，reuse 與 PE utilization 會下降，需要彈性 NoC/dataflow/sparsity。                  | DPVO 很多 module shape 小且 irregular，不能假設 Gemmini 永遠高 utilization。                                          |
| MAESTRO              | 用 data-centric directive 描述 dataflow，分析 reuse、NoC traffic、buffer access、latency/energy。                  | 建立 DPVO dense module 的 dataflow/action count，可先用類 MAESTRO 的 loop/reuse 分析。                                |
| ScaleSim             | Cycle-accurate systolic-array simulator，輸出 SRAM/DRAM trace、utilization、bandwidth。                            | Gemmini dense operator 可用類 ScaleSim 模型估 array utilization、scratchpad/DRAM traffic。                            |
| Timeloop             | Architecture + workload + mapping search，mapping 會讓同一個 workload energy 差很多。                              | `M_k` 必須納入模型，不能只固定一種 tiling 或只算理論 MAC。                                                          |
| Accelergy            | 用 component/action energy table，把 action counts 轉 energy。                                                     | 建 Gemmini/CPU/DRAM 的 energy reference table，尤其區分 random/repeated/gated/idle。                                  |
| ZigZag               | Memory-centric DSE，支援 operand-specific hierarchy、uneven mapping、bandwidth/stall 分析。                        | DPVO 各 operand 的 memory behavior 很不同，feature/correlation/edge metadata 不應混成單一 memory amount。             |
| SparseLoop           | Sparse acceleration features: representation、gating、skipping；稀疏效果取決於 density、format、dataflow。         | Soft aggregation、BA、graph edge traversal 要把 metadata、irregular access、skipping/gating 分開建模。                |
| Ruby                 | Perfect factorization 會因 tensor dimension 和硬體資源不整除而低 utilization；imperfect factorization 可改善 EDP。 | DPVO 的 small MLP、correlation、patch batch size 可能和 Gemmini`DIM` 對不齊，padding/tail/utilization 必須算。      |
| ScaleSimV3           | 加入 multi-core、sparsity、Ramulator DRAM、data layout、Accelergy power。                                          | SoC-level model 要包含 DRAM stalls、bank conflict、request queue、idle/static state，否則會錯判 dataflow/array size。 |

---

## 6. 對 DPVO 的 module-wise 建模

DPVO 不是單一 CNN。建議先拆 module，再為每個 module 選合適的建模方法。

| DPVO module                | Workload 型態                        | 可能的`P_a`                                   | Mapping 候選                     | Energy model 重點                                                         |
| -------------------------- | ------------------------------------ | ----------------------------------------------- | -------------------------------- | ------------------------------------------------------------------------- |
| Feature / context encoder  | Conv / dense tensor                  | image resolution、channels、precision           | Gemmini high feasibility         | MAC、scratchpad、accumulator、DMA、DRAM、array utilization                |
| Patch extraction           | bilinear sampling / gather           | patch count、patch size、feature dim            | CPU high, Gemmini low            | random/strided feature read、cache hit/miss、interpolation ALU、data copy |
| Correlation lookup         | local dot product + sampling         | edge count、radius、feature dim、patch count    | batched Gemmini GEMM or CPU loop | 若 batch 成 GEMM 算 dense action；若不行，算 random gather + small dot    |
| Temporal conv / transition | small dense / MLP / recurrent update | hidden dim、iteration count、batch size         | CPU or Gemmini                   | 小矩陣 offload overhead、Gemmini utilization、padding/tail                |
| Soft aggregation           | graph aggregation                    | node/edge count、degree distribution            | CPU or custom                    | edge metadata、irregular read、scatter/add、atomic/sync                   |
| Factor head                | MLP                                  | hidden dim、edge count、precision               | Gemmini if batch large enough    | MLP MAC + activation movement + utilization                               |
| Differentiable BA          | GN / Schur / sparse block ops        | active window、landmarks/patches、BA iterations | CPU baseline, custom candidate   | sparse block matrix access、factorization/solve、gather/scatter、sync     |
| Graph management           | dynamic control/data structure       | keyframe policy、edge update policy             | CPU                              | branch/control/cache miss/allocation overhead                             |

### 6.1 Feature/context encoder

這是最接近傳統 accelerator modeling 的部分。

對每個 conv/GEMM layer：

```text
MACs = H_o * W_o * C_out * C_in * K_h * K_w

E_layer =
  N_MAC * e_MAC
  + N_spad_read * e_spad_read
  + N_spad_write * e_spad_write
  + N_acc_read * e_acc_read
  + N_acc_write * e_acc_write
  + N_DMA_read_bytes * e_DMA_read_per_byte
  + N_DMA_write_bytes * e_DMA_write_per_byte
  + N_DRAM_read_bytes * e_DRAM_read_per_byte
  + N_DRAM_write_bytes * e_DRAM_write_per_byte
```

但 `N_spad_read`、`N_DRAM_read` 不是直接由 MACs 決定，而是由 mapping 決定：

- tile size 是否放得進 scratchpad？
- weights 能不能 reuse？
- ifmap 能不能 reuse？
- psum 留在 accumulator 還是反覆 spill？
- Gemmini dataflow 對該 layer 是否友善？
- channel 數是否和 `DIM` 對齊？
- batch size 是否足夠讓 array 滿載？

這裡可以用 Timeloop/MAESTRO/ScaleSim/ZigZag 的精神。

### 6.2 Patch extraction

Patch extraction 不是 Gemmini 典型強項，因為它像：

```text
for each patch p:
  for each sample point q in patch:
    read neighboring feature pixels
    bilinear interpolate
    write patch feature
```

這裡的 energy model 不該硬套 GEMM，而是：

```text
E_patch =
  N_feature_random_read_L1 * e_L1_random_read
  + N_feature_random_read_L2 * e_L2_random_read
  + N_feature_random_read_DRAM * e_DRAM_random_read
  + N_interp_muladd * e_CPU_ALU
  + N_patch_write * e_cache_or_DRAM_write
  + N_control * e_CPU_control
```

第一版可以先用 cache hit-rate 假設：

```text
N_L1 = total_feature_reads * hit_L1
N_L2 = total_feature_reads * (1 - hit_L1) * hit_L2
N_DRAM = total_feature_reads * (1 - hit_L1) * (1 - hit_L2)
```

之後再用 FireSim 或 CPU profiling 校正。

### 6.3 Correlation lookup

Correlation lookup 有兩種建模方式。

如果能 batch 成 GEMM：

```text
E_corr = E_Gemmini_GEMM(action counts from systolic mapping)
```

這時重點是：

- batch size 是否夠大？
- feature dim 是否對齊 Gemmini `DIM`？
- search radius 對 GEMM shape 的影響？
- feature/correlation tensor 能不能 reuse？

如果不能 batch，變成大量小 dot product + gather：

```text
N_dot_MAC = N_edges * (2r + 1)^2 * C
N_feature_reads = N_edges * (2r + 1)^2 * C * operands

E_corr =
  N_dot_MAC * e_CPU_or_Gemmini_MAC
  + N_feature_random_reads * e_memory_random_read
  + N_index_reads * e_metadata_read
  + N_control * e_control
```

這就是 EyerissV2、SparseLoop、ScaleSimV3 提醒的地方：小 shape、irregular access、metadata、stall 可能吞掉理論 MAC savings。

### 6.4 Update network / factor head

這些可能是 MLP/GRU/small dense layer。

關鍵不是它們能不能用 Gemmini 做矩陣乘法，而是：

```text
useful_MACs / executed_MACs
```

如果 hidden dimension、edge count 或 batch size 和 Gemmini tile 對不齊，會出現：

- array utilization 低
- padding MAC
- padding memory access
- mvin/mvout overhead 比 compute 大
- CPU/Gemmini synchronization overhead 大

Ruby 的觀念可以放在這裡：

```text
executed_M = ceil(M / DIM) * DIM
executed_N = ceil(N / DIM) * DIM
executed_K = ceil(K / DIM) * DIM

utilization = useful_MACs / (executed_M * executed_N * executed_K)
```

如果只用 perfect factorization 或只用簡單 padding，可能錯估 EDP。你可以把 `P_a` 中的 hidden dimension、feature dimension、patch batch size 設計成比較符合 Gemmini `DIM`，或讓 mapper 支援 tail/imperfect factorization。

### 6.5 Soft aggregation

Graph aggregation 的模型應該長得像 sparse/irregular workload：

```text
for each edge (i, j):
  read node/edge feature
  compute message
  scatter/add to node i or j
```

Energy model：

```text
E_agg =
  N_edge_metadata_read * e_metadata_read
  + N_node_feature_random_read * e_random_read
  + N_edge_feature_read * e_read
  + N_message_MAC_or_ALU * e_compute
  + N_scatter_write * e_random_write
  + N_atomic_add * e_atomic
  + N_sync * e_sync
```

SparseLoop 的精神是：sparsity 不只是少做幾個 MAC，還要付出 metadata、format、intersection、load imbalance、skipping/gating logic 的成本。

所以 soft aggregation 要記錄：

- graph density
- degree distribution
- edge order 是否 locality-friendly
- 是否有 atomic add
- 是否可以 reorder edge 來改善 cache locality
- 是否有 all-zero / inactive node 可以 skip

### 6.6 Differentiable BA

BA/Gauss-Newton/Schur complement 類 workload 通常是：

- block sparse Jacobian/Hessian
- irregular gather/scatter
- small dense block operation
- iterative solver
- synchronization-heavy

第一版不要試圖把它變成一個 Gemmini GEMM。比較好的方式是 block-level action model：

```text
E_BA =
  N_block_read * e_block_read
  + N_block_write * e_block_write
  + N_small_matmul * e_small_matmul
  + N_factorization_ops * e_compute
  + N_sparse_metadata_read * e_metadata_read
  + N_scatter_atomic * e_atomic
  + N_solver_iteration * e_iteration_overhead
```

這裡可以先用 CPU baseline 建模，之後如果發現 BA 是 bottleneck，再提出 BA accelerator 或把某些 block dense kernel offload 到 Gemmini。

---

## 7. 三個小例子

### Example A: 同樣 MAC 數，energy 可能不同

假設一個 convolution layer 有：

```text
MACs = 100M
```

簡化模型會寫：

```text
E = 100M * e_MAC
```

但比較合理的是：

```text
E =
  100M * e_MAC
  + IFMAP_access_spad * e_spad
  + Weight_access_spad * e_spad
  + Psum_access_acc * e_acc
  + IFMAP_access_DRAM * e_DRAM
  + Weight_access_DRAM * e_DRAM
  + OFMAP_write_DRAM * e_DRAM
```

如果 mapping A 讓 weights 留在 scratchpad 重複使用，mapping B 每個 tile 都從 DRAM 重讀 weights，兩者 MAC 一樣，但 energy 可能差很多。

這就是 Eyeriss、MAESTRO、Timeloop 的共同訊息。

### Example B: DPVO patch 數變少，不一定等比例省電

假設你把 patch count 從 `N_p` 降到 `0.5 N_p`。

直覺：

```text
correlation MAC 下降約 50%
```

但實際 energy 要看：

- feature encoder 是否仍要跑完整張影像？如果要，encoder energy 幾乎不變。
- patch extraction 的 feature read 是否更 locality-friendly？如果 patch 分布更分散，cache hit 可能下降。
- graph edge count 是否也下降？如果 active window 不變，某些 overhead 仍存在。
- CPU/Gemmini synchronization 次數是否不變？小 batch 反而降低 Gemmini utilization。

所以真正的模型應該拆成：

```text
E_total =
  E_encoder(almost same)
  + E_patch_extract(N_p, locality)
  + E_correlation(N_edges, radius, C, mapping)
  + E_update(N_edges, hidden_dim, utilization)
  + E_BA(window, iterations, sparsity)
```

`P_a` 的效果會透過不同 module 傳遞，不是單一比例縮放。

### Example C: channel dimension 和 Gemmini `DIM` 對不齊

假設 Gemmini 是 `16 x 16` array。

一個 MLP layer 的 hidden dimension 是 113。若用 padding：

```text
executed_dim = ceil(113 / 16) * 16 = 128
```

那麼某些 tile 會有：

```text
utilization ~= 113 / 128 = 88.3%
```

如果是 2D/3D GEMM tile，浪費可能在多個 dimension 相乘後更嚴重。

Ruby 的啟發是：不要只假設 perfect factorization，也不要只假設 padding 沒代價。你要把：

- padding compute
- padding memory access
- tail handling overhead
- lower utilization
- possibly better parallelism

全部納入 EDP trade-off。

對 DPVO 的 co-design 意義是：某些 `P_a` 例如 feature dimension、hidden dimension、patch batch size，可以刻意選成 Gemmini-friendly 的數值。但如果這樣會傷 accuracy，就要回到 accuracy/power Pareto front 評估。

---

## 8. 建立 DPVO-on-Gemmini energy model 的實作流程

建議分成四個版本做，不要一開始就追求完美。

### Version 0: First-order analytical model

目的：快速知道大概誰是 bottleneck。

需要：

- DPVO module list
- 每個 module 的 rough operations
- 每個 module 的 rough memory bytes
- 粗略 pJ/action table
- 粗略 runtime estimate

輸出：

```text
module, MACs, bytes, estimated energy, estimated time, dynamic power
```

缺點：

- cache hit/miss 可能很粗。
- Gemmini utilization 可能估錯。
- irregular access 可能估太樂觀。

### Version 1: Dense module 使用 systolic/dataflow model

對 encoder、factor head、update MLP、batched correlation：

- 建 workload shape。
- 建 Gemmini hardware config。
- 搜尋或指定 mapping/dataflow/tile。
- 估 scratchpad/accumulator/DRAM action count。
- 估 array utilization。

可以用：

- Timeloop + Accelergy
- MAESTRO-style analysis
- ScaleSim-style trace
- 自己寫簡化版 loop/tile analyzer

輸出：

```text
N_MAC
N_spad_read/write
N_acc_read/write
N_DMA_read/write
N_DRAM_read/write
utilization
cycles
```

### Version 2: Irregular/sparse module 建 action taxonomy

對 patch extraction、soft aggregation、BA、graph management：

不要硬套 dense accelerator model。先定義 action：

```text
random_feature_read
sequential_feature_read
edge_metadata_read
node_feature_read
scatter_write
atomic_add
small_block_matmul
branch/control
sync
cache_miss
```

每個 module 產生自己的 action count。

例如 soft aggregation：

```text
N_edge_metadata_read = number_of_edges
N_node_feature_read = number_of_edges * features_per_edge
N_scatter_write = number_of_edges
N_atomic_add = number_of_edges if parallel scatter
```

再用 cache hit-rate、degree locality、edge ordering 去修正 L1/L2/DRAM 分布。

### Version 3: FireSim / trace calibration

對候選 design point 跑 FireSim 或 Gemmini simulator，取得：

- real cycles
- Gemmini command count
- mvin/mvout bytes
- memory stall cycles
- cache miss rate
- DMA utilization
- CPU time
- accelerator idle time

用這些數據校正：

```text
estimated action counts -> measured/trace-derived action counts
estimated timing -> measured timing
unit energy table -> calibrated unit energy
```

---

## 9. 合理假設該怎麼做

你的問題裡特別問：「各個 memory hierarchy 的 memory access 取決於 scratchpad、L1、L2、DRAM 大小與拓樸，該做怎樣的假設才合理？」

建議用「可替換假設」而不是「單一假設」。

### 9.1 對 dense Gemmini module 的合理假設

第一版可以假設：

- operator 已經被 lowering 成 Gemmini-friendly GEMM/conv。
- Gemmini scratchpad/accumulator 是顯式管理，不自動 cache。
- tile 若放得進 scratchpad，tile 內 reuse 只在 scratchpad/accumulator 發生。
- tile 放不下就要多次 DMA 或 spill。
- L1/L2 主要影響 CPU 與 DMA path，不把它當成神奇的全自動 reuse 層。
- DRAM bytes 由 mvin/mvout 與 weight/input/output tile movement 決定。

更精細時加入：

- DMA burst alignment
- scratchpad bank conflict
- accumulator spill
- TileLink contention
- CPU/Gemmini synchronization
- DRAM row buffer / bank conflict

### 9.2 對 CPU irregular module 的合理假設

第一版可以假設：

- L1/L2 hit rate 用參數表示，而不是直接固定成 100%。
- random gather 的 L1 hit rate 低於 sequential scan。
- graph edge reorder 可以提高 locality，因此要有 ordered/unordered 兩種情境。
- atomic/sync 先用每次 operation 的平均成本建模。

例如：

```text
E_random_read =
  N_read * (
    h1 * e_L1_read
    + (1 - h1) * h2 * e_L2_read
    + (1 - h1) * (1 - h2) * e_DRAM_read
  )
```

這個寫法的好處是可以做 sensitivity analysis：

```text
h1 = 0.2, 0.5, 0.8
h2 = 0.5, 0.8, 0.95
```

你可以觀察 bottleneck 判斷是否穩定。

### 9.3 對 precision 的合理假設

Precision 不只改 compute energy，也改 memory energy：

```text
N_bytes = N_elements * bits_per_element / 8
```

一般趨勢：

- MAC energy: int8 < int16/fp16 < fp32
- memory bytes: bitwidth 線性影響 data movement
- accumulator 可能需要更高 precision，例如 int8 input/weight 但 int32 accumulation

所以 `P_a` 裡的 quantization choice 要同時影響：

- accuracy
- compute energy
- memory bytes
- accumulator storage
- Gemmini supported mode
- CPU fallback cost

### 9.4 對 static/leakage 的合理假設

如果只估 dynamic power，可以先不算 leakage。但若要比較不同 `P_h`，例如 `16x16` vs `32x32` array，建議至少保留：

```text
E_static = P_active_static * T_active + P_idle_static * T_idle
```

ScaleSimV3 的重點是：大 array 可能 latency 較低，但 utilization/idle/leakage 使 energy efficiency 變差。因此若要做 hardware DSE，static term 最好不要完全省略。

---

## 10. 改版 ISC / DSE flow

原本 ISC 的精神是 algorithm 和 hardware 分開迭代，找到 minimum-resource design。對 DPVO-on-Gemmini，需要把 energy cost model 放進中間，並加上 missing-information acquisition。

建議流程如下。

### Step 1: 定義 system target

例如：

```text
Throughput: 90 pose/sec
Power: 1-2 W
Accuracy: no error increment
Latency: 待定，需由 application pipeline 定義
```

### Step 2: 定義 `P_a` design space

列出 DPVO 可調參數：

- image resolution
- feature/context encoder width
- feature dimension
- patch count
- patch size
- correlation radius / level
- active window size
- graph edge policy
- update iteration count
- BA iteration count
- precision / quantization
- module approximation choice

每個 `P_a` 都要連到：

```text
accuracy impact
workload shape impact
memory/data movement impact
mapping feasibility impact
```

### Step 3: 定義 `H` 與 `P_h` design space

`H`:

- CPU + Gemmini baseline
- CPU + Gemmini + BA accelerator
- CPU + Gemmini + correlation accelerator
- CPU-only baseline

`P_h`:

- Gemmini DIM
- scratchpad size/banks
- accumulator size
- precision
- DMA bandwidth
- L2 size
- DRAM bandwidth
- frequency
- optional sparsity/gating support

### Step 4: 建立 mapping feasibility table

這一步回應你筆記中的 Q6。

對每個 module 問：

- Dense or sparse/irregular?
- Regular or irregular memory access?
- Static shape or dynamic graph?
- GEMM-friendly or many small kernels?
- Precision requirement?
- CPU control frequency?
- Input/output data movement 是否大於 compute benefit?

輸出：

```text
module -> CPU / Gemmini / custom / mixed
```

### Step 5: 建立 module-wise action count model

這一步是 `Energy cost model` 的核心。

對每個 module 產生：

```text
N_MAC
N_CPU_ALU
N_spad_read/write
N_acc_read/write
N_L1/L2/DRAM_read/write
N_DMA_bytes
N_metadata_read/write
N_random_access
N_atomic/sync
N_idle/gated/skipped
```

### Step 6: 建立 per-action energy table

用 Accelergy/CACTI/paper/synthesis/measurement 建表：

| Component    | Action          | Unit energy | Source                    |
| ------------ | --------------- | ----------: | ------------------------- |
| Gemmini MAC  | int8 MAC        |      TBD pJ | Accelergy/synthesis/paper |
| Scratchpad   | read/write      |      TBD pJ | CACTI/Accelergy           |
| Accumulator  | read/write      |      TBD pJ | CACTI/Accelergy           |
| L1/L2        | read/write hit  |      TBD pJ | CACTI                     |
| DRAM         | read/write byte |      TBD pJ | DRAM model/paper          |
| CPU          | ALU/FPU op      |      TBD pJ | paper/measurement         |
| Interconnect | byte transfer   |      TBD pJ | Accelergy/estimate        |
| Atomic/sync  | operation       |      TBD pJ | calibration               |

第一版不知道數字沒關係，但表格格式要先固定。

### Step 7: 搜尋 `P_a`，但用 energy model 而不是只用 ops

對每組 `P_a`：

```text
accuracy(P_a) >= baseline_accuracy
E_dynamic(P_a, H, P_h)
T(P_a, H, P_h)
P_dynamic = E_dynamic / T
```

得到 Pareto frontier：

```text
accuracy vs energy
accuracy vs latency
power vs throughput
```

這比只比較 operations 更接近你的目標。

### Step 8: 搜尋 `P_h` 與 mapping

固定候選 `P_a`，探索：

- Gemmini array size
- scratchpad/accumulator size
- dataflow/tile mapping
- CPU/Gemmini partition
- precision
- possible custom accelerator

目標不是最大 TOPS，而是：

```text
minimize power / EDP
subject to throughput and accuracy
```

### Step 9: Failure diagnosis

如果 design point 不滿足規格，不要只說「加大硬體」。要診斷是哪一種 failure：

| Failure type           | 現象                                   | 回饋方向                                                             |
| ---------------------- | -------------------------------------- | -------------------------------------------------------------------- |
| Compute-bound          | PE utilization 高，compute cycles 主導 | 增加 PE、降 precision、減 workload、增加 parallelism                 |
| Memory bandwidth-bound | DRAM/DMA stall 高                      | 改 dataflow、增 scratchpad、壓縮、重排 layout、減 data movement      |
| Low utilization        | array 很閒、batch 太小、tail 多        | 調`P_a` dimension、batch/fuse kernel、Ruby-style imperfect mapping |
| Irregular access-bound | cache miss、random DRAM、atomic 多     | reorder graph、blocking、custom gather/scatter、改資料結構           |
| CPU control-bound      | Gemmini 等 CPU，kernel 太碎            | fuse kernels、減 command overhead、改 runtime                        |
| Accuracy-bound         | 省電參數傷 accuracy                    | 回到`P_a`，找替代壓縮/quantization/approximation                   |

### Step 10: FireSim calibration and iterate

把候選 design point 放到 FireSim 或更低階 simulator：

- 驗證 cycles。
- 驗證 memory traffic。
- 驗證 CPU/Gemmini overlap。
- 驗證 cache/memory stall。
- 更新 energy model。

這可以成為你修改版 ISC 的新增步驟：

```text
Missing information acquisition:
  if action count unknown -> instrumentation/profiling
  if unit energy unknown -> Accelergy/CACTI/synthesis/measurement
  if timing unknown -> simulator/FireSim
  if accuracy impact unknown -> DPVO experiment
```

---

## 11. 最終你應該交付哪些表

如果要把這個模型變成研究中的可用工具，建議至少產生六張表。

### Table A: Algorithm parameter table

| Parameter     | Meaning            | Range | Accuracy impact | Workload impact                      |
| ------------- | ------------------ | ----: | --------------- | ------------------------------------ |
| `N_p`       | patch count        |   TBD | TBD             | patch/correlation/graph cost         |
| `r`         | correlation radius |   TBD | TBD             | local candidates grow as`(2r+1)^2` |
| `C`         | feature dimension  |   TBD | TBD             | MAC and memory grow with`C`        |
| `W`         | active window      |   TBD | TBD             | graph/BA cost                        |
| `I_update`  | update iterations  |   TBD | TBD             | recurrent update cost                |
| `precision` | int8/fp16/etc.     |   TBD | TBD             | compute and memory energy            |

### Table B: Hardware parameter table

| Parameter    | Meaning                 | Range |
| ------------ | ----------------------- | ----: |
| `DIM`      | Gemmini array dimension |   TBD |
| `SP_size`  | scratchpad size         |   TBD |
| `ACC_size` | accumulator size        |   TBD |
| `SP_banks` | scratchpad banks        |   TBD |
| `BW_dma`   | DMA bandwidth           |   TBD |
| `BW_dram`  | DRAM bandwidth          |   TBD |
| `freq`     | clock frequency         |   TBD |
| `dtype`    | supported precision     |   TBD |

### Table C: Mapping feasibility table

| Module            | CPU      | Gemmini     | Custom            | Main risk                |
| ----------------- | -------- | ----------- | ----------------- | ------------------------ |
| Feature encoder   | possible | good        | no need initially | memory traffic           |
| Patch extraction  | good     | poor        | maybe             | random access            |
| Correlation       | possible | conditional | maybe             | batching/locality        |
| Update/factor MLP | good     | conditional | no need initially | small kernel utilization |
| Aggregation       | good     | poor        | maybe             | sparse graph access      |
| BA                | baseline | poor        | candidate         | sparse block solver      |

### Table D: Action count table

| Module           | Component   | Action                  | Count expression                   |
| ---------------- | ----------- | ----------------------- | ---------------------------------- |
| Encoder          | Gemmini MAC | int8 MAC                | from conv shape and mapping        |
| Encoder          | Scratchpad  | read/write              | from tile analysis                 |
| Patch extraction | L1/L2/DRAM  | random read             | from patch count and hit-rate      |
| Correlation      | Gemmini/CPU | MAC                     | from edges/radius/C                |
| Aggregation      | CPU cache   | metadata read           | from edge count                    |
| BA               | CPU/cache   | block sparse read/write | from nonzero blocks and iterations |

### Table E: Energy reference table

| Component  | Action        | Energy | Confidence      |
| ---------- | ------------- | -----: | --------------- |
| MAC        | int8 MAC      |    TBD | low/medium/high |
| Scratchpad | random read   |    TBD | low/medium/high |
| Scratchpad | repeated read |    TBD | low/medium/high |
| L2         | read hit      |    TBD | low/medium/high |
| DRAM       | read byte     |    TBD | low/medium/high |
| DMA        | byte transfer |    TBD | low/medium/high |
| CPU        | ALU op        |    TBD | low/medium/high |

### Table F: Model vs measurement calibration table

| Metric         | Model | FireSim/profiling | Error | Fix                      |
| -------------- | ----: | ----------------: | ----: | ------------------------ |
| Encoder cycles |   TBD |               TBD |   TBD | mapping/bandwidth        |
| DMA bytes      |   TBD |               TBD |   TBD | tile model               |
| CPU cycles     |   TBD |               TBD |   TBD | instruction/cache model  |
| DRAM accesses  |   TBD |               TBD |   TBD | cache/memory model       |
| Dynamic energy |   TBD |               TBD |   TBD | unit energy/action count |

---

## 12. 常見錯誤

1. 用 FLOPs 當 energy。
   FLOPs 只算 compute，沒有算 memory hierarchy。
2. 只算 DRAM bytes。
   Timeloop 指出，即使 DRAM access 一樣，on-chip buffer/NoC access 也可能讓 energy 差很多。
3. 忽略 mapping。
   沒有 mapping，就沒有正確 action count。
4. 假設 Gemmini 永遠滿載。
   DPVO 有很多 small/irregular module，array utilization 可能很低。
5. 忽略 padding/tail。
   Ruby 顯示 tensor dimension 與 array size 不對齊會造成 utilization 和 memory access 問題。
6. 把 sparse 視為免費省電。
   SparseLoop 顯示 metadata、format、intersection、load imbalance、skipping hardware 都有成本。
7. 把所有 memory read 視為同一種。
   Accelergy 和 ScaleSimV3 都強調 random/repeated/idle/gated action energy 不同。
8. 忽略 DRAM stall 和 bank conflict。
   ScaleSimV3 顯示 DRAM modeling 可能改變 dataflow 選擇。
9. 忽略 CPU control 與 synchronization。
   DPVO deployment 不是純 accelerator kernel，CPU/Gemmini interaction 可能是瓶頸。
10. 沒有 sensitivity analysis。
    早期模型一定有不確定性，要知道結論是否對 unit energy/cache hit-rate/latency 假設敏感。

---

## 13. 一句話總結

你要建立的不是：

```text
Energy = MACs * pJ/MAC
```

而是：

```text
P_a -> workload shape/statistics
P_a + H + P_h -> mapping/dataflow/tiling/partition
mapping -> action counts at every compute and memory component
action counts * per-action energy -> dynamic energy
dynamic energy / execution time -> dynamic power
```

對 DPVO-on-Gemmini，最關鍵的研究價值在於：你要把 ISC 從「algorithm parameters 降 operations，再用 hardware 補 throughput」改成「algorithm parameters、mapping、memory traffic、irregularity、utilization、timing、energy」共同建模的 DSE flow。這樣才能回答真正的問題：

```text
在 accuracy 不下降的條件下，
哪一組 P_a、H、P_h、mapping，
能用最低 dynamic power 達到 90 pose/sec？
```
