# DPVO-on-Gemmini Energy Cost Model

這個資料夾實作一個 first-order analytical model：

```text
Energy = F(H, P_a, P_h)
P_dynamic = E_dynamic_per_frame / T_per_frame
```

它不是用 MAC 數直接估 power，而是把 DPVO 拆成 module，將每個 module 轉成 workload，再依照 CPU/Gemmini mapping 產生 action counts，最後乘上每種 action 的 pJ/action。

## 實作檔案

| 檔案 | 用途 |
| --- | --- |
| `parameters.py` | 定義 `AlgorithmParams`、`HardwareParams`、`MappingParams`、`EnergyTable`。可讀 DPVO config，也可從 `gemmini_params.h` 讀 Gemmini DIM、SPAD/ACC 容量與資料型別。 |
| `workloads.py` | 把 DPVO 的 `P_a` 轉成 module workload：encoder conv、patch extraction、correlation、update linears、soft aggregation、BA、graph management。 |
| `actions.py` | 估 Gemmini dense op 與 CPU irregular op 的 action counts、memory hierarchy access、cycles。 |
| `model.py` | 聚合 action counts，計算 energy、latency、dynamic power，並輸出 markdown/json/csv report。 |
| `cli.py` | 命令列入口。 |

## 模型分層

目前的實作遵循下面的資料流：

```text
P_a
  -> DPVO module workload shape
  -> mapping choice M_k
  -> action counts N(component, action)
  -> energy = sum(N * pJ/action)
  -> timing model T
  -> dynamic power = energy / T
```

DPVO module 拆分如下：

| Module | 預設 mapping | 建模方式 |
| --- | --- | --- |
| `feature_context_encoder` | Gemmini | `BasicEncoder4` fnet + inet conv lowering 成 GEMM。 |
| `patch_extraction` | CPU | bilinear gather / random feature read / interpolation / metadata。 |
| `correlation_lookup` | CPU | local dot product + bilinear sampling；也可用 `--corr-mapping gemmini` 測 batched GEMM 情境。 |
| `update_dense_linears` | Gemmini | Update block 裡 corr MLP、c1/c2、SoftAgg projection、GRU、delta/weight heads。 |
| `soft_aggregation_scatter` | CPU | `scatter_softmax`、`scatter_sum`、atomic/scatter、metadata。 |
| `bundle_adjustment` | CPU | BA Jacobian/Hessian assembly + Schur complement first-order model。 |
| `graph_management` | CPU | append/remove factors、keyframe/state bookkeeping。 |

## 預設參數

`P_a` 預設讀：

```text
DPVO/config/default.yaml
```

所以目前預設是：

```text
PATCHES_PER_FRAME = 96
REMOVAL_WINDOW = 22
OPTIMIZATION_WINDOW = 10
PATCH_LIFETIME = 13
MIXED_PRECISION = True
```

`P_h` 預設會讀目前 workspace 中產生的：

```text
chipyard/generators/gemmini/software/libgemmini/gemmini_params.h
```

這份 header 目前是 `DIM=8`、`elem_t=float`、`acc_t=float`、SPAD 約 `256 KiB`、ACC 約 `64 KiB`。若想看 Scala `GemminiConfigs.defaultConfig` 的 int8 16x16 設定，可以加：

```bash
python3 DPVO/tools/energy_cost_model/cli.py --hardware-profile default-int8
```

## 使用方式

預設報告：

```bash
python3 DPVO/tools/energy_cost_model/cli.py
```

用 int8 16x16 Gemmini profile，且只估新增 edge：

```bash
python3 DPVO/tools/energy_cost_model/cli.py \
  --hardware-profile default-int8 \
  --edge-mode new-frame
```

掃一組較省的 DPVO algorithm parameter：

```bash
python3 DPVO/tools/energy_cost_model/cli.py \
  --hardware-profile default-int8 \
  --patches-per-frame 64 \
  --patch-lifetime 10 \
  --removal-window 16 \
  --corr-radius 2
```

輸出 JSON 並包含 aggregate action counts：

```bash
python3 DPVO/tools/energy_cost_model/cli.py \
  --format json \
  --show-actions \
  --output DPVO/tools/energy_cost_model/report.json
```

替換 calibrated energy table：

```bash
python3 DPVO/tools/energy_cost_model/cli.py \
  --energy-table DPVO/tools/energy_cost_model/my_energy_table.json
```

`my_energy_table.json` 格式：

```json
{
  "gemmini.mac": 0.2,
  "dram.random_read_byte": 140.0,
  "dram.write_byte": 120.0
}
```

只需要放要 override 的 action，其他會沿用預設值。

## 重要 CLI 參數

| 類別 | 參數 |
| --- | --- |
| DPVO `P_a` | `--patches-per-frame`、`--patch-lifetime`、`--removal-window`、`--optimization-window`、`--corr-radius`、`--corr-levels`、`--update-iterations`、`--ba-iterations`、`--nn-dtype-bytes` |
| Graph size | `--edge-mode steady`、`--edge-mode new-frame`、`--edges`、`--unique-patches` |
| Gemmini `P_h` | `--hardware-profile`、`--gemmini-dim`、`--input-bytes`、`--acc-bytes`、`--sp-capacity-kib`、`--acc-capacity-kib`、`--dma-buswidth-bits` |
| Timing | `--frequency`、`--dram-bandwidth`、`--l2-bandwidth`、`--l1-bandwidth`、`--no-overlap-dma-compute` |
| Mapping | `--encoder-mapping`、`--update-mapping`、`--corr-mapping` |
| CPU memory locality | `--random-l1-hit-rate`、`--random-l2-hit-rate`、`--sequential-l1-hit-rate`、`--sequential-l2-hit-rate` |

## 輸出欄位

Markdown table 會列：

| 欄位 | 意義 |
| --- | --- |
| `Useful MACs` | DPVO workload 真正需要的 MAC。 |
| `Exec MACs` | Gemmini padding/tile 後實際執行的 MAC；CPU module 通常等於 useful MAC。 |
| `Util.` | `Useful MACs / Exec MACs`，用來觀察 channel/edge/batch 是否和 Gemmini `DIM` 對齊。 |
| `DRAM R/W` | 估計走到 DRAM 的 read/write bytes。CPU irregular module 會經過 L1/L2 hit-rate 分解。 |
| `Energy` | 該 module 的 dynamic energy。 |
| `Time` | first-order latency。Gemmini 預設用 `max(compute, DMA, DRAM)` 代表 overlap。 |
| `Pdyn` | 該 module 自己的 `energy / time`。Summary 的 dynamic power 則是整張 frame 的 `total energy / total time`。 |

## 目前假設

1. 報告是 per-frame estimate。每張新影像包含 encoder、patch extraction、factor graph update、correlation/update/BA。
2. `--edge-mode steady` 會用 active graph size：

   ```text
   E = PATCHES_PER_FRAME * REMOVAL_WINDOW * (2 * PATCH_LIFETIME - 1)
   ```

   `--edge-mode new-frame` 只算新加入 factors：

   ```text
   E = PATCHES_PER_FRAME * (2 * PATCH_LIFETIME - 1)
   ```

3. Dense Gemmini model 使用 square `DIM x DIM` systolic array，計入 padding MAC、scratchpad/accumulator access、DMA、DRAM traffic 與 RoCC command overhead。
4. CPU irregular model 使用可替換的 L1/L2 hit-rate，把 random/sequential read 與 write 分到 L1/L2/DRAM。
5. 預設 pJ/action 是 order-of-magnitude table，主要用來比較 design points；絕對功耗需要用 FireSim/RTL/CACTI/Accelergy 或板級量測校正。
6. Static/leakage 目前沒有加進 summary，因為問題定義要求 dynamic power。若要做 `16x16` vs `32x32` hardware DSE，建議後續加入 static/idle/gating state。

## 校正建議

後續可以把 trace 或工具數字逐步接進來：

1. Gemmini simulator / Spike / Verilator：校正 `mvin/mvout` bytes、Gemmini command count、cycle。
2. FireSim counters：校正 DMA stall、cache miss、DRAM contention、CPU/Gemmini sync。
3. CACTI / Accelergy：替換 `EnergyTable` 的 SRAM/cache/MAC pJ/action。
4. RTL/VCD power 或 board measurement：校正整體 scale factor 與 missing actions。

校正時建議不要直接改公式，而是先判斷是：

```text
action counts N(c,a) 錯
unit energy e(c,a) 錯
timing T 錯
mapping M_k 與實際 runtime 不一致
```

再針對對應層修正。
