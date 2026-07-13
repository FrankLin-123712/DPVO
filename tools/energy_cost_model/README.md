# DPVO-on-Gemmini Energy Cost Model

這是一個用於 algorithm DSE 的一階分析模型。它的主要目標不是預測未校準的絕對功耗，而是在固定硬體架構 `H` 與硬體參數 `P_h` 下，合理比較不同演算法參數 `P_a` 的每幀 dynamic energy：

```text
E_dynamic/frame = F(H, P_h, P_a)
```

模型先把 DPVO 各 module 轉成 workload，再依 CPU/Gemmini mapping 計算 MAC、SRAM/cache/DRAM、DMA、atomic 與同步等 action count，最後乘上 `pJ/action`。Energy 是主要 DSE 指標；latency 與 active power 只是輔助診斷。

更完整的建模取捨與本次修訂內容請見 [`Energy_cost_model_revision.md`](Energy_cost_model_revision.md)。

## 檔案

| 檔案 | 功能 |
| --- | --- |
| `parameters.py` | `AlgorithmParams`、`HardwareParams`、`MappingParams`、`EnergyTable`，以及 DPVO config/Gemmini header 讀取與輸入驗證 |
| `workloads.py` | 將 `P_a` 轉成 encoder、patch、correlation、update、factor head、BA、geometry 等 workload |
| `actions.py` | 將 dense/irregular workload 轉成 compute、memory、DMA、atomic、sync action counts 與估計 cycles |
| `model.py` | 套用 mapping，彙總 module energy、time、active power 與可選的 fixed-rate power |
| `cli.py` | 命令列介面與 markdown/json/csv 輸出 |
| `test_energy_cost_model.py` | 參數、工作量、memory hierarchy、precision 與 energy 趨勢回歸測試 |
| `Energy_cost_model_revision.md` | DSE 最小必要模型、原實作問題、實際修改與限制 |

## 模型涵蓋的 DPVO module

| Module | 預設 mapping | 主要 workload/action |
| --- | --- | --- |
| `feature_context_encoder` | Gemmini | fnet/inet convolution lowering 成 dense GEMM |
| `feature_pyramid_pooling` | CPU | correlation pyramid 的固定 pooling/feature traffic |
| `patch_extraction` | CPU | patch centroid、bilinear gather、metadata、非連續 feature read |
| `correlation_lookup` | CPU | local dot product、bilinear sampling、cache-line-amplified random read |
| `update_dense_linears` | Gemmini | corr/motion encoder、GRU 與 SoftAgg projection |
| `factor_heads` | Gemmini | delta/weight heads；可與 update block 分開 mapping |
| `soft_aggregation_scatter` | CPU | 依 unique patch 與 unique `(ii,jj)` group 計算 scatter softmax/sum |
| `update_geometry_elementwise` | CPU | reprojection、layer normalization、gating/residual、target 與 point-cloud elementwise work |
| `bundle_adjustment` | CPU | BA assembly、atomic accumulation、Schur complement |
| `graph_management` | CPU | factor append/remove 與 state bookkeeping |

`correlation_lookup` 目前只允許 CPU mapping。DPVO 的 correlation operands 是 edge/row-specific tensor，不能直接當成共享 `B` 的普通 batched GEMM；在沒有正確 block-batched mapping 前，CLI 不提供會低估資料搬移的 Gemmini 選項。

## 使用方式

從 repository root 執行：

```powershell
python impl/DPVO/tools/energy_cost_model/cli.py
```

預設 hardware profile 是此 workspace 的 `FP16DefaultConfig` 對應模型：`DIM=4`、2-byte input/weight、4-byte accumulator。若要比較 int8，必須同時明確指定演算法 tensor precision，避免把 fp16 workload 靜默套進 int8 accelerator：

```powershell
python impl/DPVO/tools/energy_cost_model/cli.py `
  --hardware-profile default-int8 `
  --nn-dtype-bytes 1
```

修改一組 `P_a`：

```powershell
python impl/DPVO/tools/energy_cost_model/cli.py `
  --patches-per-frame 64 `
  --patch-lifetime 10 `
  --removal-window 16 `
  --corr-radius 2
```

輸出包含 action counts 的 JSON：

```powershell
python impl/DPVO/tools/energy_cost_model/cli.py `
  --format json `
  --show-actions `
  --output impl/DPVO/tools/energy_cost_model/report.json
```

若有指定的輸入率，可另外查看 `E/frame * fps` 與即時可行性；這不會改變 energy ranking：

```powershell
python impl/DPVO/tools/energy_cost_model/cli.py --target-fps 30
```

## 主要 CLI 參數

| 類別 | 參數 |
| --- | --- |
| DPVO `P_a` | `--patches-per-frame`、`--patch-lifetime`、`--removal-window`、`--optimization-window`、`--corr-radius`、`--corr-levels`、`--update-iterations`、`--ba-iterations`、`--nn-dtype-bytes` |
| Graph size | `--edge-mode steady/new-frame`、`--edges`、`--unique-patches`、`--unique-frame-pairs` |
| Gemmini `P_h` | `--hardware-profile`、`--gemmini-dim`、`--input-bytes`、`--acc-bytes`、`--sp-capacity-kib`、`--acc-capacity-kib`、`--dma-buswidth-bits`、`--dma-maxbytes` |
| Memory | `--random-l1-hit-rate`、`--random-l2-hit-rate`、`--sequential-l1-hit-rate`、`--sequential-l2-hit-rate`、`--cache-line-bytes` |
| Mapping | `--encoder-mapping`、`--update-mapping`、`--factor-head-mapping`、`--dataflow WS/OS` |
| 輔助 timing | `--frequency`、各層 bandwidth、`--no-overlap-dma-compute`、`--target-fps` |

`--hardware-profile generated-header` 預設讀取此 workspace 的 `impl/gemmini/software/gemmini-rocc-tests/include/gemmini_params.h`，而且只會在檔案存在且能解析時使用；找不到 header 會直接報錯，不再靜默改用另一個硬體設定。
若使用 `--input-bytes` 或 `--acc-bytes`，CLI 會同步更新 precision label；底層也會拒絕 `fp16` 搭配 1-byte storage 之類不一致的硬體參數。

## Graph 規模

`new-frame` 模式估計新幀加入的 factors：

```text
E_new = PATCHES_PER_FRAME * (2 * PATCH_LIFETIME - 1)
```

`steady` 模式會按實際 source-frame age 截斷 lifetime，而非把所有 source frame 都假設為完整 lifetime。概念上：

```text
E_steady = PATCHES_PER_FRAME
           * sum over active source frames(valid target frames at that age)
```

如果 `PATCH_LIFETIME > REMOVAL_WINDOW + 1`，DPVO 新增 forward factors 時會短暫重引入較老的source patch；模型只為這些patch計入連到當前target的一條新edge，不會把先前已移除的factors加回來。

模型也分別追蹤 active edges、unique patches 與 unique frame pairs，因為 SoftAgg、BA 與 dense layers 不會全部以同一個 `E` 等比例成長。

## Dense Gemmini action model

- useful MAC 來自真實 tensor shape；executed MAC 包含 `DIM` padding。
- Scratchpad/accumulator access 以 tile 與 WS/OS reuse 計算，不再以每個 MAC 固定讀兩個 operand 的方式重複計費。
- 模型會快速搜尋可同時放入 SPAD/ACC 的 `M/N/K` block，再由 block 數計算 operand reload 與 off-chip bytes；不做逐 cycle 搜尋。
- DMA bytes 與 transaction count 分開估計，transaction 受 `dma_maxbytes` 影響。
- Array cycle 使用 padded `K` 加每個 output tile 一次 fill/drain，避免把 fill/drain 誤乘成 `K_tiles^2`。
- 當 dense op 的 utilization 低於 mapping threshold，模型可回退到 CPU，report 會顯示實際 mapping。

## CPU irregular memory model

CPU workload 會區分 sequential/random read、write、metadata、atomic、sync 與 precision-specific compute。Random access 以 cache line 為傳輸單位：

```text
transferred bytes = unique cache lines * cache_line_bytes
```

因此 patch/correlation 的稀疏 gather 不會只按「真正用到的 scalar bytes」低估 L1/L2/DRAM traffic。Hit rate 是條件式機率：L2 hit rate 只作用在 L1 miss 上，所有 rate 都必須落在 `[0,1]`。

## Energy table

可用 JSON 覆寫任意 action 的 `pJ/action`：

```json
{
  "gemmini.mac": 0.2,
  "cpu.mac.fp16": 1.0,
  "dram.random_read_byte": 140.0,
  "dram.write_byte": 120.0,
  "dma.transaction": 25.0
}
```

舊的 `"cpu.mac"` key 仍相容，會同時覆寫所有 CPU MAC precision；新模型建議使用 `cpu.mac.int8/int16/fp16/fp32/fp64`。

## 如何解讀輸出

| 欄位 | 意義 |
| --- | --- |
| `Useful MACs` | workload 真正需要的 MAC 數 |
| `Exec MACs` | 加入 Gemmini tile padding 後的 MAC 數；CPU module 通常等於 useful MAC |
| `Util.` | `Useful MACs / Exec MACs` |
| `DRAM R/W` | 經 locality、cache-line amplification 與 Gemmini tiling/reuse 後的 DRAM bytes |
| `Energy` | 該 module 的 dynamic energy；本模型的主要 DSE 指標 |
| `Time` | 一階 latency 診斷值，不是 cycle-accurate simulator 結果 |
| `Active Pdyn` | `total energy / estimated active time`；不同設計 runtime 改變時，不等同於固定 workload rate 的平均功耗 |
| `Target Pdyn` | 只有指定 `--target-fps` 時才有，等於 `E/frame * target_fps` |

對本階段的 energy/accuracy DSE，應優先比較 `Energy/frame`，而不是用 `E/T` 排名。

## 校準與限制

預設 `EnergyTable` 是 order-of-magnitude prior，未校準前只適合做相對比較，不能把輸出的 Joule/Watt 當成 silicon 絕對值。相對趨勢仍應至少用少量代表性設計點驗證 rank/order；建議依序校準：

1. 用軟體或 RTL counters 檢查 edges、MAC、DMA bytes、cache miss/DRAM bytes 與 atomic 數量。
2. 用 CACTI、Accelergy 或 synthesis/power data 更新 SRAM/cache/MAC/DRAM 的 action energy。
3. 用 FireSim/RTL/board measurements 檢查少量 `P_a` 的 energy ordering，而不必對所有設計做 cycle-accurate 模擬。

目前未建模且通常不影響固定 `H/P_h` 下第一階段 ranking 的細節包括 RTL toggle、NoC 每跳、bank conflict 每 cycle、DVFS transient、clock-tree 與精細 leakage/gating。若 DSE 同時改變硬體尺寸、電壓/頻率或 idle time，這些項目才需要升級進模型。
