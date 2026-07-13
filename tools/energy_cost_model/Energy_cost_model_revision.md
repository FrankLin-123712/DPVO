# Energy Cost Model Revision

## 0. 修訂目標與適用範圍

這份 energy cost model 的主要用途是 DPVO algorithm-parameter DSE：在固定硬體架構 `H`、固定硬體參數 `P_h`、固定 mapping/precision 的條件下，比較不同 `P_a` 的每張影像 dynamic energy，找出 accuracy 可接受且 energy 較低的候選點。

這個階段不要求預測的絕對 joule 或 watt 與晶片量測完全一致，但要求模型盡量保留 design point 的相對排序：

```text
如果真實硬體上 E(P_a1) < E(P_a2)，
模型也應盡可能預測 E_model(P_a1) < E_model(P_a2)。
```

絕對倍率不影響排序；不同 action 之間的相對權重會影響排序。因此模型仍需要合理的 compute、sequential data movement、random data movement、atomic/scatter 與固定成本比例，不能只用 MAC 數。

目前仍保留 latency 與 active dynamic power，供後續使用；但本階段主要 DSE metric 是 `energy_per_frame`。若未來要在固定輸入率 `f_target` 下比較平均 dynamic power，應使用：

```text
P_dynamic_at_target = E_dynamic_per_frame * f_target
```

並另外檢查硬體是否能達到該 throughput。`E/T_active` 表示硬體持續滿載時的 active dynamic power，不一定等同固定輸入率下的平均 dynamic power。

---

## 1. 為了正確估計 energy 趨勢，應該模擬哪些東西

### 1.1 相對模型的最小形式

修正版採用以下概念模型：

```text
E_dynamic(P_a) =
    E_fixed
  + alpha_mac     * N_executed_mac
  + alpha_offchip * B_offchip
  + alpha_local   * B_local
  + alpha_random  * N_or_bytes_of_random_cache_lines
  + alpha_atomic  * N_atomic
  + alpha_sync    * N_sync_or_launch
```

其中所有係數都可以是未校正的 effective energy。若將所有係數同時乘上相同常數，design-point 排序不變；但若 compute、DRAM、random access、atomic 之間的比例錯得太多，排序仍可能反轉。

一個很重要的判準是 action-count dominance：若 `P_a1` 在所有重要 action 上都不大於 `P_a2`，且至少一項更小，那麼只要 unit energy 都是正值，`P_a1` 一定比較省 energy，不需要知道精確 pJ。當某組參數減少 MAC、卻增加 random traffic 或 synchronization 時，才需要更好的係數與 sensitivity analysis。

### 1.2 必須模擬的內容

#### A. Fixed cost 與 variable cost

DPVO encoder 主要由影像解析度決定，不會隨 patch count、edge count 等比例下降。因此需要把每張影像固定執行的 encoder/pyramid成本，與 graph-dependent成本分開：

```text
E_frame =
    E_encoder_fixed
  + E_patch(N_p)
  + E_geometry(E)
  + E_correlation(E, r, P, C)
  + E_update(E, groups)
  + E_softagg(E, groups)
  + E_BA(E, W, K, iterations)
  + E_graph(E_new, E_active)
```

若缺少 fixed cost，模型會高估降低 patch count 的百分比收益。

#### B. DPVO graph/workload 尺度

下列中間變數直接控制大部分 variable energy：

```text
E_new       : 每張新影像加入的 factors
E_active    : update/correlation實際處理的 active factors
K_unique    : BA及 agg_kk 的 unique patches/groups
G_ij        : agg_ij 的 unique frame-pair groups
W_pose      : BA optimization poses
I_update    : update重複次數
I_BA        : BA iterations
```

`P_a` 必須先轉成這些 workload statistics，再轉成 action counts。若未來可以取得 runtime trace，應允許直接覆寫 `edges`、`unique_patches`、`unique_frame_pairs`，避免所有資料集與 keyframe行為都依賴單一封閉公式。

#### C. Executed MAC、padding與array utilization

Gemmini energy應使用 padding後的 executed MAC：

```text
M_pad = ceil(M / DIM) * DIM
N_pad = ceil(N / DIM) * DIM
K_pad = ceil(K / DIM) * DIM
N_executed_mac = M_pad * N_pad * K_pad
```

這能反映 batch/edge/channel dimensions 與 `DIM` 不對齊時的 tail成本。它不需要 cycle-accurate simulation，只需要每個 DenseOp做幾個整數運算。

#### D. Dense operand reuse與off-chip traffic

對 Gemmini dense op，至少需要知道：

```text
M/N/K tile count
WS或OS dataflow
operand tile bytes
SPAD是否能容納tile pair
ACC是否能容納output tile
完整A/B operand是否跨過可保留的SPAD budget
DMA read/write bytes與transaction count
```

模型不需要追蹤每個PE；但不能把每一個MAC都當成兩次scratchpad read，因為這會消除systolic forwarding/reuse的效果。若完整tensor放不進SPAD，也不應立即退化成「完整A對每個N tile全部重讀」；修正版會搜尋容量可容納的M/N/K block，以block數推導A/B reload。

#### E. Sequential與random memory access必須分開

CPU irregular module至少分成：

```text
sequential useful bytes
random useful bytes / estimated unique cache lines
metadata bytes
write bytes
```

random miss不應只把2-byte或4-byte scalar傳到DRAM；硬體通常以cache line傳輸。修正版用每個workload的 `random_line_utilization` 估計：

```text
unique_lines = ceil(useful_random_bytes /
                    (cache_line_bytes * line_utilization))
lower_level_bytes = unique_lines * cache_line_bytes * miss_rate
```

這不是完整cache simulator，而是一個可校正、能捕捉random-access amplification的模型。

#### F. Atomic/scatter與sync/offload

SoftAgg與BA不能只用MAC/bytes表示。至少保留：

```text
N_atomic
N_branch/control
N_sync
N_DMA_transaction
```

atomic的精確contention latency可以延後，但edge或window變化造成的atomic count變化必須保留。

#### G. Precision與mapping一致性

FP16 DPVO workload不能無提示地以int8 Gemmini unit energy與1-byte traffic計算。若要使用int8 Gemmini，quantization本身應是明確的 `P_a`/implementation choice，並經accuracy驗證。

### 1.3 視DSE範圍決定是否重要的細節

| 細節 | 何時重要 | 固定H/P_h、只掃P_a時是否必須 |
| --- | --- | --- |
| Gemmini DIM | padding/utilization可能改變 | 必須使用固定DIM計算padding |
| SPAD/ACC capacity | working set跨容量或tile放不下 | 需要capacity threshold，不需逐bank |
| WS/OS | operand與accumulator traffic不同 | mapping固定時用一種closed-form即可 |
| cache size | working set可能跨cache容量 | 可先以hit-rate/line-utilization做piecewise近似 |
| precision | compute與bytes都改變 | 必須一致 |
| offload/sync | small op或mapping可能改變 | 必須計固定次數/成本 |
| static/leakage | 比較不同硬體、頻率或duty cycle | 本階段固定硬體比較energy可延後 |

### 1.4 對趨勢影響較小、可以延後的細節

在固定硬體、固定mapping、固定precision，且design points未跨越特殊capacity/bottleneck threshold時，下列細節可以先不模擬：

- 每一個PE register read/write的精確次數。
- scratchpad bank conflict與port arbitration。
- DRAM row-buffer、bank mapping、request queue與controller cycle。
- TLB/page-walk的精確行為。
- RoCC command pipeline的逐cycle狀態。
- L1/L2各自的絕對pJ是否完全準確。
- voltage/frequency與technology-node的精確模型。
- 絕對正確的joule/watt倍率。

這些項目可以在相對排序無法通過驗證、或開始進行hardware DSE時再加入。

---

## 2. 依據上述原則，原energy cost model需要修改的地方

### 2.1 硬體profile與precision

原模型在generated header不存在時，無提示地退回16x16 int8；README卻描述另一個FP設定。這會讓報告在語法上成功，但實際建模的 `P_h` 不同。

原模型也以 `hardware.input_bytes` 計算所有Gemmini operand traffic，忽略DenseOp自身dtype；CPU FP16 correlation與FP32 BA則共用由Gemmini precision決定的 `cpu.mac` energy。

需要：

- generated header不存在時fail fast。
- 提供明確FP16/FP32/int8 profile。
- Gemmini operand width與workload width不符時拒絕隱式offload。
- CPU MAC依precision分action。

### 2.2 Gemmini scratchpad與timing action count

原本：

```text
spad_read_bytes = executed_macs * 2 * input_bytes
```

這會將array內reuse重新計成SRAM traffic。原本fill/drain又把全域 `k_tiles` 放進每個tile product，形成不合理的二次成長。

需要改為tile-level operand injection與每個M/N output tile只支付一次fill/drain。

### 2.3 CPU irregular memory hierarchy

原模型將useful bytes乘hit/miss rate，沒有cache-line amplification；random gather讀取少量scalar時會低估lower-level traffic。需要加入cache-line size與workload-specific line utilization。

### 2.4 DPVO graph與SoftAgg scaling

原steady edge公式把所有active patches都當作已累積完整 `2r-1` edges，忽略最近source frames尚未取得未來edges。原SoftAgg把f/g/h都以active edges為sample數，但實際h作用在unique-group tensor。

需要：

- 依source age累加edges。
- 分別建模active edges、unique patches、unique frame pairs。
- SoftAgg f/g乘edges，h乘unique groups。

### 2.5 遺漏但會隨P_a變化的workload

原模型遺漏reprojection、LayerNorm/gating/residual、feature-pyramid pooling、target/weight elementwise及point-cloud update。這些不需要逐operator建模，但至少應以aggregate CPU workload加入，否則patch/edge/window的相對收益可能被高估。

### 2.6 BA action count

原模型固定使用96 atomics/edge/iteration，與kernel中的B/E/v/scalar atomicAdd結構不一致；並把同一個 `ba_macs_per_edge` 同時計成MAC與ALU。

需要改成依optimization-window active-pose fraction估算atomic count，並將 `ba_macs_per_edge` 視為單一effective compute count。

### 2.7 Correlation Gemmini mapping

原本把每個edge/patch sample視為共享同一個B matrix的一般GEMM，但實際上每一row對應不同candidate feature tensor。這個mapping若沒有block-batched/operand-staging專用模型，會得到錯誤reuse與traffic。

因此修正版暫時關閉CLI的Gemmini correlation mapping，保留CPU model。未來需先定義實際batched layout、staging及Gemmini kernel，再重新啟用。

### 2.8 輸入驗證與測試

原CLI允許hit rate大於1並產生負DRAM bytes，也沒有任何測試。相對模型需要基本invariants與monotonic trend tests，避免可執行但語意錯誤。

---

## 3. 本次實際修改內容與原因

### 3.1 `parameters.py`

- 新增所有正數、dtype、precision/storage-width一致性、edge mode、unique-group/edge關係、hit-rate範圍與unit-energy非負有限值驗證。
- 新增 `unique_frame_pairs` 與 `active_unique_frame_pairs`。
- steady edges改成依source age累加；模型對應DPVO在當前frame update後才移除expired factors，因此一般保留age `0..REMOVAL_WINDOW` 的source。若 `PATCH_LIFETIME` 更長，`__edges_forw()` 暫時重引入的更老patch只計當前target的一條edge，而不恢復已移除的舊factors。
- `active_unique_patches` 改為依edge mode與active source frames估計。
- feature height/width改為ceil-div，對非4倍數影像保持與stride convolution一致。
- mapping dataflow明確限制為WS或OS，預設WS；尚未實作的Gemmini mapping會在參數層直接拒絕，不會把請求靜默當成CPU執行。
- 新增cache-line size。
- generated header缺失時不再silent fallback。
- generated-header預設路徑改成此workspace實際的Gemmini輸出位置：`impl/gemmini/software/gemmini-rocc-tests/include/gemmini_params.h`。
- 新增明確 `fp16-default`；FP32 profile依目前Gemmini source改成4x4。
- CPU MAC unit energy拆成int8/int16/fp16/fp32/fp64。
- 保留舊JSON `cpu.mac` override的相容處理。
- 新增DMA transaction action。

### 3.2 `workloads.py`

- 每個irregular workload新增precision與random-line utilization。
- correlation CPU model保留local dot-grid `(2r+2)^2` 與bilinear output `(2r+1)^2`。
- SoftAgg f/g使用active edges，h使用unique kk/ij groups。
- delta/weight heads拆成獨立 `factor_heads` module，使factor-head mapping不再是dead parameter。
- BA atomic count改為依optimization-window active-pose fraction估計；fraction=1時對應最多342 atomicAdds/edge/iteration。
- BA metadata不再在一般read bytes中重複計算。
- `ba_macs_per_edge` 與Schur compute不再重複計成MAC與ALU。
- 新增 `feature_pyramid_pooling` fixed module。
- 新增 `update_geometry_elementwise`，以aggregate workload表示reprojection、LayerNorm/gating/residual、target/weight及point-cloud update。
- `update_iterations`會同步scale correlation、geometry、update dense、factor heads、SoftAgg與BA。

### 3.3 `actions.py`

- Gemmini offload會驗證DenseOp operand width與hardware input width一致。
- SPAD/ACC至少必須容納一組operand/output tile，否則明確報錯。
- Gemmini SPAD read改為tile-level operand injection，不再依每個MAC乘兩次。
- WS會讓B tile跨M tiles重用；OS會stream A/B並讓output partial sum留在array。
- 新增capacity-constrained M/N/K block搜尋：output block必須放入ACC、A/B panel必須放入SPAD，再以block數計算A/B reload；避免完整tensor一跨過容量就退化成逐tile全量重讀。
- accumulator traffic依WS/OS分開。
- DMA transaction count使用 `dma_maxbytes`。
- systolic cycles改為每個M/N output tile計算 `K_pad + fill/drain`，移除K_tiles二次項。
- 每個DenseOp加入一次CPU/Gemmini sync timing。
- CPU MAC action依workload precision選擇。
- random read的L2/DRAM traffic加入cache-line amplification與line utilization。

### 3.4 `model.py`

- report同時保留active dynamic power與optional target-rate dynamic power。
- `target_dynamic_power = energy_per_frame * target_fps`。
- 新增target throughput feasibility欄位，但不影響本階段energy-first DSE。
- Gemmini op低於minimum utilization時可自動留在CPU，mapping會顯示 `cpu-auto`。
- correlation Gemmini mapping在沒有正確batched model前明確拒絕。
- 加入feature pyramid、geometry與factor-head mapping。

### 3.5 `cli.py`

- 預設改成明確的 `fp16-default`，與DPVO mixed-precision預設operand width一致。
- 新增 `--unique-frame-pairs`、`--cache-line-bytes`、`--dataflow`、`--factor-head-mapping`與optional `--target-fps`。
- generated-header profile在檔案不存在時會失敗，不再無提示換硬體。
- 暫時只允許CPU correlation mapping。

### 3.6 測試

新增 `test_energy_cost_model.py`，目前涵蓋：

- steady/new-frame edge、短removal-window重引入行為與unique group統計。
- SoftAgg h使用unique groups。
- hit-rate範圍驗證。
- generated header缺失不silent fallback。
- 尚未實作的accelerator mapping會被拒絕。
- 負值或非有限unit energy會被拒絕。
- SPAD read為tile-level而非MAC-level。
- SPAD/ACC capacity跨threshold會改變block shape與reload traffic。
- capacity blocking不會退化成完整A對每個N tile重讀。
- Gemmini precision mismatch拒絕隱式offload。
- random cache-line amplification。
- 固定硬體下patch數下降會降低energy。
- target-rate power等於energy/frame乘input rate。

測試結果：

```text
Ran 17 tests
OK
```

也執行了基本趨勢掃描；在固定FP16 Gemmini profile下得到單調結果：

| Sweep | Values | Energy trend |
| --- | --- | --- |
| patches/frame | 48, 64, 96, 128 | 1.57, 2.05, 3.02, 3.99 J |
| correlation radius | 2, 3, 4 | 2.15, 3.02, 4.14 J |
| removal window | 14, 22, 30 | 1.85, 3.02, 4.19 J |

這些joule仍是未校正的absolute estimate；表格只用來確認主要P_a方向的action scaling與energy排序符合預期。

---

## 4. 修正版仍然保留的限制

1. Unit energy仍是order-of-magnitude effective coefficients，沒有經過目前目標製程的CACTI/Accelergy/RTL校正。
2. CPU hit rate與random-line utilization仍是參數化假設；資料layout或graph reorder可能改變它們。
3. steady graph公式未包含實際資料序列的motion rejection、keyframe deletion及loop closure；應使用trace override驗證。
4. Gemmini dense tile model是快速的capacity-constrained block搜尋加closed-form WS/OS近似，未模擬bank conflict、queue與TileLink contention。
5. point-cloud與elementwise operation使用aggregate operation count，需要由profiling校正。
6. absolute active power與target-rate power在未校正unit energy前不應作為晶片規格數字。

因此修正版的定位是：比原版更適合進行快速 `P_a` DSE與相對energy排序，但仍需用少量代表design points做ranking validation。建議後續以FireSim/Gemmini counters或RTL/board measurement驗證pairwise ranking及Spearman/Kendall rank correlation，而不是只驗證absolute watt誤差。
