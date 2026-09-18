# Update iteration replay 與 profiling

`generate_update_replay_testdata.py` 在真正的 Python DPVO inference 中擷取輸入、
中間值及輸出；`dpvo_runner/run_update` 使用同一組資料重播：

- `--scope iteration`：完整 `DPVOTracker::Update()`，包含 reproject、correlation、
  context gather、update network、target/weight 後處理、local BA、graph 與 point cloud 回寫。
- `--scope network`：直接呼叫 `OrtUpdateNetwork::Run()`，使用擷取的 net/ctx/corr，
  隔離上游計算與 BA，適合深入分析 ONNX nodes 和 Gemmini GEMM/MatMul。

兩種 scope 都使用既有 C++ 實作。完整 iteration 的中間結果由 C++ 計算並接續傳入下一階段，
不以 Python golden 替換。每次 warmup、repeat、診斷 replay 都從原始 snapshot 還原。

## 1. 在 GPU 主機產生 EUROC testdata

需要可正常跑此 DPVO repository 的 PyTorch/CUDA、CUDA extensions、NumPy、OpenCV，
使用 YAML 時也需要 PyYAML。這台 server 沒有 GPU；正式 inference 與 golden 生成請在 GPU 主機執行。

```bash
cd /path/to/DPVO
python tools/gen_testdata/generate_update_replay_testdata.py \
  --weights dpvo.pth \
  --images datasets/EUROC/MH_01_easy/mav0/cam0/data \
  --calib calib/euroc.txt \
  --frame-start 1 --frame-count 16 --frame-step 1 \
  --width 752 --height 480 \
  --patches-per-frame 16 --centroid-sel-strat RANDOM \
  --no-mixed-precision --no-undistort --seed 7 \
  --output-root testdata/update_replay_euroc_mh01_first16_p16
```

預設為 MH_01 前 16 幀、P16、seed 7、不去畸變、包含 `terminate()` 的最後 12 次 updates。
`--frame-start` 是 1-based；metadata 的 `input_frame_index` 是 0-based。
如需對照舊 workload，請使用相同 checkpoint、影像範圍、前處理及 tracker config，
可傳 `--config-yaml FILE`。CLI/YAML 的優先順序沿用既有 generator；最後強制 FP32、關閉 TF32。
不接受以 FP16 推論結果轉存 FP32 充當 FP32 golden。

每個真正的 `tracker.update()` 產生一個 case，stage 為 `initialization`、`update` 或 `terminate`。
case 數與 edge 數由實際 inference 決定，不預設一定是 31 次。

- `--include-motion-probes`：額外擷取 `motion_probe` 的 network inputs/outputs；只能做 network replay。
- `--skip-terminate-updates`：省略最後的 updates，會改變工作負載組成。
- `--ba-iterations N`：沿用 tracker 的 local BA iteration 設定。

原始呼叫只執行一次，擷取只觀察原結果。輸入在原 forward 前複製至 owned CPU storage，
BA 前的 poses/patches 與 BA 後結果分開保存。例外時還原所有包裝的方法。
Python 若走 global BA 或吞下 BA 例外，case 的 `iteration_supported=false`；
C++ 完整 replay 會明確拒絕，network scope 仍可使用該 case。

output root 必須不存在或為空。只有完整寫入的 case 會加入 `cases.txt`；
整次生成成功後 root `status=complete`，失敗時標記 `failed`，不可當作完整資料集使用。

## 2. 匯出對應 update 模型

必須使用同一份 checkpoint 與 repository exporter：

```bash
python tools/export2onnx/export_models.py \
  --weights dpvo.pth --out exported_models/update_replay \
  --opset 11 --skip-feature
```

目前支援的 exporter 使用 explicit `ix/jx` neighbors，runner 會在計時範圍內從 `kk/jj` 計算。
`E` 是 dynamic edge dimension，`--edges` 只是 export trace 的範例尺寸。
runner 沿用嚴格 ONNX input schema，支援 `ctx/inp` 名稱與既有 `kk`／`ix,jx` 路徑。
inference 的 Python `flow=None`；模型若保留 flow input，C++ 使用零 tensor。

將整個 testdata root 與模型搬到 server，例如：

```bash
rsync -aH testdata/update_replay_euroc_mh01_first16_p16/ \
  SERVER:/home/cclin/chipyard/generators/gemmini/software/onnxruntime-riscv/systolic_runner/dpvo_runner/testdata/update_replay_euroc_mh01_first16_p16/
```

gmap/fmap 以內容 hash 去重，case 下仍有完整 `.bin` 入口；`rsync -aH` 或 `tar` 可保留 hard links。
一般複製也可讀取，但可能膨脹成每次 update 都複製全部 feature maps。
資料量取決於 edge 數、active frames 與 slot 更新；生成時只保留當次 case 的 tensor snapshots。

## 3. 建置與 Spike 驗證

新 kernel hooks 位於 ORT MLAS；只重新編譯 runner 不會更新已存在的 ORT archive。
在已設定好的 ORT build directory 上先增量重建，再連結 runner：

```bash
cd /home/cclin/chipyard/generators/gemmini/software/onnxruntime-riscv
/home/cclin/chipyard/.conda-env/bin/cmake --build build/Release \
  --target onnxruntime_mlas onnxruntime_framework --parallel 4
cd systolic_runner/dpvo_runner
./build.sh --config=Release --update-benchmark -O2 --parallel=4
```

`Release` 本身不代表 runner 使用 `-O2`，請明確指定。保留 ORT 本來的 build 設定；
若還沒有 ORT build products，先依現有 ORT build 流程設定並建置。
`build.sh` 會產生 `run_update.build.json`，記錄 runner flags 與 ORT CMake 設定。

先挑一個 case 驗證 network，再驗證完整 iteration：

```bash
./run_update_spike.sh \
  --update_model /path/to/update_block.onnx \
  --case_root testdata/update_replay_euroc_mh01_first16_p16 \
  --scope network --max_cases 1 --warmup 0 --cache_policy natural \
  --profile_mode phases --profile_csv out/update_network_spike.csv -x 1

./run_update_spike.sh \
  --update_model /path/to/update_block.onnx \
  --case_root testdata/update_replay_euroc_mh01_first16_p16 \
  --scope iteration --max_cases 1 --warmup 0 --cache_policy natural \
  --profile_mode phases --profile_csv out/update_iteration_spike.csv -x 1
```

`-x 0/1/2` 分別為 CPU、Gemmini OS、Gemmini WS；wrapper 預設 1，直接執行 binary 預設 0。
可用 `SPIKE`、`PK` 指定執行工具，用 `--binary` 指定 binary。
Spike 用來檢查功能／數值；cycle 與 array activity 的效能結論請使用 FireSim。

case 選擇支援 `--case_root`、`--case_dir`、`--cases_file` 三擇一。
`--cases_file` 每行一個 case path，相對於該 list 所在目錄，旁邊需有完整 root metadata。
`--case_dir` 也須保留上層 root metadata。可用 `--stage initialization|update|terminate|motion_probe|all` 篩選，
再套用 `--skip_cases N`、`--max_cases N`。iteration scope 自動排除 motion probes。

## 4. FireSim profiling

建置可供既有 FireSim Linux workload 使用的 binary：

```bash
./build.sh --config=Release --update-benchmark -O2 --for_firesim --parallel=4
```

把 binary、對應 `.build.json`、ONNX 與選定 testdata 打包進既有 workload；下列為 guest 內命令，
路徑請對應實際掛載位置。本次提供 standalone runner，不自動修改或啟動 FireSim workload。

```bash
# 低干擾基準：完整 iteration
./run_update --update_model /root/dpvo/update_block.onnx \
  --case_root /root/dpvo/update_replay_euroc_mh01_first16_p16 \
  --scope iteration --profile_mode total --warmup 1 --repeat 3 \
  --counter_group none --profile_csv /root/dpvo/out/update_total.csv -x 1

# iteration phases
./run_update --update_model /root/dpvo/update_block.onnx \
  --case_root /root/dpvo/update_replay_euroc_mh01_first16_p16 \
  --scope iteration --profile_mode phases --warmup 1 --repeat 3 \
  --counter_group none --profile_csv /root/dpvo/out/update_phases.csv -x 1

# network nodes、Gemmini kernels、kernel hardware counters
./run_update --update_model /root/dpvo/update_block.onnx \
  --case_root /root/dpvo/update_replay_euroc_mh01_first16_p16 \
  --scope network --profile_mode nodes --warmup 1 --repeat 3 \
  --counter_group topdown --counter_scope kernel \
  --profile_csv /root/dpvo/out/update_nodes_topdown.csv -x 1
```

保持模型、case selection、execution、ORT `-O`、warmup、repeat 相同，分次換成
`--counter_group memory`、`dma`、`execute`。`im2col` 也保留，但 update 的線性層以 GEMM/MatMul 為主。
`--counter_scope total|node|kernel` 選擇硬體 counter window；node/kernel 需要 `--profile_mode nodes`。
同次執行只有一種 window 層級，避免巢狀 counter reset 破壞父層數值。

`--profile_mode nodes` 沒有 ORT node 事件時會報錯；kernel counter 模式沒有 kernel counter 事件時也會報錯，
請確認 instrumented ORT 已重建、runner 已重新連結，以及模型有執行加速路徑。
預設 trace capacity 為 16384，可用 `--profile_trace_capacity N` 增加；截斷或 counter 讀取失敗不會標記成功。

計時排除檔案讀取、session 建立、snapshot restore、golden validation 與報表 I/O。
`Update()` 本身的 copy/allocation、neighbors、ORT output copy 都保留在計時內。
iteration 中間值由額外一次 **untimed diagnostic replay** 擷取，避免 tensor 複製污染量測；
net、target、graph weight、BA 後狀態則取自實際 measured replay，計時結束後才複製。

`warm` 每次 repeat 前都做指定次數的 warmup；restore 仍可能影響 cache。
`natural --warmup 0` 表示不強制預熱，並不保證 cold cache，也不等同完整 tracker 的 cache 歷史。
細粒度 instrumentation、counter reads 與同步 fence 會增加開銷；以 total/none 作效能基準，
phases/nodes 用於定位，再比較不同模式的 total 差距。

## 5. 輸出與判讀

指定 `--profile_csv FILE` 後產生：

- `FILE`：event、counter、aggregate rows。包含 parent/event ID、stage、edge/group 數、
  operator、provider、module、inclusive/exclusive cycles、wall time、counter window 與 kernel detail。
- `FILE.validation.csv`：每 case、repeat、tensor 的 PASS/FAIL、mismatch count、max abs/relative error、worst index。
- `FILE.metadata.json`：模型路徑、checkpoint 路徑、建置資訊、cases、設定、trace 狀態與最終結果。

`aggregate` 依來源 stage、E、unique kk/ij 數與事件分組，避免不同 shape 混成一個平均值。
network module 分組辨識 `corr/norm/c1/c2/agg_kk/agg_ij/gru/d/w`；經最佳化後無法辨識的名稱為 `unassigned`，
仍保留原始 node 名稱。kernel `detail` 包含 M/N/K、execution、DIM、DIM 對齊尺寸、transpose 與邏輯 FLOPs。
`tiling=auto` 表示使用現有自動 tile 選擇，不宣稱已量到實際 PE occupancy。

分析建議：

1. 用 iteration exclusive phases 判斷時間是否落在 correlation、context gather、BA、copy，或 network。
2. 用 node provider 和 operator 的 exclusive cycles，找 CPU fallback、gather/scatter、normalization、activation 等成本。
3. 對 GEMM/MatMul 比較 M/N/K、padding、kernel cycles 與 execute/load/store/DMA/wait counters。
4. 比較相同 cases 的 OS/WS、不同 E 與 group 數，區分固定開銷、小矩陣與記憶體等待。

`update_iteration` 為 inclusive 父層，不能與子層比例相加。`corr` module 是 network 裡的 correlation encoder，
與外層 `correlation` volume construction 不同。硬體各個 active/wait counter 可能重疊，不應加成 100%。
`EXE_ACTIVE_CYCLE` 是 execute controller 的 compute 狀態時間，**不是所有 PE 有效 MAC 的比例**。
64-bit counter 使用 window 差值；窄 counter 標記 `modulo_32_unverified`，不輸出可信的比率，
無法由兩個 snapshots 證明是否多次溢位。host 測試的 cycles 不可用時維持 0，使用 wall_ns。

預設 network／前段 tolerance 為 `atol=5e-4, rtol=1e-4`；post-BA poses/patches/points 獨立使用
`--ba_atol 1e-3 --ba_rtol 1e-3`。這是起始閾值，不代表已經過本次 GPU 資料驗證。
poses 在比對前對齊 quaternion 的正負號；不做軌跡對齊或縮放來掩蓋差異。
任一 tensor 失敗會回傳非零，保留 profiling 與 validation 結果；先查第一個不一致的中間值，
不要只放寬 BA tolerance。NaN/Inf、shape、dtype、byte count 與無效 index 都會拒絕。

## 格式 `dpvo_update_replay_v1`

root 有 `metadata.json`、`cases.txt`、`features/`、前處理影像 `images/` 與 `calib.txt`。
每個 `case_NNNN` 有 metadata、manifest 與同名 binary tensors。
所有資料 little-endian、C contiguous；浮點為 FP32、索引為 int64。
manifest 沿用共用 writer 的註解，格式版本以 metadata 為準。

令 N=active frames、M=patches per frame、E=edges、Hf=H/4、Wf=W/4：

| tensor | shape | 用途 |
|---|---|---|
| net / ctx / corr | `[1,E,384]` / `[1,E,384]` / `[1,E,882]` | network 真正輸入 |
| ii / jj / kk | `[E]` | source frame / target frame / patch index |
| golden_net / golden_delta / golden_weight | `[1,E,384]` / `[1,E,2]` / `[1,E,2]` | network 原輸出 |
| poses / golden_poses | `[N,7]` | BA 前後，`tx ty tz qx qy qz qw`，tracker 內部 pose 方向 |
| patches / golden_patches | `[N,M,3,3,3]` | BA 前後，x/y/disparity |
| intrinsics | `[N,4]` | feature-grid intrinsics，不能再除以 4 |
| patch_to_frame | `[N*M]` | point cloud 使用的 patch frame mapping |
| coords | `[E,2,3,3]` | Python reproject golden，去掉 batch axis |
| golden_target / golden_graph_weight | `[1,E,2]` | BA 真正輸入 |
| golden_points | `[N*M,3]` | update 結束後 point cloud |
| gmap_slot_ids / fmap_slot_ids | `[used_slots]` | 稀疏 slot 清單 |
| imap_slot_SSS / gmap_slot_SSS | `[M,384]` / `[M,128,3,3]` | `(kk/M) % patch_memory_size` |
| fmap1_slot_SSS / fmap2_slot_SSS | `[128,Hf,Wf]` / `[128,Hf/4,Wf/4]` | `jj % frame_memory_size` |

motion probe case 只要求前三列 network tensors。ring capacities、初始化狀態、BA fixed pose count、
BA iterations、各設定與 tensor shapes 記錄於 case metadata。完整 replay 保留 captured ring capacities，
不改用 C++ 預設容量；同一 snapshot 可跨多個 repeat 使用。

## 無 GPU 的開發檢查

```bash
# 在有 NumPy 的 Python 環境，DPVO root 下執行
python -m unittest discover -s tools/gen_testdata -p test_update_replay_generator.py -v

# dpvo_runner 下執行
make test HOST_OPTFLAGS=-O2
make test_update_replay HOST_OPTFLAGS=-O2
```

第二個 make target 使用明確命名的 `run_update_test_stub` 和測試用 network double，
測試 reader、實際 Update 的非網路部分、repeat reset、CSV/metadata、損毀資料與 golden mismatch。
它不驗證真正的 ONNX inference，也不提供 `--host` benchmark。
正式 GPU 生成、Spike ONNX parity 與 FireSim 效能仍須在對應環境執行。
