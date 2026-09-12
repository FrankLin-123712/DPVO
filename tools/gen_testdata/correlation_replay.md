# Correlation replay 資料產生器

`generate_correlation_replay_testdata.py` 用 Python/CUDA 執行完整 DPVO，攔截
`DPVO.corr()` 的真實輸入與輸出，供後續 C++ correlation 單層重播。
沿用現有 BA generator 的 YAML 解析，以及 Python testdata generator 的
影像前處理、tracker config、seed 設定、`manifest.txt + *.bin` 格式。
不修改 `dpvo/dpvo.py`，離開攔截範圍時會復原原始方法。

## GPU 機器執行

使用已能執行 Python DPVO 的環境，包括 PyTorch、CUDA extensions、OpenCV、
模型權重與 EUROC 影像。以下路徑對應目前 workspace；其他機器請調整路徑。

```bash
cd /home/cclin/DPVO
python tools/gen_testdata/generate_correlation_replay_testdata.py \
  --weights /path/to/dpvo.pth \
  --images /home/cclin/chipyard/generators/gemmini/software/onnxruntime-riscv/systolic_runner/dpvo_runner/datasets/EUROC/MH_01_easy/mav0/cam0/data \
  --calib /home/cclin/chipyard/generators/gemmini/software/onnxruntime-riscv/systolic_runner/dpvo_runner/calib/euroc.txt \
  --config-yaml /home/cclin/chipyard/generators/gemmini/software/onnxruntime-riscv/systolic_runner/dpvo_runner/config/fast_p16.yaml \
  --frame-start 1 --frame-count 16 --frame-step 1 \
  --width 752 --height 480 \
  --no-undistort --no-mixed-precision --seed 7 \
  --output-root testdata/correlation_replay_euroc_mh01_first16_p16
```

輸出目錄必須不存在或為空，避免覆寫先前案例。每次 correlation 完成後會
顯示 case、stage、frame 與 edge 數。預設捕捉全部呼叫，包含 terminate 的更新；
可用 `--skip-terminate-updates` 排除最後更新。沒有固定期待 39 次：Python 與
C++ 的追蹤狀態、數值與 edge graph 可能不同。

YAML 的 tracker 參數優先於 CLI 的對應參數，但 `MIXED_PRECISION` 最後一律
覆蓋為 `False`，即使 YAML 寫 `True`。TF32 關閉，seed 同時設定 torch、NumPy
與 Python random；不同 GPU／CUDA 版本仍不保證 bitwise reproducibility。
Loop closure 關閉。影像預設不去畸變；`--undistort` 可明確啟用。前處理沿用
原工具的 resize／16 倍數對齊，對其他解析度不等同 C++ 的 crop；本案例的
752×480 不需 resize 或 crop。影像使用原 Python DPVO 的 OpenCV BGR 路徑。

## 檔案格式與重播規則

```text
output-root/
  metadata.json          # format、status、有效 config、來源、所有 case metadata
  cases.txt              # 按呼叫順序列出 case_0000 等相對目錄
  calib.txt
  images/                # 實際送入 Python DPVO 的前處理影像
  features/<sha256>.bin   # 按 dtype、shape、內容識別的不可變 feature 版本
  case_0000/
    manifest.txt         # tensor_name dtype dim0 dim1 ...
    metadata.json
    coords.bin
    kk.bin
    jj.bin
    golden_corr.bin
    ...
```

每個 case 都有完整 manifest 與 `.bin` 檔案，feature 檔案以 hard link 共用
storage；檔案系統不支援時改用 copy。各 case 的資料不依賴 symlink。
搬移整包資料時可用支援保留 hard links 的工具節省空間，普通 copy 仍可讀取。
不要原地修改任何輸出 `.bin`，hard links 可能影響其他 cases。

所有 binary 為 little-endian、C contiguous，無 header。

| Tensor | dtype / shape | 意義 |
|---|---|---|
| `coords` | float32 `[E,2,3,3]` | Python `[1,E,2,3,3]` 去掉 batch；axis 1 為 x、y |
| `kk`, `jj` | int64 `[E]` | 原始 patch / target frame indices，保留 edge 順序 |
| `patches_per_frame` | int64 `[1]` | M |
| `patch_memory_size`, `frame_memory_size` | int64 `[1]` | pmem、mem |
| `gmap_slot_ids`, `fmap_slot_ids` | int64 `[S]` | 本次用到的實體 slots，各自排序去重 |
| `gmap_slot_NNN` | float32 `[M,128,3,3]` | 該 slot 的完整 patch features |
| `fmap1_slot_NNN`, `fmap2_slot_NNN` | float32 `[128,H,W]` | 各層該 slot 的完整空間 feature map |
| `golden_corr` | float32 `[1,E,882]` | 原始 `DPVO.corr()` 回傳值，沒有重算／重排 |

重播第 e 個 edge：

```text
source slot = (kk[e] // M) % pmem
source local patch = kk[e] % M
target slot = jj[e] % mem
```

`kk` 不是 source frame index；Python `corr` 內雖命名為 `ii`，實際是 patch
indices。Motion probe 必須使用它傳入的 `indicies=(kk,jj)`，不能誤用 graph
的 indices。輸出儲存所有被引用 slots 的獨立快照，不保存未引用的 slots；
同一 slot 被覆寫／因 keyframe 移除而搬動後，新內容會取得新的 feature hash。

呼叫 C++ slots 入口時，配置長度為 pmem／mem 的 Tensor vectors，將保存的
slots 放回原始位置，其餘保持空 Tensor，再傳入原始 `kk/jj/coords/M`。
Python 預設 pmem=mem=36，但格式分別記錄兩者。
輸出 flatten 順序為 `[xoff, yoff, py, px, level]`，各軸大小 `[7,7,3,3,2]`。
第二層使用 `coords / 4`。重播不需要 poses、intrinsics、geometry patches 或
reproject。此 slots replay 格式由 C++ `run_correlation` 讀取；舊的
`correlation_parity` 不適用。下方提供 host／Spike／FireSim 操作流程。

case metadata 的 `stage` 為 `motion_probe`、`initialization`、`update` 或
`terminate`。初始化 12 次 update 以「該影像開始前尚未 initialized」辨識，
因為 Python 在第一個初始化 update 前就已把 `is_initialized` 設成 True。
`input_frame_index` 是選取影像中的 0-based 序號；terminate 為 -1。
`tracker_n` 和 `tracker_counter` 是呼叫當下的 tracker 狀態，可能與影像序號
不同。`iteration` 是該影像／terminate 階段內 0-based update 序號；probe 為 null。

## 驗證與結果回報

無 GPU 的檢查只需要 NumPy：

```bash
python -m unittest discover -s tools/gen_testdata -p 'test_correlation_replay_testdata.py' -v
```

這些測試檢查 wrapped indices、probe indices、slot 覆寫後快照不變、feature
去重與版本、binary/manifest 大小、stage 標記、例外後方法復原，以及 FP32
設定。它們使用 host tensor adapter，沒有測試真實 PyTorch／CUDA inference。

GPU 產生資料成功後，根目錄 `metadata.json` 必須是 `status: complete`；
`captured_case_count` 應等於 `cases.txt` 行數。每個 case 都要求輸入／輸出
dtype 正確、shape 正確、浮點值有限。失敗時根 metadata 會標記 `failed`，
保留已完成的 cases；強制終止可能留下 `incomplete`，不能當完整資料集使用。
請回傳終端輸出及根目錄 `metadata.json`，先核對階段、edge 數與資料規模。

數值正確性仍需後續 C++ replay 驗證。目前 C++ 與 Python CUDA 都先對中心 floor 周圍
8×8 整數位置計算 dot products，再插值為 7×7 scalar correlation；舊 C++ 版本則先插值 feature。
累加精度與浮點順序仍可能不同，維持 C++ replay 的 atol=5e-4、rtol=1e-4，不要求 bitwise 相同。
資料格式與 golden 不變，既有 replay 可直接使用。產生器不量測 Python 或 C++ 的 correlation cycles。

## 資料產生器完整參數

在 DPVO repository 根目錄執行 `python tools/gen_testdata/generate_correlation_replay_testdata.py --help`。
路徑預設相對於 DPVO repository，而不是目前 shell 目錄；output-root 若明確
指定相對路徑，則相對於目前 shell 目錄。

| 參數 | 預設與用途 |
|---|---|
| `--weights PATH` | DPVO/dpvo.pth，請指定實際 checkpoint |
| `--images DIR` | DPVO/datasets/EUROC/MH_01_easy/mav0/cam0/data |
| `--calib PATH` | DPVO/calib/euroc.txt |
| `--output-root DIR` | DPVO/testdata/correlation_replay_euroc_mh01_first16_p16，必須不存在或為空 |
| `--config-yaml PATH` | 選用 C++ runner config；對應 tracker 參數優先於 CLI |
| `--frame-start N` | 1，按檔名排序後的 1-based 起始影像 |
| `--frame-count N` | 16，至少 8 張 |
| `--frame-step N` | 1，每 N 張取一張 |
| `--max-long-edge N` | 752，未指定寬高時限制最長邊，不放大 |
| `--width N --height N` | 兩者一起指定；預設 0 自動，輸出尺寸向下對齊 16 倍數 |
| `--undistort` / `--no-undistort` | 預設不去畸變；啟用時使用校正檔的 distortion |
| `--patches-per-frame N` | 16，每張 patches 數 |
| `--buffer-size N` | 64，實際至少 frame_count+8 |
| `--removal-window N` | 16 |
| `--optimization-window N` | 7 |
| `--patch-lifetime N` | 11 |
| `--keyframe-index N` | 4 |
| `--keyframe-thresh F` | 15.0 |
| `--motion-model NAME` | DAMPED_LINEAR |
| `--motion-damping F` | 0.5 |
| `--centroid-sel-strat NAME` | RANDOM，可選 GRADIENT_BIAS |
| `--ba-iterations N` | 2 |
| `--no-mixed-precision` | 明確標記 FP32；不論有無旗標，最後皆強制 FP32 |
| `--seed N` | 7，設定 torch、NumPy、Python random |
| `--cuda-device N` | 0，CUDA device index |
| `--skip-terminate-updates` | 預設不啟用；指定後省略最後 terminate updates |
| `--help` | 顯示說明 |

沒有 `--overwrite`；若要重新產生資料，請用新的 output-root。也沒有
`--all-cases`，因為此工具預設就保存所有 correlation 呼叫。

## 在 C++ runner 使用產生的資料

以下在有 Chipyard toolchain 的機器執行；Python 資料產生才需要 GPU。
將完整資料目錄複製到該機器，再設定路徑：

```bash
DPVO_RUNNER=/home/cclin/chipyard/generators/gemmini/software/onnxruntime-riscv/systolic_runner/dpvo_runner
CORR_CASES=/home/cclin/DPVO/testdata/correlation_replay_euroc_mh01_first16_p16
cd "$DPVO_RUNNER"
./build.sh --config=Release --parallel --correlation-benchmark --host --host-tests -O2
python3 tools/test_correlation_replay.py --binary build/host/Release/run_correlation
python3 tools/test_correlation_scripts.py
./build.sh --config=Release --parallel --correlation-benchmark -O2
./run_corr_spike.sh --case_root "$CORR_CASES" --max_cases 1 -x 0 \
  --profile_mode total --cache_policy warm --warmup 1 --repeat 1
```

建置用 `--correlation-benchmark` 選目標，再自行組合旗標：`--host` 改用
x86 native compiler；`--for_firesim` 僅加入 FOR_FIRESIM definition；兩者皆不加
時預設編譯 RISC-V rv64，可交給 Spike。`--parallel` 啟用平行編譯。
最佳化須明確指定 -O0／-O2／-O3，Release 不自動設定 O2。

RISC-V 輸出為 `build/Release/run_correlation`，host 為
`build/host/Release/run_correlation`。預設 config 是 Debug；上例明確選 Release。
`--host-tests` 執行 host core tests；replay 和腳本測試使用上述 Python 命令。

Gemmini Spike 測試需明確指定 matching FP32 extension：

```bash
./run_corr_spike.sh --case_dir "$CORR_CASES/case_0000" -x 2 \
  --gemmini-lib /path/to/matching-fp32/libgemmini.so \
  --profile_mode phases --cache_policy warm --warmup 1 --repeat 1
```

工具位置可用 `--spike`、`--pk`，binary 用 `--binary`；`--dry-run` 顯示
命令但不執行。CPU 模式不用 extension。DIM、BANK_ROWS、ACC_ROWS 必須
與 binary／bitstream／extension 對齊，不要假設固定 DIM8 或 DIM16。

FireSim binary 用 `./build.sh --config=Release --parallel --correlation-benchmark --for_firesim -O2` 建置，再由原本
FireMarshal／FireSim 流程打包及執行。`total` 僅量外層，`phases` 有七段
計時，其中新演算法使用 `integer_gather_pack`、`dot_product`、`interpolate_output` 區分
整數 feature 搬移、點積與 scalar 插值；使用相同 case/execution/cache/warmup/repeat，
分開執行比較 total 差異，差異亦可能包含執行波動。

預設 warmup=1，**每個 repeat 前都重新暖機**，同一 case 的列印與 CSV
輸出延後至所有 repeats 完成。CSV 記錄 profile_mode、cache_policy、
warmup_per_repeat。warm 模式不宣稱所有資料都留在 cache；要不額外準備，
用 `--cache_policy natural --warmup 0`，這也不是 cold-cache。

完整 C++ 參數、FireSim 範例、CSV 格式與計時解讀見
`dpvo_runner/docs/correlation_replay.md`。Spike cycles 只做功能檢查，
性能結論應以 matching FireSim 硬體量測為準。
