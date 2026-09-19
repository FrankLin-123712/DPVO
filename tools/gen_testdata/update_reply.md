# Update replay testdata generator

本文件說明如何使用 `generate_update_replay_testdata.py`，從真正的 Python DPVO inference
擷取 update 的輸入、中間 tensors 與輸出 golden，以及產生資料的格式與搬移方式。

生成的資料可供完整 `DPVOTracker::Update()` 或獨立 update network replay 使用。
C++ binary 的編譯、Spike／FireSim 執行及 profiling 報表請參閱
[dpvo_runner 使用說明](../../../chipyard/generators/gemmini/software/onnxruntime-riscv/systolic_runner/dpvo_runner/README_update_replay.md)。

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

## 2. 輸出格式 `dpvo_update_replay_v1`

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
BA iterations、各設定與 tensor shapes 記錄於 case metadata。slot index 必須搭配記錄的 ring capacities 解讀。

## 3. 搬移生成的資料

在 DPVO root 下，將整個 testdata root 搬到執行 runner 的 server，例如：

```bash
rsync -aH testdata/update_replay_euroc_mh01_first16_p16/ \
  SERVER:/home/cclin/chipyard/generators/gemmini/software/onnxruntime-riscv/systolic_runner/dpvo_runner/testdata/update_replay_euroc_mh01_first16_p16/
```

gmap/fmap 以內容 hash 去重，case 下仍有完整 `.bin` 入口；`rsync -aH` 或 `tar` 可保留 hard links。
一般複製也可讀取，但可能膨脹成每次 update 都複製全部 feature maps。
資料量取決於 edge 數、active frames 與 slot 更新；生成時只保留當次 case 的 tensor snapshots。

請保留 root `metadata.json`、`cases.txt` 與 case 目錄結構。
資料生成完成後，依 [dpvo_runner 使用說明](../../../chipyard/generators/gemmini/software/onnxruntime-riscv/systolic_runner/dpvo_runner/README_update_replay.md) 準備相同 checkpoint 的 ONNX 模型、
編譯 binary 並執行 replay。

## 4. 無 GPU 的 generator 開發檢查

在有 NumPy 的 Python 環境、DPVO root 下執行：

```bash
python -m unittest discover -s tools/gen_testdata -p test_update_replay_generator.py -v
```

這些測試檢查 snapshot ownership、case 標記、BA 失敗處理與 hook 還原。
正式 inference 與 golden 生成仍需要 GPU 環境。
