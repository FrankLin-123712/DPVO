# Patchify replay testdata

`generate_patchify_replay_testdata.py` 在 **真正的 Python DPVO tracker inference** 中觀察
`network.patchify.forward`，每次呼叫存一個 case。它不重新執行 encoder、不重新選 centers，
也不以另外寫的 Python patchify 實作取代 golden。

## GPU 上產生 EUROC 資料

使用已能執行此 repository DPVO 的 GPU 環境：PyTorch/CUDA、DPVO CUDA extensions、
NumPy、OpenCV；讀取 YAML 設定時也需要 PyYAML。此 server 沒有 GPU，正式生成請在 GPU 主機執行。

```bash
cd /path/to/DPVO
python tools/gen_testdata/generate_patchify_replay_testdata.py \
  --weights dpvo.pth \
  --images datasets/EUROC/MH_01_easy/mav0/cam0/data \
  --calib calib/euroc.txt \
  --frame-start 1 --frame-count 16 --frame-step 1 \
  --width 752 --height 480 \
  --patches-per-frame 16 --centroid-sel-strat RANDOM \
  --no-mixed-precision --no-undistort --seed 7 \
  --output-root testdata/patchify_replay_euroc_mh01_first16_p16
```

預設也是 MH_01 前 16 幀、P16、最長邊 752、seed 7，不做 undistortion。
共用 loader 要求至少 8 幀；`--frame-start` 為 **1-based**，輸出 `input_frame_index` 為
**0-based**。若要對照既有 run，請使用相同 frame range、resize、undistortion、tracker config
與 checkpoint。`--config-yaml FILE` 的語意沿用 correlation generator；YAML 的 FP16 設定仍會被
覆寫成 FP32，TF32 也關閉。這不是把 FP16 計算結果轉存成 FP32。

預設包含 `slam.terminate()`；它不產生新的 patchify cases。可加 `--skip-terminate-updates`
省去最後的 tracker updates，之前擷取的 patchify calls 不變。`GRADIENT_BIAS` 可擷取實際 centers，
但 C++ replay 固定使用它們，不包含重新評分／抽樣成本。

每個 480×752 case 約 60 MiB，16 幀約 1 GiB（另加前處理影像）。逐 case 寫入，不保留全部 dense
features 在 RAM。output root 必須不存在或為空；不會覆蓋既有資料。

## 對應 ONNX 模型

模型必須來自**同一份 checkpoint**，輸出 encoder 的 raw fmap／imap，不能先除以 4。
現有 `FeatureExtractor` exporter 符合這個介面：

```bash
python tools/export2onnx/export_models.py \
  --weights dpvo.pth --out exported_models/patchify_replay \
  --height 480 --width 752 --opset 11 --skip-update
sha256sum dpvo.pth exported_models/patchify_replay/feature_extractor.onnx
```

Generator 的 root metadata 記錄 checkpoint SHA-256。請把 ONNX hash 與實驗結果一起保存；
不同 checkpoint、exporter 或 graph optimization 都可能改變 parity／效能。

## 格式 `dpvo_patchify_replay_v1`

```text
metadata.json                  # status, checkpoint hash, config, resize, seed, cases
cases.txt                      # case_0000、case_0001...，順序等於呼叫順序
images/000000.png              # tracker 使用的前處理影像（BGR tensor 由 OpenCV 載入）
calib.txt
case_0000/
  metadata.json                # frame/call、dimensions、tensor_shapes、semantics
  manifest.txt                 # <name> <dtype> <dim0> <dim1> ...
  image.bin
  centers.bin
  raw_fmap.bin
  ...
```

所有 `.bin` 都是 little-endian、C contiguous；float tensors 為 FP32，index 為 int64。
`manifest.txt` 沿用共用 writer 的 `dpvo_python_testdata_v1` 註解，**replay 格式版本以 metadata 為準**。

令 `Hf=H/4`、`Wf=W/4`、`M=patches_per_frame`：

| 名稱 | Shape | 語意 |
|---|---|---|
| image | `[1,1,3,H,W]` | 實際 patchify 輸入，`2*(BGR/255)-0.5` |
| centers | `[M,2]` | `[x,y]`，feature-grid 座標，取自 golden patch 中心 |
| raw_fmap | `[1,1,128,Hf,Wf]` | fnet hook，未乘 0.25 |
| raw_imap | `[1,1,384,Hf,Wf]` | inet hook，未乘 0.25 |
| golden_fmap | `[1,1,128,Hf,Wf]` | patchify 回傳的完整 feature map，已乘 0.25 |
| golden_imap | `[M,384,1,1]` | 已取 patch 的 context，不是 dense raw_imap |
| golden_gmap | `[M,128,3,3]` | feature patches |
| golden_patches | `[M,3,3,3]` | x/y/disparity grid，disparity 此時為 1 |
| golden_colors | `[M,3]` | 原輸入 channel order 的正規化浮點顏色 |
| golden_index | `[M]` | Python 回傳 index；單幀 inference 全部為 0 |

除完整 fmap 保留 `[B,N,...]` 外，patch tensors 去除 Python 的 singleton batch axis，直接符合
C++ `PatchifyResult`。第一版要求單幀輸入、3×3 patches、`disps=None`、`return_color=True`，
H/W 可被 4 整除。顏色尚未 RGB reorder、uint8 轉換，depth 尚未被 tracker 隨機／median 初始化覆寫。

兩個 encoder 的 forward hooks 只觀察原呼叫；所有輸出立即複製到 owned CPU storage。
即使 tracker 後續修改原 tensor，也不影響 case。例外時會恢復原 forward 與 hooks。
只有完成寫入的 case 才加入 `cases.txt`；root `status=complete` 才可作完整 dataset replay。
`failed` 或 `incomplete` 資料不可冒充成功生成，請改用新的 output root 重跑。

## 搬移與驗證

保留整個資料夾結構，連同對應 `feature_extractor.onnx` 搬到 runner 可讀的位置。
C++ 使用 `--case_root DIR`，也可用 `--case_dir DIR/case_0000` 做單 case 檢查。
單 case 模式允許檢查中斷資料集中已完成的 case，不代表整份資料生成成功。

Runner 的完整建置、Spike parity、FireSim profiling 與報表說明在另一個 repository：
`systolic_runner/dpvo_runner/docs/patchify_benchmark.md`。

先跑 CPU extraction 對照五個 patchify 輸出，再跑 `feature` 比對 raw fnet/inet，最後跑 `full`。
`feature` mode 可將模型差異與 patch sampling 差異分開定位。
預設 `atol=5e-4, rtol=1e-4`；失敗時先核對 checkpoint、輸入、scaling 與 dtype，不要直接放寬 tolerance。

無 GPU 的 capture 行為檢查（只需 NumPy，不會宣稱驗證 CUDA inference）：

```bash
python tools/gen_testdata/test_patchify_replay_generator.py
python tools/gen_testdata/generate_patchify_replay_testdata.py --help
```
