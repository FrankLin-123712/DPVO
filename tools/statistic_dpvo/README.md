# statistic_dpvo

本資料夾提供一組 DPVO algorithm-parameter sweep 工具，用來把不同
`P_a` candidate 的 **估計 workload** 與 **EuRoC trajectory accuracy** 放在同一套
實驗輸出裡比較。

三個主要程式的分工如下：

- `sweep_dpvo.py`：整體 sweep runner。負責產生 candidate configs、呼叫
  `statistic_dpvo.py` 的 estimator、必要時呼叫 `evaluate_euroc_sweep.py` 跑
  EuRoC ATE，最後彙整 CSV、JSON、SVG plots 與 Markdown report。
- `statistic_dpvo.py`：DPVO 靜態 workload estimator。使用
  **layer-boundary DRAM materialization model** 統計一個 steady-state accepted
  processed frame 的 useful operations 與 logical memory traffic。
- `evaluate_euroc_sweep.py`：單一 candidate 的 EuRoC evaluator。使用原本 DPVO
  tracker 跑 EuRoC image sequences，並用 `evo` 的 APE translation RMSE 得到 ATE。

這套工具的目的不是取代 runtime profiling，而是先在同一套統計規則下比較不同
algorithm parameters，協助找出 accuracy 可接受且 ops / logical bytes 較低的候選點。

## 整體資料流

`sweep_dpvo.py` 是總控程式。完整流程可以看成：

```text
DPVO/config/default.yaml
        |
        v
sweep_dpvo.py
        |
        +--> statistic_result/generated_configs/*.yaml
        |        每個 candidate 一份 DPVO config
        |
        +--> statistic_dpvo.py internal API
        |        build_rows() -> aggregate_modules() -> sum_rows()
        |        |
        |        v
        |   statistic_result/per_module/*.json
        |   statistic_result/per_module/*.csv
        |
        +--> evaluate_euroc_sweep.py subprocess  [只有 --run-eval 時]
        |        使用指定 Python / CUDA DPVO environment
        |        |
        |        v
        |   statistic_result/euroc_eval/*.json
        |   statistic_result/euroc_eval/*_scenes.csv
        |
        v
statistic_result/sweep_summary.csv
statistic_result/module_summary.csv
statistic_result/sequence_errors.csv
statistic_result/sweep_plots/*_sweep.svg
one_at_a_time_sweep_report.md
```

預設 sweep 是 one-at-a-time：每次只改一個 parameter，其他值都維持 base config。
因此每個 candidate 的 `total_ops`、`total_memory_bytes` 與 `ate_m` 可以直接對應到
「單一參數改變」造成的 workload / accuracy 變化。

預設 sweep points：

- `PATCHES_PER_FRAME`: `96, 80, 64, 48, 32`
- `PATCH_LIFETIME`: `13, 11, 9, 7, 5`
- `REMOVAL_WINDOW`: `22, 18, 14, 10`
- `OPTIMIZATION_WINDOW`: `10, 8, 6, 4`
- `BA_ITERATIONS`: `20, 18, 16, 14, 12, 10, 8, 6, 4, 2`
- `IMAGE_SIZE`: `480x640, 384x512, 320x416, 240x320, 192x256`

## `sweep_dpvo.py`

`sweep_dpvo.py` 的重點是把整個實驗串起來，而不是自己做 workload 統計或 ATE
計算。它主要做四件事：

1. 產生 candidate config。
2. 對每個 candidate 收集 static workload statistics。
3. 在 `--run-eval` 時對每個 candidate 跑 EuRoC ATE。
4. 把所有結果彙整成表格、圖與報告。

### 執行流程

1. 解析 CLI arguments。

   重要參數包含：

   - `--config`：base DPVO config，預設是 `DPVO/config/default.yaml`。
   - `--result-dir`：輸出目錄，預設是
     `DPVO/tools/statistic_dpvo/statistic_result/`。
   - `--parameters`：指定要 sweep 哪些 parameters。
   - `--sweep-file`：用 JSON 自訂 sweep points。
   - `--run-eval`：除了 static estimator，也跑 EuRoC ATE。
   - `--python`：`evaluate_euroc_sweep.py` subprocess 使用的 Python。
   - `--reuse-eval`：重用既有 `euroc_eval/*.json`，只重新彙整與畫圖。

2. 讀取 base config。

   程式用 `statistic_dpvo.load_simple_yaml()` 讀 top-level YAML scalar values。
   這裡只需要知道可 sweep 的 algorithm parameters，不做完整 YAML schema 驗證。

3. 建立 one-at-a-time candidates。

   對每個 sweep parameter 與 value，產生一個 `Candidate`：

   - `candidate_id`：例如 `patches_per_frame_064`、`image_size_384x512`。
   - `sweep_parameter`：這個 candidate 改的是哪個參數。
   - `sweep_value`：被測的值。
   - `height` / `width`：`IMAGE_SIZE` candidate 會改這兩個欄位。
   - `config_path`：此 candidate 對應的 generated YAML。
   - `overrides`：寫入 YAML 的 parameter override。

   對一般 algorithm parameters，`sweep_dpvo.py` 會把 override 寫進 generated YAML。
   對 `IMAGE_SIZE`，DPVO config 不一定有對應欄位，所以它不改 YAML，而是把
   `height` / `width` 傳給 estimator 與 evaluator。

4. 呼叫 `statistic_dpvo.py` 的內部 API。

   `sweep_dpvo.py` 不用 subprocess 跑 estimator，而是直接 import：

   ```python
   import statistic_dpvo as statistic
   ```

   對每個 candidate，它組出類似下面的 estimator arguments：

   ```text
   --config <generated_config>
   --height <candidate.height>
   --width <candidate.width>
   --active-frames <active_frames>
   --per-module
   --format json
   ```

   然後直接呼叫：

   ```python
   parsed = statistic.parse_args(statistic_args)
   pa, graph, layer_rows = statistic.build_rows(parsed)
   module_rows = statistic.aggregate_modules(layer_rows)
   total = statistic.sum_rows(layer_rows)
   ```

   這一步會產生：

   - `statistic_result/per_module/<candidate>.json`
   - `statistic_result/per_module/<candidate>.csv`

5. 如果有 `--run-eval`，先做 evaluation preflight。

   preflight 會檢查：

   - network checkpoint 是否存在，預設 `DPVO/dpvo.pth`；
   - EuRoC image data 是否存在，預設
     `DPVO/datasets/EUROC/<scene>/mav0/cam0/data/*.png`；
   - 指定的 `--python` 是否能 import `cv2, evo, torch`；
   - 指定的 `--python` 是否能 import compiled DPVO extensions：
     `dpvo.fastba, dpvo.altcorr`。

   如果 preflight 失敗，所有 candidate 的 `evaluation_status` 會標成
   `preflight_failed`，但 static workload 結果仍然會被輸出。

6. 對每個 candidate 呼叫 `evaluate_euroc_sweep.py`。

   `sweep_dpvo.py` 用 subprocess 執行 evaluator。實際命令概念上是：

   ```bash
   <args.python> DPVO/tools/statistic_dpvo/evaluate_euroc_sweep.py \
     --network <network> \
     --config <generated_config> \
     --eurocdir <eurocdir> \
     --output statistic_result/euroc_eval/<candidate>.json \
     --per-scene-csv statistic_result/euroc_eval/<candidate>_scenes.csv \
     --height <candidate.height> \
     --width <candidate.width> \
     --stride <stride> \
     --trials <trials> \
     --seed <seed> \
     --backend-thresh <backend_thresh> \
     --scenes <scene list>
   ```

   這裡 `--python` 很重要：外層 runner 可以用任意 Python 啟動，但 evaluator
   必須使用具備 CUDA、PyTorch、OpenCV、evo、DPVO compiled extensions 的 runtime
   environment。

7. 讀回 evaluator JSON 並合併。

   evaluator 成功後，`sweep_dpvo.py` 讀取 `avg_ate_m`，填入 `StatsResult`。
   如果 evaluator return code 非 0，該 candidate 的 `evaluation_status` 會變成
   `failed:<returncode>`。

8. 輸出 summary 與 plots。

   最後會產生：

   - `sweep_summary.csv`：每個 candidate 一列，包含 `P_a`、derived graph size、
     `total_ops`、`total_memory_bytes`、`ate_m`、evaluation status。
   - `module_summary.csv`：所有 candidate 的 per-module long-form rows。
   - `sequence_errors.csv`：所有 candidate 的 per-scene ATE rows。
   - `sweep_plots/*_sweep.svg`：每個 parameter 一張圖。
   - `one_at_a_time_sweep_report.md`：自動產生的實驗摘要。

### 結果彙整方式

`sweep_summary.csv` 是最重要的總表。每列代表一個 candidate，主要欄位是：

- `candidate_id`
- `sweep_parameter`
- `sweep_value`
- `height`
- `width`
- `patches_per_frame`
- `patch_lifetime`
- `removal_window`
- `optimization_window`
- `ba_iterations`
- `edge_count`
- `unique_patches`
- `edge_groups`
- `free_pose_count`
- `total_ops`
- `total_memory_bytes`
- `ate_m`
- `evaluation_status`
- `config_path`
- `evaluation_json`

`module_summary.csv` 則把每個 candidate 拆成 module 粒度，方便找出 workload
主要來自 feature extractor、update block、correlation、bundle adjustment 或其他
DPVO runtime module。

`sequence_errors.csv` 只有在有 ATE JSON 時才會有內容。它保留每個 scene 的
median ATE、mean ATE 與 trial ATE list。

### 畫圖方式

`sweep_dpvo.py` 直接產生 SVG，不依賴 matplotlib。每個 sweep parameter 會有一張
`*_sweep.svg`。

每張圖同時畫三個 metric：

- 藍線：`total_ops`
- 橘色虛線：`total_memory_bytes`
- 綠線：`ate_m`

所有 sweep plots 對同一種 metric 使用共同 y-axis range。這樣不同 parameter 的圖
可以直接比較趨勢。如果沒有跑 `--run-eval`，ATE 欄位是空的，圖上會顯示
ATE pending 的提示線。

## `statistic_dpvo.py`

`statistic_dpvo.py` 是靜態 estimator。它的輸入是 DPVO config 與 workload shape
設定，輸出是 ops / memory 的估算表。

它統計的是 **一個 steady-state accepted processed frame**，不是整段 dataset 的總
workload，也不是實際硬體 profile。

### 核心假設：layer-boundary DRAM materialization model

本工具使用固定 accounting model：

- 每個 canonical layer 開始時，從 logical DRAM 讀取完整 input tensor。
- 每次 layer invocation 都重新讀取 weights、parameters、metadata。
- 每個 canonical layer 結束時，把完整 output tensor 寫回 logical DRAM。
- 下一個 layer 必須重新讀取前一層 output。
- 不允許跨 layer fusion。
- 不允許跨 layer cache / SPAD / register persistence。
- layer 內部允許正常 accumulation 與 data reuse。
- 所有 `P_a` candidates 使用同一套 accounting rule。

因此 `total_memory_bytes` 代表 logical tensor traffic，不代表實際晶片上的 DRAM
transaction 數，也不代表 Gemmini scratchpad/cache 行為。實際 latency、energy、
tiling、DMA overlap、kernel launch、atomic、address generation、container overhead
都不在這個 estimator 的範圍內。

### 統計範圍

`statistic_dpvo.py` 包含 initialized steady-state frame 的主要 DPVO workload：

- feature extraction ONNX graph；
- patch center selection；
- patchify；
- feature pyramid pooling；
- factor graph append / gather / remove；
- patch reprojection；
- correlation volume construction；
- update block ONNX graph；
- update postprocess；
- local bundle adjustment；
- point-cloud update；
- keyframe motion test。

不包含：

- initialization / bootstrap path；
- motion probe；
- loop closure；
- data-dependent long-term behavior；
- runtime scheduler overhead；
- GPU / CPU kernel launch overhead；
- Gemmini-specific tiling、cache、SPAD、DMA policy。

### 如何建立 workload shape

1. 解析 `P_a`。

   `P_a` 來源依序合併：

   - `--config` 指定的 YAML；
   - repeatable `--pa KEY=VALUE`；
   - explicit CLI flags，例如 `--patches-per-frame`、`--ba-iterations`。

   支援的 algorithm parameters 包含：

   - `PATCHES_PER_FRAME`
   - `REMOVAL_WINDOW`
   - `OPTIMIZATION_WINDOW`
   - `PATCH_LIFETIME`
   - `BA_ITERATIONS`
   - `CENTROID_SEL_STRAT`

2. 固定 `active_frames`。

   `active_frames` 是統計用的 reference horizon，表示假設 DPVO 已經跑到第幾個
   accepted processed frame。它是為了讓 synthetic factor graph 進入穩態。

   它不是 active graph window size。做 sweep 時應固定 `active_frames`，否則會把
   「參數造成的 workload 變化」和「統計時間點不同造成的 workload 變化」混在一起。

3. 建立 synthetic factor graph。

   程式用 `PATCH_LIFETIME` 與 `REMOVAL_WINDOW` 模擬 DPVO 的 factor append/prune
   policy，再乘上 `PATCHES_PER_FRAME` 得到 active factors。

   會推導出：

   - `edge_count`：update-time active factors；
   - `post_prune_edge_count`：keyframe pruning 後保留的 factors；
   - `new_edge_count`：目前 frame 新增的 factors；
   - `unique_patches`：active factors 參考到的 unique source patches；
   - `edge_groups`：unique `(source frame, target frame)` groups；
   - `free_pose_count`：local BA 會最佳化的 pose 數。

4. 讀取 ONNX graphs。

   會讀取：

   - `DPVO/exported_models/feature_extractor.onnx`
   - `DPVO/exported_models/update_block.onnx`

   程式內建 dependency-free protobuf reader，所以在 minimal deployment
   environment 裡不需要額外依賴完整 ONNX Python package。它會對 ONNX nodes 做
   static shape inference，然後依據 op type 與 tensor shape 算 ops / bytes。

5. 用 source-level formula 統計非 ONNX runtime operators。

   對 C++ / DPVO runtime path，例如 projective ops、correlation、BA、keyframe
   motion test，程式用手寫公式估算 useful scalar operations 與 logical memory
   read/write。

### Operation counting convention

- 一個 MAC 算 2 個 scalar operations：multiply + accumulate。
- comparison、cast、divide、sqrt、nonlinear function 各算 1 個 scalar operation。
- tensor view / copy / indexed gather 不算 arithmetic ops，但會計 logical memory
  traffic。
- address generation 與 implementation-specific overhead 不計。
- BA source-level formulas 使用 valid-residual path。
- ONNX precision 依照 exported TensorProto dtype。目前 checked-in ONNX models 是
  float32。
- `FP16 Ops` 欄位目前通常是 0，只是 output schema 保留 precision bucket。
- `FP64 Ops` 主要來自部分 C++ projective ops / BA formulas，不是 NN layer。

### 輸出欄位

每個 output row 會包含：

- `FP16 Ops`
- `FP32 Ops`
- `FP64 Ops`
- `INT/Bool Ops`
- `Total Ops`
- `Mem Read (Bytes)`
- `Mem Write (Bytes)`
- `Total Memory (Bytes)`
- `Op Intensity (Ops/Byte)`
- `Memory Access Pattern`

JSON 輸出保留 exact integers。Markdown / CSV 顯示用 decimal SI units，例如 `K`,
`M`, `G`。

## `evaluate_euroc_sweep.py`

`evaluate_euroc_sweep.py` 用來評估單一 DPVO candidate 的 trajectory accuracy。
它是 machine-readable 版本的 EuRoC evaluator，輸出 JSON / CSV，方便
`sweep_dpvo.py` 合併結果。

### 跑的 dataset

預設使用 EuRoC ASL-format image data：

```text
DPVO/datasets/EUROC/<sequence>/mav0/cam0/data/*.png
```

預設 ground truth 位置：

```text
DPVO/datasets/euroc_groundtruth/<sequence>.txt
```

預設 calibration：

```text
DPVO/calib/euroc.txt
```

預設評估 11 個 EuRoC sequences：

- `MH_01_easy`
- `MH_02_easy`
- `MH_03_medium`
- `MH_04_difficult`
- `MH_05_difficult`
- `V1_01_easy`
- `V1_02_medium`
- `V1_03_difficult`
- `V2_01_easy`
- `V2_02_medium`
- `V2_03_difficult`

### 如何跑 DPVO

對每個 scene 與 trial，evaluator 會：

1. 讀取 candidate config。
2. 設定 `BACKEND_THRESH`。
3. 用 `seed + trial` 設定 PyTorch random seed。
4. 讀取 scene image paths，並依 `--stride` 取樣。
5. 讀取 calibration。
6. 如果 calibration 有 distortion coefficients，先用 OpenCV undistort。
7. 如果 image size 與 candidate 的 `--height` / `--width` 不同，resize image。
8. resize 後同步縮放 intrinsics：`fx, cx` 依 width 比例縮放，`fy, cy` 依 height
   比例縮放。
9. 把 image tensor 與 intrinsics tensor 放到 CUDA。
10. 建立 `dpvo.dpvo.DPVO(cfg, network, ht, wd, viz=False)`。
11. 對每個 frame 呼叫 `slam(t, image_tensor, intrinsics_tensor)`。
12. sequence 結束後呼叫 `slam.terminate()` 取得 estimated poses。

`IMAGE_SIZE` sweep 能成立，是因為 evaluator 在這裡支援固定 `--height` 與 `--width`
resize，不需要手動修改 DPVO stream code。

### 如何得到 ATE

DPVO 產生 estimated trajectory 後，evaluator 會：

1. 把 DPVO poses 轉成 `evo.core.trajectory.PoseTrajectory3D`。
2. 讀取 TUM-format ground truth trajectory。
3. 用 `evo.core.sync.associate_trajectories()` 對齊 timestamps。
4. 用 `evo.main_ape.ape()` 計算 APE。
5. `pose_relation` 使用 `translation_part`。
6. `align=True`，會做 trajectory alignment。
7. `correct_scale=True`，會允許 scale correction。
8. 使用 `result.stats["rmse"]` 作為該 trial 的 ATE，單位是 meter。

每個 scene 會跑 `--trials N` 次。輸出時：

- `trial_ate_m`：此 scene 每個 trial 的 ATE。
- `median_ate_m`：此 scene 的 trial median。
- `mean_ate_m`：此 scene 的 trial mean。
- `avg_ate_m`：所有 scenes 的 `median_ate_m` 再取平均。

`sweep_dpvo.py` 最後放進 `sweep_summary.csv` 的 `ate_m` 就是 evaluator JSON 裡的
`avg_ate_m`。

## 使用說明

以下命令假設從 repository root 執行。

### 只跑 static estimator

```bash
python3 DPVO/tools/statistic_dpvo/statistic_dpvo.py
```

輸出 JSON：

```bash
python3 DPVO/tools/statistic_dpvo/statistic_dpvo.py --format json
```

用 module 粒度輸出：

```bash
python3 DPVO/tools/statistic_dpvo/statistic_dpvo.py \
  --per-module \
  --format json
```

覆寫 algorithm parameters：

```bash
python3 DPVO/tools/statistic_dpvo/statistic_dpvo.py \
  --pa PATCHES_PER_FRAME=48 \
  --pa PATCH_LIFETIME=11 \
  --pa REMOVAL_WINDOW=16 \
  --pa OPTIMIZATION_WINDOW=7 \
  --pa BA_ITERATIONS=4
```

改變 estimator image size：

```bash
python3 DPVO/tools/statistic_dpvo/statistic_dpvo.py \
  --height 384 \
  --width 512
```

輸出到檔案：

```bash
python3 DPVO/tools/statistic_dpvo/statistic_dpvo.py \
  --per-module \
  --format csv \
  --output results/statistic_dpvo_default.csv
```

固定 `active_frames`：

```bash
python3 DPVO/tools/statistic_dpvo/statistic_dpvo.py --active-frames 64
```

執行 estimator regression tests：

```bash
python3 -m unittest DPVO/tools/statistic_dpvo/test_statistic_dpvo.py
```

### 只跑 one-at-a-time static sweep

```bash
python3 DPVO/tools/statistic_dpvo/sweep_dpvo.py
```

這個命令只跑 estimator，不跑 EuRoC ATE。`sweep_summary.csv` 仍會產生，但 `ate_m`
會是空值，`evaluation_status` 會是 `not_run`。

限制 sweep parameters：

```bash
python3 DPVO/tools/statistic_dpvo/sweep_dpvo.py \
  --parameters PATCHES_PER_FRAME IMAGE_SIZE
```

使用自訂 sweep file：

```bash
python3 DPVO/tools/statistic_dpvo/sweep_dpvo.py \
  --sweep-file my_sweep.json
```

`my_sweep.json` 範例：

```json
{
  "PATCHES_PER_FRAME": [96, 64, 48],
  "IMAGE_SIZE": [[480, 640], [384, 512], [320, 416]]
}
```

### 準備 EuRoC dataset

可以使用本資料夾的 downloader：

```bash
python3 DPVO/tools/statistic_dpvo/download_euroc.py
```

下載後目錄應符合：

```text
DPVO/datasets/EUROC/<sequence>/mav0/cam0/data/*.png
```

也要確認 ground truth 存在：

```text
DPVO/datasets/euroc_groundtruth/<sequence>.txt
```

### 跑完整 sweep 加 EuRoC ATE

如果外層 shell 已經在 DPVO conda environment：

```bash
conda activate dpvo
python3 DPVO/tools/statistic_dpvo/sweep_dpvo.py \
  --run-eval \
  --trials 3
```

如果外層 Python 不是 DPVO environment，可以指定 evaluator subprocess 使用 DPVO
environment：

```bash
python3 DPVO/tools/statistic_dpvo/sweep_dpvo.py \
  --run-eval \
  --trials 3 \
  --python /home/remote/chiehchihlin/miniconda3/envs/dpvo/bin/python
```

常用縮小測試：

```bash
python3 DPVO/tools/statistic_dpvo/sweep_dpvo.py \
  --run-eval \
  --trials 1 \
  --scenes MH_01_easy V1_01_easy \
  --parameters PATCHES_PER_FRAME IMAGE_SIZE \
  --python /home/remote/chiehchihlin/miniconda3/envs/dpvo/bin/python
```

讓 log 乾淨一點：

```bash
python3 DPVO/tools/statistic_dpvo/sweep_dpvo.py \
  --run-eval \
  --trials 3 \
  --no-progress \
  --python /home/remote/chiehchihlin/miniconda3/envs/dpvo/bin/python
```

完整預設 sweep 會跑約 33 個 candidates。若使用所有 11 個 EuRoC scenes 且
`--trials 3`，會執行約 `33 x 11 x 3 = 1089` 個 scene-level trial，時間成本很高。
建議先用少量 scenes / parameters 確認環境與輸出格式。

### 單獨跑 evaluator

通常 evaluator 由 `sweep_dpvo.py` 呼叫。若要手動測單一 config：

```bash
/home/remote/chiehchihlin/miniconda3/envs/dpvo/bin/python \
  DPVO/tools/statistic_dpvo/evaluate_euroc_sweep.py \
  --network DPVO/dpvo.pth \
  --config DPVO/config/default.yaml \
  --eurocdir DPVO/datasets/EUROC \
  --output DPVO/tools/statistic_dpvo/statistic_result/euroc_eval/default.json \
  --per-scene-csv DPVO/tools/statistic_dpvo/statistic_result/euroc_eval/default_scenes.csv \
  --height 480 \
  --width 640 \
  --stride 2 \
  --trials 3
```

限制 scenes：

```bash
/home/remote/chiehchihlin/miniconda3/envs/dpvo/bin/python \
  DPVO/tools/statistic_dpvo/evaluate_euroc_sweep.py \
  --config DPVO/config/default.yaml \
  --output DPVO/tools/statistic_dpvo/statistic_result/euroc_eval/default_small.json \
  --scenes MH_01_easy V1_01_easy \
  --trials 1
```

### 重用既有 ATE JSON

如果 `statistic_result/euroc_eval/*.json` 已經存在，可以不重跑 DPVO，只重新合併
summary 與重畫 plots：

```bash
python3 DPVO/tools/statistic_dpvo/sweep_dpvo.py --reuse-eval
```

若只想重用部分 parameters 的 evaluator JSON，也要給相同的 `--parameters`，讓本次
產生的 candidate IDs 對得上既有檔案。

## 建議分析方式

1. 先跑 stats-only sweep，確認 workload trend。
2. 用小範圍 `--run-eval` 確認 runtime environment、dataset 與 ATE output。
3. 跑完整 `--run-eval` 或分批跑 subsets。
4. 看 `sweep_summary.csv`，先篩掉 `evaluation_status != ok` 的 rows。
5. 以 default candidate 的 `ate_m` 當 baseline，設定可接受的 `Delta Error`。
6. 在 accuracy 可接受的 candidates 中，比較 `total_ops` 與 `total_memory_bytes`。
7. 用 `module_summary.csv` 找出 workload 下降主要來自哪些 module。
8. 用 `sweep_plots/*.svg` 檢查每個 parameter 的 ops / memory / ATE 趨勢。

最後要注意：one-at-a-time sweep 只能觀察單一參數改變的局部趨勢。若要選 final
`P_a`，仍需要把多個 candidate 組合成 combined parameter set，再用同樣 evaluator
重新量測 ATE。
