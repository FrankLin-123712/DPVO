# 依部署 ONNX 模型產生 DPVO 混合精度測資

更新：2026-09-23。

這裡的「FP16 測資」是：**按照 runner 目前的 ONNX graph，讓 Conv／MatMul／Gemm 使用 half 輸入、權重與輸出，其他浮點算子保留 FP32。** 不是把整個 PyTorch model `.half()`，也不是把舊 FP32 golden 檔案轉成 half。

目前已實際生成、可直接放進 runner 的小型資料位於：

```text
/home/cclin/chipyard/generators/gemmini/software/onnxruntime-riscv/systolic_runner/dpvo_runner/testdata/dpvo_onnx_mixed_smoke/
```

完整的 component／tracker 產生器也已接上這個 backend；本機缺少 PyTorch、CUDA DPVO extensions，所以這兩條 GPU 流程尚未實際驗收。小型 NN 資料不需要這些依賴。詳細結果見第 4 節。

## 1. 怎麼實作

### 1.1 直接以部署模型決定精度與權重

`fp16_onnx_reference.py` 使用 ONNX ReferenceEvaluator 逐節點執行，沒有呼叫 ORT／Systolic kernel，也沒有啟用 graph optimization。

```text
models/fp16/*.onnx 的 initializers、Cast、算子
    → Python／NumPy 參考執行
    → feature／update 的結果
    → component helper 或 CUDA DPVO tracker
    → manifest.txt、*.bin、metadata.json
```

必須使用 metadata 含 `dpvo_precision_policy=compute_half_cpu_float_v1` 的兩個模型：

- `feature_extractor_opset11.onnx`
- `update_block_opset11.onnx`

NN 的唯一權重來源是這兩個 ONNX 檔案。FP16 模式不讀取 `--weights` 指定的 checkpoint，因此不需要額外的 `dpvo.pth`，也不會發生 Python 載入另一個 checkpoint 卻拿來比較的情況。FP32 模式保持原本 checkpoint 路徑。

### 1.2 三種加速算子用 FP32 累加作為獨立參考

Conv、MatMul、Gemm 覆寫為以下數值規則：

```text
half operands（包含 graph 所指定的 bias）
    → 精確擴寬成 float32
    → NumPy float32 計算
    → 捨入成 half 輸出
    → 後續是否轉回 FP32，由 graph 的 Cast 決定
```

Conv 使用 NCHW 2D im2col＋NumPy matmul，支援 group、padding、stride、dilation；Gemm 處理 transpose、alpha、beta、bias broadcast。這些參考實作與 C++ Gemmini kernel 分開。

其他節點使用 ONNX 的 reference operators。`dpvo::scatter_sum`／`scatter_max` 另以 NumPy 實作：以 edge 為順序累加；max 遇到相同值保留第一個位置；空 group 為 `-inf`／argmax=-1，與目前 runner 的規則一致。只接受目前使用的 `[B,E,C]`、dim=1（或 -2），以及每個 edge 對應一個 group index。

執行時會檢查輸入名稱、dtype、shape、共享動態維度；FP32 區域若意外接到 half／double 會報錯。輸出的 dtype 必須符合 graph，且不可含 NaN／Inf。不支援的 operator 不會默默改走其他精度。

**這不是 Gemmini PE 的逐次 half 累加模擬。** 例如 2048+1-2048，FP32 累加參考是 1，依序每次捨入成 half 則可能是 0。測試刻意保留這個差別，才能衡量硬體數值損失，而不只是讓同一份 kernel 和自己比較。

### 1.3 模型介面與模型外的資料處理

| 資料 | 精度 |
|---|---|
| feature 的 images | FP32 |
| feature 的 fmap、imap | FP16 |
| update 的 net、ctx、corr | FP16 |
| update 的 ii、jj、kk、ix、jx | INT64 |
| update 的 net_out | FP16 |
| update 的 delta、weight | FP32 |
| 幾何、座標、深度與 BA | 維持既有 FP32 路徑 |

Component 產生器中的 scaling、插值、pooling 使用 float32 中間值，NN 儲存邊界做 half 捨入。correlation 的 FP16 模式先計算四個整數格點的 float32 dot product，各自捨入成 half，再做 float32 bilinear interpolation，最後捨入成 half；不把「先插值再 dot」視為有限精度下完全相同。這仍是 CPU float32 dot reference，不是 PE 累加模型。

完整 tracker 使用 `fp16_tracker_adapter.py` 把兩個 ONNX 網路接到 DPVO：保留既有 Patchifier 的選點／幾何邏輯，但取代 encoder 與 update 的 NN 執行。`MIXED_PRECISION=True` 用於既有 half cache；`NN_FP16_WEIGHTS=False`，因為網路權重由 ONNX 按算子分配，不能再呼叫 whole-network `.half()`。每次 NN 輸入／輸出會在 CUDA tensor 與 CPU NumPy 間搬移。

**完整 tracker 的 correlation、feature pyramid、geometry、BA 仍由既有 CUDA DPVO 執行。** 這份資料是「相同 NN graph＋Python tracker」參考，不是 runner 全部運算的逐 bit 複製，也不是原版 PyTorch autocast network 的輸出。

### 1.4 保存格式與修改檔案

為了讓既有 runner parity reader 直接讀取，浮點 `.bin` 仍使用 float32；half 結果只是精確擴寬後保存，沒有重新用 FP32 network 計算。索引保存 int64，幾何等欄位沿用原格式。`golden_weight` 是 update 輸出的 confidence weight，不是 NN parameters。

`metadata.json` 記錄模型 SHA256、I/O dtype、initializers dtype 統計、operator 清單、NumPy／ONNX／Python 版本、執行政策及 NN 外的 backend。完整 GPU 流程另記錄 PyTorch／CUDA／GPU；tracker 原本的 seed、centers、bootstrap depths 與預處理紀錄繼續保留。

| 檔案 | 變更 |
|---|---|
| `gen_testdata.sh` | 新增 NN_PRECISION 與 ONNX_MODEL_DIR，兩種精度分開輸出 |
| `generate_dpvo_python_testdata.py` | CLI、graph adapter、metadata 與完整性檢查 |
| `generate_dpvo_runner_parity_testdata.py` | CLI、graph reference 與 component metadata |
| `dpvo_runner_parity_common.py` | NN 呼叫與 half 儲存邊界；保留原 FP32 分支 |
| `fp16_onnx_reference.py`（新增） | 獨立 NN graph reference；亦可產生 CPU-only 小型 NN cases |
| `fp16_tracker_adapter.py`（新增） | graph backend 與 CUDA tracker 的介面 |
| `testdata_precision.py`（新增） | 共用 CLI／環境／輸出目錄檢查 |
| `test_fp16_onnx_reference.py`（新增） | 獨立數值、精度邊界、CLI、真實模型測試 |

## 2. 怎麼使用

以下命令使用 Bash，先設定路徑：

```bash
DPVO=/home/cclin/DPVO
RUNNER=/home/cclin/chipyard/generators/gemmini/software/onnxruntime-riscv/systolic_runner/dpvo_runner
MODEL_DIR="$RUNNER/models/fp16"
```

### 2.1 先生成不需要 CUDA 的小型 NN 測資

需要 Python、NumPy、ONNX；本輪驗證使用 Python 3.10、NumPy 1.26.4、ONNX 1.16.2。本機已有可用的 interpreter：

```bash
PY=/home/cclin/.venvs/dpvo-fp16-py310/bin/python
OPENBLAS_NUM_THREADS=1 "$PY" "$DPVO/tools/gen_testdata/fp16_onnx_reference.py" \
  --onnx-model-dir "$MODEL_DIR" \
  --output-root "$RUNNER/testdata/dpvo_onnx_mixed_smoke"
```

會生成：

```text
dpvo_onnx_mixed_smoke/
├── metadata.json
├── patchify_small/   # manifest + image、centers、feature／patch／color goldens
└── update_small/     # manifest + net、ctx、corr、indices、三個輸出 goldens
```

輸入是固定公式的 32×32 tensor、兩個整數 patch centers，以及 4 條 update edges；輸出使用真實模型權重計算。這裡沒有 correlation／BA golden，也沒有完整 trajectory。

用已建好的 native FP16 runner 測試：

```bash
cd "$RUNNER"
build/native/fp16/patchify_parity \
  --feature_model "$MODEL_DIR/feature_extractor_opset11.onnx" \
  --case_dir testdata/dpvo_onnx_mixed_smoke/patchify_small -x 0
build/native/fp16/update_block_parity \
  --update_model "$MODEL_DIR/update_block_opset11.onnx" \
  --case_dir testdata/dpvo_onnx_mixed_smoke/update_small -x 0
```

若尚未建置這兩個 native binaries，先建置 host ORT，再在 runner 執行：

```bash
ORT="$RUNNER/../.."
(cd "$ORT" && bash build_fp16.sh host)
make -C "$RUNNER" -j3 PRECISION=fp16 CONFIG=Release CXX=g++ \
  TARGET_CXXFLAGS='-std=c++17 -O2 -g -MMD -MP' \
  TARGET_LDLIBS='-ldl -lpthread -latomic -lrt' \
  PARITY_LDLIBS='-ldl -lpthread -latomic -lrt' \
  TARGET_BUILD_DIR=build/native/fp16 BUILD_PATH="$ORT/build/fp16-host/Release" \
  patchify_parity update_block_parity
```

這两个 parity 命令可能因數值誤差回傳 1；這和「載入／推論失敗」不同，請讀比較報告。本輪原 tolerance 下的 FAIL 詳列在第 4 節，沒有自動放寬它。到 FPGA 時改用 RISC-V binary、相符 bitstream 與 `-x 2`。

### 2.2 在有 CUDA DPVO 的機器產生完整測資

需要原本 DPVO 的 PyTorch、Pillow／OpenCV、torch_scatter、lietorch、altcorr、fastba 等依賴，再加上 NumPy／ONNX。FP16 backend 的 NN 本身在 CPU，但選點、correlation、BA 仍需要 CUDA。先指定實際影像與 calibration：

```bash
NN_PRECISION=fp16 \
ONNX_MODEL_DIR="$MODEL_DIR" \
PYTHON_BIN=/path/to/dpvo/environment/bin/python \
IMAGES=/path/to/source/images \
CALIB=/path/to/calib.txt \
TESTDATA_ROOT="$RUNNER/testdata" \
bash "$DPVO/tools/gen_testdata/gen_testdata.sh" 0
```

MODE 0 產生兩組，MODE 1 只產生 component，MODE 2 只產生完整 tracker。輸出為：

```text
testdata/
├── dpvo_runner_parity_small_fp16/
│   ├── metadata.json
│   ├── patchify_small/
│   ├── correlation_small/
│   ├── update_small/
│   └── bundle_adjustment_small/
└── dpvo_python_fast_p16_fp16/
    ├── metadata.json
    ├── images/
    ├── calib.txt
    ├── centers_manifest.txt
    ├── bootstrap_depth_manifest.txt
    └── golden/
```

Shell 原有參數仍保留：component 使用 4 幀、256×144、每幀 8 patches；tracker 使用 32 幀、long edge ≤256、每幀 16 patches、seed=7。`p16` 指 patches 數，尾端 `_fp16` 才指精度。

需要調整尺寸／幀數或保存逐次 update cases，直接呼叫 Python：

```bash
/path/to/dpvo/environment/bin/python "$DPVO/tools/gen_testdata/generate_dpvo_python_testdata.py" \
  --nn-precision fp16 --onnx-model-dir "$MODEL_DIR" \
  --images /path/to/source/images --calib /path/to/calib.txt \
  --output-root "$RUNNER/testdata/my_tracker_fp16" \
  --frame-count 4 --max-long-edge 64 --patches-per-frame 8 \
  --seed 7 --dump-state --dump-update-parity-cases
```

建議先用小案例檢查環境，再生成 32 幀資料。FP16 模式的 `--weights` 不使用，無須指定。直接呼叫 component Python 時使用相同的 `--nn-precision fp16 --onnx-model-dir ...`，其他尺寸／影像參數見 `--help`。

完整 runner 推論時，要選同一份 `models/fp16/`、FP16 build，並傳入產物中的 `--workload-metadata`、`--centers-manifest`、`--bootstrap-depth-manifest`。不要只複製 images 而漏掉固定選點與深度資料。模型 SHA256 應與 metadata 一致；目前沒有在 C++ reader 自動強制核對這個 hash。

### 2.3 保留 FP32 與安全重跑

`NN_PRECISION` 預設為 fp32，既有 `gen_testdata.sh 0/1/2` 與 checkpoint 流程保留。FP32 目錄沒有 `_fp16` 後綴。兩個 Python CLI 的預設目錄分別仍是 `dpvo_runner_parity_small`、`dpvo_python_small`；shell 自己指定 tracker 的 `dpvo_python_fast_p16`。

不同精度不能共用已標記的 output root。FP16 模式也拒絕覆寫「非空且沒有 metadata」的目錄；若上次生成中斷，請選新目錄，或確認舊資料可刪後自行清理再重跑。相同精度的完整重跑可更新既有輸出。這個檢查不會自動刪除另一組資料。

## 3. 限制與假設

1. **精度政策相同，不代表硬體逐 bit 相同。** 加速算子的 reference 累加是 FP32，Gemmini PE 是 FP16；一般 reduction 順序也受 NumPy／BLAS 影響。不能用本工具證明 RTL bit parity。
2. **本次不是全 FP16 網路。** graph 中的 norm、Add、scatter 等區域維持 FP32；沒有修改部署模型、Cast 或 PE。
3. **ONNX reference 執行器版本有依賴。** 已驗證 ONNX 1.16.2，node loop 使用其 `rt_nodes_`／`rt_inits_`。升級後要重跑測試。外部權重資料格式（external data）明確拒絕，需先合併成單檔模型。
4. **只驗證目前兩個 opset11 graph 與四個算子 fixtures。** Conv reference 只支援 2D；scatter 只支援目前 DPVO 的 shape／index 規則。通用模型或新的 operator 需要另外驗證。
5. **完整 reference 並非全程 CPU-only。** 兩個原有產生器都仍需要 CUDA DPVO；獨立 NN smoke CLI 才不需要 PyTorch／CUDA。圖參考不需要 checkpoint，但影像、calibration 與 CUDA extensions 仍需使用者提供。
6. **參考生成時間不代表硬體效能。** NumPy im2col 需要暫存空間，完整 tracker 每次 NN 呼叫也會搬移 CPU/GPU 資料；本工具用於 golden，不用來量測 DPVO speedup。
7. **目前產生的是原有 parity／golden 格式。** 沒有增加原生 float16 檔案格式，也沒有替其他 correlation/update/patchify replay recorder 新增 FP16 模式。
8. **CPU 檔案寫入以目前 little-endian 環境驗證。** 小型 NN writer 明確輸出 little-endian；原有兩個 generator 沿用既有 writer，不宣稱 big-endian 相容。

## 4. 本輪驗證結果

2026-09-23，Python 3.10、NumPy 1.26.4、ONNX 1.16.2，OPENBLAS_NUM_THREADS=1：

- 原有與新增的 CPU-only Python tests：29 項通過。包含 FP16 捨入後再做 FP32 Add、FP32 累加與 half 累加差異、Conv bias、dtype／shape 拒絕、scatter、CLI 與輸出目錄保護。
- 四個既有 MatMul／Gemm／Conv fixtures：獨立 reference 與預先生成的 Y.bin 精確相同。
- 真實 feature 模型、不同 edge 數的 update 模型：執行成功，shape／dtype 正確、輸出 finite。
- 已實際生成 `testdata/dpvo_onnx_mixed_smoke`，兩個 native FP16 parity binaries 均能載入並完成推論／比較。
- **原 tolerance（atol=5e-4、rtol=1e-4）未全部通過。** 沒有放寬 tolerance 或把 FAIL 改標成功：

| 比較項目 | max absolute error | 結果 |
|---|---:|---|
| patchify fmap | 0.002746582 | FAIL |
| patchify imap | 0.001220703 | FAIL |
| patchify gmap | 0.002685547 | FAIL |
| patches／colors | 0 | PASS |
| update net | 0.015625 | FAIL |
| update delta | 0.05468750 | FAIL |
| update weight | 6.334812e-8 | PASS |

這些是獨立 FP32 累加 reference 與現有 ORT／FP16 CPU reference 的端到端差異；不能只憑總誤差把所有原因歸為 PE 捨入。需要的話，下一步可逐節點定位與制定模型品質容忍範圍。

本機沒有 PyTorch／CUDA DPVO，所以 **完整 component（含 BA）、完整 tracker、GPU adapter 實際執行與新資料的 FPGA 比較均未驗證**；單元測試通過不能替代這些檢查。這次也没有重新跑原先 FireSim smoke。

重跑 Python tests：

```bash
cd "$DPVO"
OPENBLAS_NUM_THREADS=1 DPVO_RUNNER_ROOT="$RUNNER" "$PY" \
  -m unittest discover -s tools/gen_testdata -p 'test_*.py' -v
```

不設 DPVO_RUNNER_ROOT 時，兩項需要 checkout 模型／fixtures 的測試會明確 skip；其餘 CPU-only tests 仍可執行。暫存建置與比較 log 位於 `/tmp/dpvo-mixed-reference/`，關鍵結果已記錄在本文，未建立另一個 review 資料夾。
