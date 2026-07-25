# DPVO One-at-a-Time Algorithmic Parameter Sweep 報告

## 範圍

這份報告由 `tools/statistic_dpvo/sweep_dpvo.py` 產生。Sweep 方式是從
`config/default.yaml` 出發，每次只改動一個 tunable parameter。對
`{H,W}` 而言，基準點定義為
`480x640`，也就是 `statistic_dpvo.py` 使用的預設輸入尺寸。
`BA_ITERATIONS` 依本次需求改為由 `20` 逐步下降到 default `2`。

EuRoC sequences 依照 DPVO 論文 Table 2，以及本 repository 的
`evaluate_euroc.py`：MH_01_easy, MH_02_easy, MH_03_medium, MH_04_difficult, MH_05_difficult, V1_01_easy, V1_02_medium, V1_03_difficult, V2_01_easy, V2_02_medium, V2_03_difficult。

## 重現方式

先準備資料與 runtime 環境：

```bash
python3 DPVO/tools/statistic_dpvo/download_euroc.py
conda activate dpvo
```

執行完整 sweep：

```bash
python3 DPVO/tools/statistic_dpvo/sweep_dpvo.py --run-eval --trials 3 --parameters BA_ITERATIONS
```

產生的輸出：

- `statistic_result/generated_configs/*.yaml`
- `statistic_result/per_module/*.json`
- `statistic_result/per_module/*.csv`
- `statistic_result/sweep_summary.csv`
- `statistic_result/module_summary.csv`
- `statistic_result/sequence_errors.csv`
- `statistic_result/sweep_plots/*_sweep.svg`

## Sweep 離散點

- `PATCHES_PER_FRAME`: 96, 80, 64, 48, 32
- `PATCH_LIFETIME`: 13, 11, 9, 7, 5
- `REMOVAL_WINDOW`: 22, 18, 14, 10
- `OPTIMIZATION_WINDOW`: 10, 8, 6, 4
- `BA_ITERATIONS`: 20, 18, 16, 14, 12, 10, 8, 6, 4, 2
- `{H,W}`: 480x640, 384x512, 320x416, 240x320, 192x256

## 目前結果摘要

| parameter | 最低 candidate | 相對 sweep 起點的 ops 降幅 | 相對 sweep 起點的 mem 降幅 | ATE 狀態 |
| --- | ---: | ---: | ---: | --- |
| `BA_ITERATIONS` | `20` | -0.3% | -1.9% | 完成 |

ATE 狀態：已完成

所有 sweep plots 針對同一個 metric 共用同一組 y-axis range：藍線的 total ops 軸
在所有圖一致，橘色虛線的 total mem 軸在所有圖一致，ATE 軸也會在所有圖一致。若尚未
執行 ATE evaluation，綠色虛線只代表 ATE pending；等 `--run-eval` 產生 `ate_m`
後會改畫實際 ATE 曲線。

## 分析

Static estimator 顯示，當各參數逐步下降時，logical workload 符合預期地下降。
`PATCHES_PER_FRAME`、`PATCH_LIFETIME` 與 `REMOVAL_WINDOW` 會直接降低 active
factor count，因此會同時影響 update、correlation 與 BA-heavy modules。
`OPTIMIZATION_WINDOW` 主要縮小 BA 中 free pose 的維度，所以對 front-end
neural-network workload 的影響較小，但仍可能影響 trajectory consistency。
新的 `BA_ITERATIONS` sweep 從 `20` 下降到 `2`；這能量化 solver refinement 次數
對 BA workload 的線性影響，也能在後續 ATE 補齊時判斷 iteration 是否有 accuracy
收益。較小的 `{H,W}` 會降低 feature extraction 與
correlation traffic，但也會改變輸入影像訊號，並可能和 patch selection 產生強交互作用。

## 候選 Algorithmic Parameter Sets P_a

ATE 補齊後，建議先驗證下列候選組合：

- `P_a_default`: `PATCHES_PER_FRAME=96`, `PATCH_LIFETIME=13`,
  `REMOVAL_WINDOW=22`, `OPTIMIZATION_WINDOW=10`, `BA_ITERATIONS=2`,
  `H,W=480x640`.
- `P_a_balanced`: `PATCHES_PER_FRAME=64`, `PATCH_LIFETIME=11`,
  `REMOVAL_WINDOW=18`, `OPTIMIZATION_WINDOW=8`, `BA_ITERATIONS=2`,
  `H,W=384x512`.
- `P_a_aggressive`: `PATCHES_PER_FRAME=48`, `PATCH_LIFETIME=9`,
  `REMOVAL_WINDOW=14`, `OPTIMIZATION_WINDOW=6`, `BA_ITERATIONS=2`,
  `H,W=320x416`.

最終選擇規則：保留 EuRoC average ATE 增幅仍在 project tolerance 內的 candidates，
再從這些 survivors 中選擇 total ops / total memory 最低的點。在目前尚未補齊 ATE
前，`P_a_balanced` 是較適合作為第一個 combined candidate 的保守選擇，因為它避開
最容易影響 accuracy 的變更（過低 `BA_ITERATIONS` 與過低 image size），同時仍能降低
factor-graph size。
