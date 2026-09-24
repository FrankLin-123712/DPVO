# DPVO One-at-a-Time Algorithmic Parameter Sweep 報告

> 這是先前執行留下的歷史報告，不是本次文件檢查重新產生的結果。本機未保留對應的 `sweep_summary.csv` 與完整 evaluator JSON，以下數值按原紀錄保留，無法重新核對。舊 report template 曾固定填入預設參數與「已完成」文字；新的產生器已改用實際設定、執行順序與完成數。再次執行 sweep 會覆寫本檔。

## 範圍

這份報告由 `tools/statistic_dpvo/sweep_dpvo.py` 產生。Sweep 方式是從
`config/default.yaml` 出發，每次只改動一個 tunable parameter。對
`{H,W}` 而言，基準點定義為
`480x640`，也就是 `statistic_dpvo.py` 使用的預設輸入尺寸。
`BA_ITERATIONS` 依本次需求改為由 `20` 逐步下降到 default `2`。

EuRoC sequences 依照 DPVO 論文 Table 2，以及本 repository 的
`evaluate_euroc.py`：MH_01_easy, MH_02_easy, MH_03_medium, MH_04_difficult, MH_05_difficult, V1_01_easy, V1_02_medium, V1_03_difficult, V2_01_easy, V2_02_medium, V2_03_difficult。

## 重現方式

以下命令從 DPVO repository 根目錄執行，示範 BA_ITERATIONS 子集合；原始 run 的其他選項未完整保存，不能視為精確重現命令。先準備資料與 runtime 環境：

```bash
python3 tools/statistic_dpvo/download_euroc.py
conda activate dpvo
```

執行 BA_ITERATIONS sweep：

```bash
python3 tools/statistic_dpvo/sweep_dpvo.py --run-eval --trials 3 --parameters BA_ITERATIONS
```

產生的輸出：

- `statistic_result/generated_configs/*.yaml`
- `statistic_result/per_module/*.json`
- `statistic_result/per_module/*.csv`
- `statistic_result/sweep_summary.csv`
- `statistic_result/module_summary.csv`
- `statistic_result/sequence_errors.csv`
- `statistic_result/sweep_plots/*_sweep.svg`

## 當時文件列出的預設 Sweep 離散點（結果表僅含 BA_ITERATIONS）

- `PATCHES_PER_FRAME`: 96, 80, 64, 48, 32
- `PATCH_LIFETIME`: 13, 11, 9, 7, 5
- `REMOVAL_WINDOW`: 22, 18, 14, 10
- `OPTIMIZATION_WINDOW`: 10, 8, 6, 4
- `BA_ITERATIONS`: 20, 18, 16, 14, 12, 10, 8, 6, 4, 2
- `{H,W}`: 480x640, 384x512, 320x416, 240x320, 192x256

## 目前結果摘要

| parameter | 最後 candidate（原紀錄） | 相對 sweep 起點的 ops 降幅 | 相對 sweep 起點的 mem 降幅 | ATE 狀態 |
| --- | ---: | ---: | ---: | --- |
| `BA_ITERATIONS` | `20` | -0.3% | -1.9% | 完成 |

原報告標記 ATE 已完成；目前沒有對應 artifacts 可確認各 candidate 的完成情況。表中負降幅代表 endpoint 比起點增加，不能解讀為 workload 降低。

所有 sweep plots 針對同一個 metric 共用同一組 y-axis range：藍線的 total ops 軸
在所有圖一致，橘色虛線的 total mem 軸在所有圖一致，ATE 軸也會在所有圖一致。若尚未
執行 ATE evaluation，綠色虛線只代表 ATE pending；等 `--run-eval` 產生 `ate_m`
後會改畫實際 ATE 曲線。

## 解讀限制

目前結果表僅含 BA_ITERATIONS，不能據此推論其他參數的結果。上方預設點順序為 20 到 2，但原表 endpoint 是 20，兩者不一致；保留原數值供追查，需以原 CSV 確認實際順序後才能計算正確降幅。

Ops／memory 是 estimator 的固定 layer-boundary accounting，不是硬體量測。Correlation accounting 尚未反映 runner 目前先算整數格點 dot、再插值 scalar correlation 的實作。參數組合後的 accuracy 需另跑 evaluator，不能由單參數 sweep 推定。

目前工具操作與結果欄位請見 [README](README.md)。
