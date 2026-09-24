# Experiment results and analysis

> 歷史實驗紀錄：下列數字按原紀錄保留，本次未重跑 GPU 評估；本機缺少對應的完整 CSV／evaluator JSON 與圖檔，不能重新核對其模型、參數及 precision。`statistic_result/` 是生成的實驗輸出，Git 不保存該目錄。現在的操作方式請見 [工具 README](README.md)。

## Experiment methodology ?
- 利用 [statistic_dpvo.py](statistic_dpvo.py) 搭配 [sweep_dpvo.py](sweep_dpvo.py) 與 [evaluate_euroc_sweep.py](evaluate_euroc_sweep.py) 產生 one-at-a-time sweep 圖表，幫助我們決定最終合適的Algorithmic parameter。
- 實驗怎麼進行?
    - tunable parameters 包含`PATCHES_PER_FRAME`, `PATCH_LIFETIME`, `REMOVAL_WINDOW`, `OPTIMIZATION_WINDOW`, `BA_ITERATIONS`, `{H,W} pair`
    - 每次跑的時候，只選定一個 tunable parameters，並且選定合理的離散數值點，以 default.yaml 為基準，每次只改一個參數。BA_ITERATIONS 是例外：預設 sweep 從 20 降到 config 的 2。
        - step 1 : 利用 [statistic_dpvo.py](statistic_dpvo.py) 得到 total ops 以及 total mem。Per module 的做統計就好。
        - step 2 : 透過 [evaluate_euroc_sweep.py](evaluate_euroc_sweep.py) 呼叫 DPVO tracker，得到在 EuRoC dataset 上的 Trajectory error (m)。
        - step 3 : 最終結果輸出在 [statistic_result](statistic_result/) 資料夾，並用 .csv file 紀錄。
        - step 4: 最後再把這些結果做成圖表，同樣放進 [statistic_result](statistic_result/)。

- EuRoC dataset 請參考此網站 https://projects.asl.ethz.ch/datasets/euroc-mav/要選用哪些 sequence 可以查看 dpvo 論文中的 table 2。 要記得把dataset 下載到 [datasets](../../datasets/)

- 圖表解釋 : x-axis 是 tunable parameters 的離散數值，由原本default.yaml中的預設值往下降。
BA_ITERATIONS 的 sweep 起點為 20，其餘預設點見下表。y-axis 有三個刻度軸，分別是 total ops(number of count), total mem(bytes), error(ATE(m))。可以用三種不同顏色標註。

## Experiment results
### Sweep discrete points
- `PATCHES_PER_FRAME`: 96, 80, 64, 48, 32
- `PATCH_LIFETIME`: 13, 11, 9, 7, 5
- `REMOVAL_WINDOW`: 22, 18, 14, 10
- `OPTIMIZATION_WINDOW`: 10, 8, 6, 4
- `BA_ITERATIONS`: 20, 18, 16, 14, 12, 10, 8, 6, 4, 2
- `{H,W}`: 480x640, 384x512, 320x416, 240x320, 192x256

### Resource reduction by decreasing algorithm parameters
| parameter | 最低 candidate | 相對 sweep 起點的 ops 降幅 | 相對 sweep 起點的 mem 降幅 | ATE 狀態 |
| --- | ---: | ---: | ---: | --- |
| `PATCHES_PER_FRAME` | `32` | 62.4% | 61.1% | 完成 |
| `PATCH_LIFETIME` | `5` | 56.4% | 55.6% | 完成 |
| `REMOVAL_WINDOW` | `10` | 56.0% | 55.4% | 完成 |
| `OPTIMIZATION_WINDOW` | `4` | 0.0% | 0.0% | 完成 |
| `BA_ITERATIONS` | `2` | 0.3% | 2.1% | 完成 |
| `{H,W}` | `192x256` | 5.3% | 6.9% | 完成 |


### Visualization

下列為原實驗圖檔的位置；本 checkout 未保留這些產物，需從原實驗備份還原。圖說為當時判讀，不是本次重新驗證的結論。
- BA iteration sweep - No impact
原圖：`./statistic_result/sweep_plots/ba_iterations_sweep.svg`
- Image size sweep - 320x416
原圖：`./statistic_result/sweep_plots/image_size_sweep.svg`
- Optimization window sweep - ATE knee point 6
原圖：`./statistic_result/sweep_plots/optimization_window_sweep.svg`
- Patch number per frame sweep - ATE knee point 64
原圖：`./statistic_result/sweep_plots/patches_per_frame_sweep.svg`
- Patch lifetime sweep - ATE knee point 7
原圖：`./statistic_result/sweep_plots/patch_lifetime_sweep.svg`
- Removal window sweep - ATE knee point 14
原圖：`./statistic_result/sweep_plots/removal_window_sweep.svg`

### Analysis
- 從圖表可以看出對於資源開銷最有影響的就是 `PATCH_LIFETIME`, `PATCHES_PER_FRAME`, `REMOVAL_WINDOW`，透過 Visualization 可以得到一個 knee.yaml 的 algorithm parameters，但由於我們在實驗時，都是只動一個變數，無法確定當多變數都往較少運算資源的配置改動時，是否會彼此互相影響到置 accuracy 大幅下降，我們嘗試去比較 `fast`（320×416）與 `knee_p64`（320×416） 之間的 ATE(m)。實驗之後可以得到 fast 配置的 ATE 為 0.137672(m)，而 `knee_p64` 配置的 ATE 為 0.190128(m)，還是 fast 較佳。
- 如果 feature point 數量變少的情況下，我是否能夠提高 BA_ITERATION 來補回精準度 ? NO, 從圖片就可以看多出，即使我增加 BA_ITERATION 的次數也沒辦法保證誤差會下降。
原圖：`./statistic_result/fast_ba_iterations_2_to_20/sweep_plots/ba_iterations_sweep.svg`
- 確認 default.yaml 配置下在 Kitti dataset 上的誤差是多少，並以DPDM的誤差值當做是我們可容許的誤差值上限。
    - 實驗5種配置在 Kitti dataset 上的誤差值。

| Config | resolution | Total_Ops | Total_mem (bytes) | $t_{err}\%$@Kitti | rotation（原紀錄，單位待核對） |
| --- | ---: | ---: | ---: |---:| ---: |
|DFVO     | Native |      |      |      3.97 |     0.77 |
|DPDM     | Native |      |      |      3.39 |     1.09 |
|`default`| Native |      |      | 15.029336 | 0.222875 |
|`fast`   | Native |      |      | 15.485609 | 0.233350 |
|`fast`   | Low    |      |      | 13.793390 | 0.257918 |
|`knee`   | Native |      |      | 15.782408 | 0.249360 |
|`knee`   | Low    |      |      | 14.335763 | 0.262674 |

KITTI evaluator 現在明確輸出 translation percent、rotation deg/m 與 deg/100m，rotation 不是百分比。原表未標明選用了哪個 rotation 欄位，故不能直接與 DFVO／DPDM 的數字比較；需先核對原始來源、對齊與 scale 設定。

`knee_p64`、`knee_p16` 是歷史實驗名稱，目前沒有同名 checked-in YAML。若要重跑，必須還原原始 config；不能直接視為目前 `config/knee.yaml`。下方 0.137672 與 0.190128 對應的是 `fast` 與 `knee_p64` 在 320×416 的紀錄。

- EuRoC evaluation uses all 11 sequences, stride=2, trials=3, ATE is averaged from per-sequence median ATE.

| Config   | resolution | Total_Ops | Total_mem (bytes) | ATE(m)@EuRoC |
| ---      | ---:      |   ---:   | ---:     |  ---:    |
| DFVO     |           |          |          |          |
| DPDM     | 480x640   |          |          | 3.263949 |
|`default` | 480x640   | 386.897G |  21.790G |    0.105 |
|`fast`    | 480x640   | 134.592G |   7.798G |    0.129 |
|`fast`    | 320x416   | 120.785G |   6.831G | 0.137672 |
|`fast`    | 240x320   | 116.318G |   6.519G | 0.302300 |
|`fast`    | 192x256   | 114.125G |   6.365G | 0.460346 |
|`fast_p16`| 480x640   |  61.167G |   3.750G | 0.293785 |
|`knee_p64`| 480x640   | 109.126G |   6.335G | 0.182255 |
|`knee_p64`| 320x416   |  95.320G |   5.404G | 0.190128 |
|`knee`    | 480x640   |  66.771G |   3.998G | 0.254978 |
|`knee`    | 320x416   |  52.965G |   3.067G | 0.290327 |
|`knee`    | 240x320   |  48.498G |   2.765G | 0.363238 |
|`knee`    | 192x256   |  46.305G |   2.617G | 0.561292 |
|`knee_p16`| 480x640   |  45.593G |   2.829G | 0.365000 |
