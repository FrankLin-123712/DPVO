# Experiment results and analysis

## Experiment methodology ? 
- 利用 [statistic_dpvo.py](DPVO/tools/statistic_dpvo/statistic_dpvo.py) 以及 [dpvo.py](DPVO/dpvo/dpvo.py) 產生 one-at-a-time sweep 圖表，幫助我們決定最終合適的Algorithmic parameter。 
- 實驗怎麼進行?
    - tunable parameters 包含`PATCHES_PER_FRAME`, `PATCH_LIFETIME`, `REMOVAL_WINDOW`, `OPTIMIZATION_WINDOW`, `BA_ITERATIONS`, `{H,W} pair`
    - 每次跑的時候，只選定一個 tunable parameters，並且選定合理的離散數值點，以default.yaml 為初始數值，並逐步減少。
        - step 1 : 利用 [statistic_dpvo.py](DPVO/tools/statistic_dpvo/statistic_dpvo.py) 得到 total ops 以及 total mem。Per module 的做統計就好。 
        - step 2 : 利用 [dpvo.py](DPVO/dpvo/dpvo.py) 得到在 EuRoC dataset 上的 Trajectory error (m)。 
        - step 3 : 最終結果輸出在 [statistic_result](DPVO/tools/statistic_dpvo/statistic_result/) 資料夾，並用 .csv file 紀錄。 
        - step 4: 最後再把這些結果做成圖表，同樣放進 [statistic_result](DPVO/tools/statistic_dpvo/statistic_result/)。

- EuRoC dataset 請參考此網站 https://projects.asl.ethz.ch/datasets/euroc-mav/要選用哪些 sequence 可以查看 dpvo 論文中的 table 2。 要記得把dataset 下載到 [datasets](DPVO/datasets/)

- 圖表解釋 : x-axis 是 tunable parameters 的離散數值，由原本default.yaml中的預設值往下降。
y-axis 有三個刻度軸，分別是 total ops(number of count), total mem(bytes), error(ATE(m))。可以用三種不同顏色標註。

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
- BA iteration sweep - No impact
![](./statistic_result/sweep_plots/ba_iterations_sweep.svg)
- Image size sweep - 320x416
![](./statistic_result/sweep_plots/image_size_sweep.svg)
- Optimization window sweep - ATE knee point 6
![](./statistic_result/sweep_plots/optimization_window_sweep.svg)
- Patch number per frame sweep - ATE knee point 64
![](./statistic_result/sweep_plots/patches_per_frame_sweep.svg)
- Patch lifetime sweep - ATE knee point 7
![](./statistic_result/sweep_plots/patch_lifetime_sweep.svg)
- Removal window sweep - ATE knee point 14
![](./statistic_result/sweep_plots/removal_window_sweep.svg)

### Analysis
- 從圖表可以看出對於資源開銷最有影響的就是 `PATCH_LIFETIME`, `PATCHES_PER_FRAME`, `REMOVAL_WINDOW`，透過 Visualization 可以得到一個 knee.yaml 的 algorithm parameters，但由於我們在實驗時，都是只動一個變數，無法確定當多變數都往較少運算資源的配置改動時，是否會彼此互相影響到置 accuracy 大幅下降，我們嘗試去比較 fast.yaml(`480x640`) 與 knee.yaml(`320x416`) 之間的 ATE(m)。實驗之後可以得到 fast 配置的 ATE 為 0.137672(m)，而 knee 配置的 ATE 為 0.190128(m)，還是 fast 較佳。
- 如果 feature point 數量變少的情況下，我是否能夠提高 BA_ITERATION 來補回精準度 ? NO, 從圖片就可以看多出，即使我增加 BA_ITERATION 的次數也沒辦法保證誤差會下降。
![](./statistic_result/fast_ba_iterations_2_to_20/sweep_plots/ba_iterations_sweep.svg)
- 確認 default.yaml 配置下在 Kitti dataset 上的誤差是多少，並以DPDM的誤差值當做是我們可容許的誤差值上限。
    - 實驗5種配置在 Kitti dataset 上的誤差值。

| Config | resolution | Total_Ops | Total_mem (bytes) | $t_{err}\%$@Kitti | $r_{err}\%$ @Kitti |
| --- | ---: | ---: | ---: |---:| ---: |
|DFVO     | Native |      |      |      3.97 |     0.77 | 
|DPDM     | Native |      |      |      3.39 |     1.09 | 
|`default`| Native |      |      | 15.029336 | 0.222875 | 
|`fast`   | Native |      |      | 15.485609 | 0.233350 | 
|`fast`   | Low    |      |      | 13.793390 | 0.257918 | 
|`knee`   | Native |      |      | 15.782408 | 0.249360 | 
|`knee`   | Low    |      |      | 14.335763 | 0.262674 | 

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