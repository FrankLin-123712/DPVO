
# Energy cost model

## Problem definition

- 根據 [Deep-patch visual odometry](https://arxiv.org/pdf/2208.04726) ([DPVO-github-repo](https://github.com/princeton-vl/DPVO)) 的演算法參數 $P_a$ 來估計它在 [RoCC-Gemmini SoC](https://github.com/ucb-bar/gemmini) 上的 Operations, memory access (bytes) of regfile, scratch pad, L1/L2 cache, DRAM, data movement (bytes)，並依據 Energy cost model 估計一張圖片計算所消耗的 Energy，最後得到 Dynamic Power。
- Input :

  - $P_a$ : DPVO 中可以變動的參數
  - $H$ : 暫時以 Rocket Chip + Gemmini 的架構為主
  - $P_h$ : RoCC-Gemmini SoC 中可以變動的參數
    - <font color="#F54927"> 該做怎樣的假設才是合理的? 各個memory hierarchy 的 memory access 取決於硬體的 scratch pad, L1,L2 cache 還有 DRAM 大小，以及它們之間的拓樸結構 </font>
- Output :

  - Dynamic Power (W)

## Purpose

- Energy cost model 可以為後續我們在評估如何降低功耗上提供取捨的方向，我們希望在保持精準度的情況下，盡可能找到一組 $P_a$ 可以讓功耗最低。
