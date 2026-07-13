## Our DPVO deployment on Gemmini-SoC case
- 將 ISC flow 套用到 DPVO-on-Gemmini，以下是初步把 ISC 套用在 DPVO-on-Gemmini 的修改版本。
- 問題如果很複雜，就先做假設，將情況簡化，先處理簡化版本，接著思考如果假設拿掉，我改怎麼改變當前作法 ? 

### Algorithm & hardware co-design
1. 定義應用場景，並且將場景需求轉化為硬體指標 resource( power & chip area) 和 performance(accuracy, latency, throughput) :
    - Latency : ??
    - Throughput : 90 pose/sec
    - Power : 1 ~ 2 w
    - Error : No error increments
2. 找出可變動的設計參數，找到 Design Space : 
    - DPVO 
        - <font color="#F54927"> Q1 : DPVO 有哪些可變動參數? </font>
    - RoCC-Gemmini System 
        - <font color="#F54927"> Q2 : RoCC-Gemmini 有哪些可變動參數? </font>
3. 依照 DSE 流程找出使用最少資源但可以符合效能需求的 Design point : <font color="#F7A004"> 把大問題拆分成 algorithm + hardware implementation 兩個子問題處理</font>
    > - 批判1: DPVO 這類型演算法，前後端耦合度高，真的可以用這種方式去找到對的 design point ? 耦合度高的情況下，用這種方式會遇到甚麼問題 ? 有哪些情況下，我們可能反而找到不對的 design point ? 
    1. 選擇一種硬體的組態 <font color="#F7A004"> $h$ = (CPU + Gemmini), (CPU + Gemmini-BA-adaptive)</font>
    2. 找到 minimum-resource design d = (h,a,i,p) 但符合效能指標
        - 在保持誤差 Error 不增加的情況下，<font color="#F7A004">找到 algorithm $a$ 以及 algorithm $p_a$ 使其最小化 operatios 以及 memory amounts</font>，以此降低 power。
            > - 批判2: 僅僅只用 ops 與 memory amount 來反應系統的功耗，太過粗淺，DPVO上有大量的 data movement、irregular memory access、atomic add (synchronization)，這些都是會對系統的 latency、throughput 以及 power 造成極大影響，但是卻沒有被 proxy metric 很好的捕捉到。
            > - 批判3: 論文在這裡直接使用 execution time 當作 operations 的 proxy metrics，但是 execution time 是 Compute + Data movement + Synchronization 綜合運作的結果，並不能把 execution time 作為反映 operation 數量的指標，倒果為因。
            - <font color="#F54927"> Q3 : 我們不應該只有將 operatios, memory amount 作為評估 power 的指標 ? 還有哪些指標也是重要的 ? </font> data movement, memor access of each memory hierarchy, etc...
            - <font color="#F54927"> Q4 : 我們該怎麼評估不同 algorithm parameters 對 power 的影響 ? </font> 利用 [energy cost model](/LXFUDBp4RaWGzsA5hWUrbA) 將 algorithm parameters 轉化為系統的 computation, data movement, memory access 再統計一整個運算總共消耗的 energy 以此得出 power。 ![image](https://hackmd.io/_uploads/HyNuGVYQMx.png)
            - <font color="#F54927"> Q5 : 該怎麼接著用 [energy cost model](/LXFUDBp4RaWGzsA5hWUrbA) 來決定最終的 algorithm parameters ? </font> 利用 cost model 我們可以找出在accuracy 不掉的情況下，最少 power 的 $p_a$ 組合
        - 固定 algorithm $a$, $p_a$，<font color="#F7A004">透過實作方法 $i$ 以及硬體參數 $p_i$ 將系統 Throughput 補回</font>
            - <font color="#F54927">Q6: DPVO 的哪些 workload 可以放到 Gemmini？哪些不能？</font> 分析DPVO運算模組，建立 mapping feasibility table，決定每個模組適合在哪一個硬體上面運算。
            :::spoiler
            - 回答以下問題:
                1. 它是 dense compute 還是 sparse / irregular compute？
                2. 它的資料存取是 regular 還是 irregular？
                3. 它有沒有 dynamic shape / dynamic graph？
                4. 它主要需要 int8 / fp16 / fp32？
                5. 它是大矩陣運算，還是很多小 kernel？
                6. 它需要頻繁 CPU 控制嗎？
                7. 它的輸入輸出資料量大不大？
            
            | DPVO workload              | 運算型態                             | Gemmini mapping  | CPU mapping | 可能需要新硬體嗎？                     | 主要疑問                                   |
            | -------------------------- | -------------------------------- | ---------------- | ----------- | ----------------------------- | -------------------------------------- |
            | Feature / context encoder  | Conv / dense tensor              | high feasibility | 可跑但慢        | 不一定                           | Gemmini utilization 多少？                |
            | Patch extraction           | bilinear sampling / gather       | low              | high        | maybe                         | data movement 是否大？                     |
            | Correlation lookup         | local dot product + sampling     | medium/low       | 可跑但可能慢      | maybe correlation accelerator | 能不能 batch 成 Gemmini-friendly GEMM？     |
            | Temporal conv / transition | small dense layer / MLP          | medium           | high        | 不一定                           | kernel 太小會不會 Gemmini overhead 很大？      |
            | Soft aggregation           | graph aggregation                | low/medium       | high        | maybe                         | irregular edge access 是否造成 bottleneck？ |
            | Factor head                | MLP                              | medium/high      | high        | 不一定                           | 是否值得 offload？                          |
            | Differentiable BA          | GN / Schur / sparse block ops    | low for Gemmini  | baseline    | BA accelerator candidate      | CPU 是否變瓶頸？                             |
            | Graph management           | control / dynamic data structure | low              | high        | 不適合 accelerator               | CPU overhead 多大？                       |
            :::
            - 透過 workload analysis 我們可以知道一張圖的運算過程中，在最理想情況下以及已知的 algorithm parameters $P_a$，如果要達到特定Performance(throughput/latency)，硬體參數 $p_h$ 的 lower bound，以此決定 RoCC-Gemmini SoC 的 configuration。 
            - <font color="#F54927"> Q7 : 我們該怎麼重新設計 RoCC-Gemmini SoC 的 microarchitecture 使其可以針對 DPVO 加速 ? </font> 找出系統 bottleneck 擬定因應策略，但要怎麼知道系統 bottleneck 在哪裡呢 ?
            - <font color="#F54927"> Q8 : 如何在還沒有跑 FireSim Profiling 之前，就知道系統的 bottleneck 可能出現在哪裡 ? </font> 利用 [SimPy event simulation](/6hRfUB_NSAu27eHcUyZ8QQ) 去估計 inference latency, system throughput, bottleneck analysis，並且用 firesim 給予的真正的數值去調整 Simpy simulation 讓他更準。
    3. 嘗試驗證 Design point d = (h,a,i,p) 可以符合硬體規格。
        - <font color="#F54927">如果 design point 不滿足目標，我要回到哪一步？</font> Failure diagnosis and feedback path
4. 實際在 FireSim 上進行驗證