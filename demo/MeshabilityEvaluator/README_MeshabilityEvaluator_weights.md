# MeshabilityEvaluator — 找出評分權重 config.json

`MeshabilityEvaluator` 執行後會輸出一份 JSON（`-o metrics.json`），裡面包含一組描述
tet mesh 可六面體化程度的特徵指標（`AverageFrameFieldEnergy`、`NumberSingularEdges`、
`PercentageMeshableEdges` 等，完整清單見 `find_weights.py` 的 `FEATURE_KEYS`）。

`find_weights.py` 和 `predict.py` 這兩個腳本就是用這些已標註成功/失敗的 JSON 結果，
找出哪些特徵最能分辨「能不能六面體化」，並產生 `scorer_config.json` 供後續預測使用。

## 整體流程

```
1. 跑 MeshabilityEvaluator 收集多份 metrics.json
        ↓
2. 手動在每份 JSON 加上 "Success": true / false 標籤
        ↓
3. 把標註好的 JSON 放進同一個資料夾（如 training_data/）
        ↓
4. python find_weights.py training_data/ -o scorer_config.json
        ↓
5. python predict.py <新資料.json> -c scorer_config.json
```

## Step 1：收集訓練資料

對每個 tet mesh 跑一次 evaluator，輸出 JSON：

```bash
./build/Build/bin/MeshabilityEvaluator -i input.vtk -o demo/MeshabilityEvaluator/training_data/case_name.json
```

## Step 2：手動標註成功/失敗

在每份輸出的 JSON 裡加上一個布林欄位（預設欄位名為 `Success`，可用
`--label-key` 改名），標記這個 mesh 實際上六面體化是否成功：

```json
{
    "AverageFrameFieldEnergy": 0.0544,
    "NumberSingularEdges": 110,
    ...
    "Success": true
}
```

標註依據通常是：完整跑過 `HexMeshing` pipeline（或人工檢查）後，是否能產出可用的
hex mesh。`demo/MeshabilityEvaluator/training_data/` 已有現成範例可參考格式。

> 至少需要 2 筆資料，且成功/失敗都要各有幾筆，否則無法分析特徵的區分力。

## Step 3：用 find_weights.py 算出權重

```bash
python find_weights.py demo/MeshabilityEvaluator/training_data/ \
    -o demo/MeshabilityEvaluator/scorer_config.json \
    --top-k 6 \
    --label-key Success
```

參數說明：

| 參數 | 說明 | 預設 |
|---|---|---|
| `data_folder` | 已標註 JSON 檔案的資料夾 | （必填） |
| `-o, --output` | 輸出的 scorer config 路徑 | `scorer_config.json` |
| `--top-k` | 保留分離度最高的前 K 個特徵 | `6` |
| `--label-key` | JSON 裡代表成功/失敗的欄位名稱 | `Success` |

腳本做的事：

1. 讀取資料夾內所有 JSON，依 `Success` 分成成功組／失敗組。
2. 對每個特徵計算兩組平均值的「分離度」（兩組平均值差的絕對值 / 全體標準差）。
3. 取分離度最高的 `top-k` 個特徵，依分離度比例算出權重，寫入 `scorer_config.json`。

輸出格式（`scorer_config.json`）：

```json
[
    {
        "feature": "NumberSingularEdges",
        "success_mean": 898.67,
        "failure_mean": 6067.75,
        "direction": "lower_better",
        "weight": 0.186
    },
    ...
]
```

`direction` 表示該特徵數值越高還是越低代表更容易六面體化，`weight` 為正規化後的
加權比例（所有保留特徵的權重總和為 1）。

> ⚠️ 這份權重只是少量資料的統計觀察，不是嚴謹的機器學習結果。資料量少時很容易
> 出現巧合性的分離度，務必搭配領域知識（例如哪些特徵理論上就該影響可六面體化程度）
> 一起判斷，不要照單全收。

## Step 4：用 predict.py 驗證或預測新資料

```bash
# 預測單一檔案
python predict.py new_case.json -c demo/MeshabilityEvaluator/scorer_config.json

# 批次預測整個資料夾
python predict.py training_data/ -c scorer_config.json

# 調整成功門檻（預設 0.8）
python predict.py new_case.json -c scorer_config.json -t 0.75

# 用已知標籤驗證準確率
python predict.py training_data/ -c scorer_config.json --with-label
```

`predict.py` 會用每個特徵的 `success_mean`／`failure_mean` 中點作為 sigmoid 中心，
依 `weight` 加權平均出一個 0~1 的成功機率；高於門檻判定為「成功」，可用
`--with-label` 在已有標籤的資料上驗證準確率，藉此評估這份 `scorer_config.json`
好不好用，必要時回到 Step 1 補更多資料再重新跑一次。
