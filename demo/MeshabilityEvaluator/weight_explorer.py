"""
小資料權重探索工具
==================
給定少量已標註資料，協助你「肉眼判斷」哪些特徵有區分力，
並提供半自動的權重建議。

不是真的在「訓練」，而是在輔助你做決策。
"""

import math
import statistics
from typing import List, Dict, Tuple


FEATURE_KEYS = [
    "AverageFrameFieldEnergy", "FrameFieldEnergy",
    "LengthComplexSingularEdges", "MinimumDihedralAngle",
    "NumberComplexSingularEdges", "NumberEdges", "NumberFaces",
    "NumberFeatureEdges", "NumberSingularEdges",
    "NumberSingularNodes", "NumberSingularVertices",
    "NumberTetraCells", "NumberVertices", "NumberZipperNodes",
    "PercentageMeshableEdges", "PercentageMeshableVertices",
]


def analyze_separation(samples: List[Dict], label_key: str = "Success") -> List[Dict]:
    """
    對每個特徵計算「分離度」：成功組與失敗組的平均距離 / 整體標準差。
    分離度越大，這個特徵越有區分力。
    """
    pos = [s for s in samples if s[label_key]]
    neg = [s for s in samples if not s[label_key]]

    if not pos or not neg:
        print("⚠️  警告：你的資料只有單一類別，無法分析區分力。")
        print(f"   成功={len(pos)} 筆，失敗={len(neg)} 筆")
        print("   建議：至少收集 2~3 筆失敗案例再進行分析。\n")
        return []

    results = []
    for key in FEATURE_KEYS:
        pos_vals = [s[key] for s in pos]
        neg_vals = [s[key] for s in neg]

        mean_pos = statistics.mean(pos_vals)
        mean_neg = statistics.mean(neg_vals)

        # 整體標準差（避免除以 0）
        all_vals = pos_vals + neg_vals
        std = statistics.stdev(all_vals) if len(all_vals) > 1 else 1.0
        if std == 0:
            std = 1.0

        # Cohen's d 風格的分離度
        separation = abs(mean_pos - mean_neg) / std

        results.append({
            "feature": key,
            "mean_success": mean_pos,
            "mean_fail": mean_neg,
            "separation": separation,
            "direction": "higher_better" if mean_pos > mean_neg else "lower_better",
        })

    results.sort(key=lambda r: r["separation"], reverse=True)
    return results


def suggest_weights(analysis: List[Dict], top_k: int = 5) -> Dict[str, float]:
    """
    根據分離度，建議前 K 個特徵的權重（其餘設為 0）。
    權重 = 分離度 / 總分離度，使其加總為 1。
    """
    if not analysis:
        return {}
    top = analysis[:top_k]
    total_sep = sum(r["separation"] for r in top)
    if total_sep == 0:
        return {r["feature"]: 1.0 / len(top) for r in top}
    return {r["feature"]: r["separation"] / total_sep for r in top}


def print_report(samples: List[Dict], label_key: str = "Success"):
    print("=" * 70)
    print(f"資料總覽：共 {len(samples)} 筆")
    pos_count = sum(1 for s in samples if s[label_key])
    print(f"  成功: {pos_count} 筆，失敗: {len(samples) - pos_count} 筆")
    print("=" * 70)

    analysis = analyze_separation(samples, label_key)
    if not analysis:
        return

    print("\n各特徵區分力（由高到低）：\n")
    print(f"{'特徵名稱':<35} {'成功平均':>12} {'失敗平均':>12} {'分離度':>8}")
    print("-" * 70)
    for r in analysis:
        print(f"{r['feature']:<35} {r['mean_success']:>12.4f} "
              f"{r['mean_fail']:>12.4f} {r['separation']:>8.3f}")

    print("\n" + "=" * 70)
    print("建議權重（取分離度最高的前 6 名）：")
    print("=" * 70)
    weights = suggest_weights(analysis, top_k=6)
    for feat, w in weights.items():
        direction = next(r["direction"] for r in analysis if r["feature"] == feat)
        print(f"  {feat:<35} 權重 {w:.3f}  方向: {direction}")

    print("\n💡 重要提醒：")
    print("   - 上述「建議權重」只是統計上的觀察，不是真正的學習結果")
    print("   - 5 筆資料的觀察極可能是巧合，務必用領域知識最終判斷")
    print("   - 若某特徵在成功組變化很大（例如 0.001 ~ 1000），")
    print("     代表它可能不是關鍵因素，不要因為平均值差異大就採信")


# ============================================================
# 成功機率預測器（基於 6 個核心特徵 + sigmoid 加權）
# ============================================================

# 每個特徵的參數來自分析結果（成功/失敗平均值 + 分離度作為原始權重）
SCORER_CONFIG: List[Dict] = [
    {
        "feature": "NumberFeatureEdges",
        "success_mean": 495.6,
        "failure_mean": 3777.0,
        "direction": "lower_better",
        "weight": 0.944,
    },
    {
        "feature": "NumberSingularEdges",
        "success_mean": 925.0,
        "failure_mean": 5007.6,
        "direction": "lower_better",
        "weight": 0.937,
    },
    {
        "feature": "PercentageMeshableEdges",
        "success_mean": 1.0,
        "failure_mean": 0.9999887486734098,
        "direction": "higher_better",
        "weight": 0.897,
    },
    {
        "feature": "NumberZipperNodes",
        "success_mean": 25.6,
        "failure_mean": 136.2,
        "direction": "lower_better",
        "weight": 0.892,
    },
    {
        "feature": "MinimumDihedralAngle",
        "success_mean": 87.5338,
        "failure_mean": 80.8752,
        "direction": "higher_better",
        "weight": 0.879,
    },
    {
        "feature": "NumberSingularNodes",
        "success_mean": 62.0,
        "failure_mean": 477.4,
        "direction": "lower_better",
        "weight": 0.768,
    },
]


def _sigmoid(x: float) -> float:
    """數值穩定的 sigmoid。"""
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def score_sample(sample: Dict, config: List[Dict] = None) -> float:
    """
    給定一筆資料，輸出成功機率（0~1）。

    原理：
      對每個特徵，以成功/失敗平均值的中點為 sigmoid 中心，
      以兩組距離的 1/4 為斜率單位，映射到 0~1 的成功傾向分數，
      再依分離度加權平均。

    回傳值接近 1 代表預測「成功」，接近 0 代表「失敗」。
    """
    if config is None:
        config = SCORER_CONFIG

    total_weight = sum(c["weight"] for c in config)
    weighted_sum = 0.0

    for c in config:
        x = sample[c["feature"]]
        midpoint = (c["success_mean"] + c["failure_mean"]) / 2.0
        spread = abs(c["failure_mean"] - c["success_mean"])
        # scale = spread/4：使中點到任一均值約 2 個 scale，
        # 對應 sigmoid 輸出約 0.88（靠近邊界但不飽和）
        scale = spread / 4.0 if spread > 0 else 1.0

        if c["direction"] == "lower_better":
            z = -(x - midpoint) / scale   # 數值越低 → z 越大 → 分數越高
        else:
            z = (x - midpoint) / scale    # 數值越高 → z 越大 → 分數越高

        weighted_sum += _sigmoid(z) * c["weight"]

    return weighted_sum / total_weight


def print_score_report(samples: List[Dict], thresh: float = 0.8, label_key: str = "Success"):
    """對所有資料輸出逐筆的成功機率預測，並與真實標籤對比。"""
    print("\n" + "=" * 70)
    print("成功機率預測（6 個核心特徵 sigmoid 加權）")
    print("=" * 70)
    print(f"{'#':<4} {'預測機率':>8}  {'判斷':>6}  真實結果")
    print("-" * 50)

    correct = 0
    for i, s in enumerate(samples):
        prob = score_sample(s)
        actual = s.get(label_key, None)
        predicted_label = "成功" if prob >= thresh else "失敗"
        actual_label = "成功" if actual else "失敗"
        hit = prob >= thresh
        match = "✓" if (hit == actual) else "✗"
        if hit == actual:
            correct += 1
        print(f"{i+1:<4} {prob:>8.3f}  {predicted_label:>6}  {actual_label} {match}")

    print(f"\n準確率：{correct}/{len(samples)} = {correct/len(samples)*100:.0f}%")
    print("\n⚠️  注意：模型是用「這批資料本身」校準的，準確率會高估真實泛化能力。")
    print("   請把這個分數當作「相對風險排序」，而非絕對機率。")
# 範例：假設你有 5 筆資料（這裡用模擬資料示意）
# ============================================================

if __name__ == "__main__":
    samples = [
        {
# Bellows
    "AverageFrameFieldEnergy": 0.05447867316465183,
    "FrameFieldEnergy": 385056.6783732614,
    "LengthComplexSingularEdges": 0.0,
    "MinimumDihedralAngle": 89.99219082645618,
    "NumberComplexSingularEdges": 0,
    "NumberEdges": 7068026,
    "NumberFaces": 11524053,
    "NumberFeatureEdges": 104,
    "NumberSingularEdges": 110,
    "NumberSingularNodes": 6,
    "NumberSingularVertices": 0,
    "NumberTetraCells": 5685211,
    "NumberVertices": 1229184,
    "NumberZipperNodes": 2,
    "PercentageMeshableEdges": 1.0,
    "PercentageMeshableVertices": 0.8751041341247526,
    "Success": True
},
# Cold_plate ear
{
    "AverageFrameFieldEnergy": 0.0957519258062653,
    "FrameFieldEnergy": 542.7219154699118,
    "LengthComplexSingularEdges": 0.0,
    "MinimumDihedralAngle": 89.98224100791059,
    "NumberComplexSingularEdges": 0,
    "NumberEdges": 5668,
    "NumberFaces": 8947,
    "NumberFeatureEdges": 127,
    "NumberSingularEdges": 136,
    "NumberSingularNodes": 6,
    "NumberSingularVertices": 0,
    "NumberTetraCells": 4322,
    "NumberVertices": 1043,
    "NumberZipperNodes": 0,
    "PercentageMeshableEdges": 1.0,
    "PercentageMeshableVertices": 0.8398849472674976,
    "Success": True
},
# Cold_plate Full
{
    "AverageFrameFieldEnergy": 0.01635693540910752,
    "FrameFieldEnergy": 7693.599068221587,
    "LengthComplexSingularEdges": 0.0,
    "MinimumDihedralAngle": 85.6256757235107,
    "NumberComplexSingularEdges": 0,
    "NumberEdges": 470357,
    "NumberFaces": 731079,
    "NumberFeatureEdges": 3604,
    "NumberSingularEdges": 3948,
    "NumberSingularNodes": 139,
    "NumberSingularVertices": 0,
    "NumberTetraCells": 342089,
    "NumberVertices": 81363,
    "NumberZipperNodes": 52,
    "PercentageMeshableEdges": 1.0,
    "PercentageMeshableVertices": 0.9487850742966705,
    "Success": False
},
# Cold_plate Full fused
{
    "AverageFrameFieldEnergy": 0.05573482779676397,
    "FrameFieldEnergy": 6052.356420106193,
    "LengthComplexSingularEdges": 0.0,
    "MinimumDihedralAngle": 67.78422747687321,
    "NumberComplexSingularEdges": 0,
    "NumberEdges": 108592,
    "NumberFaces": 176604,
    "NumberFeatureEdges": 672,
    "NumberSingularEdges": 802,
    "NumberSingularNodes": 123,
    "NumberSingularVertices": 0,
    "NumberTetraCells": 86912,
    "NumberVertices": 18896,
    "NumberZipperNodes": 33,
    "PercentageMeshableEdges": 0.9999631648740239,
    "PercentageMeshableVertices": 0.9619496189669772,
    "Success": False
},
# Cold_plate Simple fused
{
    "AverageFrameFieldEnergy": 0.03170362080016417,
    "FrameFieldEnergy": 3246.4507699368114,
    "LengthComplexSingularEdges": 0.0,
    "MinimumDihedralAngle": 77.70182749897795,
    "NumberComplexSingularEdges": 0,
    "NumberEdges": 102400,
    "NumberFaces": 166639,
    "NumberFeatureEdges": 578,
    "NumberSingularEdges": 629,
    "NumberSingularNodes": 47,
    "NumberSingularVertices": 0,
    "NumberTetraCells": 82066,
    "NumberVertices": 17825,
    "NumberZipperNodes": 7,
    "PercentageMeshableEdges": 1.0,
    "PercentageMeshableVertices": 0.9751472650771389,
    "Success": True
},
# Cold_plate Simple
{
    "AverageFrameFieldEnergy": 0.030550512534982765,
    "FrameFieldEnergy": 3178.7502787524218,
    "LengthComplexSingularEdges": 0.0,
    "MinimumDihedralAngle": 77.88622991239062,
    "NumberComplexSingularEdges": 0,
    "NumberEdges": 104049,
    "NumberFaces": 168728,
    "NumberFeatureEdges": 710,
    "NumberSingularEdges": 767,
    "NumberSingularNodes": 51,
    "NumberSingularVertices": 0,
    "NumberTetraCells": 82928,
    "NumberVertices": 18247,
    "NumberZipperNodes": 11,
    "PercentageMeshableEdges": 1.0,
    "PercentageMeshableVertices": 0.9741327341480791,
    "Success": False
},
# CP_Bracket
{
    "AverageFrameFieldEnergy": 0.025088396061522262,
    "FrameFieldEnergy": 42498.26271285108,
    "LengthComplexSingularEdges": 0.0,
    "MinimumDihedralAngle": 83.56430501635899,
    "NumberComplexSingularEdges": 0,
    "NumberEdges": 1693941,
    "NumberFaces": 2746261,
    "NumberFeatureEdges": 2424,
    "NumberSingularEdges": 5116,
    "NumberSingularNodes": 287,
    "NumberSingularVertices": 0,
    "NumberTetraCells": 1348835,
    "NumberVertices": 296512,
    "NumberZipperNodes": 190,
    "PercentageMeshableEdges": 1.0,
    "PercentageMeshableVertices": 0.9393211741851932,
    "Success": False
},
# Frame
{
    "AverageFrameFieldEnergy": 0.07980138478775185,
    "FrameFieldEnergy": 238317.2595024464,
    "LengthComplexSingularEdges": 59.92732273149199,
    "MinimumDihedralAngle": 89.51551714401768,
    "NumberComplexSingularEdges": 58,
    "NumberEdges": 2986380,
    "NumberFaces": 4746996,
    "NumberFeatureEdges": 11475,
    "NumberSingularEdges": 14405,
    "NumberSingularNodes": 1787,
    "NumberSingularVertices": 0,
    "NumberTetraCells": 2287876,
    "NumberVertices": 527125,
    "NumberZipperNodes": 395,
    "PercentageMeshableEdges": 0.999980578493025,
    "PercentageMeshableVertices": 0.7610244249466446,
    "Success": False
},
# HP
{
    "AverageFrameFieldEnergy": 0.325073250284427,
    "FrameFieldEnergy": 10763.500390167663,
    "LengthComplexSingularEdges": 0.0,
    "MinimumDihedralAngle": 89.99943171029805,
    "NumberComplexSingularEdges": 0,
    "NumberEdges": 33111,
    "NumberFaces": 48993,
    "NumberFeatureEdges": 68,
    "NumberSingularEdges": 1252,
    "NumberSingularNodes": 50,
    "NumberSingularVertices": 0,
    "NumberTetraCells": 22082,
    "NumberVertices": 6201,
    "NumberZipperNodes": 40,
    "PercentageMeshableEdges": 1.0,
    "PercentageMeshableVertices": 0.16868247056926303,
    "Success": True
},
# Stiffener
{
    "AverageFrameFieldEnergy": 0.04070543515882589,
    "FrameFieldEnergy": 12260.762007884468,
    "LengthComplexSingularEdges": 0.0,
    "MinimumDihedralAngle": 89.99327464344601,
    "NumberComplexSingularEdges": 0,
    "NumberEdges": 301207,
    "NumberFaces": 485148,
    "NumberFeatureEdges": 1601,
    "NumberSingularEdges": 2498,
    "NumberSingularNodes": 201,
    "NumberSingularVertices": 0,
    "NumberTetraCells": 237316,
    "NumberVertices": 53364,
    "NumberZipperNodes": 79,
    "PercentageMeshableEdges": 1.0,
    "PercentageMeshableVertices": 0.9068473127951427,
    "Success": True
}
    ]

    print_report(samples)
    print_score_report(samples)