"""
find_weights.py — 從已標註的 JSON 資料夾找出特徵權重
=====================================================
用法:
    python find_weights.py <data_folder> [-o scorer_config.json] [--top-k 6] [--label-key Success]

每個 JSON 檔案需包含：
  - 16 個特徵欄位（見 FEATURE_KEYS）
  - "Success": true 或 false  （手動標註）

輸出：
  - stdout：特徵區分力報告 + 建議權重
  - <output>：SCORER_CONFIG JSON 檔案（供 predict.py 使用）
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List


FEATURE_KEYS = [
    "AverageFrameFieldEnergy", "FrameFieldEnergy",
    "LengthComplexSingularEdges", "MinimumDihedralAngle",
    "NumberComplexSingularEdges", "NumberEdges", "NumberFaces",
    "NumberFeatureEdges", "NumberSingularEdges",
    "NumberSingularNodes", "NumberSingularVertices",
    "NumberTetraCells", "NumberVertices", "NumberZipperNodes",
    "PercentageMeshableEdges", "PercentageMeshableVertices",
]


# ── 資料載入 ─────────────────────────────────────────────────

def load_samples(folder: Path, label_key: str) -> List[Dict]:
    samples = []
    missing_label = []
    missing_features = []

    for json_path in sorted(folder.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"⚠️  跳過 {json_path.name}：JSON 解析錯誤 ({e})")
            continue

        if label_key not in data:
            missing_label.append(json_path.name)
            continue

        absent = [k for k in FEATURE_KEYS if k not in data]
        if absent:
            missing_features.append((json_path.name, absent))
            continue

        samples.append(data)

    if missing_label:
        print(f"⚠️  以下檔案缺少 '{label_key}' 欄位，已略過：")
        for name in missing_label:
            print(f"   - {name}")

    if missing_features:
        print("⚠️  以下檔案缺少特徵欄位，已略過：")
        for name, keys in missing_features:
            print(f"   - {name}：{keys}")

    return samples


# ── 核心分析 ─────────────────────────────────────────────────

def analyze_separation(samples: List[Dict], label_key: str = "Success") -> List[Dict]:
    pos = [s for s in samples if s[label_key]]
    neg = [s for s in samples if not s[label_key]]

    if not pos or not neg:
        print("⚠️  警告：資料只有單一類別，無法分析區分力。")
        print(f"   成功={len(pos)} 筆，失敗={len(neg)} 筆")
        print("   建議：至少收集 2~3 筆失敗案例再進行分析。\n")
        return []

    results = []
    for key in FEATURE_KEYS:
        pos_vals = [s[key] for s in pos]
        neg_vals = [s[key] for s in neg]

        mean_pos = statistics.mean(pos_vals)
        mean_neg = statistics.mean(neg_vals)

        all_vals = pos_vals + neg_vals
        std = statistics.stdev(all_vals) if len(all_vals) > 1 else 1.0
        if std == 0:
            std = 1.0

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


def build_scorer_config(analysis: List[Dict], top_k: int) -> List[Dict]:
    top = analysis[:top_k]
    total_sep = sum(r["separation"] for r in top)
    if total_sep == 0:
        total_sep = 1.0

    config = []
    for r in top:
        config.append({
            "feature": r["feature"],
            "success_mean": r["mean_success"],
            "failure_mean": r["mean_fail"],
            "direction": r["direction"],
            "weight": r["separation"] / total_sep,
        })
    return config


# ── 報告輸出 ─────────────────────────────────────────────────

def print_report(samples: List[Dict], analysis: List[Dict], scorer_config: List[Dict], label_key: str):
    print("=" * 70)
    print(f"資料總覽：共 {len(samples)} 筆")
    pos_count = sum(1 for s in samples if s[label_key])
    print(f"  成功: {pos_count} 筆，失敗: {len(samples) - pos_count} 筆")
    print("=" * 70)

    print("\n各特徵區分力（由高到低）：\n")
    print(f"{'特徵名稱':<35} {'成功平均':>12} {'失敗平均':>12} {'分離度':>8}")
    print("-" * 70)
    for r in analysis:
        print(f"{r['feature']:<35} {r['mean_success']:>12.4f} "
              f"{r['mean_fail']:>12.4f} {r['separation']:>8.3f}")

    print("\n" + "=" * 70)
    print(f"建議權重（取分離度最高的前 {len(scorer_config)} 名）：")
    print("=" * 70)
    for c in scorer_config:
        print(f"  {c['feature']:<35} 權重 {c['weight']:.3f}  方向: {c['direction']}")

    print("\n💡 重要提醒：")
    print("   - 上述「建議權重」只是統計上的觀察，不是真正的學習結果")
    print("   - 少量資料的觀察極可能是巧合，務必用領域知識最終判斷")
    print("   - 若某特徵在成功組變化很大（例如 0.001 ~ 1000），")
    print("     代表它可能不是關鍵因素，不要因為平均值差異大就採信")


# ── 主程式 ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="從已標註的 JSON 資料夾分析特徵區分力，輸出 SCORER_CONFIG。"
    )
    parser.add_argument("data_folder", type=Path, help="包含已標註 JSON 檔案的資料夾")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("scorer_config.json"),
        help="輸出的 scorer config 路徑（預設：scorer_config.json）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="保留分離度最高的前 K 個特徵（預設：6）",
    )
    parser.add_argument(
        "--label-key",
        type=str,
        default="Success",
        help="JSON 裡代表成功/失敗的欄位名稱（預設：Success）",
    )
    args = parser.parse_args()

    if not args.data_folder.is_dir():
        print(f"錯誤：找不到資料夾 '{args.data_folder}'", file=sys.stderr)
        sys.exit(1)

    samples = load_samples(args.data_folder, args.label_key)
    if len(samples) < 2:
        print(f"錯誤：有效資料不足（讀到 {len(samples)} 筆），至少需要 2 筆。", file=sys.stderr)
        sys.exit(1)

    analysis = analyze_separation(samples, args.label_key)
    if not analysis:
        sys.exit(1)

    top_k = min(args.top_k, len(analysis))
    scorer_config = build_scorer_config(analysis, top_k)

    print_report(samples, analysis, scorer_config, args.label_key)

    args.output.write_text(
        json.dumps(scorer_config, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )
    print(f"\n✅  SCORER_CONFIG 已儲存至：{args.output}")


if __name__ == "__main__":
    main()
