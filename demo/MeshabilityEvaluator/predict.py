"""
predict.py — 用 SCORER_CONFIG 預測新資料的成功機率
===================================================
用法:
    # 預測單一檔案
    python predict.py <input.json> -c scorer_config.json

    # 批次預測整個資料夾
    python predict.py <data_folder> -c scorer_config.json

    # 調整判斷門檻（預設 0.8）
    python predict.py <input.json> -c scorer_config.json -t 0.75

輸出：
  - 每筆資料的成功機率與成功/失敗判斷
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List


# ── Sigmoid ──────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


# ── 評分 ─────────────────────────────────────────────────────

def score_sample(sample: Dict, config: List[Dict]) -> float:
    """
    給定一筆資料與 SCORER_CONFIG，輸出成功機率（0~1）。

    原理：
      對每個特徵，以成功/失敗平均值的中點為 sigmoid 中心，
      以兩組距離的 1/4 為斜率單位，映射到 0~1 的成功傾向分數，
      再依分離度加權平均。
    """
    total_weight = sum(c["weight"] for c in config)
    if total_weight == 0:
        return 0.0

    weighted_sum = 0.0
    for c in config:
        if c["feature"] not in sample:
            continue
        x = sample[c["feature"]]
        midpoint = (c["success_mean"] + c["failure_mean"]) / 2.0
        spread = abs(c["failure_mean"] - c["success_mean"])
        scale = spread / 4.0 if spread > 0 else 1.0

        if c["direction"] == "lower_better":
            z = -(x - midpoint) / scale
        else:
            z = (x - midpoint) / scale

        weighted_sum += _sigmoid(z) * c["weight"]

    return weighted_sum / total_weight


# ── 資料載入 ─────────────────────────────────────────────────

def load_config(config_path: Path) -> List[Dict]:
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"錯誤：無法讀取 config '{config_path}'：{e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, list) or not data:
        print(f"錯誤：'{config_path}' 格式不正確，需為非空 JSON 陣列。", file=sys.stderr)
        sys.exit(1)

    required_keys = {"feature", "success_mean", "failure_mean", "direction", "weight"}
    for i, item in enumerate(data):
        missing = required_keys - item.keys()
        if missing:
            print(f"錯誤：config 第 {i+1} 項缺少欄位：{missing}", file=sys.stderr)
            sys.exit(1)

    return data


def load_json_file(path: Path) -> Dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"⚠️  跳過 {path.name}：{e}")
        return None


def collect_inputs(input_path: Path) -> List[tuple[str, Dict]]:
    """回傳 (名稱, 資料) 的清單。"""
    if input_path.is_file():
        data = load_json_file(input_path)
        if data is None:
            sys.exit(1)
        return [(input_path.name, data)]
    elif input_path.is_dir():
        results = []
        for json_path in sorted(input_path.glob("*.json")):
            data = load_json_file(json_path)
            if data is not None:
                results.append((json_path.name, data))
        if not results:
            print(f"錯誤：資料夾 '{input_path}' 內找不到任何 JSON 檔案。", file=sys.stderr)
            sys.exit(1)
        return results
    else:
        print(f"錯誤：'{input_path}' 不是有效的檔案或資料夾。", file=sys.stderr)
        sys.exit(1)


# ── 報告輸出 ─────────────────────────────────────────────────

def print_predictions(
    inputs: List[tuple[str, Dict]],
    config: List[Dict],
    threshold: float,
    with_label: bool = False,
    label_key: str = "Success",
):
    print("=" * 70)
    print("成功機率預測")
    print(f"Config 特徵數：{len(config)}　判斷門檻：{threshold}")
    print("=" * 70)

    name_width = max(len(name) for name, _ in inputs)
    name_width = max(name_width, 8)

    header = f"{'檔案名稱':<{name_width}}  {'預測機率':>8}  {'判斷':>6}"
    if with_label:
        header += f"  {'真實':>4}  判斷"
    print(header)
    print("-" * (name_width + (30 if with_label else 20)))

    correct = 0
    scoreable = 0

    for name, sample in inputs:
        missing = [c["feature"] for c in config if c["feature"] not in sample]
        if missing:
            print(f"{name:<{name_width}}  {'N/A':>8}  {'無法預測':>6}  (缺少特徵: {missing})")
            continue

        prob = score_sample(sample, config)
        verdict = "成功" if prob >= threshold else "失敗"
        line = f"{name:<{name_width}}  {prob:>8.3f}  {verdict:>6}"

        if with_label:
            actual = sample.get(label_key)
            if actual is None:
                line += f"  {'N/A':>4}  ?"
            else:
                actual_label = "成功" if actual else "失敗"
                hit = (prob >= threshold) == bool(actual)
                match = "✓" if hit else "✗"
                line += f"  {actual_label:>4}  {match}"
                scoreable += 1
                if hit:
                    correct += 1

        print(line)

    print()

    if with_label and scoreable > 0:
        print(f"準確率：{correct}/{scoreable} = {correct / scoreable * 100:.0f}%")
        print()


# ── 主程式 ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="用 SCORER_CONFIG 預測 JSON 資料的成功機率。"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="單一 JSON 檔案路徑，或包含多個 JSON 的資料夾",
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        required=True,
        help="SCORER_CONFIG JSON 檔案路徑（由 find_weights.py 產生）",
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.8,
        help="成功機率門檻，高於此值判斷為成功（預設：0.8）",
    )
    parser.add_argument(
        "--with-label",
        action="store_true",
        help="輸入資料含有真實標籤（Success 欄位），同時計算預測準確率",
    )
    parser.add_argument(
        "--label-key",
        type=str,
        default="Success",
        help="真實標籤的欄位名稱（預設：Success，搭配 --with-label 使用）",
    )
    args = parser.parse_args()

    if not (0.0 < args.threshold < 1.0):
        print("錯誤：--threshold 必須介於 0 到 1 之間（不含端點）。", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.config)
    inputs = collect_inputs(args.input)
    print_predictions(inputs, config, args.threshold, with_label=args.with_label, label_key=args.label_key)


if __name__ == "__main__":
    main()
