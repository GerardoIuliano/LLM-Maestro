"""Evaluate LLM detector suites using the original MAESTRO union scorer."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

if __package__:
    from .sft_common import load_jsonl
else:
    from sft_common import load_jsonl


DATA_DIR = Path("datasets/maestro/seed_42")
RUN_DIR = Path("runs/LLM-maestro")

# Scoring rules from Hit it, Maestro, EXPERIMENT/scripts/4_orchestrator_x_swc_union.py.
TARGET_SWCS = [
    "SWC-101",  
    "SWC-104", 
    "SWC-107", 
    "SWC-113", 
    "SWC-114", 
    "SWC-116",
]
NO_TOOL_ALIASES = {"nessun_tool", "none", "no_tool", "notool", "n/a", "na", "null", "nan", ""}


def _normalize_tool(value: object) -> str:
    return re.sub(r"\s+", "", str(value)).strip().lower()


def _normalize_contract(value: object) -> str:
    return str(value).strip()


def _contract_lookup_key(value: object) -> str:
    return _normalize_contract(value).lower()


def _calculate_all_stats(m: dict[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = m["TP"], m["FP"], m["FN"], m["TN"]
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    mcc_denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = ((tp * tn) - (fp * fn)) / mcc_denominator if mcc_denominator > 0 else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "mcc": mcc,
    }


def _round_metric_columns(metrics: pd.DataFrame) -> pd.DataFrame:
    rounded = metrics.copy()
    metric_cols = ["accuracy", "precision", "recall", "specificity", "f1", "mcc"]
    for col in metric_cols:
        if col in rounded.columns:
            rounded[col] = rounded[col].round(2)
    return rounded


def _json_index(per_contract_dir: Path) -> dict[str, Path]:
    return {_contract_lookup_key(path.stem): path for path in per_contract_dir.glob("*.json")}


def _evaluate_union(
    recommendations: pd.DataFrame,
    per_contract_dir: Path,
    targets: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    json_by_contract = _json_index(per_contract_dir)
    swc_metrics = {swc: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for swc in targets}
    global_metrics = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    contract_rows = []
    missing_json = []

    for _, rec_row in recommendations.iterrows():
        contract = _normalize_contract(rec_row["contractname_safe"])
        key = _contract_lookup_key(contract)
        allowed_tools = {
            _normalize_tool(tool)
            for tool in str(rec_row.get("recommended_tools", "")).split(";")
            if _normalize_tool(tool) not in NO_TOOL_ALIASES
        }
        json_path = json_by_contract.get(key)
        if json_path is None:
            missing_json.append(contract)
            continue

        data = json.loads(json_path.read_text(encoding="utf-8"))
        union_findings = {swc: 0 for swc in targets}
        ground_truth: dict[str, int] = {}

        for entry in data:
            tool = _normalize_tool(entry.get("tool"))
            if tool not in allowed_tools:
                continue

            findings = entry.get("findings_SWC", {}) or {}
            gts = entry.get("ground_truth_SWC", {}) or {}
            for swc in targets:
                if findings.get(swc) == 1:
                    union_findings[swc] = 1
                if swc in gts:
                    ground_truth[swc] = int(gts[swc])

        # If no tool is recommended, still read ground truth from any entry.
        if not allowed_tools:
            for entry in data:
                gts = entry.get("ground_truth_SWC", {}) or {}
                for swc in targets:
                    if swc in gts:
                        ground_truth[swc] = int(gts[swc])

        contract_result = {
            "contractname_safe": contract,
            "recommended_tools": ";".join(sorted(allowed_tools)),
            "n_recommended_tools": len(allowed_tools),
        }

        for swc in targets:
            if swc not in ground_truth:
                continue

            finding = union_findings.get(swc, 0)
            gt = ground_truth[swc]
            if finding == 1 and gt == 1:
                result = "TP"
            elif finding == 1 and gt == 0:
                result = "FP"
            elif finding == 0 and gt == 1:
                result = "FN"
            else:
                result = "TN"

            swc_metrics[swc][result] += 1
            global_metrics[result] += 1
            contract_result[f"{swc}_gt"] = gt
            contract_result[f"{swc}_finding"] = finding
            contract_result[f"{swc}_result"] = result

        contract_rows.append(contract_result)

    metrics_rows = []
    for swc in targets:
        counts = swc_metrics[swc]
        metrics_rows.append({"class": swc, **counts, **_calculate_all_stats(counts)})

    per_swc_stats = [_calculate_all_stats(swc_metrics[swc]) for swc in targets]
    macro_row = {
        "class": "Macro Avg",
        "TP": None,
        "FP": None,
        "FN": None,
        "TN": None,
        "accuracy": sum(row["accuracy"] for row in per_swc_stats) / len(per_swc_stats),
        "precision": sum(row["precision"] for row in per_swc_stats) / len(per_swc_stats),
        "recall": sum(row["recall"] for row in per_swc_stats) / len(per_swc_stats),
        "specificity": sum(row["specificity"] for row in per_swc_stats) / len(per_swc_stats),
        "f1": sum(row["f1"] for row in per_swc_stats) / len(per_swc_stats),
        "mcc": sum(row["mcc"] for row in per_swc_stats) / len(per_swc_stats),
    }
    metrics_rows.append(macro_row)
    micro_row = {"class": "Micro Avg", **global_metrics, **_calculate_all_stats(global_metrics)}
    metrics_rows.append(micro_row)

    return _round_metric_columns(pd.DataFrame(metrics_rows)), pd.DataFrame(contract_rows), {
        "evaluated_contracts": len(contract_rows),
        "missing_json_contracts": missing_json,
        "missing_json_count": len(missing_json),
    }


def load_recommendations(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, keep_default_na=False)[["contractname_safe", "recommended_tools"]]

    recommendations = []
    for row in load_jsonl(path):
        tools = row["selected_tools"]
        if not isinstance(tools, list) or any(not isinstance(tool, str) for tool in tools):
            raise ValueError(f"{path}: selected_tools must be a list of tool names")
        recommendations.append({
            "contractname_safe": row["contractname_safe"],
            "recommended_tools": ";".join(tools),
        })
    return pd.DataFrame(recommendations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recommendations", type=Path, default=RUN_DIR / "merged/test.jsonl",
                        help="Merged LLM JSONL or MAESTRO recommendations CSV for the seed-42 test set")
    parser.add_argument("--output-dir", type=Path, default=RUN_DIR / "evaluation_union")
    args = parser.parse_args()

    try:
        recommendations = load_recommendations(args.recommendations)
        baseline = load_recommendations(DATA_DIR / "union_recommendations_test.csv")
        keys = recommendations["contractname_safe"].map(_contract_lookup_key)
        baseline_keys = baseline["contractname_safe"].map(_contract_lookup_key)
        if keys.duplicated().any() or set(keys) != set(baseline_keys):
            raise ValueError(f"{args.recommendations}: expected one suite for each MAESTRO seed-42 test contract")

        metrics, per_contract, report = _evaluate_union(recommendations, DATA_DIR / "per_contract", TARGET_SWCS)
        baseline_metrics = pd.read_csv(DATA_DIR / "union_metrics.csv")
        comparison = baseline_metrics.merge(metrics, on="class", suffixes=("_maestro", "_llm_maestro"))
        scored = metrics.loc[metrics["class"] == "Micro Avg", ["TP", "FP", "FN", "TN"]]
        report.update({
            "recommendations": str(args.recommendations),
            "baseline_seed": 42,
            "expected_contract_swc_pairs": len(recommendations) * len(TARGET_SWCS),
            "evaluated_contract_swc_pairs": int(scored.to_numpy().sum()),
        })

        args.output_dir.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(args.output_dir / "union_metrics.csv", index=False)
        per_contract.to_csv(args.output_dir / "union_per_contract_results.csv", index=False)
        comparison.to_csv(args.output_dir / "comparison_seed_42.csv", index=False)
        (args.output_dir / "union_evaluation_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))
    print(metrics.to_string(index=False))
    print(f"Scored {report['evaluated_contract_swc_pairs']}/{report['expected_contract_swc_pairs']} contract–SWC pairs.")
    print(f"Saved results in {args.output_dir}")


if __name__ == "__main__":
    main()
