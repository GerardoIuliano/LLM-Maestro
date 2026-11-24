"""Combine per-SWC detector predictions into one tool suite per contract."""

import argparse
import json
from pathlib import Path

if __package__:
    from .sft_common import load_jsonl, load_swc_evaluation_split, resolve_swc_config
else:
    from sft_common import load_jsonl, load_swc_evaluation_split, resolve_swc_config


def index_predictions(path, swc, split, expected_ids):
    predictions = {}
    for row in load_jsonl(path):
        contract = row.get("contractname_safe")
        prediction = row.get("prediction")
        if row.get("swc") != swc or row.get("split") != split:
            raise ValueError(f"{path}: expected predictions for {swc} {split}")
        if contract not in expected_ids or contract in predictions:
            raise ValueError(f"{path}: unexpected or duplicate contract {contract!r}")
        if not isinstance(prediction, str):
            raise ValueError(f"{path}: missing prediction for {contract!r}")
        predictions[contract] = prediction
    missing = expected_ids - predictions.keys()
    if missing:
        raise ValueError(f"{path}: missing predictions for {len(missing)} contracts")
    return predictions


def merge_predictions(cfg, split, prediction_files):
    swcs = cfg["swcs"]
    if not swcs or len(set(swcs)) != len(swcs):
        raise ValueError("Configure a nonempty list of distinct SWCs")
    detector_labels = set(cfg["detector_labels"]) - {"no_tool"}
    if not detector_labels:
        raise ValueError("Configure at least one detector label")

    expected_ids = None
    by_swc = {}
    for swc in swcs:
        swc_cfg = resolve_swc_config(cfg, swc)
        split_path = swc_cfg[f"{split}_jsonl_path"]
        rows = load_swc_evaluation_split(split_path, swc)
        ids = {row["contractname_safe"] for row in rows}
        if len(ids) != len(rows) or (expected_ids is not None and ids != expected_ids):
            raise ValueError(f"{split_path}: expected the same unique contract IDs across all SWCs")
        expected_ids = ids
        default_path = Path(swc_cfg["output_dir"]) / f"eval-{split}" / "predictions.jsonl"
        path = prediction_files.get(swc, default_path)
        by_swc[swc] = index_predictions(path, swc, split, expected_ids)

    merged = []
    for contract in sorted(expected_ids):
        predictions = {swc: by_swc[swc][contract] for swc in swcs}
        merged.append({
            "contractname_safe": contract,
            "predictions_by_swc": predictions,
            "selected_tools": sorted(set(predictions.values()) & detector_labels),
            "invalid_predictions": {
                swc: prediction for swc, prediction in predictions.items()
                if prediction != "no_tool" and prediction not in detector_labels
            },
        })
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, default="config/lora-swc-104.yaml")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path, help="Merged JSONL destination")
    parser.add_argument(
        "--prediction-file", action="append", default=[], metavar="SWC=PATH",
        help="Use a specific prediction file for one SWC, for example from a checkpoint",
    )
    args = parser.parse_args()

    import yaml

    try:
        with args.cfg.open(encoding="utf-8") as source:
            cfg = yaml.safe_load(source)
        prediction_files = {}
        for override in args.prediction_file:
            swc, _, path = override.partition("=")
            if swc not in cfg["swcs"] or not path or swc in prediction_files:
                raise ValueError(f"Invalid or duplicate --prediction-file: {override!r}")
            prediction_files[swc] = Path(path)

        merged = merge_predictions(cfg, args.split, prediction_files)
        output = args.output or Path("runs/LLM-maestro/merged") / f"{args.split}.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as destination:
            for row in merged:
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))

    invalid_count = sum(len(row["invalid_predictions"]) for row in merged)
    print(f"{output}: {len(merged)} contract suites, {invalid_count} invalid predictions excluded")


if __name__ == "__main__":
    main()
