"""Run the six test adapters, merge their suites and score them against MAESTRO."""

import argparse
import csv
from pathlib import Path
import subprocess
import sys

if __package__:
    from .sft_common import load_swc_evaluation_split, resolve_swc_config
    from .train_swc_batch import verify_adapter
else:
    from sft_common import load_swc_evaluation_split, resolve_swc_config
    from train_swc_batch import verify_adapter


REPO_ROOT = Path(__file__).resolve().parents[1]


def check_inputs(cfg):
    if __package__:
        from .eval_maestro_union import DATA_DIR, TARGET_SWCS
    else:
        from eval_maestro_union import DATA_DIR, TARGET_SWCS

    swcs = cfg["swcs"]
    if len(swcs) != len(TARGET_SWCS) or set(swcs) != set(TARGET_SWCS):
        raise ValueError("The MAESTRO test comparison requires the six seed-42 SWCs")

    data_dir = REPO_ROOT / DATA_DIR
    with (data_dir / "union_recommendations_test.csv").open(encoding="utf-8", newline="") as source:
        baseline_ids = {row["contractname_safe"].strip().lower() for row in csv.DictReader(source)}
    if not (data_dir / "union_metrics.csv").is_file():
        raise ValueError(f"Missing MAESTRO baseline metrics in {data_dir}")
    reports = {path.stem.strip().lower() for path in (data_dir / "per_contract").glob("*.json")}
    if len(baseline_ids) != 200 or not baseline_ids.issubset(reports):
        raise ValueError(f"Expected the 200 MAESTRO seed-42 reports in {data_dir}")

    expected_ids = None
    for swc in swcs:
        settings = resolve_swc_config(cfg, swc)
        verify_adapter(REPO_ROOT / settings["output_dir"])
        test_path = REPO_ROOT / settings["test_jsonl_path"]
        rows = load_swc_evaluation_split(test_path, swc)
        ids = {row["contractname_safe"] for row in rows}
        if len(ids) != len(rows) or {name.strip().lower() for name in ids} != baseline_ids:
            raise ValueError(f"{test_path}: expected one row per seed-42 contract")
        if expected_ids is not None and ids != expected_ids:
            raise ValueError(f"{test_path}: contract IDs differ from the other SWC splits")
        expected_ids = ids
    return swcs


def run_script(script, *arguments):
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), *map(str, arguments)],
        cwd=REPO_ROOT,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, default=REPO_ROOT / "config/lora-swc-104.yaml")
    parser.add_argument("--base-model", help="Local directory for the same CodeLlama base model")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs/LLM-maestro/test_pipeline",
                        help="New directory for all outputs from this run")
    args = parser.parse_args()

    import yaml

    config_path = args.cfg.resolve()
    output_dir = args.output_dir.resolve()
    stage = "Preflight"
    try:
        with config_path.open(encoding="utf-8") as source:
            swcs = check_inputs(yaml.safe_load(source))
        base_args = ["--base-model", args.base_model] if args.base_model else []
        if args.base_model and not Path(args.base_model).is_dir():
            raise ValueError(f"Local base model directory does not exist: {args.base_model}")
        if args.base_model:
            base_args[-1] = str(Path(args.base_model).resolve())

        output_dir.mkdir(parents=True, exist_ok=False)
        prediction_args = []
        for index, swc in enumerate(swcs, start=1):
            stage = f"Inference for {swc}"
            print(f"[{index}/{len(swcs)}] {stage}", flush=True)
            swc_output = output_dir / "predictions" / swc
            run_script("eval_swc.py", "--cfg", config_path, "--swc", swc, "--split", "test",
                       "--output-dir", swc_output, *base_args)
            prediction_args.extend(["--prediction-file", f"{swc}={swc_output / 'predictions.jsonl'}"])

        stage = "Merge test predictions"
        print(stage, flush=True)
        merged = output_dir / "merged" / "test.jsonl"
        run_script("merge_swc_predictions.py", "--cfg", config_path, "--split", "test",
                   "--output", merged, *prediction_args)

        stage = "Score suites against MAESTRO"
        print(stage, flush=True)
        scoring = output_dir / "evaluation_union"
        run_script("eval_maestro_union.py", "--recommendations", merged, "--output-dir", scoring)
    except KeyboardInterrupt:
        parser.exit(130, f"Interrupted during {stage}.\n")
    except (ImportError, OSError, KeyError, ValueError, subprocess.CalledProcessError, yaml.YAMLError) as error:
        parser.exit(1, f"{stage} failed: {error}\n")

    print(f"Comparison saved to {scoring / 'comparison_seed_42.csv'}")


if __name__ == "__main__":
    main()
