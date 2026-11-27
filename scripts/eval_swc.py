"""Evaluate an SWC detector-selection adapter with MAESTRO's Precision@1."""

import argparse
from collections import Counter
import json
from pathlib import Path

if __package__:
    from .sft_common import codellama_user_message, load_base_model, load_swc_evaluation_split, resolve_swc_config
else:
    from sft_common import codellama_user_message, load_base_model, load_swc_evaluation_split, resolve_swc_config


def summarize_predictions(predictions):
    totals = Counter(row["swc_present"] for row in predictions)
    hits = Counter(row["swc_present"] for row in predictions if row["correct"])
    positive = hits[1] / totals[1] if totals[1] else None
    negative = hits[0] / totals[0] if totals[0] else None
    return {
        "num_contracts": len(predictions),
        "num_positive": totals[1],
        "num_negative": totals[0],
        "precision_at_1_positive": positive,
        "precision_at_1_negative": negative,
        "balanced_accuracy": (positive + negative) / 2 if positive is not None and negative is not None else None,
        "accuracy": sum(hits.values()) / len(predictions),
    }


def evaluate_records(rows, swc, predict):
    flag = swc.lower().replace("-", "_")
    results = []
    for row in rows:
        prediction = predict(codellama_user_message(row)).strip()
        results.append({
            "contractname_safe": row["contractname_safe"],
            "swc_present": row[flag],
            "prediction": prediction,
            "valid_answers": row["valid_answers"],
            "correct": prediction in row["valid_answers"],
        })
    return results, summarize_predictions(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", default="config/lora-swc-104.yaml", type=Path)
    parser.add_argument("--swc", help="Target category (default: SWC from the config)")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--adapter", type=Path, help="Adapter or checkpoint directory (default: config output_dir)")
    parser.add_argument("--base-model", help="Override the saved model ID with a local base model directory")
    parser.add_argument("--output-dir", type=Path, help="Write predictions and metrics to this directory")
    args = parser.parse_args()

    import yaml

    with args.cfg.open(encoding="utf-8") as source:
        try:
            cfg = resolve_swc_config(yaml.safe_load(source), args.swc)
        except ValueError as error:
            parser.error(str(error))
    if args.base_model:
        cfg["base_model"] = args.base_model
    swc = cfg.get("swc")
    split_path = cfg.get(f"{args.split}_jsonl_path")
    if not swc or not split_path:
        parser.error(f"Config must specify swc and {args.split}_jsonl_path")
    adapter = args.adapter or Path(cfg["output_dir"])
    if not (adapter / "adapter_config.json").is_file():
        parser.error(f"No LoRA adapter found in {adapter}")
    rows = load_swc_evaluation_split(split_path, swc)

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter, use_fast=True)
    model = PeftModel.from_pretrained(load_base_model(cfg, for_inference=True), adapter)
    model.eval()

    def predict(message):
        inputs = tokenizer.apply_chat_template(
            [message], tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt"
        ).to(model.device)
        prompt_length = inputs["input_ids"].shape[1]
        if prompt_length >= cfg["max_length"]:
            raise ValueError(f"Evaluation prompt exceeds the input budget of {cfg['max_length'] - 1} tokens")
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=cfg.get("gen_max_new_tokens", 16),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=True)

    predictions, metrics = evaluate_records(rows, swc, predict)
    output_dir = args.output_dir or adapter / f"eval-{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as output:
        for row in predictions:
            result = {"swc": swc, "split": args.split, **row}
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
    report = {"swc": swc, "split": args.split, **metrics}
    text = json.dumps(report, indent=2)
    (output_dir / "metrics.json").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"Saved results in {output_dir}")


if __name__ == "__main__":
    main()
