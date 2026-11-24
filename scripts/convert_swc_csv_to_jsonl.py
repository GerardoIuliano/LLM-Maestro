"""Convert one SWC's CSV splits into SFT and evaluation JSONL."""

import argparse
import csv
import json
import re
from pathlib import Path


SPLITS = ("train", "validation", "test")
ID_COLUMN = "contractname_safe"
TARGET_COLUMN = "best_tool_tp"
TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "prompt_templates"


def load_split(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, strict=True)
        columns = reader.fieldnames or []
        if not {ID_COLUMN, TARGET_COLUMN}.issubset(columns) or len(columns) != len(set(columns)):
            raise ValueError(f"{path}: expected unique columns including {ID_COLUMN} and {TARGET_COLUMN}")
        features = [
            name for name in columns
            if name not in {ID_COLUMN, TARGET_COLUMN, "swc"} and not name.startswith("swc_")
        ]
        records = []
        for row in reader:
            location = f"{path}:{reader.line_num}"
            if None in row or None in row.values() or not row[ID_COLUMN].strip():
                raise ValueError(f"{location}: malformed row or missing contract ID")
            labels = list(dict.fromkeys(label.strip() for label in row[TARGET_COLUMN].split(";")))
            if "" in labels or ("no_tool" in labels and len(labels) > 1):
                raise ValueError(f"{location}: invalid {TARGET_COLUMN}: {row[TARGET_COLUMN]!r}")
            records.append((row, labels))
    if not features or not records:
        raise ValueError(f"{path}: expected feature columns and at least one data row")
    return features, records


def convert(args):
    swc = args.swc.upper()
    if not re.fullmatch(r"SWC-\d+", swc):
        raise ValueError("--swc must use a code such as SWC-104")
    swc_flag = swc.lower().replace("-", "_")
    output_dir = args.output_dir or args.input_dir
    outputs = {split: output_dir / f"{split}.jsonl" for split in SPLITS}
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise ValueError(f"Output already exists: {', '.join(existing)}. Use --overwrite to replace it.")

    splits = {split: load_split(args.input_dir / f"{split}.csv") for split in SPLITS}
    train_features, train_records = splits["train"]
    candidate_labels = ", ".join(sorted({label for _, labels in train_records for label in labels}))
    context = {"swc": swc, "candidate_labels": candidate_labels}
    system = args.system_prompt_file.read_text(encoding="utf-8").strip().format(**context)
    user_template = args.prompt_file.read_text(encoding="utf-8").strip()
    if "{features}" not in user_template:
        raise ValueError(f"{args.prompt_file}: include {{features}} where the contract features should appear")

    seen_ids = set()
    payloads = {}
    for split, (features, records) in splits.items():
        if set(features) != set(train_features):
            raise ValueError(f"{split}.csv: feature columns differ from train.csv")
        ids = {row[ID_COLUMN] for row, _ in records}
        if seen_ids & ids:
            raise ValueError(f"{split}.csv: contract IDs overlap with another split")
        if split != "train" and len(ids) != len(records):
            raise ValueError(f"{split}.csv: expected one row per contract")
        seen_ids.update(ids)

        metadata = {}
        if split != "train" and args.metadata_dir:
            metadata_path = args.metadata_dir / f"{split}.csv"
            _, metadata_rows = load_split(metadata_path)
            metadata = {row[ID_COLUMN]: row for row, _ in metadata_rows}
            if len(metadata) != len(metadata_rows) or not ids.issubset(metadata):
                raise ValueError(f"{metadata_path}: expected one metadata row per contract in {split}.csv")

        payload = []
        for row, labels in records:
            feature_text = "\n".join(f"{name}={row[name]}" for name in train_features)
            record = {
                "system": system,
                "user": user_template.format(features=feature_text, **context),
            }
            if split == "train":
                payload.extend(dict(record, assistant=label) for label in labels)
                continue

            record[ID_COLUMN] = row[ID_COLUMN]
            flag_source = row if swc_flag in row else metadata.get(record[ID_COLUMN])
            if flag_source is not None:
                flag = flag_source.get(swc_flag, "").strip()
                if flag not in {"0", "1"}:
                    raise ValueError(f"{split}.csv: invalid {swc_flag} for {row[ID_COLUMN]}: {flag!r}")
                record[swc_flag] = int(flag)
            payload.append(dict(record, valid_answers=labels))
        payloads[split] = payload

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, path in outputs.items():
        payload = payloads[split]
        with path.open("w" if args.overwrite else "x", encoding="utf-8", newline="\n") as output:
            for record in payload:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{path}: {len(payload)} JSONL rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory with all three CSV splits")
    parser.add_argument("--swc", required=True, help="Target category, such as SWC-104")
    parser.add_argument(
        "--metadata-dir", type=Path,
        help="Directory with source validation/test CSVs for missing SWC flags",
    )
    parser.add_argument("--output-dir", type=Path, help="JSONL destination (default: input directory)")
    parser.add_argument(
        "--prompt-file", type=Path, default=TEMPLATE_DIR / "swc_user.txt",
        help="User template with {features} and optional {swc} or {candidate_labels}",
    )
    parser.add_argument(
        "--system-prompt-file", type=Path, default=TEMPLATE_DIR / "swc_system.txt",
        help="System template with optional {swc} or {candidate_labels}",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing JSONL outputs")
    args = parser.parse_args()
    try:
        convert(args)
    except (OSError, ValueError, KeyError, csv.Error) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
