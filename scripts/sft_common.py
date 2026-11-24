import json
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile


def resolve_swc_config(cfg, swc_override=None):
    default_swc = cfg.get("swc")
    if not default_swc and swc_override is None:
        return cfg
    if not default_swc:
        raise ValueError("--swc requires an SWC config")
    swc = (swc_override or default_swc).upper()
    if not re.fullmatch(r"SWC-\d+", swc):
        raise ValueError(f"Invalid SWC code: {swc}")
    resolved = dict(cfg, swc=swc)
    for key in ("train_jsonl_path", "validation_jsonl_path", "test_jsonl_path", "output_dir"):
        value = cfg.get(key)
        if value is None:
            continue
        if swc != default_swc.upper() and "{swc}" not in value:
            raise ValueError(f"{key} must include {{swc}} to select a different SWC")
        resolved[key] = value.replace("{swc}", swc)
    return resolved


def load_jsonl(path, *, digest=None):
    rows = []
    with Path(path).open("rb") as source:
        for line in source:
            if digest is not None:
                digest.update(line)
            if line.strip():
                rows.append(json.loads(line.decode("utf-8")))
    if not rows:
        raise ValueError(f"{path}: empty JSONL file")
    return rows


def save_swc_run_config(cfg):
    import yaml

    path = Path(cfg["output_dir"]) / "training_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            yaml.safe_dump(cfg, output, sort_keys=False, allow_unicode=True)
            output.close()
            try:
                os.link(temporary, path)
            except FileExistsError:
                with path.open(encoding="utf-8") as existing:
                    if yaml.safe_load(existing) != cfg:
                        raise ValueError(
                            f"{path}: configuration or training data differs from the saved run. "
                            "Choose a different output_dir."
                        )
        finally:
            temporary.unlink(missing_ok=True)
    return path


def codellama_user_message(example):
    system = example["system"].strip()
    user = example["user"].strip()
    return {"role": "user", "content": f"<<SYS>>\n{system}\n<</SYS>>\n\n{user}"}


def load_base_model(cfg, *, for_inference=False):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    use_4bit = bool(cfg.get("load_in_4bit", False))
    dtype = getattr(torch, cfg.get("torch_dtype", "bfloat16"))
    quantization = None
    if use_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=getattr(torch, cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
        )
    return AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype=dtype if not use_4bit else None,
        quantization_config=quantization,
        device_map="auto" if use_4bit or for_inference else None,
    )


def load_swc_evaluation_split(path, swc, *, digest=None):
    flag = swc.lower().replace("-", "_")
    rows = load_jsonl(path, digest=digest)
    for line, row in enumerate(rows, 1):
        answers = row.get("valid_answers")
        if (
            row.get(flag) not in (0, 1)
            or not isinstance(answers, list)
            or not answers
            or any(not isinstance(answer, str) or not answer.strip() for answer in answers)
        ):
            raise ValueError(f"{path}: row {line} needs {flag} (0 or 1) and a nonempty valid_answers list")
    return rows
