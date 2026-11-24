import glob
import hashlib
import os
import argparse
import random
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
import torch
import yaml

if __package__:
    from .sft_common import (
        codellama_user_message, load_base_model, load_jsonl, load_swc_evaluation_split,
        resolve_swc_config, save_swc_run_config,
    )
else:
    from sft_common import (
        codellama_user_message, load_base_model, load_jsonl, load_swc_evaluation_split,
        resolve_swc_config, save_swc_run_config,
    )

# -----
# Main
# -----
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="config/lora-codellama13b.yaml")
    p.add_argument("--swc", help="Target category for an SWC config")
    p.add_argument("--jsonl_path", default="")
    p.add_argument("--validation-jsonl-path", help="Predefined SWC validation JSONL (overrides the config)")
    args = p.parse_args()

    with open(args.cfg, encoding="utf-8") as source:
        try:
            cfg = resolve_swc_config(yaml.safe_load(source), args.swc)
        except ValueError as error:
            p.error(str(error))
    swc = cfg.get("swc")
    if swc:
        train_path = args.jsonl_path or cfg.get("train_jsonl_path")
        validation_path = args.validation_jsonl_path or cfg.get("validation_jsonl_path")
        if not train_path or not validation_path:
            p.error("SWC training requires separate train and validation JSONL files")
        if Path(train_path).resolve() == Path(validation_path).resolve():
            p.error("Train and validation must use different files")
        train_digest = hashlib.sha256()
        validation_digest = hashlib.sha256()
        train = load_jsonl(train_path, digest=train_digest)
        validation = load_swc_evaluation_split(validation_path, swc, digest=validation_digest)
        cfg = dict(
            cfg,
            train_jsonl_path=train_path,
            validation_jsonl_path=validation_path,
            dataset_sha256={
                "train": train_digest.hexdigest(),
                "validation": validation_digest.hexdigest(),
            },
        )
        # SFTTrainer needs one completion per row to report validation loss.
        ds = DatasetDict({
            "train": Dataset.from_list(train),
            "validation": Dataset.from_list([
                {"system": row["system"], "user": row["user"], "assistant": label}
                for row in validation
                for label in row["valid_answers"]
            ]),
        })
    else:
        if args.validation_jsonl_path:
            p.error("--validation-jsonl-path requires an SWC config")
        if args.jsonl_path:
            data = Dataset.from_list(load_jsonl(args.jsonl_path))
        else:
            data = load_dataset(cfg["hf_dataset_repo"], split="train")
        split = data.train_test_split(test_size=cfg.get("val_split_ratio", 0.1), seed=cfg["seed"])
        ds = DatasetDict({"train": split["train"], "validation": split["test"]})
    print(f"Dataset rows: {len(ds['train'])} train, {len(ds['validation'])} validation loss")

    random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])

    tok = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # SWC completions learn to terminate directly after the detector name.
    assistant_template = (
        "{% generation %}{{ m['content'] }}{{ eos_token }}{% endgeneration %}"
        if swc else "{% generation %}{{ m['content'] }}{% endgeneration %}{{ eos_token }}"
    )
    tok.chat_template = (
        "{{ bos_token }}"
        "{% for m in messages %}"
        "{% if m['role'] == 'user' %}"
        "[INST] {{ m['content'] }} [/INST]"
        "{% elif m['role'] == 'assistant' %}"
        + assistant_template
        + "{% endif %}"
        "{% endfor %}"
    )

    stop_seq = None if swc else cfg["stop_sequence"]

    peft_cfg = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias=cfg["bias"],
        task_type="CAUSAL_LM"
    )

    def to_messages(ex):
        assistant = ex["assistant"].strip()
        if stop_seq and not assistant.endswith(stop_seq):
            assistant += stop_seq

        return {
            "messages": [
                codellama_user_message(ex),
                {"role": "assistant", "content": assistant},
            ]
        }

    if swc:
        ds = DatasetDict({
            split: dataset.map(to_messages, remove_columns=dataset.column_names)
            for split, dataset in ds.items()
        })
    else:
        ds = ds.map(to_messages)

    def _ok_len(batch):
        return [
            len(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False))
            <= cfg["max_length"]
            for msgs in batch["messages"]
        ]

    if swc:
        for split, dataset in ds.items():
            fitting = dataset.filter(_ok_len, batched=True)
            if len(fitting) != len(dataset):
                raise ValueError(f"{split}: {len(dataset) - len(fitting)} rows exceed max_length={cfg['max_length']}")
    else:
        ds["train"] = ds["train"].filter(_ok_len, batched=True)

    if swc:
        try:
            snapshot_path = save_swc_run_config(cfg)
        except ValueError as error:
            p.error(str(error))
        print(f"Training configuration saved to {snapshot_path}")

    model = load_base_model(cfg)
    model.config.pad_token_id = tok.pad_token_id
    if cfg.get("gradient_checkpointing", False):
        model.config.use_cache = False

    sft_cfg = SFTConfig(
        max_length=cfg["max_length"],
        packing=cfg["packing"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=cfg["train_batch_size"],
        gradient_accumulation_steps=cfg["grad_accum_steps"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg["warmup_ratio"],
        output_dir=cfg["output_dir"],
        report_to=["tensorboard"],
        bf16=(str(cfg.get("torch_dtype", "bfloat16")) == "bfloat16"),
        gradient_checkpointing=cfg["gradient_checkpointing"],
        assistant_only_loss=True,
        save_total_limit=cfg.get("save_total_limit", 3),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        peft_config=peft_cfg,
        processing_class=tok,
    )

    last_ckpt = None
    if os.path.isdir(cfg["output_dir"]):
        candidates = sorted(glob.glob(os.path.join(cfg["output_dir"], "checkpoint-*")),
                            key=os.path.getmtime)
        if candidates:
            last_ckpt = candidates[-1]

    if last_ckpt:
        print(f"Resuming from checkpoint: {last_ckpt}")
        trainer.train(resume_from_checkpoint=last_ckpt)
    else:
        trainer.train()

    trainer.save_model(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])

if __name__ == "__main__":
    main()
