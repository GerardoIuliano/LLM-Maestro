"""Train one LoRA adapter per SWC, one training process at a time."""

import argparse
from pathlib import Path
import subprocess
import sys

if __package__:
    from .sft_common import resolve_swc_config
else:
    from sft_common import resolve_swc_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def verify_adapter(output_dir):
    files = ("adapter_config.json", "training_config.yaml", "tokenizer_config.json")
    missing = [name for name in files if not has_contents(output_dir / name)]
    if not any(has_contents(output_dir / name) for name in ("adapter_model.safetensors", "adapter_model.bin")):
        missing.append("adapter weights")
    if missing:
        raise ValueError(f"{output_dir}: missing or empty artifacts: {', '.join(missing)}")


def has_contents(path):
    return path.is_file() and path.stat().st_size > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, default=REPO_ROOT / "config/lora-swc-104.yaml")
    parser.add_argument("--swc", nargs="+", help="Run only these categories, in the given order")
    args = parser.parse_args()

    import yaml

    cfg_path = args.cfg.resolve()
    try:
        with cfg_path.open(encoding="utf-8") as source:
            cfg = yaml.safe_load(source)
        configured_swcs = cfg["swcs"]
        swcs = [swc.upper() for swc in (args.swc or configured_swcs)]
        if not swcs or len(swcs) != len(set(swcs)) or set(swcs) - set(configured_swcs):
            raise ValueError("Choose distinct SWCs from the configuration's swcs list")

        jobs = {}
        for swc in swcs:
            settings = resolve_swc_config(cfg, swc)
            for key in ("train_jsonl_path", "validation_jsonl_path"):
                path = REPO_ROOT / settings[key]
                if not has_contents(path):
                    raise ValueError(f"{swc}: missing or empty dataset: {path}")
            jobs[swc] = (REPO_ROOT / settings["output_dir"]).resolve()
        if len(set(jobs.values())) != len(jobs):
            raise ValueError("Each SWC must have a different output directory")
    except (OSError, KeyError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))

    for index, (swc, output_dir) in enumerate(jobs.items(), start=1):
        print(f"[{index}/{len(jobs)}] Training {swc}", flush=True)
        try:
            subprocess.run(
                [sys.executable, str(REPO_ROOT / "scripts/train.py"), "--cfg", str(cfg_path), "--swc", swc],
                cwd=REPO_ROOT,
                check=True,
            )
            verify_adapter(output_dir)
        except KeyboardInterrupt:
            parser.exit(130, f"Interrupted during {swc}.\n")
        except (OSError, subprocess.CalledProcessError, ValueError) as error:
            parser.exit(1, f"Batch stopped at {swc}: {error}\n")
        print(f"[{index}/{len(jobs)}] {swc} saved in {output_dir}", flush=True)

    print(f"Completed {len(jobs)} adapter(s).")


if __name__ == "__main__":
    main()
