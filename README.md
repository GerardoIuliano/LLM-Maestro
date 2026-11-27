# Prompt2OpenSCENARIO: Fine-Tuning CodeLlama-13B for Scenario Generation

This repository provides the full pipeline for **fine-tuning, validating, and evaluating a Large Language Model (LLM)** on the task of **generating OpenSCENARIO 1.0 (`.xosc`) files from natural language descriptions of driving scenes**.  

The work leverages **CodeLlama-13B-Instruct** as the base model and applies **LoRA/QLoRA parameter-efficient fine-tuning**. The resulting adapter weights are released on [Hugging Face Hub](https://huggingface.co/anto0699/Prompt2OpenSCENARIO-CodeLlama13B-LoRA).

---

## Repository Structure

- **`train.py`**  
  Fine-tunes the base model on a JSONL/Hugging Face dataset.  
  - LoRA configuration is read from `config/lora-codellama13b.yaml`.  
  - Training uses `trl.SFTTrainer` with `assistant_only_loss` for stable single-turn alignment.  
  - Supports mixed precision (`bfloat16` / `fp16`) and 4-bit quantization.

- **`eval_validate.py`**  
  Performs **systematic evaluation of the fine-tuned model**:
  - Generation of `.xosc` files given system and user prompts.  
  - Validation of XML well-formedness and XSD compliance.  
  - Multiple evaluation metrics: BLEU, ROUGE-L, chrF++, METEOR, BERTScore, edit similarity, jaccard overlap of XML tags, slot-based precision/recall/F1, and perplexity.  
  - Logs memory (RAM/VRAM) usage and throughput.  
  - Produces:
    - `metrics.json` (aggregated scores)  
    - `predictions.jsonl` (per-sample generations)

- **`plot_training.py`**  
  Parses Hugging Face `Trainer` log history and plots:
  - Training vs. evaluation **loss**  
  - Training vs. evaluation **perplexity** (log-scaled)

- **`plot_eval.py`**  
  Visualizes evaluation metrics from `metrics.json` in grouped bar charts:
  - XML/XSD validity  
  - BLEU & ROUGE  
  - chrF++ & METEOR  
  - Slot metrics (precision/recall/accuracy/F1)  
  - Edit similarity, jaccard, and length ratio  
  - Memory usage (VRAM/RAM)

- **`config/lora-codellama13b.yaml`**  
  Central configuration file controlling training, quantization, optimizer settings, evaluation parameters, and Hugging Face Hub integration.

- **`prompt_templates/codellama_inst.txt`**  
  Instruction-style prompt template aligned with CodeLlama’s expected format.

---

## Workflow

1. **Dataset Preparation**  
   Training data is stored in JSONL format with three fields:
   - `system` → System instruction (e.g., *"Act as an OpenSCENARIO generator for CARLA..."*)  
   - `user` → Natural language description of a driving scenario  
   - `assistant` → Gold-standard `.xosc` file  

   Alternatively, a Hugging Face dataset repository can be specified in the config file.

2. **Fine-Tuning**  
   ```bash
   python train.py 
   ```

The script automatically handles dataset splitting, tokenization, LoRA adapter setup, and checkpointing.

3. **Evaluation**
   ```bash
   python eval_validate.py --cfg config/lora-codellama13b.yaml --use_jsonl --jsonl_path ./datasets/test.jsonl 
   ```
This produces:
- runs/coherence/metrics.json
- runs/coherence/predictions.jsonl
4. **Plotting**
- Training curves:
   ```bash
   python plot_training.py --log_history runs/prompt2openscenario/codellama13b-lora/trainer_state.json
   ```
- Evaluation metrics:
    ```bash
    python plot_eval.py --input_path runs/coherence/metrics.json
     ```

## SWC dataset conversion

Convert the three CSV splits for one SWC from the repository root:

```bash
python3 scripts/convert_swc_csv_to_jsonl.py \
  --swc SWC-104 \
  --input-dir datasets/SWC-104/selected_splits \
  --metadata-dir datasets/SWC-104/full_splits
```

Edit [the user prompt](prompt_templates/swc_user.txt) or pass a different file with
`--prompt-file`. Place `{features}` where the contract features should appear.
Use `{swc}` for the target category and `{candidate_labels}` for the labels observed
in train. You can also edit [the system prompt](prompt_templates/swc_system.txt) or
override it with `--system-prompt-file`.

The converter writes one training row per valid answer, with only `system`,
`user`, and `assistant`. Validation and test retain one row per contract with
a `valid_answers` list, the contract ID, and the target SWC flag when available.
For `selected_splits`, provide `--metadata-dir` to recover the missing SWC flags
from the matching validation and test CSVs in `full_splits` by contract ID.
The flags are evaluation metadata. Use `--output-dir` to write to a separate
directory or `--overwrite` to replace existing JSONL files.

## SWC fine-tuning and evaluation

Run these commands from the repository root. Pass the target category with
`--swc` to select its dataset and adapter directory.

```bash
python3 scripts/train.py --cfg config/lora-swc-104.yaml --swc SWC-104
python3 scripts/eval_swc.py --swc SWC-104 --split validation
python3 scripts/eval_swc.py --swc SWC-104 --split test
```

To train all configured SWCs in order, run:

```bash
python3 scripts/train_swc_batch.py
```

The batch waits for each training process and checks that it saved the adapter,
tokenizer, and training configuration before starting the next SWC. It stops
at the first failure. Use `--swc SWC-107 SWC-113 SWC-114 SWC-116`, for example,
to run only the remaining categories after an interruption.

The SWC config selects the base model, the three fixed split files, and a dedicated
adapter directory. Training uses the predefined validation set for loss. Each
valid answer is expanded into a separate loss example in memory. The evaluator
generates one answer per contract and reports Precision@1 for positive and negative
contracts grouped by the target SWC flag, their mean as balanced accuracy, and
overall accuracy. It saves predictions and metrics under the adapter directory.
Use `--adapter` to compare saved checkpoints on validation before evaluating a
selected adapter on test.

Repeat for SWC-101, SWC-107, SWC-113, SWC-114, and SWC-116. Each invocation uses
one adapter under `runs/LLM-maestro/<SWC>/codellama13b-lora` by default. The trainer
saves `training_config.yaml` in that directory with the resolved hyperparameters,
dataset paths, and SHA-256 hashes of the train and validation JSONL files. The
current configuration uses `selected_splits` with a balanced train split and
selected features. Set a different `output_dir` when running a different configuration
or dataset for the same SWC. The saved YAML can be passed to `--cfg` when
evaluating that adapter.

For an end-to-end test run, use the trained adapter directories and run this
from the repository root in your inference environment:

```powershell
python scripts/run_maestro_test.py
```

The pipeline runs one adapter at a time, merges the predictions and scores the
tool suites against the MAESTRO seed-42 baseline. Its comparison is saved in
`runs/LLM-maestro/test_pipeline/evaluation_union/comparison_seed_42.csv`.
Use `--base-model "D:\models\CodeLlama-13b-Instruct-hf"` if the same model is stored
in a local directory outside the Hugging Face cache. Use `--output-dir` with a
new directory for a separate run. The adapter directories and archive in
`runs/LLM-maestro` stay local to each machine.

To run the stages separately, merge the predictions once the six evaluations
are complete:


```bash
python3 scripts/merge_swc_predictions.py --split test
```

The merge checks that all six SWCs cover the same contracts. It writes one row
per contract to `runs/LLM-maestro/merged/test.jsonl`, with the per-SWC predictions and
the union of recognized detector labels. A `no_tool` response contributes no
detector. Unrecognized responses are recorded in `invalid_predictions`. Use
`--prediction-file SWC-104=path/to/predictions.jsonl` to merge results from an
alternative checkpoint, and `--output` to save a separate comparison run.

Score the merged tool suites using MAESTRO's original seed-42 protocol:

```bash
python3 scripts/eval_maestro_union.py
```

The scorer reads the original SmartBugs reports for the 200 test contracts and
the seed-42 MAESTRO baseline in `datasets/maestro/seed_42/`. It takes the union of
findings from all selected tools for each contract. It reads ground truth from
the selected tools' report entries, or from all entries when the suite is empty.
It scores contract–SWC pairs with available ground truth and reports coverage.
The metrics, per-contract results, and side-by-side seed-42 comparison are saved
under `runs/LLM-maestro/evaluation_union/`.

## Key Features
Parameter-efficient tuning (LoRA/QLoRA) for large-scale models.
Comprehensive evaluation pipeline including structural XML validation against the OpenSCENARIO 1.0 XSD.
Multi-metric assessment combining NLP quality metrics (BLEU, ROUGE, chrF++, METEOR, BERTScore) with domain-specific slot-based metrics.
Visualization scripts for reproducible analysis.
Direct Hugging Face Hub integration for model push/pull.

## Citation
If you use this work, please cite:
@misc{prompt2openscenario2025,
  title        = {Empty Title},
  author       = {No Authors},
  year         = {2025},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/anto0699/Prompt2OpenSCENARIO-CodeLlama13B-LoRA}}
}
