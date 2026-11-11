# Data

## Dataset

AIGCodeSet (Demirok & Kutlu, 2025): https://huggingface.co/datasets/basakdemirok/AIGCodeSet

Python solutions to 317 CodeNet problems. Human code comes from CodeNet, LLM code from Gemini, CodeLlama and Codestral.
License: CDLA Permissive 2.0.

Note: the hub `test` split is an exact copy of `train`, so only `train` is used (7,583 rows). We build our own problem-grouped splits.

The processed version used in this project is on the Hugging Face Hub: https://huggingface.co/datasets/mparts/CODer-aigcodeset

## Reproduce

Only `processed/meta.json` is in git. To build the rest from the original dataset, run from the repository root:

```bash
PYTHONPATH=src python3 -m code_origin.data.download     # HF hub -> data/raw
PYTHONPATH=src python3 -m code_origin.data.preprocess   # data/raw -> data/processed
PYTHONPATH=src python3 -m code_origin.data.audit        # sanity checks
```

`bash run/RUN.sh og.conf` runs these three steps as part of the full pipeline. `bash run/RUN.sh hub.conf` fetches the processed files from the Hub instead.

## Layout

```
data/
├── raw/                    from download.py, not in git
│   ├── train.parquet       original rows (without the Ada embedding)
│   └── ada_train.npy       Ada embeddings, row i belongs to uid train_i
└── processed/              from preprocess.py or the Hub
    ├── all.parquet         all cleaned rows
    ├── variants/*.parquet  one file per task
    └── meta.json           class names and sample counts per split (in git)
```

## Preprocessing

- Duplicates (the same code once whitespace runs are collapsed, with the same label) are cut down to one row, leaving 7,531 rows. Code that appears with both labels is kept and flagged in `conflict`. Training and evaluation drop these rows, 478 in all (`load_variant(clean=True)`).
- `code_raw` is the original code, `code_norm` has normalised whitespace (line endings, tabs, trailing spaces, outer blank space).
- Splits: 10 folds grouped by `problem_id` and stratified by LLM (human or one of the three models), so a problem never appears in two splits. Folds 0-1 are test, fold 2 is dev, the rest is train (1,525 / 739 / 5,267 rows before the conflict rows are dropped).
- The target is `y` in the variant files, an index into that variant's classes in `meta.json`. `all.parquet` has `origin` (human or ai) instead.
- Columns starting with `meta_` (status, source, parent submission, LLM) reveal the label. They are for analysis only and must not be used as model input.

## Variants

| Variant | Classes |
|---|---|
| `binary` | human vs ai |
| `binary_generate` | human vs LLM code written from scratch |
| `multiclass` | Human, GEMINI, LLAMA, CODESTRAL |
| `multiclass_generate` | Human, GEMINI, LLAMA, CODESTRAL, only code written from scratch |

The `_generate` variants drop the LLM edits of human submissions. Sample counts per split are in `processed/meta.json`, counted before the conflict rows are dropped.

## Citation

```bibtex
@inproceedings{demirok2025aigcodeset,
  title={Aigcodeset: A new annotated dataset for ai generated code detection},
  author={Demirok, Basak and Kutlu, Mucahid},
  booktitle={2025 33rd Signal Processing and Communications Applications Conference (SIU)},
  pages={1--4},
  year={2025},
  organization={IEEE}
}
```

An English preprint is available at https://arxiv.org/abs/2412.16594.
