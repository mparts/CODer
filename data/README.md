# Data

## Dataset

AIGCodeSet (Demirok & Kutlu, 2025): https://huggingface.co/datasets/basakdemirok/AIGCodeSet

Python solutions to 317 CodeNet problems. Human code comes from CodeNet, LLM code from Gemini, CodeLlama and Codestral.
License: CDLA Permissive 2.0.

Note: the hub `test` split is an exact copy of `train`, so only `train` is used (7,583 rows). We build our own problem-grouped splits.

## Reproduce

Run from the repository root:

```bash
PYTHONPATH=src python -m code_origin.data.download     # HF hub -> data/raw
PYTHONPATH=src python -m code_origin.data.preprocess   # data/raw -> data/processed
PYTHONPATH=src python -m code_origin.data.audit        # sanity checks
```

## Layout

```
data/
├── raw/
│   ├── train.parquet       original rows (without the Ada embedding)
│   ├── ada_train.npy       Ada embeddings, row i belongs to uid train_i
└── processed/
    ├── all.parquet         all cleaned rows
    ├── variants/*.parquet  one file per task
    └── meta.json           class names and sample counts per variant
```

## Preprocessing

- Rows with identical code and the same label are dropped. Code that appears with both labels is kept and flagged in `conflict`.
- `code_raw` is the original code, `code_norm` has normalised whitespace (line endings, tabs, trailing spaces, outer blank space).
- Splits: 10 folds grouped by `problem_id` and stratified by LLM (human or one of the three models), so a problem never appears in two splits. Folds 0-1 are test, fold 2 is dev, the rest is train.
- Columns starting with `meta_` (status, source, parent submission, LLM) reveal the label. They are for analysis only and must not be used as model input.

## Variants

| Variant | Classes |
|---|---|
| `human_vs_ai` | human, ai |
| `multiclass` | Human, GEMINI, LLAMA, CODESTRAL |
| `human_vs_gemini`, `human_vs_llama`, `human_vs_codestral` | Human vs one LLM |
| `gemini_vs_llama`, `gemini_vs_codestral`, `llama_vs_codestral` | LLM vs LLM |
| `human_vs_generate` | human vs LLM code written from scratch |
| `accepted_vs_generate` | accepted human solutions vs LLM code written from scratch |
| `wrongruntime_human_vs_edit` | human Wrong/Runtime submissions vs LLM edits of them |

Load them with:

```python
from code_origin.data.loader import load_variant
df = load_variant("human_vs_ai", clean=True)   # clean=True drops conflict rows
```

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