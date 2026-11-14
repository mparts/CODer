# code_origin

The project's Python package. Run its modules from the repository root with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python3 -m code_origin.training.train_baseline --variant binary --model char_LogReg
```

`run/RUN.sh` runs the whole pipeline in order. Each module's docstring gives its usage. The notes under each table record how things are implemented and why.

## Top level

| Module | What it does |
|---|---|
| `config.py` | Paths, dataset variants, models and hyperparameters |
| `helpers.py` | Logging, run setup and archiving to `docs/experiments/archive/` |

- `SEED` is read from the environment when `config` is imported, default 0. So every seed needs its own process (`sweep.py` and `RUN.sh` do this). The data splits and the LogReg baselines are run with the default 0.
- The `ARGS` defaults are the final UniXcoder config: 4 epochs, 256 tokens, lr 2e-5, batch 16, weight decay 0.01, warmup 0.1, no class weights, dropout 0.0.
- `HYPERPARAMETERS` is `ARGS` from `epochs` on. `archive_run` joins those values and the seed into the run folder name, and `summary.py` splits the name back. Keep the two orders the same.
- `VARIANTS` defines each task as a pandas query, a target column and a class list. Class 0 is human in every variant, and `breakdown.py` relies on it.
- `cache_data` loads a variant with the conflict rows dropped and the `meta_` columns kept. Models only ever see `code_norm`. Only `meta_LLM` and `meta_source` go on, into `predictions.parquet`, for the per-source plots and tables.

## data/

| Module | What it does |
|---|---|
| `download.py` | Downloads AIGCodeSet from the Hugging Face Hub into `data/raw/` |
| `preprocess.py` | Cleans the raw data and builds the problem-grouped splits and task variants in `data/processed/` |
| `audit.py` | Checks the processed data: composition, split leakage, formatting artifacts |
| `loader.py` | Loads one task variant |

- `download.py` stops with an error if the Hub's `train` and `test` splits ever differ.
- Splitting comes first, on the raw rows: `StratifiedGroupKFold` with 10 folds, grouped by `problem_id`, stratified by `LLM`, seed 0. Deduplication comes after, on an MD5 of the whitespace-collapsed code, keeping one row per (code, label). Two asserts check that no problem and no code hash crosses splits.
- `meta_source` is derived. AI code whose `submission_id` is "unrelated" is `generate`. Otherwise it is `edit`, and the human submission it edits goes in `meta_parent_id`.
- `code_norm` unifies line endings, expands tabs (tab stops of 4) and strips trailing spaces and outer blank space, with one final newline. Indentation stays. The raw-format audit found such tells (21% of LLAMA files start with whitespace), so no model sees `code_raw`.
- The `artifacts` check in `audit.py` gives each raw formatting feature's human-vs-AI AUC: how much formatting alone gives away.
- `load_variant(balance=True)` and `load_ada` are unused by the final runs. The Ada embeddings only exist in `data/raw/`.

## models/

| Module | What it does |
|---|---|
| `baseline.py` | TF-IDF logistic regression on word or character n-grams |
| `transformer.py` | Fine-tuning and prediction for the transformer classifiers |

- word LogReg: tokens are `\w+` or single punctuation, 1-3 grams. It is blind to layout.
- char LogReg: 1-6 grams, capped at 300k features. The `char` analyzer collapses whitespace runs, so `show_whitespace` first maps newline to `¶`, tab to `→` and space to `·`, which keeps the layout visible. (`code_norm` has no tabs left, so `→` never appears.)
- Both keep case (in the char model, the custom `preprocessor` replaces sklearn's lowercasing). Both use sublinear TF, `min_df=2`, C=10 and `class_weight="balanced"`. For the char model, the (1,6) range and C=10 came from 7-fold CV on the train folds. The word model reuses C=10.
- `fit` uses AdamW with linear warmup then decay, clips the grad norm at 1.0 and runs bf16 autocast on CUDA with the loss in float32. Weight decay applies to all parameters, biases and LayerNorm included. Class weights are `none`, `sqrt` (square root of the inverse frequency) or anything else for the full inverse frequency.
- Inputs are truncated to `max_length` tokens from the start of the file, so the final model reads the first 256. Half of the files exceed 128 tokens. Batches are padded dynamically.
- Dev is scored 5 times per epoch (`EVALS_PER_EPOCH`) for the learning curves. Only the epoch-end scores pick the checkpoint: the best dev macro-F1, kept in memory and restored before saving.
- `--dropout` sets both the hidden and the attention dropout.

## training/

| Module | What it does |
|---|---|
| `train_baseline.py` | Trains a LogReg baseline on one variant and evaluates it on dev or test |
| `train_codebert.py` | Fine-tunes a transformer on one variant and evaluates it on dev or test |
| `sweep.py` | Trains transformers over several seeds, models and variants |

- A dev run trains on train only, scores dev and saves the model to `artifacts/<variant>__<model>/`, overwriting the last one. A `--test` run only loads and predicts, from `artifacts/` or, in `train_codebert.py`, from `--model_dir` (how `RUN.sh` uses the Hub checkpoints). Each final run was evaluated on test once.
- Every run writes `proba.npy`, `predictions.parquet` (uid, problem, label, prediction, LLM, source, plus `score` for binary), `metrics.json` (with all CLI args) and plots, then copies them to the archive. `--save_ckpt` also puts the model under `checkpoint/` in the archived copy, not in the top-level run folder.
- A transformer run's archive folder name comes from its CLI hyperparameters and `SEED`. So a `--test` run needs the same flags and `SEED` as its training run, or its results land in another folder. `sweep.py` passes them on. The PREDICT stage of `RUN.sh` sets `SEED` and relies on the `ARGS` defaults being the final config.
- `sweep.py` starts one `train_codebert` subprocess per (seed, model, variant) with `SEED` in the environment. It passes on only the hyperparameters given to it, so the rest keep the `ARGS` defaults. `--with_test` runs the test pass after each successful training run. It logs to `logs/sweep_<time>.log`, with the tqdm lines stripped. The logs of the tuning and final runs are in git.
- Its defaults (UniXcoder, the two `_generate` variants, seeds 0-2) are the tuning setup. A change was adopted only if the mean dev gain was at least .01 on both variants and at most one of the six paired seed comparisons was negative. The final runs use fresh seeds 3-8.
- `torch.manual_seed(SEED)` covers the head init and the shuffling. GPU training is still not bit-exact, so retrained numbers come close to the paper's but are not identical.

## evaluation/

| Module | What it does |
|---|---|
| `evaluate.py` | Classification metrics |
| `plots.py` | Plots saved with every run |
| `summary.py` | Collects run metrics into `summary.csv` tables |
| `ensemble.py` | Dev macro-F1 of the UniXcoder seed average mixed with char LogReg at several weights |
| `final_eval.py` | Table 2: test macro-F1 with bootstrap intervals |
| `breakdown.py` | Table 5 and section 4.4: per-class and per-source test results |

- The metric is macro-F1, with accuracy, per-class P/R/F1 and the confusion matrix alongside (scikit-learn, `zero_division=0`).
- The seed ensemble averages the softmax outputs of the six UniXcoder runs. The mix is 0.5 × ensemble + 0.5 × char LogReg. The 0.5 was not tuned: `ensemble.py`'s 0.3 and 0.7 on dev were exploratory only.
- The table scripts read the shipped `proba.npy` and `predictions.parquet` straight from `archive/`, so they need no model. `final_eval.py` asserts that every run lists the files in the same uid order before stacking them.
- `final_eval.py` uses a paired cluster bootstrap over problems: all files of a problem are resampled together, 1,000 resamples, seed 0, 95% percentile intervals on the gap to char LogReg. Seeds are held fixed, so the intervals leave out seed variation. A single seed is reported as mean ± std over seeds (ddof 1).
- `breakdown.py` and the accuracy-by-source plot group test files by true source: Human, or LLM × generate/edit. "Flagged AI" means any prediction other than class 0.
- `summary.py` reads the hyperparameters back from the run folder names. Its seed-aggregated tables write `mean ± std` and list the seeds best first.
