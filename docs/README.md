# Docs

The paper, the project proposal and slides, and the archived runs behind Tables 2, 3 and 5.

## Layout

```
docs/
├── reports/
│   ├── coder_paper.pdf         the paper
│   ├── coder_paper_latex/      its LaTeX source (ACL template), references and figures
│   └── AI_log.md               how AI assistance was used
├── notebooks/
│   ├── Project_Proposal_Bartsiokas.md   project proposal
│   └── bartsiokas.odp, .pdf             presentation slides
└── experiments/
    ├── archive/                final runs, the source of Tables 2 and 5
    └── archive_tuning/         tuning runs on dev, the source of Table 3
```

## Experiments

```
experiments/archive/
├── summary.csv                                    every run, seeds averaged
├── baseline/0/<variant>__<model>/{dev,test}/      word and char LogReg, seed 0
└── transformer/<run>/<variant>__unixcoder/{dev,test}/   UniXcoder, seeds 3-8
```

- `<run>` joins the hyperparameters and the seed: `epochs_maxlen_lr_batch_wd_warmup_classweight_dropout_seed`. For example, `4_256_2e-05_16_0.01_0.1_none_0.0_3` is seed 3 of the final config.
- Each `dev/` or `test/` folder has `metrics.json`, `predictions.parquet`, `proba.npy` (class probabilities) and plots: confusion matrix, accuracy by source, ROC (binary variants), plus training curves (`curve.csv` and a plot, UniXcoder dev) or top features (LogReg).
- `summary.csv` files sit at every level, from one run up to the whole archive (`code_origin.evaluation.summary --archive`).
- `final_eval.py`, `breakdown.py` and `ensemble.py` in `src/code_origin/evaluation/` read `archive/` directly, so Tables 2 and 5 rebuild without training (`bash run/RUN.sh`).
- The UniXcoder checkpoints are not in git. They are on the Hugging Face Hub: https://huggingface.co/mparts/CODer-unixcoder

`archive_tuning/` has the same layout, dev only: 36 transformer run folders (12 configs × seeds 0-2, UniXcoder in all of them, CodeBERT and GraphCodeBERT in a few) and the LogReg dev runs. Its plots are not in git.

Every new run, dev or test, writes to `experiments/<baseline|transformer>/<variant>__<model>/<split>/`, overwriting the last run there. It is then copied into `archive/`, overwriting the shipped run of the same name. The top-level folders are not in git.
