# CODer: Code Origin Detector

Final project for the course Machine Learning for Statistical NLP: Advanced (LT2326/LT2926).

Can code written by an LLM be told apart from human code, and can we tell which LLM wrote it? CODer studies this on [AIGCodeSet](https://huggingface.co/datasets/basakdemirok/AIGCodeSet), Python solutions to CodeNet problems written by humans or by Gemini, CodeLlama and Codestral. It compares word and character n-gram logistic regression with fine-tuned code encoders (CodeBERT, GraphCodeBERT, UniXcoder) on four task variants, using splits grouped by problem and a test set used only once.

The paper: [coder_paper.pdf](docs/reports/coder_paper.pdf) ([LaTeX source](docs/reports/coder_paper_latex/)).

## Results

Test macro-F1, from Table 2 of the paper (which also has word LogReg and the bootstrap intervals). UniXcoder, 1 seed is the mean ± std over seeds 3-8. The seed ensemble averages their class probabilities, and the mix averages the ensemble with char LogReg. The mix was added after seeing dev and its weight was not tuned, so the paper gives it no interval.

| Variant                                        | char LogReg | UniXcoder, 1 seed | Seed ensemble | Mix 0.5 |
| ---------------------------------------------- | ----------- | ----------------- | ------------- | ------- |
| `binary` (human vs AI)                         | .743        | .772 ± .002       | .778          | .783    |
| `multiclass` (human, GEMINI, LLAMA, CODESTRAL) | .554        | .571 ± .026       | .595          | .614    |
| `binary_generate`                              | .923        | .939 ± .008       | .943          | .948    |
| `multiclass_generate`                          | .808        | .837 ± .016       | .839          | .867    |

The `_generate` variants keep only LLM code written from scratch and drop the LLM edits of human submissions.

- UniXcoder is ahead of the char baseline on every variant. In a paired bootstrap over problems, a single model is significantly ahead only on `binary`, the seed ensemble on three of four.
- Edited code is the main source of error. On `binary`, the UniXcoder ensemble flags 27-54% of the LLM-edited files as AI, depending on the LLM, against 94-98% of the code written from scratch.
- Gemini is the hardest LLM, and the models mistake it for human rather than for another LLM: in `multiclass_generate` the ensemble labels 37% of the GEMINI files as Human and 8% as another LLM.

## Setup

The project runs on mltgpu, the university GPU server, with Python 3.12 and the server-wide packages. There is no virtual environment and nothing to install.

```bash
git clone https://github.com/mparts/CODer.git
cd CODer
```

A GPU is only needed to train or re-predict. The versions below are the server's on 2026-11-14. The `transformers` version is the one all 24 UniXcoder checkpoints record.

| Package           | Version                                        |
| ----------------- | ---------------------------------------------- |
| Python            | 3.12.16                                        |
| `torch`           | 2.4.1+cu121 (CUDA 12.1)                        |
| `transformers`    | 4.44.2                                         |
| `datasets`        | 4.8.5                                          |
| `huggingface_hub` | 0.36.2                                         |
| `scikit-learn`    | 1.5.2                                          |
| `pandas`          | 2.2.2                                          |
| `pyarrow`         | 24.0.0                                         |
| `numpy`           | 1.26.4                                         |
| `matplotlib`      | 3.9.2                                          |
| `tqdm`            | 4.66.5                                         |
| GPU               | NVIDIA GeForce GTX 1080 Ti (driver 580.178.04) |

## Reproduce

[run/RUN.sh](run/RUN.sh) runs the pipeline. Run it from the repository root, with or without a preset:

| Command                    | What it does                                                                                                                                                                               | Needs                                                                                                         |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| `bash run/RUN.sh`          | Rebuilds Tables 2 and 5 from the test predictions shipped in `docs/experiments/archive/`                                                                                                   | Minutes, no download, no GPU                                                                                  |
| `bash run/RUN.sh hub.conf` | Fetches the processed data and the 24 UniXcoder checkpoints from the Hugging Face Hub, predicts test again, then rebuilds the tables. The LogReg numbers come from the shipped predictions | About 12 GB of download, a GPU                                                                                |
| `bash run/RUN.sh og.conf`  | The whole experiment from scratch: download, preprocessing, data checks, both LogRegs, UniXcoder over 6 seeds and 4 variants, then the tables                                              | Hours on the GPU. The numbers come out close to the paper, not identical, since GPU training is not bit-exact |

- Stages and settings live in [run/run.conf](run/run.conf). A preset is read after it and overrides it. The output is logged to `run/run.log`, which stays local and is not in git.
- The `hub.conf` and `og.conf` runs overwrite the shipped results in `docs/experiments/archive/`. `git checkout -- docs/experiments/archive` restores them.
- Each step can also be run on its own, for example `PYTHONPATH=src python3 -m code_origin.evaluation.final_eval binary`. The modules are listed in [src/code_origin/README.md](src/code_origin/README.md).
- The tuning runs behind Table 3 are shipped in `docs/experiments/archive_tuning/`, but `RUN.sh` does not rerun them. Table 4 (cross-validation and ablation of the char baseline) cannot be rerun, as its code was not kept.

## Layout

```text
.
├── run/                RUN.sh and its settings: run.conf, og.conf, hub.conf
├── src/code_origin/    the Python package: data, models, training, evaluation
├── data/               the dataset, built or fetched by RUN.sh (only processed/meta.json is in git)
├── logs/               training logs of the tuning and final runs, written by sweep.py
└── docs/
    ├── reports/        the paper and the AI-use log
    ├── notebooks/      project proposal and presentation slides
    └── experiments/    archived runs behind Tables 2, 3 and 5
```

Runs also write models to `artifacts/`, which is not in git. More detail in [data/README.md](data/README.md), [docs/README.md](docs/README.md) and [src/code_origin/README.md](src/code_origin/README.md).

## Links

- Checkpoints, 24 UniXcoder models (seeds 3-8 × 4 variants): https://huggingface.co/mparts/CODer-unixcoder
- Processed dataset, the splits and variants used here: https://huggingface.co/datasets/mparts/CODer-aigcodeset
- Original dataset, AIGCodeSet (Demirok & Kutlu, 2025): https://huggingface.co/datasets/basakdemirok/AIGCodeSet, preprint at https://arxiv.org/abs/2412.16594
- How AI assistance was used: [docs/reports/AI_log.md](docs/reports/AI_log.md)

## License

The code is under the MIT License ([LICENSE](LICENSE)). AIGCodeSet, and the processed version on the Hub, are under CDLA-Permissive-2.0 (see [data/README.md](data/README.md)).
