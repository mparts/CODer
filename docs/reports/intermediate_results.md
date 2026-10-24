# CODer: Findings So Far

Human vs LLM code detection on AIGCodeSet. Status as of 2026-10-24. All numbers are **dev** macro-F1 unless stated; test has not been touched.

## 1. Task and data

- **Dataset:** AIGCodeSet (CodeNet human code + LLM code from CodeLlama 34B, Codestral 22B, Gemini 1.5 Flash). Raw 7,583 samples; 7,531 after dedup (train 5,267 / dev 739 / test 1,525).
- **Three LLM scenarios** (paper): *generate* from scratch, *fix a runtime error*, *correct a wrong answer*. The last two are **edits of a human submission**.
- **Our variants:** `binary` and `multiclass` (all data) vs `binary_generate` and `multiclass_generate` (human + from-scratch LLM code only).
- **Splits:** grouped by problem, no shared problems or identical code across splits (audit confirms 0 overlap). Training uses `clean=True`, which drops 478 label-conflict rows (same code under both labels, likely unchanged edits). Dev therefore has 681 rows, not 739.
- **Dev class counts:** binary 436 human / 245 AI; binary_generate 436 / 93; multiclass 436 / 85 GEMINI / 68 LLAMA / 92 CODESTRAL; multiclass_generate 436 / 31 / 31 / 31.
- **Preprocessing:** `code_norm` strips trailing spaces, tabs, leading blank space and final-newline differences. The audit found these raw tells (e.g. 21% of LLAMA files start with whitespace, only 13% end with a newline vs ~35% for human code); normalization neutralizes them. Single raw-format features reach only AUC 0.50-0.60.

## 2. What we did, why, and what it gave

| # | Step | Why | Outcome |
|---|---|---|---|
| 1 | CodeBERT, 128 tokens, 3 epochs, inverse class weights, lr 2e-5 | First reference run | binary .714, binary_generate .861, multiclass .404, multiclass_generate .637. Overfits after epoch 1-3. GEMINI nearly unlearned (F1 .08). |
| 2 | Same at 512 tokens | Test whether truncation hurts | Gains were small (+.01 on binary variants), 4x slower. Only `multiclass` moved (.404 to .500), and that was later shown to be confounded with weighting. |
| 3 | Class weights: inverse vs sqrt vs none | Inverse weights flooded the Human-heavy dev set with false GEMINI predictions (GEMINI precision .16) | **No weights is best or tied everywhere.** multiclass .404 / .477 / .515; multiclass_generate .637 / .678 / .689. Binary unaffected (within .007). |
| 4 | 10 epochs, lr 1e-5, no weights, 3 seeds | Test whether slower training stabilizes it | **No gain** over the 3-epoch run, 3x the cost. Dev loss is lowest at epoch 2-4 in every run; train loss falls to .03-.08 on the generate variants (memorization). Peak epoch varies 3-8 by seed. |
| 5 | Baselines: word and char n-gram LogReg | The proposal's baseline | **Char LogReg beats CodeBERT on all four variants.** |
| 6 | Read the dataset paper and AICD Bench | Compare with published work | Our patterns match theirs (see section 5). |

## 3. Results

### CodeBERT, every configuration (macro-F1 / accuracy)

| Config | binary | binary_generate | multiclass | multiclass_generate |
|---|---|---|---|---|
| 128, inverse, 3 ep | .714 / .721 | .861 / .911 | .404 / .652 | .637 / .847 |
| 512, inverse, 3 ep | .722 / .730 | .867 / .930 | .500 / .665 | .643 / .735 |
| 128, sqrt, 3 ep | .709 / .718 | .860 / .911 | .477 / .599 | .678 / .800 |
| 128, none, 3 ep | .716 / .730 | .866 / .926 | .515 / .684 | .689 / .860 |
| 128, none, 10 ep, lr 1e-5, 3 seeds (mean ± std) | .701 ± .018 / .711 | .863 ± .007 / .913 | .513 ± .019 / .667 | .705 ± .011 / .859 |
| same, **last-epoch** F1 (no selection) | .670 | .785 | .475 | .655 |

Seed noise is about ±.02 on dev macro-F1 (±.04-.07 per epoch on the small multiclass_generate dev set), so differences under ~.03 are ties.

### Baselines vs CodeBERT (macro-F1 / accuracy)

| Variant | word LogReg | char LogReg | CodeBERT (3 seeds) |
|---|---|---|---|
| binary | .669 / .721 | **.733 / .764** | .701 / .711 |
| binary_generate | .851 / .915 | **.900 / .943** | .863 / .913 |
| multiclass | .474 / .636 | **.539 / .680** | .513 / .667 |
| multiclass_generate | .678 / .871 | **.796 / .926** | .705 / .859 |

The comparison *favors* CodeBERT: its scores pick the best epoch on dev, it sees only 128 tokens, and LogReg is fit once with no selection. The gap on both generate variants is larger than the seed noise.

### Per-class F1, CodeBERT 3-seed mean

| Variant | Classes |
|---|---|
| binary | human .757, ai .644 |
| binary_generate | human .946, ai .779 (precision .704, recall .874) |
| multiclass | Human .806, GEMINI .319, LLAMA .507, CODESTRAL .422 |
| multiclass_generate | Human .930, GEMINI .412 (precision .297, recall .688), LLAMA .863, CODESTRAL .612 |

## 4. Conclusions

1. **Char n-gram LogReg is the strongest model so far.** It beats CodeBERT by .03-.09 macro-F1. Likely reason (a hypothesis, not yet tested): it sees layout (indentation, spacing, newlines) that CodeBERT's pretraining largely normalized away, and it reads the whole file.
2. **More CodeBERT tuning won't help.** Learning rate, epochs, length and weights all landed in the same ±.02 band per variant. The model overfits a training set of ~4-5k samples.
3. **Class weighting hurt the multiclass variants.** Inverse weights over-predicted rare classes; removing them raised both macro-F1 and accuracy.
4. **The binary vs binary_generate gap (.70 vs .86) is a data property.** Edited code stays close to its human parent, so it is much harder to detect. The paper reports the same: recall on from-scratch code is .5-.99, on fixed/corrected code .05-.77.
5. **GEMINI is the hardest LLM to identify,** in our multiclass results (F1 .32-.41, precision ~.30) and in the paper (lowest recall of the three).
6. **LLAMA is easiest.** F1 .86 on multiclass_generate; it also produces the most conflict rows (unchanged edits).
7. **Reported best-epoch scores are optimistic.** The checkpoint is chosen on the same dev set. Fixing the epoch count instead, binary_generate is closer to .83 than .86.

## 5. Relation to the literature

- **AIGCodeSet paper:** random 80/20 split (grouping not stated), Ada-embedding or 5K-feature TF-IDF with RF/XGBoost/SVM plus a Bayes classifier. Best F1 .63 (Bayes), best accuracy .71 (SVM on Ada). Binary only; no multiclass attribution. Not directly comparable to ours (different split, our dev excludes conflict rows).
- **AICD Bench (EACL 2026):** fine-tuned encoders (CodeBERT, UniXcoder, ModernBERT), TF-IDF models, zero-shot detectors. Domain shift is the main problem, encoders overfit their training setting, hybrid/edited code is mostly labeled fully AI, and model attribution is the hardest task. Consistent with our results. (Read via a summarizing fetch tool; verify numbers against the tables before citing.)
- **Our multiclass results go beyond the dataset paper,** which has no attribution experiment.

## 6. Caveats

- Dev is small (93 AI samples in binary_generate, 31 per class in multiclass_generate), and every CodeBERT number is single-run-noisy.
- Dev doubles as the checkpoint selector for CodeBERT, so its scores are optimistic. Test is untouched.
- Two earlier claims were corrected during the work: the apparent benefit of 512 tokens on `multiclass` (confounded by weighting), and the expectation that lr 1e-5 would stabilize training (it did not).

## 7. Next steps

1. **Breakdown by scenario and LLM** (recall for scratch / runtime-fix / wrong-fix and per LLM) from the saved predictions: the like-for-like comparison with the paper's Tables 4 and 5.
2. **Tune char LogReg** on the two generate variants only (`C`, n-gram range, word+char union), by cross-validation on the training folds, not on dev.
3. **Whitespace ablation** of char LogReg to test the layout hypothesis.
4. **Final CodeBERT protocol:** 4 epochs, lr 2e-5, no weights, 3 seeds, report the last epoch.
5. **Average LogReg and CodeBERT probabilities** and keep it only if it beats LogReg in cross-validation.
6. **Evaluate once on test** with the chosen configuration per variant.
7. **Stretch:** a small MBPP-based dataset used only for evaluation. Expect domain shift (short functions vs contest programs); contamination is possible.