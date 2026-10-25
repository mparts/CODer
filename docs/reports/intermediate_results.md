# CODer: Findings So Far

Human vs LLM code detection on AIGCodeSet. Status as of 2026-10-25 (after the night sweep and the first ensemble test). All numbers are **dev** macro-F1 unless stated (steps 7-9 use 7-fold cross-validation on the training folds); test has not been touched.

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
| 7 | Tuned char LogReg (n-gram range x C), 7-fold CV on train folds, generate variants only | The strongest model deserves a fair tuning, without touching dev | Best `(1,6)`, `C=10`: binary_generate .909 ± .018 (default .896), multiclass_generate .791 ± .034 (default .768). Top settings tie, so it is a plateau. CV agrees with dev (.900 / .796), so dev is not a fluke. |
| 8 | Whitespace ablation | Test the idea that CodeBERT loses because it cannot see layout | Removing indentation/newlines costs -.002 / -.013 (ties). **Layout hypothesis not supported.** |
| 9 | Content ablation: remove comments; mask identifier names | What separates LLM from human code (the proposal's core question) | Comments -.007 / -.004 (ties). **Masking names -.027 / -.026**, consistent in both variants. Most signal remains with names masked. |
| 10 | Final CodeBERT protocol: 4 epochs, lr 2e-5, no weights, 128 tokens, 3 seeds, last-epoch F1 | An honest transformer number without epoch selection | Last epoch: binary .711 ± .012, binary_generate .801 ± .012, multiclass .511 ± .012, multiclass_generate .700 ± .017. Higher than the 10-epoch run's last epoch (.670 / .785 / .475 / .655) because training stops before it overfits, but still .02-.03 (non-generate) and ~.10 (generate variants) below char LogReg. |
| 11 | Second encoder: UniXcoder, same protocol (4 ep, lr 2e-5, 128 tokens, no weights, 3 seeds) | Test whether the CodeBERT result is specific to CodeBERT | **UniXcoder is far better than CodeBERT** (+.03 to +.09) and close to char LogReg (see below). The pretrained checkpoint matters much more than any hyperparameter we tried. |
| 12 | Tuned char LogReg rerun on dev (presumably `(1,6)`, `C=10`) | Confirm the CV tuning holds on dev | .911 / .794 / .739 / .568 (binary_generate / multiclass_generate / binary / multiclass), vs the default's .900 / .796 / .733 / .539. Word LogReg shifted slightly too (.856 / .688 / .672 / .455), so the new C probably applied to it as well. |
| 13 | Length test: UniXcoder and CodeBERT at 256 tokens (generate variants, seeds 0-1), paired with the 128-token runs of the same seeds | Half of the files are cut at 128 tokens | UniXcoder +.011 (binary_generate) / +.010 (multiclass_generate), 3 of 4 paired comparisons positive; CodeBERT +.022 / +.027. Mixed by the strict rule we set, but leaning positive. At 256 UniXcoder is level with char LogReg (.906 vs .911; .787 vs .794). The 384-token runs have not been done yet. |
| 14 | Token-length audit (UniXcoder tokenizer, train split) | How much does truncation cut? | Median 128 tokens, p90 ~365, max 4,360. 49.8% of files exceed 128 tokens, 20.7% exceed 256, 9.1% exceed 384, 5.1% exceed 512. Same in both classes (median 129 human / 123 AI), so truncation creates no class shortcut. |
| 15 | More checkpoints at 128 tokens: GraphCodeBERT (3 seeds, generate variants); ModernBERT | Is the pretrained checkpoint the main lever? | GraphCodeBERT .869 ± .012 / .739 ± .015: between CodeBERT (.834 / .712) and UniXcoder (.896 / .778). **Ranking: UniXcoder > GraphCodeBERT > CodeBERT.** ModernBERT could not run (installed `transformers` too old). |
| 16 | Leakage audit of the data pipeline | Make sure the `meta_*` columns never reach a model | **No leak found:** models receive only `code_norm` and `y`; vectorizers fit on train only; `meta_*` columns are used only to write analysis outputs. The baseline LogReg is deterministic (seeds 0 and 42 gave identical scores). |
| 17 | Night sweep on UniXcoder (generate variants, seeds 0-2): 384 tokens, dropout 0.2, lr 3e-5, 6 epochs, lr 1e-5 with 6 epochs, each against the 256-token reference | Last attempt to find a better configuration before the final run | **Nothing clears the adoption rule; the reference stays** (256 tokens, lr 2e-5, 4 epochs, dropout 0.1). Dropout 0.2 costs about .045. See the table below. |
| 18 | Ensemble: average char LogReg and UniXcoder class probabilities (UniXcoder averaged over 3 seeds, 6-epoch runs), dev only | Two models built differently may make different mistakes | Mix beats both single models at every weight on both variants: binary_generate .919-.927 vs .914 / .911; multiclass_generate .824-.827 vs .822 / .794. Gain over the better single model is small (+.005 to +.013). |


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
| 128, none, **4 ep, lr 2e-5, 3 seeds**, best epoch (mean ± std) | .724 ± .009 / .740 | .834 ± .009 / .893 | .512 ± .011 / .671 | .712 ± .028 / .897 |
| same, **last epoch = final protocol** (macro-F1 only) | .711 ± .012 | .801 ± .012 | .511 ± .012 | .700 ± .017 |

Seed noise is about ±.02 on dev macro-F1 (±.04-.07 per epoch on the small multiclass_generate dev set), so differences under ~.03 are ties.

### Baselines vs CodeBERT (dev macro-F1 / accuracy)

| Variant | word LogReg | char LogReg (default) | CodeBERT (3 seeds) |
|---|---|---|---|
| binary | .669 / .721 | **.733 / .764** | .701 / .711 |
| binary_generate | .851 / .915 | **.900 / .943** | .863 / .913 |
| multiclass | .474 / .636 | **.539 / .680** | .513 / .667 |
| multiclass_generate | .678 / .871 | **.796 / .926** | .705 / .859 |

The comparison *favors* CodeBERT: its scores pick the best epoch on dev, it sees only 128 tokens, and LogReg is fit once with no selection. The gap on both generate variants is larger than the seed noise. Tuning the char LogReg adds only ~.01-.02 (step 7). With the final protocol (last epoch, 3 seeds) CodeBERT scores .711 / .801 / .511 / .700, which is .022 / .099 / .028 / .096 below char LogReg's default dev scores.

### Ablations (char LogReg, (1,6), C=10, 7-fold CV on train; macro-F1 mean ± std)

| Input | binary_generate | multiclass_generate |
|---|---|---|
| Full code | .909 ± .018 | .791 ± .034 |
| Layout removed | .907 ± .017 | .778 ± .031 |
| Comments removed | .902 ± .020 | .787 ± .026 |
| Identifier names masked | .882 ± .021 | .765 ± .020 |

### UniXcoder vs CodeBERT vs char LogReg (dev macro-F1; same protocol, 3 seeds)

| Variant | UniXcoder best epoch | UniXcoder last epoch | CodeBERT last epoch | char LogReg (tuned) | UniXcoder last minus char |
|---|---|---|---|---|---|
| binary | .749 ± .006 | .744 ± .005 | .711 | .739 | +.005 |
| binary_generate | .897 ± .005 | .892 ± .008 | .801 | .911 | -.019 |
| multiclass | .540 ± .011 | .538 ± .011 | .511 | .568 | -.030 |
| multiclass_generate | .778 ± .004 | .772 ± .018 | .700 | .794 | -.022 |

UniXcoder gains +.03 / +.09 / +.03 / +.07 over CodeBERT (last epoch). GEMINI F1 in multiclass_generate rises from ~.41 (CodeBERT) to .57. Unlike CodeBERT, UniXcoder's dev F1 keeps improving to epoch 3-4, so it overfits later. The last-epoch comparison with char LogReg is the fair one (char is fit once, no selection): UniXcoder is level on binary and .02-.03 behind elsewhere.

### Checkpoint and length comparison (generate variants, best-epoch dev macro-F1, mean ± std)

| Model / setting | Seeds | binary_generate | multiclass_generate |
|---|---|---|---|
| char LogReg (tuned, deterministic) | n/a | .911 | .794 |
| UniXcoder, 256 tokens | 0, 1 | .906 ± .015 | .787 ± .023 |
| UniXcoder, 128 tokens | 0, 1, 2 | .896 ± .005 | .778 ± .004 |
| GraphCodeBERT, 128 tokens | 0, 1, 2 | .869 ± .012 | .739 ± .015 |
| CodeBERT, 256 tokens | 0, 1 | .853 ± .028 | .726 ± .001 |
| CodeBERT, 128 tokens | 0, 1, 2 | .834 ± .009 | .712 ± .028 |

Paired differences, 256 minus 128 tokens (same seed, best-epoch dev macro-F1):

| Model / variant | Seed 0 | Seed 1 | Mean |
|---|---|---|---|
| UniXcoder, binary_generate | +.004 | +.018 | +.011 |
| UniXcoder, multiclass_generate | -.003 | +.023 | +.010 |
| CodeBERT, binary_generate | +.049 | -.006 | +.022 |
| CodeBERT, multiclass_generate | +.044 | +.009 | +.027 |

GraphCodeBERT vs UniXcoder at 128, same seed: all six paired differences are negative (binary_generate -.035 / -.019 / -.030; multiclass_generate -.025 / -.036 / -.058). GraphCodeBERT vs CodeBERT: five of six positive (about +.03). GEMINI F1 in multiclass_generate: CodeBERT ~.41, GraphCodeBERT ~.45, UniXcoder ~.57 (256 tokens: ~.60 on two seeds).

### Night sweep: UniXcoder, generate variants, seeds 0-2 (best-epoch dev macro-F1, mean)

| Setting | binary_generate | multiclass_generate | vs reference |
|---|---|---|---|
| **Reference** (256, lr 2e-5, 4 ep, dropout 0.1) | .906 (.895 / .916 / .908) | .798 (.771 / .804 / .819) | |
| 384 tokens | .911 | .774 | +.005 / -.024 |
| Dropout 0.2 | .862 | .751 | -.044 / -.047 |
| lr 3e-5 | .904 | .793 | -.002 / -.005 |
| 6 epochs, lr 2e-5 | .907 | .816 | +.001 / +.018 |
| 6 epochs, lr 1e-5 | .901 | .780 | -.005 / -.018 |

Every change is a tie or worse except dropout 0.2, which clearly hurts. More regularization is not what the model needs. A dropout 0.0 block is running; result pending.

### Ensemble of char LogReg and UniXcoder (dev macro-F1)

| Variant | UniXcoder, 3-seed avg | char LogReg | mix w=0.3 | mix w=0.5 | mix w=0.7 |
|---|---|---|---|---|---|
| binary_generate | .914 | .911 | .927 | .922 | .919 |
| multiclass_generate | .822 | .794 | .827 | .825 | .824 |

w is the UniXcoder weight. The UniXcoder side uses the 6-epoch runs because only those saved probabilities. Seed averaging alone lifts UniXcoder by about +.007 over a single seed.

### Per-class F1, CodeBERT 3-seed mean

| Variant | Classes |
|---|---|
| binary | human .757, ai .644 |
| binary_generate | human .946, ai .779 (precision .704, recall .874) |
| multiclass | Human .806, GEMINI .319, LLAMA .507, CODESTRAL .422 |
| multiclass_generate | Human .930, GEMINI .412 (precision .297, recall .688), LLAMA .863, CODESTRAL .612 |

## 4. Conclusions

1. **Char n-gram LogReg and UniXcoder are the strongest models; CodeBERT is clearly worse.** Char LogReg beats CodeBERT by .03-.10 macro-F1; UniXcoder closes most of that gap (level on binary, .02-.03 behind char LogReg elsewhere). The reason is **not** layout (step 8). Untested explanations: it reads the whole file while CodeBERT sees 128 tokens, a sparse linear model suits a ~4-5k-sample training set better than a 125M-parameter fine-tune, and it memorizes lexical fragments.
2. **No single cue drives detection.** Identifier names are the largest single contributor (about -.026 when masked), layout at most -.013, comments at most -.007. With names masked the model still reaches .88 / .77, so most of the signal sits in code structure and idioms. There is no cheap shortcut.
3. **More CodeBERT tuning won't help.** Learning rate, epochs, length and weights all landed in the same ±.02 band per variant. The model overfits a training set of ~4-5k samples. The 4-epoch final protocol confirmed it: last-epoch scores rose .02-.04 over the 10-epoch run simply because training stops earlier, but the gap to char LogReg stayed. The baseline is also near its ceiling: tuning gave only +.01-.02.
4. **Class weighting hurt the multiclass variants.** Inverse weights over-predicted rare classes; removing them raised both macro-F1 and accuracy.
5. **The binary vs binary_generate gap (.70 vs .86) is a data property.** Edited code stays close to its human parent, so it is much harder to detect. The paper reports the same: recall on from-scratch code is .5-.99, on fixed/corrected code .05-.77.
6. **GEMINI is the hardest LLM to identify,** in our multiclass results (F1 .32-.41, precision ~.30) and in the paper (lowest recall of the three). **LLAMA is the easiest** (F1 .86 on multiclass_generate).
7. **Reported best-epoch CodeBERT scores are optimistic.** The checkpoint is chosen on the same dev set. Fixing the epoch count instead, binary_generate is closer to .83 than .86.
8. **Checkpoint ranking: UniXcoder > GraphCodeBERT > CodeBERT**, consistent across seeds (UniXcoder beats GraphCodeBERT in all six paired comparisons). The choice of pretrained model moved results far more than any hyperparameter.
9. **Truncation matters a little.** Half of the files exceed 128 tokens. 256 tokens gives UniXcoder about +.01 (not conclusive with two seeds) and brings it level with char LogReg on the generate variants.
10. **No data leakage was found,** and the baseline is deterministic, so its only uncertainty is dev-set size.
11. **UniXcoder's configuration is settled.** 384 tokens, lr 3e-5, 6 epochs and lr 1e-5 are ties or worse; dropout 0.2 costs about .045. The only mild positive is 6 epochs on multiclass_generate (+.018), but it does not repeat on binary_generate.
12. **Combining the two models helps a little and never hurts.** The mix beats both single models at every weight on both variants, but only by +.005 to +.013 over the better one. It is a consistent gain, not a large one; dev is small, so treat it as promising until test confirms it.

## 5. Relation to the literature

- **AIGCodeSet paper:** random 80/20 split (grouping not stated), Ada-embedding or 5K-feature TF-IDF with RF/XGBoost/SVM plus a Bayes classifier. Best F1 .63 (Bayes), best accuracy .71 (SVM on Ada). Binary only; no multiclass attribution. Not directly comparable to ours (different split, our dev excludes conflict rows).
- **AICD Bench (EACL 2026):** fine-tuned encoders (CodeBERT, UniXcoder, ModernBERT), TF-IDF models, zero-shot detectors. Domain shift is the main problem, encoders overfit their training setting, hybrid/edited code is mostly labeled fully AI, and model attribution is the hardest task. Consistent with our results. Their stylistic-cue discussion (identifier naming) fits our name-masking result. (Read via a summarizing fetch tool; verify numbers against the tables before citing.)
- **Our multiclass results go beyond the dataset paper,** which has no attribution experiment.

## 6. Caveats

- Dev is small (93 AI samples in binary_generate, 31 per class in multiclass_generate), and every CodeBERT number is single-run-noisy.
- Dev doubles as the checkpoint selector for CodeBERT, so its scores are optimistic. Test is untouched.
- Ablations were run on the two generate variants only, are not additive, and tell us how much a cue contributes, not its direction. Masking names also changes the n-gram statistics. The layout ablation collapses runs of whitespace but keeps single-space habits such as `a = b` vs `a=b`. The comment ablation removes `#` comments only.
- Tuning and ablations used cross-validation on the training folds; dev was not used to choose anything for the baseline.
- The earlier single-run 3-epoch CodeBERT numbers (e.g. binary_generate .866) were unseeded single draws and look lucky next to the 3-seed means (.834 best epoch, .801 last epoch). The 3-seed means are the reliable ones.
- The ensemble weight (0.3) was not chosen on a held-out set; report w=0.5. The UniXcoder probabilities come from 6-epoch best-on-dev runs, so the ensemble numbers are slightly optimistic.
- ModernBERT could not be evaluated: the installed `transformers` version is too old, and upgrading mid-project was avoided so earlier results stay comparable.
- I predicted GraphCodeBERT would behave like CodeBERT; it landed about +.03 above it (still clearly below UniXcoder). I predicted UniXcoder would land within ±.03 of CodeBERT; it landed +.03 to +.09 above it. Two earlier claims were also corrected during the work: the apparent benefit of 512 tokens on `multiclass` (confounded by weighting), and the expectation that lr 1e-5 would stabilize training (it did not). The "CodeBERT cannot see layout" explanation was tested and not supported.

## 7. Next steps

1. **Wait for the dropout 0.0 result.** Adopt it only under the usual rule; otherwise the configuration is final.
2. **Final run:** reference configuration, all four variants x 3 seeds, test evaluated in the same run (best-on-dev checkpoint), `proba.npy` saved for both UniXcoder and char LogReg on test. The baseline uses the normal `--test` path.
3. **Report on test:** UniXcoder (mean ± std over seeds), char LogReg, and the 50/50 ensemble (UniXcoder averaged over seeds). Do not tune the weight on test.
4. **Optional:** read the char LogReg `top_features.png`; per-scenario and per-LLM recall breakdown (the paper's Tables 4-5); a code LLM with LoRA; the MBPP evaluation set (expect domain shift and contamination).
