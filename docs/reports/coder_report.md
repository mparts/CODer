# Detecting LLM-Written Code: Fine-Tuned Encoders vs N-gram Baselines on AIGCodeSet

*Interim report, 2026-10-25. LT2326 Machine Learning. All scores are dev macro-F1 unless stated; the test split has not been touched.*

## Summary

Classification of code as human or LLM (and, in the multiclass setting, which LLM wrote it) on AIGCodeSet. A character n-gram logistic regression is a very strong baseline: fine-tuned UniXcoder only matches it, and CodeBERT and GraphCodeBERT fall clearly below it. Hyperparameter tuning of UniXcoder is exhausted (everything is a tie or worse). Identifier names are the largest single cue; layout and comments barely matter. Averaging the two best models gives a small, consistent gain. Final test numbers are still to come.

## 1. Task and data

- **Data:** AIGCodeSet: human submissions from CodeNet plus code written by CodeLlama 34B, Codestral 22B and Gemini 1.5 Flash. 7,531 samples after deduplication (train 5,267 / dev 739 / test 1,525), split by programming problem so no problem appears in two splits.
- **Scenarios:** the LLM either writes code from scratch (*generate*) or *edits* a human submission (fix an error, correct a wrong answer). Edits stay close to the human original and are much harder to detect.
- **Variants:** `binary` and `multiclass` use all data; `binary_generate` and `multiclass_generate` keep only human and from-scratch code. Rows whose identical code appears under both labels (478) are dropped from training.
- **Input:** normalized code only (trailing whitespace and final newline differences removed). Raw formatting differences existed in the data but are neutralized. An audit confirmed that metadata columns never reach a model.

## 2. Models

- **Baselines:** word and character n-gram TF-IDF + logistic regression. Tuned on 7-fold CV over the training folds: char n-grams (1,6), C=10.
- **Encoders:** CodeBERT, GraphCodeBERT, UniXcoder, fully fine-tuned (4 epochs, lr 2e-5, batch 16, no class weights, best epoch chosen on dev). Seeds 0-2.
- **Ensemble:** average of class probabilities from char LogReg and seed-averaged UniXcoder.

## 3. Findings

### 3.1 The simple baseline is hard to beat

| Variant | word LogReg | char LogReg (tuned) | CodeBERT (128 tok) | UniXcoder (128 tok) |
|---|---|---|---|---|
| binary | .672 | **.739** | .711 | .744 |
| binary_generate | .856 | **.911** | .801 | .892 |
| multiclass | .455 | **.568** | .511 | .538 |
| multiclass_generate | .688 | **.794** | .700 | .772 |

CodeBERT and UniXcoder rows are last-epoch (no selection), the fair comparison against a model fit once. UniXcoder at 256 tokens with best-epoch selection reaches .906 / .798 on the generate variants, level with char LogReg.

### 3.2 The pretrained checkpoint matters more than any hyperparameter

At 128 tokens on the generate variants: UniXcoder .896 / .778, GraphCodeBERT .869 / .739, CodeBERT .834 / .712. UniXcoder beats GraphCodeBERT in all six paired seed comparisons. Class weights (hurt multiclass), learning rate, epochs, batch size, 384 tokens, and dropout 0.2 (cost ~.045) never moved UniXcoder beyond the ±.02 seed noise. The reference configuration is therefore fixed: 256 tokens, lr 2e-5, 4 epochs, dropout 0.1.

| UniXcoder, generate variants, 3 seeds | binary_generate | multiclass_generate |
|---|---|---|
| Reference | .906 | .798 |
| 384 tokens | .911 | .774 |
| Dropout 0.2 | .862 | .751 |
| lr 3e-5 | .904 | .793 |
| 6 epochs | .907 | .816 |
| 6 epochs, lr 1e-5 | .901 | .780 |

Training loss falls to ~.03-.08 while dev loss bottoms out at epoch 2-4: the models memorize the 4-5k training samples.

### 3.3 What carries the signal

Ablating the char LogReg input (7-fold CV, generate variants):

| Input | binary_generate | multiclass_generate |
|---|---|---|
| Full code | .909 | .791 |
| Layout removed | .907 | .778 |
| Comments removed | .902 | .787 |
| Identifier names masked | .882 | .765 |

Masking identifier names costs the most (~.026), yet the model keeps most of its accuracy, so signal is spread over structure and idioms. There is no cheap shortcut. The hypothesis that CodeBERT loses because it cannot see layout was tested and not supported.

### 3.4 Attribution and data difficulty

- GEMINI is the hardest LLM to identify (F1 ~.4 for CodeBERT, ~.57-.60 for UniXcoder); LLAMA is the easiest (F1 .86).
- `binary` (.74) is far below `binary_generate` (.91) because edited code stays close to its human parent. The dataset paper reports the same pattern.

### 3.5 Combining the models

| Variant | UniXcoder (3-seed avg) | char LogReg | 50/50 mix | best weight tried |
|---|---|---|---|---|
| binary_generate | .914 | .911 | .922 | .927 |
| multiclass_generate | .822 | .794 | .825 | .827 |

The mix beats both models at every weight tried on both variants. The gain over the better single model is small (+.003 to +.013), so we treat it as promising, not proven, until test.

## 4. Relation to prior work

The AIGCodeSet paper uses TF-IDF/embedding features with classical classifiers on binary detection only (best F1 .63) and reports that edited code is hardest and Gemini the hardest generator; our results agree and add model attribution and fine-tuned encoders. AICD Bench reports that encoders overfit and generalize poorly across domains, and that attribution is hardest; this matches our results.

## 5. Limitations

- Dev is small (93 AI samples in binary_generate, 31 per class in multiclass_generate): differences under ~.03 are ties.
- Best-epoch scores are selected on dev and are slightly optimistic; ensemble weights were not tuned on held-out data.
- Ablations are not additive and ran on the generate variants only.
- ModernBERT could not be run (outdated library); a code LLM with LoRA is untested.
- No out-of-distribution evaluation yet (the proposal's MBPP stretch goal).

## 6. Remaining work

1. Result of the dropout 0.0 run (adopt only if it clears the +.01 rule).
2. Final run: reference configuration, four variants x 3 seeds, test evaluated once with the best-on-dev checkpoint; baseline on test.
3. Test-split ensemble (50/50, no tuning).
4. Optional: per-scenario and per-LLM breakdown; top char n-gram features; out-of-distribution test set.
