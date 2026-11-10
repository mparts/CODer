#!/usr/bin/env bash
#
# Script: RUN.sh
# Description: Runs the CODer pipeline. Logs all output.
# Usage: bash RUN.sh            stages as set in run.conf
#        bash RUN.sh og.conf    the original experiment from scratch
#        bash RUN.sh hub.conf   test predictions from the Hugging Face data and checkpoints
#

cd "$(dirname "$0")" || exit 1
source "./run.conf"
if [ -n "$1" ]; then
    source "./$1" || exit 1
fi


LINE="|==========|==========|==========|==========|==========|==========|==========|==========|==========|"
if $APPEND_LOG; then
    echo "$LINE" | tee -a "$LOG_FILE"
else
    echo "$LINE" | tee "$LOG_FILE"
fi

section() {
    printf "    :    :\n    |    |\n%s\n    |    |\n    :    :\n" "$1" | tee -a "$LOG_FILE"
}

run_step() {
    local enabled=$1 script=$2
    shift 2

    if $enabled; then
        section "~ Running $script ..."
        PYTHONPATH=../src $PYTHON -m "code_origin.$script" "$@" 2>&1 | tee -a "$LOG_FILE"
    else
        section "<!> $script skipped. <!>"
    fi
}

fetch() {
    local enabled=$1 repo=$2 type=$3 files=$4 dir=$5

    if $enabled; then
        section "~ Fetching $files from $repo ..."
        $PYTHON -c "from huggingface_hub import snapshot_download; snapshot_download('$repo', repo_type='$type', allow_patterns='$files', local_dir='$dir')" 2>&1 | tee -a "$LOG_FILE"
    else
        section "<!> fetch from $repo skipped. <!>"
    fi
}


section "CODer Pipeline"

if ! $DOWNLOAD_DATASET && ! $PREPROCESS_DATASET && ! $FETCH_DATA && ! $PRINT_DATASET_STATS; then
    echo "$LINE" | tee -a "$LOG_FILE"
    section "<!> Part 1. Dataset download and preprocessing skipped.<!>"
else
    echo "$LINE" | tee -a "$LOG_FILE"
    section "Part 1. Dataset download and preprocessing"
    echo "$LINE" | tee -a "$LOG_FILE"
    run_step $DOWNLOAD_DATASET data.download
    run_step $PREPROCESS_DATASET data.preprocess
    fetch $FETCH_DATA "$DATA_REPO" dataset "*.parquet" ../data/processed
    run_step $PRINT_DATASET_STATS data.audit
fi


if ! $TRAIN_BASELINE; then
    echo "$LINE" | tee -a "$LOG_FILE"
    section "<!> Part 2. Baseline training and evaluation skipped. <!>"
else
    echo "$LINE" | tee -a "$LOG_FILE"
    section "Part 2. Baseline training and evaluation"
    echo "$LINE" | tee -a "$LOG_FILE"
    for v in $DATASETS; do
        for m in $BASE_MODELS; do
            run_step true training.train_baseline --variant "$v" --model "$m"
            run_step true training.train_baseline --variant "$v" --model "$m" --test
        done
    done
fi


if ! $TRAIN_CODEBERT; then
    echo "$LINE" | tee -a "$LOG_FILE"
    section "<!> Part 3. Transformer training and evaluation skipped. <!>"
else
    echo "$LINE" | tee -a "$LOG_FILE"
    section "Part 3. Transformer training and evaluation"
    echo "$LINE" | tee -a "$LOG_FILE"
    for v in $DATASETS; do
        run_step true training.sweep --model $TRANSFORMER_MODELS --seeds $SEEDS --variants "$v" \
            --max_length 256 --lr 2e-5 --epochs 4 --dropout 0.0 --batch_size 16 \
            --weight_decay 0.01 --warmup_ratio 0.1 --class_weight none --with_test --save_ckpt
        sed -i '/^[[:space:]]*Epoch/d' "$LOG_FILE"
    done
fi


if ! $PREDICT_CODEBERT; then
    echo "$LINE" | tee -a "$LOG_FILE"
    section "<!> Part 4. Test predictions from the Hub checkpoints skipped. <!>"
else
    echo "$LINE" | tee -a "$LOG_FILE"
    section "Part 4. Test predictions from the Hub checkpoints"
    echo "$LINE" | tee -a "$LOG_FILE"
    for s in $SEEDS; do
        section "Seed $s"
        for v in $DATASETS; do
            fetch true "$CKPT_REPO" model "seed$s/$v/*" ../artifacts/CODer-unixcoder
            SEED=$s run_step true training.train_codebert --variant "$v" --model unixcoder --test \
                --model_dir "../artifacts/CODer-unixcoder/seed$s/$v"
        done
    done
fi


if ! $PAPER_TABLES; then
    echo "$LINE" | tee -a "$LOG_FILE"
    section "<!> Part 5. Paper tables skipped. <!>"
else
    echo "$LINE" | tee -a "$LOG_FILE"
    section "Part 5. Paper tables (Tables 2 and 5) from the test predictions in docs/experiments/archive/"
    echo "$LINE" | tee -a "$LOG_FILE"
    for v in $DATASETS; do
        run_step true evaluation.final_eval "$v"
        run_step true evaluation.breakdown "$v"
    done
fi

echo "$LINE" | tee -a "$LOG_FILE"
echo "^^^                                       ^^^" | tee -a "$LOG_FILE"
echo "^^^    End of RUN.sh pipeline.            ^^^" | tee -a "$LOG_FILE"
echo "^^^                                       ^^^" | tee -a "$LOG_FILE"
echo "$LINE" | tee -a "$LOG_FILE"

exit 0