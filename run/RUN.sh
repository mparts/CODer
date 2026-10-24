#!/usr/bin/env bash
#
# Script: RUN.sh
# Description: Runs the CODer pipeline. Logs all output.
# Usage: bash RUN.sh
#

cd "$(dirname "$0")" || exit 1
source "./run.conf"


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


section "CODer Pipeline"

if ! $DOWNLOAD_DATASET && ! $PREPROCESS_DATASET && ! $PRINT_DATASET_STATS; then
    echo "$LINE" | tee -a "$LOG_FILE"
    section "<!> Part 1. Dataset download and preprocessing skipped.<!>"
else
    echo "$LINE" | tee -a "$LOG_FILE"
    section "Part 1. Dataset download and preprocessing"
    echo "$LINE" | tee -a "$LOG_FILE"
    run_step $DOWNLOAD_DATASET data.download
    run_step $PREPROCESS_DATASET data.preprocess
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
    section "Seed $s"
    for v in $DATASETS; do
        for m in $TRANSFORMER_MODELS; do
            run_step true training.train_codebert --variant "$v" --model "$m"
            sed -i '/^[[:space:]]*Epoch/d' "$LOG_FILE"
        done
    done
fi

echo "$LINE" | tee -a "$LOG_FILE"
echo "^^^                                       ^^^" | tee -a "$LOG_FILE"
echo "^^^    End of RUN.sh pipeline.            ^^^" | tee -a "$LOG_FILE"
echo "^^^                                       ^^^" | tee -a "$LOG_FILE"
echo "$LINE" | tee -a "$LOG_FILE"

exit 0