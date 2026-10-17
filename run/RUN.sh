#!/usr/bin/env bash
#
# Script: RUN.sh
# Description: Runs the CODer pipeline. Logs all output.
# Usage: bash RUN.sh
#

cd "$(dirname "$0")" || exit 1
source "./run.conf"

LINE="===================================================================================================="

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
echo "$LINE" | tee -a "$LOG_FILE"


section "Part 1. Dataset download and preprocessing"
run_step $DOWNLOAD_DATASET data.download
run_step $PREPROCESS_DATASET data.preprocess
run_step $PRINT_DATASET_STATS data.audit

# for t in "1 2 3 4"; do
#     run_step $DOWNLOAD_DATASET data.download
# done
for v in binary binary_generate multiclass multiclass_generate; do 
    for m in tfidf_word tfidf_char; do 
        run_step $TRAIN_BASELINE training.train --variant $v --model $m
    done; 
done 

echo "$LINE" | tee -a "$LOG_FILE"
echo "^^^                                       ^^^" | tee -a "$LOG_FILE"
echo "^^^    End of RUN.sh pipeline.            ^^^" | tee -a "$LOG_FILE"
echo "^^^                                       ^^^" | tee -a "$LOG_FILE"
echo "$LINE" | tee -a "$LOG_FILE"

exit 0