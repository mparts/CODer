#!/usr/bin/env bash
# run/night.sh: final UniXcoder runs, one sweep (and one log) per variant
cd "$(dirname "$0")/.."
export PYTHONPATH=src

for v in binary binary_generate multiclass multiclass_generate; do
  python3 -m code_origin.training.sweep --model unixcoder --seeds 3 4 5 6 7 8 \
    --variants $v \
    --max_length 256 --lr 2e-5 --epochs 4 --dropout 0.0 --batch_size 16 \
    --weight_decay 0.01 --warmup_ratio 0.1 --class_weight none --with_test --save_ckpt
done