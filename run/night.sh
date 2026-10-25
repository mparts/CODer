#!/usr/bin/env bash
cd "$(dirname "$0")/.."            # project root (this file lives in run/)
export PYTHONPATH=src
# SW="python3 -m code_origin.training.sweep --model unixcoder --seeds 0 1 2 --class_weight none --batch_size 16"
# $SW --max_length 256 --lr 2e-5 --epochs 6 --dropout 0.0
# $SW --max_length 256 --lr 2e-5 --epochs 4 --dropout 0.0
# $SW --max_length 384 --lr 2e-5 --epochs 4                    # 2. longer context
# $SW --max_length 256 --lr 2e-5 --epochs 4 --dropout 0.2      # 3. dropout
# $SW --max_length 256 --lr 3e-5 --epochs 4                    # 4. higher lr
# $SW --max_length 256 --lr 2e-5 --epochs 6                    # 5. more epochs
# $SW --max_length 256 --lr 1e-5 --epochs 6                    # 6. lower lr, more epochs

SW="python3 -m code_origin.training.sweep --model unixcoder --seeds 0 1 2 --class_weight none --max_length 256 --lr 2e-5 --epochs 4 --dropout 0.0"
$SW --weight_decay 0.1
$SW --batch_size 8
$SW --batch_size 32