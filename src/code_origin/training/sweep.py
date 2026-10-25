"""Run models over several variants and seeds, logging everything to one file.

    PYTHONPATH=src python -m code_origin.training.sweep --model unixcoder --seeds 0 1 2 3 4 \
        --variants binary_generate multiclass_generate --with_test --save_ckpt
"""
import argparse, os, shlex, subprocess, sys
from datetime import datetime

from code_origin.config import ARGS, ROOT


ap = argparse.ArgumentParser()
ap.add_argument("--variants", nargs="+", default=[ARGS["variant"][1], ARGS["variant"][-1]])
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
ap.add_argument("--model", nargs="+", default=[ARGS["transformer_models"][1]])
ap.add_argument("--epochs", type=int)
ap.add_argument("--lr", type=float)
ap.add_argument("--max_length", type=int)
ap.add_argument("--batch_size", type=int)
ap.add_argument("--class_weight")
ap.add_argument("--dropout", type=float)
ap.add_argument("--weight_decay", type=float)
ap.add_argument("--warmup_ratio", type=float)
ap.add_argument("--with_test", action="store_true", help="after each training run, evaluate the saved model on test")
ap.add_argument("--save_ckpt", action="store_true", help="keep the checkpoint next to the run's metrics and plots")
a = ap.parse_args()

overrides = [x for k in ("epochs", "lr", "max_length", "batch_size", "class_weight", "dropout", "weight_decay", "warmup_ratio")
             if getattr(a, k) is not None for x in (f"--{k}", str(getattr(a, k)))]
log_dir = ROOT / "logs"
log_dir.mkdir(exist_ok=True)
log = log_dir / f"sweep_{datetime.now():%Y%m%d_%H%M}.log"


def run(cmd, env, header):
    print(header, flush=True)
    with log.open("a") as f:
        f.write(header)
    r = subprocess.run(["bash", "-c", f"set -o pipefail; {shlex.join(cmd)} 2>&1 | tee -a {shlex.quote(str(log))}"], env=env)
    subprocess.run(["sed", "-i", r"/^[[:space:]]*Epoch/d", str(log)], check=True)
    return r.returncode


for seed in a.seeds:
    for model in a.model:
        for variant in a.variants:
            base = [sys.executable, "-m", "code_origin.training.train_codebert", "--variant", variant, "--model", model, *overrides]
            env = {**os.environ, "SEED": str(seed), "PYTHONPATH": str(ROOT / "src")}
            train_cmd = base + (["--save_ckpt"] if a.save_ckpt else [])
            rc = run(train_cmd, env, f"\n##### seed {seed} | {variant} | {' '.join(overrides)}\n")
            if a.with_test:
                if rc != 0:
                    print(f"!!! training failed (exit {rc}), skipping test for seed {seed} {variant}", flush=True)
                    continue
                run(base + ["--test"], env, f"\n##### seed {seed} | {variant} | TEST\n")