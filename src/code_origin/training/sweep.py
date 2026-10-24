"""Run CodeBERT over several variants and seeds, logging everything to one file.

    PYTHONPATH=src python -m code_origin.training.sweep --epochs 4 --lr 2e-5 --class_weight none --max_length 128
"""
import argparse, os, shlex, subprocess, sys
from datetime import datetime

from code_origin.config import ARGS, ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--variants", nargs="+", default=ARGS["variant"])
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
ap.add_argument("--model", default=ARGS["transformer_models"][1])
ap.add_argument("--epochs", type=int)
ap.add_argument("--lr", type=float)
ap.add_argument("--max_length", type=int)
ap.add_argument("--batch_size", type=int)
ap.add_argument("--class_weight")
a = ap.parse_args()

overrides = [x for k in ("epochs", "lr", "max_length", "batch_size", "class_weight")
             if getattr(a, k) is not None for x in (f"--{k}", str(getattr(a, k)))]
log_dir = ROOT / "logs"
log_dir.mkdir(exist_ok=True)
log = log_dir / f"sweep_{datetime.now():%Y%m%d_%H%M}.log"

for seed in a.seeds:
    for variant in a.variants:
        cmd = [sys.executable, "-m", "code_origin.training.train_codebert",
               "--variant", variant, "--model", a.model, *overrides]
        env = {**os.environ, "SEED": str(seed), "PYTHONPATH": str(ROOT / "src")}
        header = f"\n##### seed {seed} | {variant} | {' '.join(overrides)}\n"
        print(header, flush=True)
        with log.open("a") as f:
            f.write(header)
        subprocess.run(f"{shlex.join(cmd)} 2>&1 | tee -a {shlex.quote(str(log))}", shell=True, env=env)