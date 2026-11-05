#!/usr/bin/env python3
"""Train Tabula on the SMOTE-balanced training set and draw candidate synthetic records.

The only long-running step of the reproduction (hours on one GPU), so it lives outside
the notebook. Training and sampling are separate stages so that training can be spread
over several GPUs with torchrun while sampling stays on one:

    # single GPU, everything in one go
    python tools/tabula_synthesis.py

    # two GPUs: distribution is handled by torchrun, the global batch size is kept
    torchrun --nproc_per_node=2 tools/tabula_synthesis.py --stage train --devices 2
    python tools/tabula_synthesis.py --stage sample --from-pickle tabula_weights/run/tabula.pkl

Outputs
-------
    data/processed/synthetic_pool.csv     candidate records, label-valid, rounded
    data/processed/synthetic_pool.json    run metadata (parameters, timings, counts)
    tabula_weights/<run>/                 checkpoint (git-ignored)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")   # see README
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["all", "train", "sample"], default="all")
    p.add_argument("--train", default=str(ROOT / "data" / "processed" / "train_smote_panel_a.csv"))
    p.add_argument("--out", default=str(ROOT / "data" / "processed" / "synthetic_pool.csv"))
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=32,
                   help="global batch size; divided across the visible devices")
    p.add_argument("--n-samples", type=int, default=10000)
    p.add_argument("--k", type=int, default=256,
                   help="sequences generated per forward pass (memory permitting)")
    p.add_argument("--max-length", type=int, default=0,
                   help="0 = derive from the token spans of the training table")
    p.add_argument("--experiment-dir", default=str(ROOT / "tabula_weights" / "run_current"))
    p.add_argument("--llm", default=str(ROOT / "tabula_weights" / "distilgpt2"),
                   help="local distilgpt2 directory (see README); falls back to the Hub id")
    p.add_argument("--init", choices=["pretrained", "scratch"], default="pretrained")
    p.add_argument("--from-pickle", default=None)
    p.add_argument("--devices", type=int, default=1,
                   help="informational: number of torchrun workers actually used")
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16",
                   help="mixed precision for training")
    p.add_argument("--workers", type=int, default=8,
                   help="DataLoader worker processes (the collator is Python-heavy)")
    p.add_argument("--smoke", action="store_true", help="1 epoch, 64 records")
    return p.parse_args()


def read_training_table(args):
    """Load the SMOTE-balanced table and order its columns the way Tabula learns best.

    The first column of a Tabula record is encoded as "<column name> <value>" and the
    generator starts from the conditional label, so sparse integer columns placed first are
    reproduced badly (docs/devlog.md 8.1). Ordering the record as binary symptoms ->
    laboratory values -> age -> day variables reproduces the 2025 layout and fixes it.
    """
    data = pd.read_csv(Path(args.train).resolve(), index_col=0)
    manifest = json.loads((ROOT / "data" / "processed" / "part2_manifest.json").read_text())
    bins, days = manifest["binary"], manifest["day_vars"]
    middle = [c for c in data.columns if c not in bins and c not in days + ["Age", "Label"]]
    order = bins + middle + ["Age"] + days + ["Label"]
    assert set(order) == set(data.columns), "column order does not cover the table"
    return data[order], manifest


def build_model(args, exp_dir):
    from tabula_middle_padding import Tabula
    llm = args.llm if Path(args.llm).exists() else "distilgpt2"
    per_device = max(1, args.batch_size // max(WORLD_SIZE, 1))
    precision = {} if args.dtype == "fp32" else {args.dtype: True}
    if args.workers:
        precision.update({"dataloader_num_workers": args.workers,
                          "dataloader_pin_memory": True})
    model = Tabula(llm=llm, experiment_dir=str(exp_dir),
                   batch_size=per_device, epochs=args.epochs, **precision)
    if args.init == "pretrained":
        from transformers import AutoModelForCausalLM
        model.model = AutoModelForCausalLM.from_pretrained(llm, config=model.config)
        if RANK == 0:
            print(f"model initialised from pretrained weights: {llm}", flush=True)
    return model, llm


def stage_train(args, data, exp_dir):
    model, llm = build_model(args, exp_dir)
    if RANK == 0:
        print(f"world size {WORLD_SIZE}, per-device batch size {max(1, args.batch_size // max(WORLD_SIZE, 1))}, "
              f"global batch size {args.batch_size}", flush=True)
    t0 = time.time()
    # The vendored Tabula builds the LM with AutoModelForCausalLM.from_config(),
    # i.e. from *random* weights; see docs/devlog.md section 8.1.
    model.fit(data, conditional_col="Label")
    fit_seconds = time.time() - t0

    if RANK == 0:
        print(f"\ntraining finished in {fit_seconds / 60:.1f} min", flush=True)
        import dill, torch
        model.model.to("cpu")
        with open(exp_dir / "tabula.pkl", "wb") as fh:
            dill.dump(model, fh)
        torch.save(model.model.state_dict(), exp_dir / "tabula_state_dict.pt")
        print(f"checkpoint saved to {exp_dir}", flush=True)
    return model, fit_seconds, fit_seconds, llm


def stage_sample(args, model, data, manifest, llm, fit_minutes):
    import torch
    span_total = int(sum(model.token_list_length))
    max_length = args.max_length or span_total + 16
    print(f"token span per record: {span_total} -> max_length={max_length}", flush=True)

    t0 = time.time()
    synthetic = model.sample(n_samples=args.n_samples, k=args.k, max_length=max_length)
    sample_seconds = time.time() - t0
    print(f"sampling finished in {sample_seconds / 60:.1f} min", flush=True)

    n_drawn = len(synthetic)

    # --- repair the conditional column -------------------------------------
    # The vendored sampler prepends the conditional token pair ("Label 0") to the
    # generated sequence, but the parser assigns spans column by column from the first
    # column onward. The first column therefore receives the conditional token and the
    # generated label at the end of the sequence is not the conditioning label
    # (docs/devlog.md, section 8.2). Recover the conditioning label from that first
    # column, write it back as Label, and redraw the displaced column from the empirical
    # marginal observed in the training corpus.
    # the training table is reordered so the conditional label sits last and the first
    # data column is the first binary symptom
    first_col = manifest["binary"][0]
    priming = synthetic[first_col].round().clip(0, 1).astype(int)
    synthetic["Label"] = priming
    rng = np.random.default_rng(42)
    train_dist = pd.read_csv(Path(args.train).resolve(), index_col=0)[first_col]
    synthetic[first_col] = rng.choice(sorted(train_dist.unique()),
                                      size=len(synthetic),
                                      p=[(train_dist == v).mean() for v in sorted(train_dist.unique())])

    synthetic = synthetic[synthetic["Label"].isin([0, 1])].copy()
    for col in manifest["binary"]:
        synthetic[col] = synthetic[col].round().clip(0, 1).astype(int)
    for col in manifest["day_vars"]:
        synthetic[col] = synthetic[col].round().astype(int)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canonical = manifest["panel_a"] + ["Label"]
    synthetic = synthetic[canonical].reset_index(drop=True)
    synthetic.to_csv(out_path, index=False)

    meta = {
        "source": str(Path(args.train).resolve().relative_to(ROOT)),
        "output": str(out_path.relative_to(ROOT)),
        "llm": llm,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "devices": args.devices,
        "n_requested": args.n_samples,
        "n_drawn": n_drawn,
        "n_label_valid": int(len(synthetic)),
        "init": args.init if not args.from_pickle else "from_pickle",
        "k": args.k,
        "max_length": max_length,
        "token_span": span_total,
        "label_counts": {int(k): int(v) for k, v in
                         synthetic["Label"].value_counts().sort_index().items()},
        "fit_minutes": round(fit_minutes, 1),
        "sample_minutes": round(sample_seconds / 60, 1),
        "smoke": bool(args.smoke),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    print(f"\nwritten: {out_path}", flush=True)


def main():
    args = parse_args()
    if args.smoke:
        args.epochs, args.n_samples, args.k = 1, 64, 64

    data, manifest = read_training_table(args)
    exp_dir = Path(args.experiment_dir).resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)

    if RANK == 0:
        print(f"training corpus : {Path(args.train).name} {data.shape}", flush=True)
        print(f"class balance   : {dict(data['Label'].value_counts().sort_index())}", flush=True)
        print(f"tabula          : distilgpt2, epochs={args.epochs}, batch_size={args.batch_size}, "
              f"precision={args.dtype}", flush=True)
        print(f"generation      : n_samples={args.n_samples}, k={args.k}", flush=True)

    if args.stage in ("all", "train"):
        model, fit_seconds, fit_minutes, llm = stage_train(args, data, exp_dir)
        if args.stage == "train":
            return
        if WORLD_SIZE > 1:
            raise SystemExit("stage=all is single-process; use --stage train and --stage sample")
    else:
        import dill
        if not args.from_pickle:
            raise SystemExit("--stage sample needs --from-pickle")
        with open(args.from_pickle, "rb") as fh:
            model = dill.load(fh)
        model.model.to("cuda" if __import__("torch").cuda.is_available() else "cpu")
        llm, fit_minutes = "from_pickle", 0.0
        print(f"loaded Tabula from {args.from_pickle}", flush=True)

    if RANK == 0:
        stage_sample(args, model, data, manifest, llm, fit_minutes)


if __name__ == "__main__":
    main()
