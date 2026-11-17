"""Command-line entry point: `wct2g1 <stage> [--limit N] [--ids ...]` and `wct2g1 report`."""
from __future__ import annotations

import argparse
import importlib
import logging
import os
import random
import sys
from datetime import datetime

import numpy as np

from .config import load_config
from .manifest import STAGE_ORDER, read_manifest, status_kind

# CLI stage name -> module implementing `run(cfg, limit=, ids=, ...)`.
STAGES = {
    "acquire": "wct2g1.acquire",
    "segment": "wct2g1.segment",
    "track": "wct2g1.track",
    "reconstruct": "wct2g1.reconstruct",
    "clean": "wct2g1.clean",
    "retarget": "wct2g1.retarget",
    "train_filter": "wct2g1.train_filter",
}


def report(cfg) -> None:
    print(f"config {cfg.path} (hash {cfg.hash})\n")
    print(f"{'stage':<8} {'total':>7} {'ok':>7} {'rejected':>9} {'failed':>7}")
    details = []
    for stage in STAGE_ORDER:
        df = read_manifest(cfg, stage)
        if df.empty:
            print(f"{stage:<8} {'-':>7}")
            continue
        kinds = df["status"].map(status_kind).value_counts()
        print(f"{stage:<8} {len(df):>7} {kinds.get('ok', 0):>7} "
              f"{kinds.get('rejected', 0):>9} {kinds.get('failed', 0):>7}")
        bad = df.loc[df["status"] != "ok", "status"].value_counts()
        if len(bad):
            details.append((stage, bad))
    for stage, bad in details:
        print(f"\n{stage}: reasons")
        for reason, n in bad.head(15).items():
            print(f"  {n:>6}  {reason}")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wct2g1")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in STAGES:
        sp = sub.add_parser(name)
        sp.add_argument("--limit", type=int, default=None, help="process at most N items")
        sp.add_argument("--ids", nargs="+", default=None, help="only these item ids")
        sp.add_argument("--shard", default=None,
                        help="i/n: only items whose id hashes to shard i of n (SLURM arrays)")
        if name == "acquire":
            sp.add_argument("--manifest-only", action="store_true",
                            help="skip downloading; rebuild raw manifest from disk")
    sub.add_parser("report")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    log_dir = cfg.path_of("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / f"{args.cmd}_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(logfile)],
    )
    logging.getLogger(__name__).info("config hash %s", cfg.hash)
    seed_all(cfg["seed"])

    if args.cmd == "report":
        report(cfg)
        return 0

    if getattr(args, "shard", None):
        os.environ["WCT2G1_SHARD"] = args.shard
        logging.getLogger(__name__).info("shard %s", args.shard)
    mod = importlib.import_module(STAGES[args.cmd])
    kwargs = {"limit": args.limit, "ids": args.ids}
    if args.cmd == "acquire":
        kwargs["manifest_only"] = args.manifest_only
    mod.run(cfg, **kwargs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
