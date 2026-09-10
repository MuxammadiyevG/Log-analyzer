"""
CLI entry point for the evaluation harness.

Examples:
    # Baseline the current IsolationForest on the project's own web logs
    python -m evaluation.run --dataset weblog --data data/sample_logs.log \
        --report reports/weblog_eval.md --json reports/weblog_eval.json

    # Same, but with the autoencoder
    python -m evaluation.run --dataset weblog --model autoencoder

    # Inspect an HDFS session dataset (scaffold; for future sequence models)
    python -m evaluation.run --dataset hdfs --hdfs-log HDFS.log --hdfs-label anomaly_label.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.datasets import DEFAULT_WEBLOG, load_hdfs_sessions  # noqa: E402
from evaluation.report import render_console, save_json, save_markdown  # noqa: E402
from evaluation.runner import (  # noqa: E402
    run_hdfs_deeplog,
    run_sequence_hard,
    run_sequence_synth,
    run_weblog,
    run_weblog_compare,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Log anomaly detection — evaluation harness")
    p.add_argument(
        "--dataset",
        choices=["weblog", "hdfs", "seqsynth", "seqmatched"],
        default="weblog",
    )
    p.add_argument("--epochs", type=int, default=30, help="DeepLog training epochs")
    p.add_argument("--data", type=str, default=str(DEFAULT_WEBLOG), help="Web log path")
    p.add_argument(
        "--model",
        choices=["isolation_forest", "autoencoder", "semantic", "deeplog"],
        default="isolation_forest",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="Train baseline + semantic on the same benchmark, one table",
    )
    p.add_argument("--test-fraction", type=float, default=0.3)
    p.add_argument("--anomaly-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report", type=str, default=None, help="Markdown output path")
    p.add_argument("--json", type=str, default=None, help="JSON output path")
    # HDFS
    p.add_argument("--hdfs-log", type=str, default=None)
    p.add_argument("--hdfs-label", type=str, default=None)
    p.add_argument("--hdfs-max-train", type=int, default=6000)
    p.add_argument("--hdfs-max-test", type=int, default=50000)
    args = p.parse_args()

    if args.dataset in ("seqsynth", "seqmatched"):
        if args.dataset == "seqmatched":
            bench, results = run_sequence_hard(seed=args.seed, epochs=args.epochs)
        else:
            bench, results = run_sequence_synth(seed=args.seed, epochs=args.epochs)
        print()
        print(render_console(results))
        print()
        if args.report:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            save_markdown(
                results,
                Path(args.report),
                title="Sequence benchmark — DeepLog vs Markov (Phase 3)",
                meta=bench.meta,
            )
            logger.info(f"Markdown report -> {args.report}")
        if args.json:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            save_json(results, Path(args.json), meta=bench.meta)
        return

    if args.dataset == "hdfs":
        if not args.hdfs_log or not args.hdfs_label:
            p.error("--dataset hdfs requires --hdfs-log and --hdfs-label")
        if args.model == "deeplog":
            meta, results = run_hdfs_deeplog(
                Path(args.hdfs_log),
                Path(args.hdfs_label),
                epochs=args.epochs,
                max_train_sessions=args.hdfs_max_train,
                max_test_normal=args.hdfs_max_test,
                seed=args.seed,
            )
            print()
            print(render_console(results))
            print()
            if args.report:
                Path(args.report).parent.mkdir(parents=True, exist_ok=True)
                save_markdown(
                    results, Path(args.report), title="HDFS sequence benchmark", meta=meta
                )
            if args.json:
                Path(args.json).parent.mkdir(parents=True, exist_ok=True)
                save_json(results, Path(args.json), meta=meta)
            return
        ds = load_hdfs_sessions(Path(args.hdfs_log), Path(args.hdfs_label))
        logger.info(
            f"Loaded HDFS: {len(ds.sessions)} sessions, {int(ds.labels.sum())} "
            "anomalous. Add --model deeplog to run the sequence benchmark; the "
            "point-wise models do not consume sessions."
        )
        return

    if args.compare:
        bench, results = run_weblog_compare(
            Path(args.data),
            test_fraction=args.test_fraction,
            anomaly_fraction=args.anomaly_fraction,
            seed=args.seed,
        )
        title = "Baseline vs Semantic (Phase 1 vs Phase 2)"
    else:
        bench, results = run_weblog(
            Path(args.data),
            model_type=args.model,
            test_fraction=args.test_fraction,
            anomaly_fraction=args.anomaly_fraction,
            seed=args.seed,
        )
        title = f"Web-log evaluation ({args.model})"

    print()
    print(render_console(results))
    print()

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        save_markdown(results, Path(args.report), title=title, meta=bench.meta)
        logger.info(f"Markdown report -> {args.report}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        save_json(results, Path(args.json), meta=bench.meta)
        logger.info(f"JSON report -> {args.json}")


if __name__ == "__main__":
    main()
