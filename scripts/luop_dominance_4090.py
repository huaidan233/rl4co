import argparse
import csv
import math
import subprocess
from pathlib import Path


DEFAULT_COMMAND = [
    "python",
    "run.py",
    "experiment=cityplan/dominance",
    "trainer.max_epochs=3",
    "trainer.accelerator=gpu",
    "+trainer.devices=1",
    "trainer.precision=16-mixed",
    "model.batch_size=256",
    "model.val_batch_size=256",
    "model.test_batch_size=256",
    "model.train_data_size=1024",
    "model.val_data_size=128",
    "model.test_data_size=128",
    "model.generate_default_data=false",
    "env.val_file=null",
    "env.test_file=null",
    "callbacks.rich_progress_bar=null",
    "callbacks.learning_rate_monitor=null",
    "+trainer.log_every_n_steps=1",
    "logger=csv",
]

REQUIRED_METRICS = {
    "train/dominance_reward",
    "train/hv_contribution",
    "val/pareto_hypervolume",
    "val/pareto_front_size",
}


def build_command(extra_args=None):
    command = list(DEFAULT_COMMAND)
    if extra_args:
        command.extend(extra_args)
    return command


def _float_or_none(value):
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def scan_csv_metrics(metrics_path):
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics file not found: {metrics_path}")

    seen = set()
    finite_values = {}
    with metrics_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            for metric in REQUIRED_METRICS:
                if metric not in row:
                    continue
                value = _float_or_none(row.get(metric))
                if value is None:
                    continue
                seen.add(metric)
                finite_values.setdefault(metric, []).append(value)

    missing = REQUIRED_METRICS - seen
    if missing:
        raise ValueError(f"missing finite metrics: {sorted(missing)}")
    if len(set(finite_values["train/dominance_reward"])) <= 1:
        raise ValueError("train/dominance_reward collapsed to a constant value")
    return finite_values


def main():
    parser = argparse.ArgumentParser(description="Run or inspect LUOP dominance smoke.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command only.")
    parser.add_argument("--metrics", type=Path, help="Scan a Lightning CSV metrics file.")
    parser.add_argument("overrides", nargs="*", help="Additional Hydra overrides.")
    args = parser.parse_args()

    if args.metrics is not None:
        values = scan_csv_metrics(args.metrics)
        for metric, metric_values in sorted(values.items()):
            print(f"{metric}: count={len(metric_values)} last={metric_values[-1]:.6f}")
        return

    command = build_command(args.overrides)
    if args.dry_run:
        print(" ".join(command))
        return
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
