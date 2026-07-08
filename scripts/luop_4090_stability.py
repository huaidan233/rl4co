import argparse
import csv
import math
import os
import re
import subprocess
import sys
import threading

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf


STABILITY_PATTERNS = {
    "cuda_oom": re.compile(r"cuda out of memory|outofmemoryerror", re.IGNORECASE),
    "nan_or_inf": re.compile(
        r"(^|[^a-z])(nan|inf|non[-_ ]?finite)([^a-z]|$)", re.IGNORECASE
    ),
    "all_false_mask": re.compile(
        r"all[-_ ]false|no[_ ]valid[_ ]action|no valid .*joint action",
        re.IGNORECASE,
    ),
    "infeasible_action": re.compile(
        r"infeasible .*action|mismatched .*joint actions|"
        r"component actions must be in valid .*ranges|"
        r"flat actions must be in valid .*range|"
        r"LUOP replay .*actions .*must be in|"
        r"LUOP replay .*actions .*integer ids|"
        r"LUOP replay type_actions and parcel_actions .*padding|"
        r"evaluate_pareto_front requires .*actions .*range|"
        r"evaluate_pareto_front requires .*actions padding|"
        r"evaluate_pareto_front requires type_actions and parcel_actions "
        r"to encode the returned flat actions|"
        r"evaluate_pareto_front requires returned actions to include "
        r"one valid action|"
        r"evaluate_pareto_front requires returned actions to satisfy "
        r"the current LUOP action_mask|sampled bad values",
        re.IGNORECASE,
    ),
    "invalid_distribution": re.compile(
        r"invalid probability distribution|probability tensor contains|"
        r"probabilities contain|invalid multinomial distribution",
        re.IGNORECASE,
    ),
    "invalid_plan": re.compile(
        r"violates .*ratios|fixed parcels|all parcels must "
        r"(be assigned|use a known land-use type)|"
        r"evaluate_pareto_front requires current_plan",
        re.IGNORECASE,
    ),
}

REQUIRED_SIGNAL_PATTERNS = {
    "missing_loss_metric": re.compile(
        r"(?<![a-z0-9_])(?:train|val|test)/loss(?:_(?:step|epoch))?"
        r"(?=\s*[=,]|\b)|"
        r"(?<![a-z0-9_/])loss(?=\s*=)",
        re.IGNORECASE,
    ),
    "missing_scalar_reward_metric": re.compile(
        r"(?<![a-z0-9_])(?:train|val|test)/reward(?:_(?:step|epoch))?"
        r"(?=\s*[=,]|\b)|"
        r"(?<![a-z0-9_/])reward(?=\s*=)",
        re.IGNORECASE,
    ),
    "missing_compatibility_metric": re.compile(r"compatibility_reward", re.IGNORECASE),
    "missing_accessibility_metric": re.compile(r"accessibility_reward", re.IGNORECASE),
    "missing_pareto_hypervolume": re.compile(r"pareto_hypervolume", re.IGNORECASE),
    "missing_pareto_front_size": re.compile(r"pareto_front_size", re.IGNORECASE),
    "missing_checkpoint_score": re.compile(r"checkpoint_score", re.IGNORECASE),
}

REQUIRED_METRIC_TOKENS = {
    "missing_loss_metric": re.compile(r"(^|/)loss(?:_(?:step|epoch))?$", re.IGNORECASE),
    "missing_scalar_reward_metric": re.compile(
        r"(^|/)reward(?:_(?:step|epoch))?$", re.IGNORECASE
    ),
    "missing_compatibility_metric": re.compile(
        r"(^|/)compatibility_reward(?:_(?:step|epoch))?$", re.IGNORECASE
    ),
    "missing_accessibility_metric": re.compile(
        r"(^|/)accessibility_reward(?:_(?:step|epoch))?$", re.IGNORECASE
    ),
    "missing_pareto_hypervolume": re.compile(
        r"(^|/)pareto_hypervolume(?:_(?:step|epoch))?$", re.IGNORECASE
    ),
    "missing_pareto_front_size": re.compile(
        r"(^|/)pareto_front_size(?:_(?:step|epoch))?$", re.IGNORECASE
    ),
    "missing_checkpoint_score": re.compile(
        r"(^|/)checkpoint_score(?:_(?:step|epoch))?$", re.IGNORECASE
    ),
}

STABILITY_CURVE_METRIC_TOKENS = {
    "unstable_val_reward_curve": re.compile(
        r"^val/reward(?:_(?:step|epoch))?$", re.IGNORECASE
    ),
    "unstable_val_compatibility_reward_curve": re.compile(
        r"^val/compatibility_reward(?:_(?:step|epoch))?$", re.IGNORECASE
    ),
    "unstable_val_accessibility_reward_curve": re.compile(
        r"^val/accessibility_reward(?:_(?:step|epoch))?$", re.IGNORECASE
    ),
    "unstable_val_pareto_hypervolume_curve": re.compile(
        r"^val/pareto_hypervolume(?:_(?:step|epoch))?$", re.IGNORECASE
    ),
    "unstable_val_checkpoint_score_curve": re.compile(
        r"^val/checkpoint_score(?:_(?:step|epoch))?$", re.IGNORECASE
    ),
}

GPU_PEAK_MEMORY_PATTERN = re.compile(r"gpu_peak_memory_mib\s*=\s*([0-9.]+)")
GPU_MEMORY_LIMIT_EXCEEDED_PATTERN = re.compile(
    r"gpu_memory_limit_exceeded_mib\s*=\s*([0-9.]+)"
)
MIN_GPU_PEAK_MEMORY_MIB = 8000
MAX_GPU_MEMORY_MIB = 22000
MIN_STABILITY_CURVE_POINTS = 4
MIN_SUSTAINED_DRIFT_POINTS = 8
MAX_VALIDATION_METRIC_SWING = 0.05
MAX_VALIDATION_METRIC_DRIFT = 0.025
MIN_FULL_VALIDATION_POINTS = MIN_STABILITY_CURVE_POINTS
EXPECTED_EVAL_OBJECTIVE_WEIGHTS = [0.5, 0.5]
EXPECTED_CHECKPOINT_MONITOR = "val/checkpoint_score"
EXPECTED_LUOP_MODEL_TARGET = "rl4co.models.LUOPAttentionModel"
EXPECTED_DECODE_TYPE_FIRST = False
MIN_DEFAULT_CONSTRAINT_NUM_LOC = 12
SAFE_MAX_BATCH_BY_NUM_LOC = {
    200: 256,
}
NVIDIA_SMI_CANDIDATES = (
    "nvidia-smi",
    "/usr/lib/wsl/lib/nvidia-smi",
)


@dataclass
class GpuPreflightResult:
    names: list[str]
    selected_index: int

    def __eq__(self, other):
        if isinstance(other, list):
            return self.names == other
        return super().__eq__(other)

    def __iter__(self):
        return iter(self.names)


STANDARD_SMOKE_SUITE = [
    {
        "num_loc": 50,
        "target_batch": 2048,
        "min_batch": 512,
        "objective_scalarization": "linear",
    },
    {
        "num_loc": 100,
        "target_batch": 1024,
        "min_batch": 256,
        "objective_scalarization": "linear",
    },
    {
        "num_loc": 200,
        "target_batch": 256,
        "min_batch": 128,
        "objective_scalarization": "linear",
    },
    {
        "num_loc": 50,
        "target_batch": 2048,
        "min_batch": 512,
        "objective_scalarization": "chebyshev",
    },
]


def hydra_path(path: str) -> str:
    """Return a path string that Hydra parses consistently across platforms."""
    return Path(path).as_posix()


def candidate_batches(target_batch: int, min_batch: int = 256) -> list[int]:
    """Return descending batch sizes to try before a serious 4090 run."""
    batches = []
    batch = target_batch
    while batch >= min_batch:
        batches.append(batch)
        batch //= 2
    return batches


def _query_nvidia_smi(args: list[str]) -> str:
    for executable in NVIDIA_SMI_CANDIDATES:
        try:
            return subprocess.check_output(
                [executable, *args],
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return ""


def detect_gpu_names() -> list[str]:
    """Return visible GPU names from nvidia-smi, or an empty list if unavailable."""
    output = _query_nvidia_smi(
        [
            "--query-gpu=name",
            "--format=csv,noheader",
        ]
    )
    if not output:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def detect_gpu_memory_mib() -> list[int]:
    """Return visible GPU memory totals in MiB, or an empty list if unavailable."""
    output = _query_nvidia_smi(
        [
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return []
    memory = []
    for line in output.splitlines():
        value = line.strip()
        if not value:
            continue
        try:
            memory.append(int(value))
        except ValueError:
            return []
    return memory


def detect_gpu_memory_used_mib() -> list[int]:
    """Return current visible GPU memory usage in MiB, or an empty list."""
    output = _query_nvidia_smi(
        [
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return []
    used = []
    for line in output.splitlines():
        value = line.strip()
        if not value:
            continue
        try:
            used.append(int(value))
        except ValueError:
            return []
    return used


def _selected_memory_sample(samples: list[int], cuda_visible_devices: Optional[str]):
    if not samples:
        return None
    if cuda_visible_devices is None:
        return max(samples)
    first_device = cuda_visible_devices.split(",", 1)[0].strip()
    try:
        physical_index = int(first_device)
    except ValueError:
        return max(samples)
    if 0 <= physical_index < len(samples):
        return samples[physical_index]
    return max(samples)


def assert_expected_gpu(
    expected_gpu: str = "4090",
    min_memory_mib: int = 20000,
) -> GpuPreflightResult:
    """Fail before launching training when the requested GPU class is not visible."""
    gpu_names = detect_gpu_names()
    matching_indices = [
        index
        for index, name in enumerate(gpu_names)
        if expected_gpu.lower() in name.lower()
    ]
    if not matching_indices:
        visible = ", ".join(gpu_names) if gpu_names else "none"
        raise RuntimeError(
            f"Expected a GPU containing '{expected_gpu}', but visible GPUs are: {visible}"
        )
    memory_mib = detect_gpu_memory_mib()
    if memory_mib:
        matching_memory = [
            memory_mib[index] for index in matching_indices if index < len(memory_mib)
        ]
        if matching_memory and max(matching_memory) < min_memory_mib:
            formatted = ", ".join(f"{value} MiB" for value in matching_memory)
            raise RuntimeError(
                f"Expected a GPU containing '{expected_gpu}' with at least "
                f"{min_memory_mib} MiB memory, but matching GPUs report: {formatted}"
            )
    if memory_mib:
        best_index = max(
            matching_indices,
            key=lambda index: memory_mib[index] if index < len(memory_mib) else -1,
        )
    else:
        best_index = matching_indices[0]
    return GpuPreflightResult(gpu_names, best_index)


def build_command(
    num_loc: int,
    batch_size: int,
    max_epochs: int = 1,
    train_data_size: int = 4096,
    eval_data_size: int = 512,
    smoke: bool = False,
    run_dir: Optional[str] = None,
    objective_scalarization: str = "linear",
    objective_ideal: Optional[list[float]] = None,
) -> list[str]:
    if num_loc < MIN_DEFAULT_CONSTRAINT_NUM_LOC:
        raise ValueError(
            "LUOP default grouped constraints require num_loc to be at least "
            f"{MIN_DEFAULT_CONSTRAINT_NUM_LOC}; got num_loc={num_loc}. "
            "Use a larger probe size so generated instances remain feasible."
        )
    safe_max_batch = SAFE_MAX_BATCH_BY_NUM_LOC.get(num_loc)
    if safe_max_batch is not None and batch_size > safe_max_batch:
        raise ValueError(
            f"LUOP{num_loc} batch_size={batch_size} exceeds the 4090-safe cap "
            f"of {safe_max_batch}. Use batch_size={safe_max_batch} or lower; "
            "batch_size=512 exhausted device memory in a real LUOP200 probe."
        )
    if objective_ideal is None:
        objective_ideal = [1.0, 1.0]
    objective_ideal_override = (
        "[" + ",".join(str(value) for value in objective_ideal) + "]"
    )
    cmd = [
        sys.executable,
        "run.py",
        "experiment=cityplan/am",
        "logger=csv",
        f"trainer.max_epochs={max_epochs}",
        "trainer.accelerator=gpu",
        "+trainer.devices=1",
        "trainer.precision=16-mixed",
        f"env.generator_params.num_loc={num_loc}",
        f"model.batch_size={batch_size}",
        f"model.val_batch_size={batch_size}",
        f"model.test_batch_size={batch_size}",
        f"model.train_data_size={train_data_size}",
        f"model.val_data_size={eval_data_size}",
        f"model.test_data_size={eval_data_size}",
        "model.optimizer_kwargs.lr=5e-5",
        "model.policy_kwargs.decode_type_first=false",
        "env.sample_objective_weights=false",
        "env.sample_eval_objective_weights=false",
        "env.eval_objective_weights=[0.5,0.5]",
        f"env.objective_scalarization={objective_scalarization}",
        f"env.objective_ideal={objective_ideal_override}",
        "model.generate_default_data=false",
        "env.val_file=null",
        "env.test_file=null",
        "callbacks.model_checkpoint.monitor=val/checkpoint_score",
        "callbacks.rich_progress_bar=null",
        "callbacks.learning_rate_monitor=null",
        "+trainer.log_every_n_steps=1",
    ]
    if run_dir is not None:
        cmd.append(f"hydra.run.dir={hydra_path(run_dir)}")
    if smoke:
        cmd.extend(
            [
                "+trainer.limit_train_batches=20",
                "+trainer.limit_val_batches=4",
                "+trainer.limit_test_batches=4",
                "+trainer.detect_anomaly=true",
            ]
        )
    return cmd


def run_command(
    cmd: list[str],
    log_path: Path,
    cuda_visible_devices: Optional[str] = None,
    gpu_sample_interval_s: float = 1.0,
    max_gpu_memory_mib: Optional[int] = MAX_GPU_MEMORY_MIB,
) -> int:
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(" ".join(cmd) + "\n\n")
        log_file.flush()
        env = os.environ.copy()
        env.setdefault("PROJECT_ROOT", str(Path.cwd()))
        if cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        stop_sampling = threading.Event()
        peak_memory_mib = 0
        memory_limit_exceeded_mib = None

        def sample_gpu_memory():
            nonlocal peak_memory_mib, memory_limit_exceeded_mib
            while not stop_sampling.is_set():
                samples = detect_gpu_memory_used_mib()
                selected_memory = _selected_memory_sample(
                    samples,
                    cuda_visible_devices,
                )
                if selected_memory is not None:
                    peak_memory_mib = max(peak_memory_mib, selected_memory)
                    if (
                        max_gpu_memory_mib is not None
                        and selected_memory > max_gpu_memory_mib
                        and memory_limit_exceeded_mib is None
                    ):
                        memory_limit_exceeded_mib = selected_memory
                        log_file.write(
                            f"\ngpu_memory_limit_exceeded_mib={selected_memory}\n"
                        )
                        log_file.flush()
                        proc.terminate()
                stop_sampling.wait(gpu_sample_interval_s)

        sampler = threading.Thread(target=sample_gpu_memory, daemon=True)
        sampler.start()
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                print(line, end="")
                log_file.write(line)
            return_code = proc.wait()
        finally:
            samples = detect_gpu_memory_used_mib()
            selected_memory = _selected_memory_sample(samples, cuda_visible_devices)
            if selected_memory is not None:
                peak_memory_mib = max(peak_memory_mib, selected_memory)
            stop_sampling.set()
            sampler.join(timeout=max(gpu_sample_interval_s, 0.1))
            log_file.write(f"\ngpu_peak_memory_mib={peak_memory_mib}\n")
            log_file.flush()
        return return_code


def _read_metrics_artifacts(run_dir: Optional[Path]) -> str:
    if run_dir is None or not run_dir.exists():
        return ""
    chunks = []
    for metrics_path in run_dir.rglob("metrics.csv"):
        chunks.append(metrics_path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


def _scan_run_config_artifacts(run_dir: Optional[Path]) -> list[str]:
    if run_dir is None:
        return []

    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        return ["missing_run_config"]

    try:
        cfg = OmegaConf.load(config_path)
    except Exception:
        return ["invalid_run_config"]

    failures = []
    model_target = OmegaConf.select(
        cfg,
        "model._target_",
        default=None,
    )
    if model_target != EXPECTED_LUOP_MODEL_TARGET:
        failures.append("wrong_luop_model_target")

    decode_type_first = OmegaConf.select(
        cfg,
        "model.policy_kwargs.decode_type_first",
        default=None,
    )
    if decode_type_first is not EXPECTED_DECODE_TYPE_FIRST:
        failures.append("type_first_luop_decode")

    sample_train = OmegaConf.select(
        cfg,
        "env.sample_objective_weights",
        default=None,
    )
    if sample_train is not False:
        failures.append("sampled_train_objective_weights")

    sample_eval = OmegaConf.select(
        cfg,
        "env.sample_eval_objective_weights",
        default=None,
    )
    if sample_eval is not False:
        failures.append("sampled_eval_objective_weights")

    eval_weights = OmegaConf.select(
        cfg,
        "env.eval_objective_weights",
        default=None,
    )
    if eval_weights is None:
        failures.append("missing_eval_objective_weights")
    else:
        try:
            actual_weights = [float(value) for value in eval_weights]
        except (TypeError, ValueError):
            actual_weights = []
        if len(actual_weights) != len(EXPECTED_EVAL_OBJECTIVE_WEIGHTS) or any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-8)
            for actual, expected in zip(
                actual_weights,
                EXPECTED_EVAL_OBJECTIVE_WEIGHTS,
            )
        ):
            failures.append("wrong_eval_objective_weights")

    monitor = OmegaConf.select(
        cfg,
        "callbacks.model_checkpoint.monitor",
        default=None,
    )
    if monitor != EXPECTED_CHECKPOINT_MONITOR:
        failures.append("wrong_checkpoint_monitor")

    return failures


def _has_unstable_metric_curve(
    values: list[float],
    max_swing: float = MAX_VALIDATION_METRIC_SWING,
    max_drift: float = MAX_VALIDATION_METRIC_DRIFT,
    min_points: int = MIN_STABILITY_CURVE_POINTS,
) -> bool:
    if len(values) < min_points:
        return False

    best_index = max(range(len(values)), key=lambda index: values[index])
    if len(values) >= MIN_SUSTAINED_DRIFT_POINTS:
        best_to_last_drop = values[best_index] - values[-1]
        if values[-1] < values[0] and best_to_last_drop > max_drift:
            return True

    if best_index < len(values) - 1:
        post_best_drop = values[best_index] - min(values[best_index + 1 :])
        if post_best_drop > max_swing:
            return True

    deltas = [values[index + 1] - values[index] for index in range(len(values) - 1)]
    sign_changes = 0
    previous_sign = 0
    max_increase = 0.0
    max_drop = 0.0
    for delta in deltas:
        if delta > 0:
            max_increase = max(max_increase, delta)
        elif delta < 0:
            max_drop = max(max_drop, -delta)
        sign = 1 if delta > 0 else -1 if delta < 0 else 0
        if sign and previous_sign and sign != previous_sign:
            sign_changes += 1
        if sign:
            previous_sign = sign
    return sign_changes >= 2 and max_increase > max_swing and max_drop > max_swing


def _scan_metric_curve_failures(
    metric_curves: dict[str, list[float]],
    min_validation_points: int = 0,
) -> list[str]:
    failures = [
        label
        for label, values in metric_curves.items()
        if _has_unstable_metric_curve(values)
    ]
    failures = [
        label
        for label in failures
        if not _is_compensated_objective_rebalancing(label, metric_curves)
    ]
    if min_validation_points > 0:
        max_validation_points = max(
            (len(values) for values in metric_curves.values()),
            default=0,
        )
        if max_validation_points < min_validation_points:
            failures.append("insufficient_validation_points")
    return failures


def _is_compensated_objective_rebalancing(
    unstable_label: str,
    metric_curves: dict[str, list[float]],
) -> bool:
    """Allow small component drift when the equal-weight objective rebalances.

    LUOP validation uses fixed [0.5, 0.5] eval weights. A small drop in one
    component can be expected when the other component improves enough to raise
    scalar reward, Pareto hypervolume, and checkpoint score while narrowing the
    component gap. This is not the same failure mode as a component collapse.
    """
    component_labels = {
        "unstable_val_compatibility_reward_curve",
        "unstable_val_accessibility_reward_curve",
    }
    if unstable_label not in component_labels:
        return False

    compat = metric_curves["unstable_val_compatibility_reward_curve"]
    access = metric_curves["unstable_val_accessibility_reward_curve"]
    reward = metric_curves["unstable_val_reward_curve"]
    hypervolume = metric_curves["unstable_val_pareto_hypervolume_curve"]
    checkpoint = metric_curves["unstable_val_checkpoint_score_curve"]
    if not all((compat, access, reward, hypervolume, checkpoint)):
        return False

    curve_lengths = {
        len(compat),
        len(access),
        len(reward),
        len(hypervolume),
        len(checkpoint),
    }
    if len(curve_lengths) != 1 or next(iter(curve_lengths)) < MIN_SUSTAINED_DRIFT_POINTS:
        return False
    if any(
        _has_unstable_metric_curve(values) for values in (reward, hypervolume, checkpoint)
    ):
        return False
    if (
        reward[-1] <= reward[0]
        or hypervolume[-1] <= hypervolume[0]
        or checkpoint[-1] <= checkpoint[0]
    ):
        return False

    declining = (
        compat if unstable_label == "unstable_val_compatibility_reward_curve" else access
    )
    improving = (
        access if unstable_label == "unstable_val_compatibility_reward_curve" else compat
    )
    component_drop = declining[0] - declining[-1]
    if component_drop <= 0 or component_drop > MAX_VALIDATION_METRIC_SWING:
        return False
    if improving[-1] - improving[0] <= component_drop:
        return False

    initial_gap = abs(compat[0] - access[0])
    final_gap = abs(compat[-1] - access[-1])
    return final_gap < initial_gap


def _scan_metrics_artifacts(
    run_dir: Optional[Path],
    fallback_text: str = "",
    min_validation_points: int = 0,
) -> list[str]:
    if run_dir is None or not run_dir.exists():
        return []

    metrics_paths = list(run_dir.rglob("metrics.csv"))
    if not metrics_paths:
        return []

    finite_seen = {
        label: bool(REQUIRED_SIGNAL_PATTERNS[label].search(fallback_text))
        for label in REQUIRED_METRIC_TOKENS
    }
    found_required_column = False
    nonfinite_required_value = False
    metric_curves = {label: [] for label in STABILITY_CURVE_METRIC_TOKENS}

    for metrics_path in metrics_paths:
        with metrics_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            required_columns = {
                label: [name for name in reader.fieldnames if pattern.search(name)]
                for label, pattern in REQUIRED_METRIC_TOKENS.items()
            }
            curve_columns = {
                label: [name for name in reader.fieldnames if pattern.search(name)]
                for label, pattern in STABILITY_CURVE_METRIC_TOKENS.items()
            }
            found_required_column = found_required_column or any(
                columns for columns in required_columns.values()
            )
            for row in reader:
                for label, columns in required_columns.items():
                    for column in columns:
                        value = (row.get(column) or "").strip()
                        if value == "":
                            continue
                        try:
                            number = float(value)
                        except ValueError:
                            nonfinite_required_value = True
                            continue
                        if math.isfinite(number):
                            finite_seen[label] = True
                        else:
                            nonfinite_required_value = True
                for label, columns in curve_columns.items():
                    for column in columns:
                        value = (row.get(column) or "").strip()
                        if value == "":
                            continue
                        try:
                            number = float(value)
                        except ValueError:
                            nonfinite_required_value = True
                            continue
                        if math.isfinite(number):
                            metric_curves[label].append(number)
                        else:
                            nonfinite_required_value = True

    failures = []
    if nonfinite_required_value:
        failures.append("nonfinite_metrics")
    failures.extend(
        _scan_metric_curve_failures(
            metric_curves,
            min_validation_points=min_validation_points,
        )
    )
    missing = [label for label, seen in finite_seen.items() if not seen]
    if found_required_column and missing:
        failures.append("missing_finite_metrics")
    elif missing:
        failures.extend(missing)
    return failures


def scan_stability_failures(
    log_path: Path,
    run_dir: Optional[Path] = None,
    min_gpu_peak_memory_mib: int = MIN_GPU_PEAK_MEMORY_MIB,
    min_validation_points: int = 0,
) -> list[str]:
    """Return stability failure labels detected in a completed run log."""
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    text = "\n".join([log_text, _read_metrics_artifacts(run_dir)])
    failures = [
        label for label, pattern in STABILITY_PATTERNS.items() if pattern.search(text)
    ]
    failures.extend(_scan_run_config_artifacts(run_dir))
    metric_failures = _scan_metrics_artifacts(
        run_dir,
        fallback_text=log_text,
        min_validation_points=min_validation_points,
    )
    if metric_failures:
        failures.extend(metric_failures)
    else:
        failures.extend(
            label
            for label, pattern in REQUIRED_SIGNAL_PATTERNS.items()
            if not pattern.search(text)
        )
    peak_values = []
    for match in GPU_PEAK_MEMORY_PATTERN.finditer(text):
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        if math.isfinite(value):
            peak_values.append(value)
    if not peak_values:
        failures.append("missing_gpu_peak_memory")
    elif max(peak_values) < min_gpu_peak_memory_mib:
        failures.append("low_gpu_peak_memory")
    if GPU_MEMORY_LIMIT_EXCEEDED_PATTERN.search(text):
        failures.append("gpu_memory_limit_exceeded")
    return failures


def _is_retryable_oom(code: int, failures: list[str]) -> bool:
    if code == 0 or "cuda_oom" not in failures:
        return False
    non_oom_failures = [failure for failure in failures if failure != "cuda_oom"]
    retryable_missing_signals = set(REQUIRED_SIGNAL_PATTERNS) | {
        "missing_run_config",
        "missing_finite_metrics",
        "missing_gpu_peak_memory",
        "low_gpu_peak_memory",
    }
    return all(failure in retryable_missing_signals for failure in non_oom_failures)


def _is_retryable_memory_cap(code: int, failures: list[str]) -> bool:
    if code == 0 or "gpu_memory_limit_exceeded" not in failures:
        return False
    retryable_missing_signals = set(REQUIRED_SIGNAL_PATTERNS) | {
        "missing_run_config",
        "missing_finite_metrics",
        "missing_gpu_peak_memory",
    }
    non_memory_failures = [
        failure for failure in failures if failure != "gpu_memory_limit_exceeded"
    ]
    return all(failure in retryable_missing_signals for failure in non_memory_failures)


def run_adaptive_probe(args, log_dir: Path, timestamp: str) -> int:
    """Try candidate batches, backing off only when the failure is pure CUDA OOM."""
    for batch_size in candidate_batches(args.target_batch, args.min_batch):
        run_dir = log_dir / f"run_luop{args.num_loc}_bs{batch_size}_{timestamp}"
        try:
            cmd = build_command(
                num_loc=args.num_loc,
                batch_size=batch_size,
                max_epochs=args.max_epochs,
                train_data_size=args.train_data_size,
                eval_data_size=args.eval_data_size,
                smoke=not args.full,
                run_dir=str(run_dir),
                objective_scalarization=args.objective_scalarization,
                objective_ideal=args.objective_ideal,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if args.dry_run:
            print(" ".join(cmd))
            continue

        log_path = log_dir / f"luop{args.num_loc}_bs{batch_size}_{timestamp}.log"
        print(f"\n=== Trying LUOP{args.num_loc} batch_size={batch_size} ===")
        code = run_command(
            cmd,
            log_path,
            cuda_visible_devices=getattr(args, "cuda_visible_devices", None),
            max_gpu_memory_mib=args.max_gpu_memory_mib,
        )
        failures = scan_stability_failures(
            log_path,
            run_dir=run_dir,
            min_gpu_peak_memory_mib=args.min_gpu_peak_memory_mib,
            min_validation_points=args.min_validation_points,
        )
        if code == 0 and not failures:
            print(f"Stable probe passed at batch_size={batch_size}; log: {log_path}")
            print(f"Selected stable LUOP batch_size={batch_size}")
            return 0
        if failures:
            print(
                "Probe failed stability scan at "
                f"batch_size={batch_size}: {', '.join(failures)}; log: {log_path}"
            )
        else:
            print(f"Probe failed at batch_size={batch_size}; log: {log_path}")

        if _is_retryable_memory_cap(code, failures):
            print(
                f"GPU memory cap exceeded at batch_size={batch_size}; "
                "trying next smaller batch."
            )
            continue
        if not _is_retryable_oom(code, failures):
            print(
                "Stopping adaptive probe because failure is not a pure CUDA OOM.",
                file=sys.stderr,
            )
            return 1
        print(f"CUDA OOM at batch_size={batch_size}; trying next smaller batch.")

    if args.dry_run:
        return 0

    print("No candidate batch size passed.", file=sys.stderr)
    return 1


def run_standard_suite(args, log_dir: Path, timestamp: str) -> int:
    """Run the standard 4090 LUOP smoke grid, stopping at the first failed probe."""
    for index, probe in enumerate(STANDARD_SMOKE_SUITE, start=1):
        suite_args = argparse.Namespace(**vars(args))
        for key, value in probe.items():
            setattr(suite_args, key, value)
        print(
            "\n=== Suite probe "
            f"{index}/{len(STANDARD_SMOKE_SUITE)}: "
            f"LUOP{suite_args.num_loc}, "
            f"batch {suite_args.target_batch}->{suite_args.min_batch}, "
            f"{suite_args.objective_scalarization} ==="
        )
        if suite_args.dry_run:
            run_dir = (
                log_dir / f"run_luop{suite_args.num_loc}_bs{suite_args.target_batch}_"
                f"{timestamp}_suite{index}"
            )
            try:
                cmd = build_command(
                    num_loc=suite_args.num_loc,
                    batch_size=suite_args.target_batch,
                    max_epochs=suite_args.max_epochs,
                    train_data_size=suite_args.train_data_size,
                    eval_data_size=suite_args.eval_data_size,
                    smoke=not suite_args.full,
                    run_dir=str(run_dir),
                    objective_scalarization=suite_args.objective_scalarization,
                    objective_ideal=suite_args.objective_ideal,
                )
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                return 2
            print(" ".join(cmd))
            continue
        code = run_adaptive_probe(
            suite_args,
            log_dir=log_dir,
            timestamp=f"{timestamp}_suite{index}",
        )
        if code != 0:
            return code
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LUOP stability probes on a 4090.")
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Run the standard LUOP50/100/200 plus Chebyshev 4090 smoke grid.",
    )
    parser.add_argument("--num-loc", type=int, default=50)
    parser.add_argument("--target-batch", type=int, default=2048)
    parser.add_argument("--min-batch", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--train-data-size", type=int, default=4096)
    parser.add_argument("--eval-data-size", type=int, default=512)
    parser.add_argument(
        "--objective-scalarization",
        choices=["linear", "chebyshev"],
        default="linear",
    )
    parser.add_argument(
        "--objective-ideal",
        type=float,
        nargs=2,
        default=[1.0, 1.0],
    )
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--expected-gpu", default="4090")
    parser.add_argument(
        "--min-gpu-memory-mib",
        type=int,
        default=20000,
        help=(
            "Minimum total memory, in MiB, required for the selected expected "
            "GPU before launching training."
        ),
    )
    parser.add_argument(
        "--min-gpu-peak-memory-mib",
        type=int,
        default=MIN_GPU_PEAK_MEMORY_MIB,
        help=(
            "Minimum observed peak GPU memory, in MiB, required for a "
            "successful stability probe."
        ),
    )
    parser.add_argument(
        "--max-gpu-memory-mib",
        type=int,
        default=MAX_GPU_MEMORY_MIB,
        help=(
            "Terminate a probe once sampled GPU memory exceeds this MiB cap. "
            "Use 0 to disable the runtime cap."
        ),
    )
    parser.add_argument(
        "--min-validation-points",
        type=int,
        default=None,
        help=(
            "Minimum finite validation metric points required by the stability "
            f"scan. Defaults to {MIN_FULL_VALIDATION_POINTS} for --full runs "
            "and 0 for smoke probes."
        ),
    )
    parser.add_argument("--log-dir", default="logs/luop_4090_stability")
    args = parser.parse_args()
    if args.max_gpu_memory_mib <= 0:
        args.max_gpu_memory_mib = None
    if args.min_validation_points is None:
        args.min_validation_points = MIN_FULL_VALIDATION_POINTS if args.full else 0

    log_dir = Path(args.log_dir)
    if not args.dry_run:
        try:
            gpu_preflight = assert_expected_gpu(
                args.expected_gpu,
                min_memory_mib=args.min_gpu_memory_mib,
            )
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        args.cuda_visible_devices = str(gpu_preflight.selected_index)
        print(f"GPU preflight passed: {', '.join(gpu_preflight.names)}")
        print(f"Using CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}")
        log_dir.mkdir(parents=True, exist_ok=True)
    else:
        args.cuda_visible_devices = None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.suite:
        return run_standard_suite(args, log_dir=log_dir, timestamp=timestamp)
    return run_adaptive_probe(args, log_dir=log_dir, timestamp=timestamp)


if __name__ == "__main__":
    raise SystemExit(main())
