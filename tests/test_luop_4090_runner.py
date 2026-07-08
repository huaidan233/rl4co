import time
import sys

from pathlib import Path

import pytest

import scripts.luop_4090_stability as runner
from scripts.luop_4090_stability import (
    assert_expected_gpu,
    build_command,
    candidate_batches,
    main,
    run_command,
    scan_stability_failures,
)


def _write_hydra_run_config(
    run_dir: Path,
    sample_objective_weights: bool = False,
    sample_eval_objective_weights: bool = False,
    eval_objective_weights=(0.5, 0.5),
    checkpoint_monitor: str = "val/checkpoint_score",
    model_target: str = "rl4co.models.LUOPAttentionModel",
    decode_type_first: bool = False,
):
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    eval_weights_yaml = ""
    if eval_objective_weights is not None:
        weights = ", ".join(str(value) for value in eval_objective_weights)
        eval_weights_yaml = f"  eval_objective_weights: [{weights}]\n"
    (hydra_dir / "config.yaml").write_text(
        "model:\n"
        f"  _target_: {model_target}\n"
        "  policy_kwargs:\n"
        f"    decode_type_first: {str(decode_type_first).lower()}\n"
        "env:\n"
        f"  sample_objective_weights: {str(sample_objective_weights).lower()}\n"
        f"  sample_eval_objective_weights: {str(sample_eval_objective_weights).lower()}\n"
        f"{eval_weights_yaml}"
        "callbacks:\n"
        "  model_checkpoint:\n"
        f"    monitor: {checkpoint_monitor}\n",
        encoding="utf-8",
    )


def _write_hydra_run_config_for_command(cmd):
    run_dir_override = next(item for item in cmd if item.startswith("hydra.run.dir="))
    _write_hydra_run_config(Path(run_dir_override.split("=", 1)[1]))


def _write_stable_metrics(run_dir: Path):
    metrics_path = run_dir / "csv" / "version_0" / "metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        "epoch,step,train/loss,val/reward,train/compatibility_reward,"
        "train/accessibility_reward,val/pareto_hypervolume,"
        "val/pareto_front_size,val/checkpoint_score\n"
        "0,1,0.1,0.60,0.4,0.5,0.20,2,0.433\n"
        "1,2,0.1,0.61,0.4,0.5,0.21,2,0.440\n"
        "2,3,0.1,0.62,0.4,0.5,0.22,2,0.447\n"
        "3,4,0.1,0.63,0.4,0.5,0.23,2,0.453\n",
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def _assume_full_4090_memory_for_runner_tests(monkeypatch, request):
    if request.node.name.startswith("test_detect_gpu_"):
        return
    monkeypatch.setattr(
        "scripts.luop_4090_stability.detect_gpu_memory_mib",
        lambda: [24564],
    )


def test_candidate_batches_scale_down_from_target_batch():
    assert candidate_batches(2048, min_batch=256) == [2048, 1024, 512, 256]
    assert candidate_batches(700, min_batch=256) == [700, 350]


def test_detect_gpu_names_falls_back_to_wsl_nvidia_smi(monkeypatch):
    calls = []

    def fake_check_output(cmd, **kwargs):
        calls.append(cmd[0])
        if cmd[0] == "nvidia-smi":
            raise FileNotFoundError("nvidia-smi")
        assert cmd[0] == "/usr/lib/wsl/lib/nvidia-smi"
        return "NVIDIA GeForce RTX 4090\n"

    monkeypatch.setattr(runner.subprocess, "check_output", fake_check_output)

    assert runner.detect_gpu_names() == ["NVIDIA GeForce RTX 4090"]
    assert calls == ["nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi"]


def test_detect_gpu_memory_falls_back_to_wsl_nvidia_smi(monkeypatch):
    queries = []

    def fake_check_output(cmd, **kwargs):
        queries.append((cmd[0], cmd[1]))
        if cmd[0] == "nvidia-smi":
            raise FileNotFoundError("nvidia-smi")
        return "24564\n"

    monkeypatch.setattr(runner.subprocess, "check_output", fake_check_output)

    assert runner.detect_gpu_memory_mib() == [24564]
    assert runner.detect_gpu_memory_used_mib() == [24564]
    assert queries == [
        ("nvidia-smi", "--query-gpu=memory.total"),
        ("/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=memory.total"),
        ("nvidia-smi", "--query-gpu=memory.used"),
        ("/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=memory.used"),
    ]


def test_4090_runner_command_uses_gpu_precision_and_stability_limits():
    cmd = build_command(
        num_loc=50,
        batch_size=1024,
        smoke=True,
        run_dir="logs/luop_4090_stability/run_probe",
    )

    assert cmd[:2] == [sys.executable, "run.py"]
    assert "experiment=cityplan/am" in cmd
    assert "trainer.accelerator=gpu" in cmd
    assert "+trainer.devices=1" in cmd
    assert "trainer.precision=16-mixed" in cmd
    assert "logger=csv" in cmd
    assert "hydra.run.dir=logs/luop_4090_stability/run_probe" in cmd
    assert "model.batch_size=1024" in cmd
    assert "model.optimizer_kwargs.lr=5e-5" in cmd
    assert "model.policy_kwargs.decode_type_first=false" in cmd
    assert "env.sample_objective_weights=false" in cmd
    assert "env.sample_eval_objective_weights=false" in cmd
    assert "env.eval_objective_weights=[0.5,0.5]" in cmd
    assert "callbacks.model_checkpoint.monitor=val/checkpoint_score" in cmd
    assert "callbacks.rich_progress_bar=null" in cmd
    assert "callbacks.learning_rate_monitor=null" in cmd
    assert "+trainer.log_every_n_steps=1" in cmd
    assert "+trainer.detect_anomaly=true" in cmd
    assert "+trainer.limit_train_batches=20" in cmd


def test_4090_runner_rejects_too_small_default_constraint_probe():
    with pytest.raises(ValueError, match="num_loc.*at least 12"):
        build_command(num_loc=8, batch_size=2, smoke=True)


def test_4090_runner_rejects_luop200_batch_above_safe_cap():
    with pytest.raises(ValueError, match="LUOP200.*batch_size.*256"):
        build_command(num_loc=200, batch_size=512, smoke=True)


def test_4090_runner_too_small_probe_dry_run_fails_before_printing_command(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "luop_4090_stability.py",
            "--num-loc",
            "8",
            "--target-batch",
            "2",
            "--min-batch",
            "2",
            "--dry-run",
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )

    assert main() == 2
    captured = capsys.readouterr()
    assert "at least 12" in captured.err
    assert "python run.py" not in captured.out


def test_4090_runner_too_large_luop200_probe_dry_run_fails_before_printing_command(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "luop_4090_stability.py",
            "--num-loc",
            "200",
            "--target-batch",
            "512",
            "--min-batch",
            "512",
            "--dry-run",
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )

    assert main() == 2
    captured = capsys.readouterr()
    assert "LUOP200" in captured.err
    assert "batch_size=512" in captured.err
    assert "python run.py" not in captured.out


def test_4090_runner_can_probe_chebyshev_scalarization():
    cmd = build_command(
        num_loc=50,
        batch_size=1024,
        smoke=True,
        objective_scalarization="chebyshev",
    )

    assert "env.objective_scalarization=chebyshev" in cmd
    assert "env.objective_ideal=[1.0,1.0]" in cmd


def test_cityplan_am_config_uses_dedicated_luop_model():
    from pathlib import Path

    config_text = Path("configs/experiment/cityplan/am.yaml").read_text(encoding="utf-8")

    assert "override /model: luop_am.yaml" in config_text
    assert "sample_objective_weights: false" in config_text
    assert "decode_type_first: false" in config_text
    assert "lr: 5e-5" in config_text


def test_4090_runner_overrides_are_valid_hydra_keys():
    from hydra import compose, initialize_config_dir
    from pathlib import Path

    overrides = build_command(num_loc=50, batch_size=512, smoke=True)[2:]
    overrides = [override for override in overrides if not override.startswith("logger=")]
    config_dir = str((Path.cwd() / "configs").resolve())

    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="main.yaml", overrides=overrides)

    assert cfg.model.policy_kwargs.decode_type_first is False
    assert cfg.model._target_ == "rl4co.models.LUOPAttentionModel"
    assert cfg.env.sample_objective_weights is False
    assert cfg.env.sample_eval_objective_weights is False
    assert cfg.env.eval_objective_weights == [0.5, 0.5]
    assert cfg.env.objective_scalarization == "linear"
    assert cfg.callbacks.model_checkpoint.monitor == "val/checkpoint_score"


def test_cityplan_luop_config_accepts_objective_weight_floor_override():
    from hydra import compose, initialize_config_dir
    from pathlib import Path

    config_dir = str((Path.cwd() / "configs").resolve())

    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(
            config_name="main.yaml",
            overrides=[
                "experiment=cityplan/am",
                "env.objective_weight_min=0.2",
            ],
        )

    assert cfg.env.objective_weight_min == 0.2


def test_4090_runner_dry_run_exits_successfully(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "luop_4090_stability.py",
            "--num-loc",
            "50",
            "--target-batch",
            "1024",
            "--min-batch",
            "512",
            "--dry-run",
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )

    assert main() == 0
    assert "trainer.accelerator=gpu" in capsys.readouterr().out
    assert not (tmp_path / "logs").exists()


def test_4090_runner_dry_run_emits_portable_hydra_run_dir(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "luop_4090_stability.py",
            "--num-loc",
            "50",
            "--target-batch",
            "1024",
            "--min-batch",
            "1024",
            "--dry-run",
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )

    assert main() == 0
    output = capsys.readouterr().out

    assert "hydra.run.dir=" in output
    assert "\\run_luop" not in output


def test_4090_runner_suite_dry_run_emits_standard_probe_grid(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "luop_4090_stability.py",
            "--suite",
            "--dry-run",
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )

    assert main() == 0
    output = capsys.readouterr().out

    assert output.count(f"{sys.executable} run.py") == 4
    assert "env.generator_params.num_loc=50" in output
    assert "model.batch_size=2048" in output
    assert "env.generator_params.num_loc=100" in output
    assert "model.batch_size=1024" in output
    assert "env.generator_params.num_loc=200" in output
    assert "model.batch_size=256" in output
    assert "env.objective_scalarization=chebyshev" in output
    assert "\\run_luop" not in output
    assert not (tmp_path / "logs").exists()


def test_4090_runner_suite_stops_after_first_failed_probe(monkeypatch, tmp_path):
    launched = []

    monkeypatch.setattr(
        "scripts.luop_4090_stability.detect_gpu_names",
        lambda: ["NVIDIA GeForce RTX 4090"],
    )

    def fake_run_command(cmd, log_path, **kwargs):
        num_loc_override = next(
            item for item in cmd if item.startswith("env.generator_params.num_loc=")
        )
        launched.append(int(num_loc_override.split("=")[1]))
        log_path.write_text("ValueError: no valid LUOP joint action\n", encoding="utf-8")
        return 1

    monkeypatch.setattr("scripts.luop_4090_stability.run_command", fake_run_command)
    monkeypatch.setattr(
        "sys.argv",
        [
            "luop_4090_stability.py",
            "--suite",
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )

    assert main() == 1
    assert launched == [50]


def test_4090_runner_scans_logs_for_stability_failures(tmp_path):
    log_path = tmp_path / "bad.log"
    log_path.write_text(
        "loss=nan\nRuntimeError: CUDA out of memory\n"
        "infeasible action selected\ninvalid probability distribution\n",
        encoding="utf-8",
    )

    failures = scan_stability_failures(log_path)

    assert "nan_or_inf" in failures
    assert "cuda_oom" in failures
    assert "infeasible_action" in failures
    assert "invalid_distribution" in failures


def test_4090_runner_scans_common_nonfinite_log_variants(tmp_path):
    log_path = tmp_path / "nonfinite.log"
    log_path.write_text(
        "RuntimeError: Function AddBackward0 returned nonfinite values\n"
        "ValueError: non-finite reward encountered\n"
        "Gradient contains nonfinite entries\n",
        encoding="utf-8",
    )

    failures = scan_stability_failures(log_path)

    assert "nan_or_inf" in failures


def test_4090_runner_scans_common_mask_failure_variants(tmp_path):
    log_path = tmp_path / "mask.log"
    log_path.write_text(
        "decoder failed with all_false mask\n"
        "RuntimeError: no_valid_action for LUOP row 7\n",
        encoding="utf-8",
    )

    failures = scan_stability_failures(log_path)

    assert "all_false_mask" in failures


def test_4090_runner_scans_luop_no_valid_joint_action(tmp_path):
    log_path = tmp_path / "luop_mask.log"
    log_path.write_text(
        "ValueError: no valid LUOP joint action for active batch rows [0]\n",
        encoding="utf-8",
    )

    failures = scan_stability_failures(log_path)

    assert "all_false_mask" in failures


def test_4090_runner_scans_new_luop_action_and_plan_errors(tmp_path):
    log_path = tmp_path / "new_luop_errors.log"
    log_path.write_text(
        "ValueError: infeasible LUOP joint action selected for active batch rows [0]\n"
        "ValueError: infeasible LUOP component action selected for batch rows [1]\n"
        "ValueError: LUOP component actions must be in valid type and parcel ranges\n"
        "ValueError: mismatched flat and explicit LUOP joint actions for batch rows [2]\n"
        "ValueError: All parcels must use a known land-use type\n",
        encoding="utf-8",
    )

    failures = scan_stability_failures(log_path)

    assert "infeasible_action" in failures
    assert "invalid_plan" in failures


def test_4090_runner_scans_luop_component_range_errors(tmp_path):
    log_path = tmp_path / "component_range.log"
    log_path.write_text(
        "ValueError: LUOP component actions must be in valid type and parcel ranges\n",
        encoding="utf-8",
    )

    failures = scan_stability_failures(log_path)

    assert "infeasible_action" in failures


def test_4090_runner_scans_luop_flat_action_range_errors(tmp_path):
    log_path = tmp_path / "flat_range.log"
    log_path.write_text(
        "ValueError: LUOP flat actions must be in valid joint type-parcel range\n",
        encoding="utf-8",
    )

    failures = scan_stability_failures(log_path)

    assert "infeasible_action" in failures


def test_4090_runner_scans_luop_replay_action_validation_errors(tmp_path):
    log_path = tmp_path / "replay_validation.log"
    log_path.write_text(
        "ValueError: LUOP replay actions must be in [0, 23] for active rows, "
        "with -1 allowed only as done-row padding\n"
        "ValueError: LUOP replay type_actions and parcel_actions must use "
        "matching -1 padding\n"
        "ValueError: LUOP replay parcel_actions must be in [0, 2] for active rows, "
        "with -1 allowed only as done-row padding\n",
        encoding="utf-8",
    )

    failures = scan_stability_failures(log_path)

    assert "infeasible_action" in failures


def test_4090_runner_scans_luop_replay_integer_validation_errors(tmp_path):
    log_path = tmp_path / "replay_integer_validation.log"
    log_path.write_text(
        "ValueError: LUOP replay actions must contain integer ids\n"
        "ValueError: LUOP replay type_actions must contain finite integer ids\n",
        encoding="utf-8",
    )

    failures = scan_stability_failures(log_path)

    assert "infeasible_action" in failures


def test_4090_runner_scans_pareto_action_artifact_errors(tmp_path):
    log_path = tmp_path / "pareto_artifacts.log"
    log_path.write_text(
        "ValueError: evaluate_pareto_front requires actions values in range [-1, 31]\n"
        "ValueError: evaluate_pareto_front requires type_actions and parcel_actions "
        "to encode the returned flat actions\n"
        "ValueError: evaluate_pareto_front requires actions padding to be a "
        "trailing suffix\n",
        encoding="utf-8",
    )

    failures = scan_stability_failures(log_path)

    assert "infeasible_action" in failures


def test_4090_runner_scans_pareto_short_action_trace_errors(tmp_path):
    log_path = tmp_path / "pareto_short_trace.log"
    log_path.write_text(
        "ValueError: evaluate_pareto_front requires returned actions to include "
        "one valid action for each initially unassigned parcel\n",
        encoding="utf-8",
    )

    failures = scan_stability_failures(log_path)

    assert "infeasible_action" in failures


def test_4090_runner_scans_pareto_action_mask_replay_errors(tmp_path):
    log_path = tmp_path / "pareto_action_mask_replay.log"
    log_path.write_text(
        "ValueError: evaluate_pareto_front requires returned actions to satisfy "
        "the current LUOP action_mask at each replay step\n",
        encoding="utf-8",
    )

    failures = scan_stability_failures(log_path)

    assert "infeasible_action" in failures


def test_4090_runner_scans_pareto_padding_artifact_errors(tmp_path):
    log_path = tmp_path / "pareto_padding_artifacts.log"
    log_path.write_text(
        "ValueError: evaluate_pareto_front requires actions padding to be a "
        "trailing suffix\n",
        encoding="utf-8",
    )

    failures = scan_stability_failures(log_path)

    assert "infeasible_action" in failures


def test_4090_runner_scans_pareto_plan_artifact_errors(tmp_path):
    log_path = tmp_path / "pareto_plan.log"
    log_path.write_text(
        "ValueError: evaluate_pareto_front requires current_plan to be complete "
        "with no unassigned parcels\n"
        "ValueError: evaluate_pareto_front requires current_plan to contain known "
        "land-use type ids in [0, 7]\n",
        encoding="utf-8",
    )

    failures = scan_stability_failures(log_path)

    assert "invalid_plan" in failures


def test_4090_runner_accepts_logs_with_required_multiobjective_metrics(tmp_path):
    log_path = tmp_path / "good.log"
    log_path.write_text(
        "train/loss=0.1\n"
        "val/reward=0.6\n"
        "train/compatibility_reward=0.4\n"
        "train/accessibility_reward=0.5\n"
        "val/pareto_hypervolume=0.2\n"
        "val/pareto_front_size=2\n"
        "val/checkpoint_score=0.433\n"
        "gpu_peak_memory_mib=16384\n",
        encoding="utf-8",
    )

    assert scan_stability_failures(log_path) == []


def test_4090_runner_requires_loss_and_scalar_reward_metrics(tmp_path):
    log_path = tmp_path / "missing_training_stability.log"
    log_path.write_text(
        "train/compatibility_reward=0.4\n"
        "train/accessibility_reward=0.5\n"
        "val/pareto_hypervolume=0.2\n"
        "val/pareto_front_size=2\n"
        "gpu_peak_memory_mib=16384\n",
        encoding="utf-8",
    )

    failures = scan_stability_failures(log_path)

    assert "missing_loss_metric" in failures
    assert "missing_scalar_reward_metric" in failures


def test_4090_runner_requires_peak_gpu_memory_signal(tmp_path):
    log_path = tmp_path / "no_gpu_peak.log"
    log_path.write_text(
        "train/compatibility_reward=0.4\n"
        "train/accessibility_reward=0.5\n"
        "val/pareto_hypervolume=0.2\n"
        "val/pareto_front_size=2\n",
        encoding="utf-8",
    )

    failures = scan_stability_failures(log_path)

    assert "missing_gpu_peak_memory" in failures


def test_4090_runner_rejects_low_peak_gpu_memory(tmp_path):
    log_path = tmp_path / "low_gpu_peak.log"
    log_path.write_text(
        "train/compatibility_reward=0.4\n"
        "train/accessibility_reward=0.5\n"
        "val/pareto_hypervolume=0.2\n"
        "val/pareto_front_size=2\n"
        "gpu_peak_memory_mib=2048\n",
        encoding="utf-8",
    )

    failures = scan_stability_failures(log_path)

    assert "low_gpu_peak_memory" in failures


def test_4090_runner_scans_gpu_memory_limit_exceeded(tmp_path):
    log_path = tmp_path / "gpu_memory_cap.log"
    log_path.write_text(
        "gpu_memory_limit_exceeded_mib=24073\n" "gpu_peak_memory_mib=24073\n",
        encoding="utf-8",
    )

    failures = scan_stability_failures(log_path)

    assert "gpu_memory_limit_exceeded" in failures


def test_4090_runner_accepts_custom_lower_peak_gpu_memory_threshold(tmp_path):
    log_path = tmp_path / "custom_low_gpu_peak.log"
    log_path.write_text(
        "train/loss=0.1\n"
        "val/reward=0.6\n"
        "train/compatibility_reward=0.4\n"
        "train/accessibility_reward=0.5\n"
        "val/pareto_hypervolume=0.2\n"
        "val/pareto_front_size=2\n"
        "val/checkpoint_score=0.433\n"
        "gpu_peak_memory_mib=4096\n",
        encoding="utf-8",
    )

    failures = scan_stability_failures(log_path, min_gpu_peak_memory_mib=4096)

    assert "low_gpu_peak_memory" not in failures


def test_run_command_writes_peak_gpu_memory(monkeypatch, tmp_path):
    class FakeProcess:
        stdout = iter(["training output\n"])

        def wait(self):
            time.sleep(0.02)
            return 0

    samples = iter([[1024], [4096], [16384]])

    def fake_memory_used():
        try:
            return next(samples)
        except StopIteration:
            return [16384]

    monkeypatch.setattr(
        "scripts.luop_4090_stability.subprocess.Popen",
        lambda *args, **kwargs: FakeProcess(),
    )
    monkeypatch.setattr(
        "scripts.luop_4090_stability.detect_gpu_memory_used_mib",
        fake_memory_used,
        raising=False,
    )
    log_path = tmp_path / "run.log"

    code = run_command(
        ["python", "run.py"],
        log_path,
        gpu_sample_interval_s=0.001,
    )

    assert code == 0
    assert "gpu_peak_memory_mib=16384" in log_path.read_text(encoding="utf-8")


def test_run_command_terminates_when_gpu_memory_limit_is_exceeded(monkeypatch, tmp_path):
    class FakeProcess:
        stdout = iter(["training output\n"])

        def __init__(self):
            self.terminated = False

        def terminate(self):
            self.terminated = True

        def wait(self):
            time.sleep(0.02)
            return -15 if self.terminated else 0

    fake_process = FakeProcess()

    monkeypatch.setattr(
        "scripts.luop_4090_stability.subprocess.Popen",
        lambda *args, **kwargs: fake_process,
    )
    monkeypatch.setattr(
        "scripts.luop_4090_stability.detect_gpu_memory_used_mib",
        lambda: [24073],
        raising=False,
    )
    log_path = tmp_path / "run.log"

    code = run_command(
        ["python", "run.py"],
        log_path,
        gpu_sample_interval_s=0.001,
        max_gpu_memory_mib=22000,
    )
    text = log_path.read_text(encoding="utf-8")

    assert code != 0
    assert fake_process.terminated is True
    assert "gpu_memory_limit_exceeded_mib=24073" in text
    assert "gpu_peak_memory_mib=24073" in text


def test_run_command_pins_child_process_to_selected_gpu(monkeypatch, tmp_path):
    captured_env = {}

    class FakeProcess:
        stdout = iter(["training output\n"])

        def wait(self):
            return 0

    def fake_popen(*args, **kwargs):
        captured_env.update(kwargs["env"])
        return FakeProcess()

    monkeypatch.setattr("scripts.luop_4090_stability.subprocess.Popen", fake_popen)
    monkeypatch.setattr(
        "scripts.luop_4090_stability.detect_gpu_memory_used_mib",
        lambda: [16384],
    )

    code = run_command(
        ["python", "run.py"],
        tmp_path / "run.log",
        cuda_visible_devices="1",
        gpu_sample_interval_s=0.001,
    )

    assert code == 0
    assert captured_env["CUDA_VISIBLE_DEVICES"] == "1"


def test_run_command_samples_peak_memory_from_pinned_physical_gpu(monkeypatch, tmp_path):
    class FakeProcess:
        stdout = iter(["training output\n"])

        def wait(self):
            return 0

    monkeypatch.setattr(
        "scripts.luop_4090_stability.subprocess.Popen",
        lambda *args, **kwargs: FakeProcess(),
    )
    monkeypatch.setattr(
        "scripts.luop_4090_stability.detect_gpu_memory_used_mib",
        lambda: [20000, 10000],
    )
    log_path = tmp_path / "run.log"

    code = run_command(
        ["python", "run.py"],
        log_path,
        cuda_visible_devices="1",
        gpu_sample_interval_s=0.001,
    )

    assert code == 0
    assert "gpu_peak_memory_mib=10000" in log_path.read_text(encoding="utf-8")


def test_4090_runner_accepts_csv_metrics_artifact_when_console_is_quiet(tmp_path):
    log_path = tmp_path / "quiet.log"
    log_path.write_text("training completed\n", encoding="utf-8")
    _write_hydra_run_config(tmp_path)
    metrics_path = tmp_path / "csv" / "version_0" / "metrics.csv"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        "epoch,step,train/loss,val/reward,train/compatibility_reward,train/accessibility_reward,"
        "val/pareto_hypervolume,val/pareto_front_size,val/checkpoint_score\n"
        "0,1,0.1,0.6,0.4,0.5,0.2,2,0.433\n",
        encoding="utf-8",
    )
    log_path.write_text(
        "training completed\ngpu_peak_memory_mib=16384\n",
        encoding="utf-8",
    )

    assert scan_stability_failures(log_path, run_dir=tmp_path) == []


def test_4090_runner_accepts_lightning_csv_metric_suffixes(tmp_path):
    log_path = tmp_path / "quiet.log"
    log_path.write_text(
        "training completed\ngpu_peak_memory_mib=16384\n", encoding="utf-8"
    )
    _write_hydra_run_config(tmp_path)
    metrics_path = tmp_path / "csv" / "version_0" / "metrics.csv"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        "epoch,step,train/loss_step,val/reward_epoch,"
        "train/compatibility_reward_step,train/accessibility_reward_step,"
        "val/pareto_hypervolume_epoch,val/pareto_front_size_epoch,"
        "val/checkpoint_score_epoch\n"
        "0,1,0.1,0.6,0.4,0.5,0.2,2,0.433\n",
        encoding="utf-8",
    )

    assert scan_stability_failures(log_path, run_dir=tmp_path) == []


def test_4090_runner_rejects_unstable_validation_reward_curve(tmp_path):
    log_path = tmp_path / "quiet.log"
    log_path.write_text(
        "training completed\ngpu_peak_memory_mib=16384\n", encoding="utf-8"
    )
    _write_hydra_run_config(tmp_path)
    metrics_path = tmp_path / "csv" / "version_0" / "metrics.csv"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        "epoch,step,train/loss,val/reward,train/compatibility_reward,"
        "train/accessibility_reward,val/pareto_hypervolume,val/pareto_front_size\n"
        "0,1,0.1,0.5365,0.4,0.5,0.300,2\n"
        "1,2,0.1,0.5114,0.4,0.5,0.305,2\n"
        "2,3,0.1,0.5883,0.4,0.5,0.310,2\n"
        "3,4,0.1,0.5703,0.4,0.5,0.315,2\n"
        "4,5,0.1,0.5294,0.4,0.5,0.320,2\n",
        encoding="utf-8",
    )

    failures = scan_stability_failures(log_path, run_dir=tmp_path)

    assert "unstable_val_reward_curve" in failures


def test_4090_runner_rejects_unstable_validation_pareto_curve(tmp_path):
    log_path = tmp_path / "quiet.log"
    log_path.write_text(
        "training completed\ngpu_peak_memory_mib=16384\n", encoding="utf-8"
    )
    _write_hydra_run_config(tmp_path)
    metrics_path = tmp_path / "csv" / "version_0" / "metrics.csv"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        "epoch,step,train/loss,val/reward,train/compatibility_reward,"
        "train/accessibility_reward,val/pareto_hypervolume,val/pareto_front_size\n"
        "0,1,0.1,0.500,0.4,0.5,0.2920,2\n"
        "1,2,0.1,0.505,0.4,0.5,0.2792,2\n"
        "2,3,0.1,0.510,0.4,0.5,0.3348,2\n"
        "3,4,0.1,0.515,0.4,0.5,0.2765,2\n"
        "4,5,0.1,0.520,0.4,0.5,0.2913,2\n",
        encoding="utf-8",
    )

    failures = scan_stability_failures(log_path, run_dir=tmp_path)

    assert "unstable_val_pareto_hypervolume_curve" in failures


def test_4090_runner_rejects_collapsing_validation_accessibility_curve(tmp_path):
    log_path = tmp_path / "quiet.log"
    log_path.write_text(
        "training completed\ngpu_peak_memory_mib=16384\n", encoding="utf-8"
    )
    _write_hydra_run_config(tmp_path)
    metrics_path = tmp_path / "csv" / "version_0" / "metrics.csv"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        "epoch,step,train/loss,val/reward,train/compatibility_reward,"
        "train/accessibility_reward,val/compatibility_reward,"
        "val/accessibility_reward,val/pareto_hypervolume,val/pareto_front_size\n"
        "0,1,0.1,0.500,0.4,0.5,0.700,0.300,0.250,2\n"
        "1,2,0.1,0.495,0.4,0.5,0.720,0.270,0.260,2\n"
        "2,3,0.1,0.490,0.4,0.5,0.740,0.230,0.270,2\n"
        "3,4,0.1,0.485,0.4,0.5,0.760,0.200,0.280,2\n",
        encoding="utf-8",
    )

    failures = scan_stability_failures(log_path, run_dir=tmp_path)

    assert "unstable_val_accessibility_reward_curve" in failures


def test_4090_runner_rejects_sustained_validation_accessibility_drift(tmp_path):
    log_path = tmp_path / "quiet.log"
    log_path.write_text(
        "training completed\ngpu_peak_memory_mib=16384\n", encoding="utf-8"
    )
    _write_hydra_run_config(tmp_path)
    metrics_path = tmp_path / "csv" / "version_0" / "metrics.csv"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        "epoch,step,train/loss,val/reward,train/compatibility_reward,"
        "train/accessibility_reward,val/compatibility_reward,"
        "val/accessibility_reward,val/pareto_hypervolume,val/pareto_front_size\n"
        "0,1,0.1,0.514,0.4,0.5,0.740,0.289,0.271,2\n"
        "1,2,0.1,0.520,0.4,0.5,0.742,0.298,0.265,2\n"
        "2,3,0.1,0.511,0.4,0.5,0.738,0.285,0.264,2\n"
        "3,4,0.1,0.513,0.4,0.5,0.737,0.290,0.260,2\n"
        "4,5,0.1,0.509,0.4,0.5,0.736,0.282,0.261,2\n"
        "5,6,0.1,0.504,0.4,0.5,0.730,0.277,0.257,2\n"
        "6,7,0.1,0.485,0.4,0.5,0.714,0.257,0.255,2\n"
        "7,8,0.1,0.493,0.4,0.5,0.711,0.274,0.258,2\n"
        "8,9,0.1,0.496,0.4,0.5,0.723,0.269,0.258,2\n"
        "9,10,0.1,0.487,0.4,0.5,0.703,0.271,0.244,2\n",
        encoding="utf-8",
    )

    failures = scan_stability_failures(
        log_path,
        run_dir=tmp_path,
        min_validation_points=8,
    )

    assert "unstable_val_accessibility_reward_curve" in failures


def test_4090_runner_accepts_small_validation_metric_noise(tmp_path):
    log_path = tmp_path / "quiet.log"
    log_path.write_text(
        "training completed\ngpu_peak_memory_mib=16384\n", encoding="utf-8"
    )
    _write_hydra_run_config(tmp_path)
    metrics_path = tmp_path / "csv" / "version_0" / "metrics.csv"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        "epoch,step,train/loss,val/reward,train/compatibility_reward,"
        "train/accessibility_reward,val/pareto_hypervolume,"
        "val/pareto_front_size,val/checkpoint_score\n"
        "0,1,0.1,0.520,0.4,0.5,0.300,2,0.373\n"
        "1,2,0.1,0.540,0.4,0.5,0.315,2,0.385\n"
        "2,3,0.1,0.535,0.4,0.5,0.318,2,0.386\n"
        "3,4,0.1,0.545,0.4,0.5,0.316,2,0.390\n",
        encoding="utf-8",
    )

    assert scan_stability_failures(log_path, run_dir=tmp_path) == []


def test_4090_runner_accepts_improvement_with_small_late_wiggle(tmp_path):
    log_path = tmp_path / "quiet.log"
    log_path.write_text(
        "training completed\ngpu_peak_memory_mib=16384\n", encoding="utf-8"
    )
    _write_hydra_run_config(tmp_path)
    metrics_path = tmp_path / "csv" / "version_0" / "metrics.csv"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        "epoch,step,train/loss,val/reward,train/compatibility_reward,"
        "train/accessibility_reward,val/compatibility_reward,"
        "val/accessibility_reward,val/pareto_hypervolume,"
        "val/pareto_front_size,val/checkpoint_score\n"
        "0,1,0.1,0.4427,0.4,0.5,0.6796,0.2059,0.2559,2,0.3015\n"
        "1,2,0.1,0.4974,0.4,0.5,0.7181,0.2767,0.2286,2,0.3342\n"
        "2,3,0.1,0.5210,0.4,0.5,0.7388,0.3032,0.2481,2,0.3574\n"
        "3,4,0.1,0.5244,0.4,0.5,0.7407,0.3081,0.2497,2,0.3607\n"
        "4,5,0.1,0.5271,0.4,0.5,0.7383,0.3160,0.2619,2,0.3683\n"
        "5,6,0.1,0.5350,0.4,0.5,0.7486,0.3213,0.2802,2,0.3788\n"
        "6,7,0.1,0.5323,0.4,0.5,0.7325,0.3322,0.2757,2,0.3801\n"
        "7,8,0.1,0.5345,0.4,0.5,0.7345,0.3344,0.2832,2,0.3840\n",
        encoding="utf-8",
    )

    assert (
        scan_stability_failures(
            log_path,
            run_dir=tmp_path,
            min_validation_points=8,
        )
        == []
    )


def test_4090_runner_accepts_compensated_objective_rebalancing(tmp_path):
    log_path = tmp_path / "quiet.log"
    log_path.write_text(
        "training completed\ngpu_peak_memory_mib=16384\n", encoding="utf-8"
    )
    _write_hydra_run_config(tmp_path)
    metrics_path = tmp_path / "csv" / "version_0" / "metrics.csv"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        "epoch,step,train/loss,val/reward,train/compatibility_reward,"
        "train/accessibility_reward,val/compatibility_reward,"
        "val/accessibility_reward,val/pareto_hypervolume,"
        "val/pareto_front_size,val/checkpoint_score\n"
        "0,1,0.1,0.4627,0.4,0.5,0.7547,0.1707,0.1854,2,0.2730\n"
        "1,2,0.1,0.4956,0.4,0.5,0.7456,0.2456,0.2141,2,0.3184\n"
        "2,3,0.1,0.4892,0.4,0.5,0.7475,0.2308,0.2291,2,0.3164\n"
        "3,4,0.1,0.4838,0.4,0.5,0.7481,0.2195,0.2648,2,0.3227\n"
        "4,5,0.1,0.4840,0.4,0.5,0.7401,0.2278,0.2783,2,0.3300\n"
        "5,6,0.1,0.4982,0.4,0.5,0.7320,0.2643,0.2825,2,0.3483\n"
        "6,7,0.1,0.5160,0.4,0.5,0.7326,0.2994,0.2863,2,0.3672\n"
        "7,8,0.1,0.5304,0.4,0.5,0.7157,0.3450,0.2900,2,0.3885\n",
        encoding="utf-8",
    )

    assert (
        scan_stability_failures(
            log_path,
            run_dir=tmp_path,
            min_validation_points=8,
        )
        == []
    )


def test_4090_runner_accepts_console_training_metrics_with_partial_csv(tmp_path):
    log_path = tmp_path / "run.log"
    _write_hydra_run_config(tmp_path)
    log_path.write_text(
        "train/reward=0.373, train/compatibility_reward=0.493, "
        "train/accessibility_reward=0.258, train/loss=2.520\n"
        "val/reward=0.373, val/compatibility_reward=0.670, "
        "val/accessibility_reward=0.100, val/pareto_hypervolume=0.157, "
        "val/pareto_front_size=2.010, val/checkpoint_score=0.210\n"
        "gpu_peak_memory_mib=12659\n",
        encoding="utf-8",
    )
    metrics_path = tmp_path / "csv" / "version_0" / "metrics.csv"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        "epoch,step,test/accessibility_reward,test/compatibility_reward,"
        "test/pareto_front_size,test/pareto_hypervolume,test/reward,"
        "time/epoch (s),val/accessibility_reward,val/compatibility_reward,"
        "val/pareto_front_size,val/pareto_hypervolume,val/reward\n"
        "0,1,,,,,,,0.09,0.670,2.008,0.157,0.373\n"
        ",2,,,,,,10.0,,,,,\n"
        "1,2,0.109,0.665,2.002,0.164,0.383,,,,,,\n",
        encoding="utf-8",
    )

    assert scan_stability_failures(log_path, run_dir=tmp_path) == []


def test_4090_runner_rejects_csv_metrics_with_nonfinite_required_values(tmp_path):
    log_path = tmp_path / "quiet.log"
    log_path.write_text("training completed\n", encoding="utf-8")
    metrics_path = tmp_path / "csv" / "version_0" / "metrics.csv"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        "epoch,step,train/compatibility_reward,train/accessibility_reward,"
        "val/pareto_hypervolume,val/pareto_front_size\n"
        "0,1,nan,0.5,0.2,2\n",
        encoding="utf-8",
    )

    failures = scan_stability_failures(log_path, run_dir=tmp_path)

    assert "nonfinite_metrics" in failures


def test_4090_runner_rejects_csv_metrics_with_header_only_required_values(tmp_path):
    log_path = tmp_path / "quiet.log"
    log_path.write_text("training completed\n", encoding="utf-8")
    metrics_path = tmp_path / "csv" / "version_0" / "metrics.csv"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        "epoch,step,train/compatibility_reward,train/accessibility_reward,"
        "val/pareto_hypervolume,val/pareto_front_size\n",
        encoding="utf-8",
    )

    failures = scan_stability_failures(log_path, run_dir=tmp_path)

    assert "missing_finite_metrics" in failures


def test_4090_runner_requires_csv_metrics_when_run_dir_is_scanned(tmp_path):
    log_path = tmp_path / "quiet.log"
    log_path.write_text("training completed\n", encoding="utf-8")

    failures = scan_stability_failures(log_path, run_dir=tmp_path)

    assert "missing_compatibility_metric" in failures
    assert "missing_pareto_hypervolume" in failures


def test_4090_runner_accepts_hydra_config_with_fixed_eval_weights(tmp_path):
    log_path = tmp_path / "quiet.log"
    log_path.write_text(
        "training completed\ngpu_peak_memory_mib=16384\n", encoding="utf-8"
    )
    _write_hydra_run_config(tmp_path)
    _write_stable_metrics(tmp_path)

    assert scan_stability_failures(log_path, run_dir=tmp_path) == []


def test_4090_runner_rejects_missing_hydra_config_when_run_dir_is_scanned(tmp_path):
    log_path = tmp_path / "quiet.log"
    log_path.write_text(
        "training completed\ngpu_peak_memory_mib=16384\n", encoding="utf-8"
    )
    _write_stable_metrics(tmp_path)

    failures = scan_stability_failures(log_path, run_dir=tmp_path)

    assert "missing_run_config" in failures


def test_4090_runner_rejects_sampled_eval_objective_weights_in_hydra_config(tmp_path):
    log_path = tmp_path / "quiet.log"
    log_path.write_text(
        "training completed\ngpu_peak_memory_mib=16384\n", encoding="utf-8"
    )
    _write_hydra_run_config(tmp_path, sample_eval_objective_weights=True)
    _write_stable_metrics(tmp_path)

    failures = scan_stability_failures(log_path, run_dir=tmp_path)

    assert "sampled_eval_objective_weights" in failures


def test_4090_runner_rejects_sampled_train_objective_weights_in_hydra_config(tmp_path):
    log_path = tmp_path / "quiet.log"
    log_path.write_text(
        "training completed\ngpu_peak_memory_mib=16384\n", encoding="utf-8"
    )
    _write_hydra_run_config(tmp_path, sample_objective_weights=True)
    _write_stable_metrics(tmp_path)

    failures = scan_stability_failures(log_path, run_dir=tmp_path)

    assert "sampled_train_objective_weights" in failures


def test_4090_runner_rejects_missing_eval_objective_weights_in_hydra_config(tmp_path):
    log_path = tmp_path / "quiet.log"
    log_path.write_text(
        "training completed\ngpu_peak_memory_mib=16384\n", encoding="utf-8"
    )
    _write_hydra_run_config(tmp_path, eval_objective_weights=None)
    _write_stable_metrics(tmp_path)

    failures = scan_stability_failures(log_path, run_dir=tmp_path)

    assert "missing_eval_objective_weights" in failures


def test_4090_runner_rejects_nonfixed_eval_objective_weights_in_hydra_config(
    tmp_path,
):
    log_path = tmp_path / "quiet.log"
    log_path.write_text(
        "training completed\ngpu_peak_memory_mib=16384\n", encoding="utf-8"
    )
    _write_hydra_run_config(tmp_path, eval_objective_weights=(0.7, 0.3))
    _write_stable_metrics(tmp_path)

    failures = scan_stability_failures(log_path, run_dir=tmp_path)

    assert "wrong_eval_objective_weights" in failures


def test_4090_runner_rejects_wrong_checkpoint_monitor_in_hydra_config(tmp_path):
    log_path = tmp_path / "quiet.log"
    log_path.write_text(
        "training completed\ngpu_peak_memory_mib=16384\n", encoding="utf-8"
    )
    _write_hydra_run_config(tmp_path, checkpoint_monitor="val/reward")
    _write_stable_metrics(tmp_path)

    failures = scan_stability_failures(log_path, run_dir=tmp_path)

    assert "wrong_checkpoint_monitor" in failures


def test_4090_runner_rejects_legacy_generic_attention_model_target(tmp_path):
    log_path = tmp_path / "quiet.log"
    log_path.write_text(
        "training completed\ngpu_peak_memory_mib=16384\n", encoding="utf-8"
    )
    _write_hydra_run_config(tmp_path, model_target="rl4co.models.AttentionModel")
    _write_stable_metrics(tmp_path)

    failures = scan_stability_failures(log_path, run_dir=tmp_path)

    assert "wrong_luop_model_target" in failures


def test_4090_runner_rejects_type_first_luop_decode_config(tmp_path):
    log_path = tmp_path / "quiet.log"
    log_path.write_text(
        "training completed\ngpu_peak_memory_mib=16384\n", encoding="utf-8"
    )
    _write_hydra_run_config(tmp_path, decode_type_first=True)
    _write_stable_metrics(tmp_path)

    failures = scan_stability_failures(log_path, run_dir=tmp_path)

    assert "type_first_luop_decode" in failures


def test_4090_runner_rejects_successful_process_with_bad_stability_log(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "scripts.luop_4090_stability.detect_gpu_names",
        lambda: ["NVIDIA GeForce RTX 4090"],
    )

    def fake_run_command(cmd, log_path, **kwargs):
        log_path.write_text("val/reward=nan\n", encoding="utf-8")
        return 0

    monkeypatch.setattr("scripts.luop_4090_stability.run_command", fake_run_command)
    monkeypatch.setattr(
        "sys.argv",
        [
            "luop_4090_stability.py",
            "--num-loc",
            "50",
            "--target-batch",
            "1024",
            "--min-batch",
            "1024",
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )

    assert main() == 1


def test_4090_runner_rejects_successful_process_without_multiobjective_metrics(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "scripts.luop_4090_stability.detect_gpu_names",
        lambda: ["NVIDIA GeForce RTX 4090"],
    )

    def fake_run_command(cmd, log_path, **kwargs):
        log_path.write_text("train/loss=0.1\nval/reward=0.2\n", encoding="utf-8")
        return 0

    monkeypatch.setattr("scripts.luop_4090_stability.run_command", fake_run_command)
    monkeypatch.setattr(
        "sys.argv",
        [
            "luop_4090_stability.py",
            "--num-loc",
            "50",
            "--target-batch",
            "1024",
            "--min-batch",
            "1024",
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )

    assert main() == 1


def test_4090_runner_full_run_rejects_single_validation_point(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "scripts.luop_4090_stability.detect_gpu_names",
        lambda: ["NVIDIA GeForce RTX 4090"],
    )

    def fake_run_command(cmd, log_path, **kwargs):
        run_dir_override = next(item for item in cmd if item.startswith("hydra.run.dir="))
        run_dir = Path(run_dir_override.split("=", 1)[1])
        metrics_path = run_dir / "csv" / "version_0" / "metrics.csv"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            "epoch,step,train/loss,val/reward,train/compatibility_reward,"
            "train/accessibility_reward,val/pareto_hypervolume,"
            "val/pareto_front_size,val/checkpoint_score\n"
            "0,1,0.1,0.6,0.4,0.5,0.2,2,0.433\n",
            encoding="utf-8",
        )
        log_path.write_text(
            "training completed\ngpu_peak_memory_mib=16384\n",
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr("scripts.luop_4090_stability.run_command", fake_run_command)
    monkeypatch.setattr(
        "sys.argv",
        [
            "luop_4090_stability.py",
            "--num-loc",
            "50",
            "--target-batch",
            "512",
            "--min-batch",
            "512",
            "--full",
            "--max-epochs",
            "1",
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )

    assert main() == 1


def test_4090_runner_backs_off_after_cuda_oom_and_keeps_successful_batch(
    monkeypatch, tmp_path, capsys
):
    launched_batches = []

    monkeypatch.setattr(
        "scripts.luop_4090_stability.detect_gpu_names",
        lambda: ["NVIDIA GeForce RTX 4090"],
    )

    def fake_run_command(cmd, log_path, **kwargs):
        batch_override = next(
            item for item in cmd if item.startswith("model.batch_size=")
        )
        batch_size = int(batch_override.split("=")[1])
        launched_batches.append(batch_size)
        if batch_size == 1024:
            log_path.write_text("RuntimeError: CUDA out of memory\n", encoding="utf-8")
            return 1
        _write_hydra_run_config_for_command(cmd)
        log_path.write_text(
            "train/loss=0.1\n"
            "val/reward=0.6\n"
            "train/compatibility_reward=0.4\n"
            "train/accessibility_reward=0.5\n"
            "val/pareto_hypervolume=0.2\n"
            "val/pareto_front_size=2\n"
            "val/checkpoint_score=0.433\n"
            "gpu_peak_memory_mib=16384\n",
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr("scripts.luop_4090_stability.run_command", fake_run_command)
    monkeypatch.setattr(
        "sys.argv",
        [
            "luop_4090_stability.py",
            "--num-loc",
            "50",
            "--target-batch",
            "1024",
            "--min-batch",
            "512",
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )

    assert main() == 0
    assert launched_batches == [1024, 512]
    assert "Selected stable LUOP batch_size=512" in capsys.readouterr().out


def test_4090_runner_backs_off_after_cuda_oom_with_header_only_metrics(
    monkeypatch, tmp_path, capsys
):
    launched_batches = []

    monkeypatch.setattr(
        "scripts.luop_4090_stability.detect_gpu_names",
        lambda: ["NVIDIA GeForce RTX 4090"],
    )

    def fake_run_command(cmd, log_path, **kwargs):
        batch_override = next(
            item for item in cmd if item.startswith("model.batch_size=")
        )
        batch_size = int(batch_override.split("=")[1])
        launched_batches.append(batch_size)
        run_dir_override = next(item for item in cmd if item.startswith("hydra.run.dir="))
        run_dir = Path(run_dir_override.split("=", 1)[1])
        metrics_path = run_dir / "csv" / "version_0" / "metrics.csv"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        if batch_size == 1024:
            metrics_path.write_text(
                "epoch,step,train/loss,val/reward,train/compatibility_reward,"
                "train/accessibility_reward,val/pareto_hypervolume,"
                "val/pareto_front_size\n",
                encoding="utf-8",
            )
            log_path.write_text("RuntimeError: CUDA out of memory\n", encoding="utf-8")
            return 1
        _write_hydra_run_config(run_dir)
        metrics_path.write_text(
            "epoch,step,train/loss,val/reward,train/compatibility_reward,"
            "train/accessibility_reward,val/pareto_hypervolume,"
            "val/pareto_front_size,val/checkpoint_score\n"
            "0,1,0.1,0.6,0.4,0.5,0.2,2,0.433\n",
            encoding="utf-8",
        )
        log_path.write_text("gpu_peak_memory_mib=16384\n", encoding="utf-8")
        return 0

    monkeypatch.setattr("scripts.luop_4090_stability.run_command", fake_run_command)
    monkeypatch.setattr(
        "sys.argv",
        [
            "luop_4090_stability.py",
            "--num-loc",
            "50",
            "--target-batch",
            "1024",
            "--min-batch",
            "512",
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )

    assert main() == 0
    assert launched_batches == [1024, 512]
    assert "Selected stable LUOP batch_size=512" in capsys.readouterr().out


def test_4090_runner_backs_off_after_cuda_oom_with_low_sampled_peak_memory(
    monkeypatch, tmp_path, capsys
):
    launched_batches = []

    monkeypatch.setattr(
        "scripts.luop_4090_stability.detect_gpu_names",
        lambda: ["NVIDIA GeForce RTX 4090"],
    )

    def fake_run_command(cmd, log_path, **kwargs):
        batch_override = next(
            item for item in cmd if item.startswith("model.batch_size=")
        )
        batch_size = int(batch_override.split("=")[1])
        launched_batches.append(batch_size)
        if batch_size == 1024:
            log_path.write_text(
                "RuntimeError: CUDA out of memory\n" "gpu_peak_memory_mib=1024\n",
                encoding="utf-8",
            )
            return 1
        _write_hydra_run_config_for_command(cmd)
        log_path.write_text(
            "train/loss=0.1\n"
            "val/reward=0.6\n"
            "train/compatibility_reward=0.4\n"
            "train/accessibility_reward=0.5\n"
            "val/pareto_hypervolume=0.2\n"
            "val/pareto_front_size=2\n"
            "val/checkpoint_score=0.433\n"
            "gpu_peak_memory_mib=16384\n",
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr("scripts.luop_4090_stability.run_command", fake_run_command)
    monkeypatch.setattr(
        "sys.argv",
        [
            "luop_4090_stability.py",
            "--num-loc",
            "50",
            "--target-batch",
            "1024",
            "--min-batch",
            "512",
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )

    assert main() == 0
    assert launched_batches == [1024, 512]
    assert "Selected stable LUOP batch_size=512" in capsys.readouterr().out


def test_4090_runner_backs_off_after_gpu_memory_cap_is_exceeded(
    monkeypatch, tmp_path, capsys
):
    launched_batches = []

    monkeypatch.setattr(
        "scripts.luop_4090_stability.detect_gpu_names",
        lambda: ["NVIDIA GeForce RTX 4090"],
    )

    def fake_run_command(cmd, log_path, **kwargs):
        batch_override = next(
            item for item in cmd if item.startswith("model.batch_size=")
        )
        batch_size = int(batch_override.split("=")[1])
        launched_batches.append(batch_size)
        if batch_size == 256:
            log_path.write_text(
                "gpu_memory_limit_exceeded_mib=22001\n" "gpu_peak_memory_mib=22001\n",
                encoding="utf-8",
            )
            return -15
        _write_hydra_run_config_for_command(cmd)
        log_path.write_text(
            "train/loss=0.1\n"
            "val/reward=0.6\n"
            "train/compatibility_reward=0.4\n"
            "train/accessibility_reward=0.5\n"
            "val/pareto_hypervolume=0.2\n"
            "val/pareto_front_size=2\n"
            "val/checkpoint_score=0.433\n"
            "gpu_peak_memory_mib=12000\n",
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr("scripts.luop_4090_stability.run_command", fake_run_command)
    monkeypatch.setattr(
        "sys.argv",
        [
            "luop_4090_stability.py",
            "--num-loc",
            "200",
            "--target-batch",
            "256",
            "--min-batch",
            "128",
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )

    assert main() == 0
    assert launched_batches == [256, 128]
    assert "Selected stable LUOP batch_size=128" in capsys.readouterr().out


def test_4090_runner_cli_passes_custom_peak_memory_threshold(
    monkeypatch, tmp_path, capsys
):
    launched_batches = []

    monkeypatch.setattr(
        "scripts.luop_4090_stability.detect_gpu_names",
        lambda: ["NVIDIA GeForce RTX 4090"],
    )

    def fake_run_command(cmd, log_path, **kwargs):
        batch_override = next(
            item for item in cmd if item.startswith("model.batch_size=")
        )
        launched_batches.append(int(batch_override.split("=")[1]))
        _write_hydra_run_config_for_command(cmd)
        log_path.write_text(
            "train/loss=0.1\n"
            "val/reward=0.6\n"
            "train/compatibility_reward=0.4\n"
            "train/accessibility_reward=0.5\n"
            "val/pareto_hypervolume=0.2\n"
            "val/pareto_front_size=2\n"
            "val/checkpoint_score=0.433\n"
            "gpu_peak_memory_mib=4096\n",
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr("scripts.luop_4090_stability.run_command", fake_run_command)
    monkeypatch.setattr(
        "sys.argv",
        [
            "luop_4090_stability.py",
            "--num-loc",
            "50",
            "--target-batch",
            "1024",
            "--min-batch",
            "1024",
            "--min-gpu-peak-memory-mib",
            "4096",
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )

    assert main() == 0
    assert launched_batches == [1024]
    assert "Selected stable LUOP batch_size=1024" in capsys.readouterr().out


def test_4090_runner_stops_on_non_oom_stability_failure(monkeypatch, tmp_path):
    launched_batches = []

    monkeypatch.setattr(
        "scripts.luop_4090_stability.detect_gpu_names",
        lambda: ["NVIDIA GeForce RTX 4090"],
    )

    def fake_run_command(cmd, log_path, **kwargs):
        batch_override = next(
            item for item in cmd if item.startswith("model.batch_size=")
        )
        launched_batches.append(int(batch_override.split("=")[1]))
        log_path.write_text("ValueError: no valid LUOP joint action\n", encoding="utf-8")
        return 1

    monkeypatch.setattr("scripts.luop_4090_stability.run_command", fake_run_command)
    monkeypatch.setattr(
        "sys.argv",
        [
            "luop_4090_stability.py",
            "--num-loc",
            "50",
            "--target-batch",
            "1024",
            "--min-batch",
            "256",
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )

    assert main() == 1
    assert launched_batches == [1024]


def test_4090_runner_refuses_to_launch_without_4090(monkeypatch, tmp_path):
    launched = False

    def fake_run_command(cmd, log_path, **kwargs):
        nonlocal launched
        launched = True
        return 0

    monkeypatch.setattr(
        "scripts.luop_4090_stability.detect_gpu_names",
        lambda: ["NVIDIA GeForce RTX 4060"],
    )
    monkeypatch.setattr("scripts.luop_4090_stability.run_command", fake_run_command)
    monkeypatch.setattr(
        "sys.argv",
        [
            "luop_4090_stability.py",
            "--num-loc",
            "50",
            "--target-batch",
            "1024",
            "--min-batch",
            "1024",
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )

    assert main() == 2
    assert launched is False


def test_4090_runner_refuses_to_launch_4090_with_too_little_memory(monkeypatch, tmp_path):
    launched = False

    def fake_run_command(cmd, log_path, **kwargs):
        nonlocal launched
        launched = True
        log_path.write_text(
            "train/compatibility_reward=0.4\n"
            "train/accessibility_reward=0.5\n"
            "val/pareto_hypervolume=0.2\n"
            "val/pareto_front_size=2\n",
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(
        "scripts.luop_4090_stability.detect_gpu_names",
        lambda: ["NVIDIA GeForce RTX 4090"],
    )
    monkeypatch.setattr(
        "scripts.luop_4090_stability.detect_gpu_memory_mib",
        lambda: [8192],
        raising=False,
    )
    monkeypatch.setattr("scripts.luop_4090_stability.run_command", fake_run_command)
    monkeypatch.setattr(
        "sys.argv",
        [
            "luop_4090_stability.py",
            "--num-loc",
            "50",
            "--target-batch",
            "1024",
            "--min-batch",
            "1024",
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )

    assert main() == 2
    assert launched is False


def test_4090_runner_accepts_custom_preflight_gpu_memory_floor(monkeypatch, tmp_path):
    launched = []

    monkeypatch.setattr(
        "scripts.luop_4090_stability.detect_gpu_names",
        lambda: ["NVIDIA GeForce RTX 4090"],
    )
    monkeypatch.setattr(
        "scripts.luop_4090_stability.detect_gpu_memory_mib",
        lambda: [16384],
        raising=False,
    )

    def fake_run_command(cmd, log_path, **kwargs):
        launched.append(cmd)
        _write_hydra_run_config_for_command(cmd)
        log_path.write_text(
            "train/loss=0.1\n"
            "val/reward=0.6\n"
            "train/compatibility_reward=0.4\n"
            "train/accessibility_reward=0.5\n"
            "val/pareto_hypervolume=0.2\n"
            "val/pareto_front_size=2\n"
            "val/checkpoint_score=0.433\n"
            "gpu_peak_memory_mib=16384\n",
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr("scripts.luop_4090_stability.run_command", fake_run_command)
    monkeypatch.setattr(
        "sys.argv",
        [
            "luop_4090_stability.py",
            "--num-loc",
            "50",
            "--target-batch",
            "1024",
            "--min-batch",
            "1024",
            "--min-gpu-memory-mib",
            "12000",
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )

    assert main() == 0
    assert len(launched) == 1


def test_4090_runner_gpu_preflight_accepts_expected_name(monkeypatch):
    monkeypatch.setattr(
        "scripts.luop_4090_stability.detect_gpu_names",
        lambda: ["NVIDIA GeForce RTX 4090"],
    )

    assert assert_expected_gpu("4090") == ["NVIDIA GeForce RTX 4090"]


def test_4090_runner_gpu_preflight_returns_selected_4090_index(monkeypatch):
    monkeypatch.setattr(
        "scripts.luop_4090_stability.detect_gpu_names",
        lambda: ["NVIDIA GeForce RTX 4060", "NVIDIA GeForce RTX 4090"],
    )
    monkeypatch.setattr(
        "scripts.luop_4090_stability.detect_gpu_memory_mib",
        lambda: [8192, 24564],
    )

    assert assert_expected_gpu("4090").selected_index == 1


def test_4090_runner_pins_training_to_selected_4090(monkeypatch, tmp_path):
    captured_visible_devices = []

    monkeypatch.setattr(
        "scripts.luop_4090_stability.detect_gpu_names",
        lambda: ["NVIDIA GeForce RTX 4060", "NVIDIA GeForce RTX 4090"],
    )
    monkeypatch.setattr(
        "scripts.luop_4090_stability.detect_gpu_memory_mib",
        lambda: [8192, 24564],
    )

    def fake_run_command(cmd, log_path, **kwargs):
        captured_visible_devices.append(kwargs.get("cuda_visible_devices"))
        _write_hydra_run_config_for_command(cmd)
        log_path.write_text(
            "train/loss=0.1\n"
            "val/reward=0.6\n"
            "train/compatibility_reward=0.4\n"
            "train/accessibility_reward=0.5\n"
            "val/pareto_hypervolume=0.2\n"
            "val/pareto_front_size=2\n"
            "val/checkpoint_score=0.433\n"
            "gpu_peak_memory_mib=16384\n",
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr("scripts.luop_4090_stability.run_command", fake_run_command)
    monkeypatch.setattr(
        "sys.argv",
        [
            "luop_4090_stability.py",
            "--num-loc",
            "50",
            "--target-batch",
            "1024",
            "--min-batch",
            "1024",
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )

    assert main() == 0
    assert captured_visible_devices == ["1"]


def test_4090_runner_gpu_preflight_rejects_4090_with_too_little_memory(monkeypatch):
    monkeypatch.setattr(
        "scripts.luop_4090_stability.detect_gpu_names",
        lambda: ["NVIDIA GeForce RTX 4090"],
    )
    monkeypatch.setattr(
        "scripts.luop_4090_stability.detect_gpu_memory_mib",
        lambda: [8192],
        raising=False,
    )

    with pytest.raises(RuntimeError, match="at least 20000 MiB"):
        assert_expected_gpu("4090", min_memory_mib=20000)
