import pytest

from scripts.luop_dominance_4090 import build_command, scan_csv_metrics


def test_luop_dominance_4090_command_avoids_large_default_data_generation():
    command = build_command()

    assert "experiment=cityplan/dominance" in command
    assert "model.generate_default_data=false" in command
    assert "env.val_file=null" in command
    assert "env.test_file=null" in command


def test_luop_dominance_4090_command_disables_nonessential_rich_progress():
    command = build_command()

    assert "logger=csv" in command
    assert "callbacks.rich_progress_bar=null" in command
    assert "callbacks.learning_rate_monitor=null" in command
    assert "+trainer.log_every_n_steps=1" in command


def test_luop_dominance_metrics_scan_accepts_finite_varying_metrics(tmp_path):
    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "train/dominance_reward,train/hv_contribution,"
        "val/pareto_hypervolume,val/pareto_front_size\n"
        "0.7,0.01,0.2,2\n"
        "0.8,0.02,0.3,3\n",
        encoding="utf-8",
    )

    values = scan_csv_metrics(metrics)

    assert values["train/dominance_reward"] == [0.7, 0.8]
    assert values["val/pareto_front_size"] == [2.0, 3.0]


def test_luop_dominance_metrics_scan_requires_all_metrics(tmp_path):
    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "train/dominance_reward,train/hv_contribution,"
        "val/pareto_hypervolume\n"
        "0.7,0.01,0.2\n"
        "0.8,0.02,0.3\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing finite metrics"):
        scan_csv_metrics(metrics)


def test_luop_dominance_metrics_scan_rejects_constant_dominance_reward(tmp_path):
    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "train/dominance_reward,train/hv_contribution,"
        "val/pareto_hypervolume,val/pareto_front_size\n"
        "0.7,0.01,0.2,2\n"
        "0.7,0.02,0.3,3\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="collapsed to a constant"):
        scan_csv_metrics(metrics)
