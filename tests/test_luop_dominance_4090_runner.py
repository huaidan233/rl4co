from scripts.luop_dominance_4090 import build_command


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
