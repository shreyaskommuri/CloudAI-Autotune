from autotune import config_mutator


def test_load_config_reads_examples():
    config = config_mutator.load_config("configs/examples/vllm_baseline.toml")

    assert config["scenario"]["name"] == "vllm_baseline"
    assert config["serving"]["batch_size"] == 1


def test_get_and_set_value_are_immutable():
    config = {"serving": {"batch_size": 1, "max_tokens": 256}}

    assert config_mutator.get_value(config, "serving.batch_size") == 1

    updated = config_mutator.set_value(config, "serving.batch_size", 8)

    assert updated["serving"]["batch_size"] == 8
    assert config["serving"]["batch_size"] == 1  # original untouched


def test_to_toml_round_trips_through_load(tmp_path):
    config = {
        "scenario": {"name": "demo", "backend": "vllm"},
        "serving": {"batch_size": 8, "max_tokens": 256},
    }

    out_path = config_mutator.write_config(config, tmp_path / "derived.toml")
    reloaded = config_mutator.load_config(out_path)

    assert reloaded == config


def test_derive_config_applies_overrides(tmp_path):
    out_path = tmp_path / "derived.toml"

    written = config_mutator.derive_config(
        "configs/examples/vllm_baseline.toml",
        overrides={"serving.batch_size": 8},
        out_path=out_path,
    )

    derived = config_mutator.load_config(written)
    assert derived["serving"]["batch_size"] == 8
    assert derived["scenario"]["name"] == "vllm_baseline"  # untouched keys preserved
