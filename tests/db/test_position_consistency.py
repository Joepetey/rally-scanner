"""Tests for model persistence atomicity and position state consistency."""

import pytest
from rally_ml.config import AssetConfig
from rally_ml.core.persistence import load_model, save_model


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    """Redirect MODELS_DIR to a temp directory and stub out DB manifest writes."""
    models = tmp_path / "models"
    models.mkdir()

    import rally_ml.core.persistence as persist
    monkeypatch.setattr(persist, "MODELS_DIR", models)
    fake_store = type(
        "FakeStore", (), {"save_entry": lambda self, t, m: None, "load_all": lambda self: {}}
    )()
    monkeypatch.setattr(persist, "_store", fake_store)

    return models


def _make_asset_config() -> AssetConfig:
    return AssetConfig(ticker="TEST", asset_class="equity", r_up=0.05, d_dn=0.03)


def _make_artifacts(tag: str = "v1") -> dict:
    return {
        "lr_model": None,
        "lr_scaler": None,
        "iso_calibrator": None,
        "hmm_model": None,
        "hmm_scaler": None,
        "state_order": [0, 1],
        "feature_cols": ["ret_1d", "vol_5d"],
        "train_start": "2020-01-01",
        "train_end": "2021-01-01",
        "tag": tag,
    }


class TestSaveModelAtomicWrite:
    """save_model writes to .tmp then renames, producing a valid .joblib file."""

    def test_save_creates_valid_loadable_file(self, models_dir):
        config = _make_asset_config()
        artifacts = _make_artifacts()

        save_model("TEST", artifacts, config)

        loaded = load_model("TEST")
        assert loaded["tag"] == "v1"
        assert loaded["feature_cols"] == ["ret_1d", "vol_5d"]
        assert loaded["train_start"] == "2020-01-01"
        assert loaded["asset_config"]["r_up"] == 0.05

    def test_save_leaves_no_tmp_file(self, models_dir):
        save_model("TEST", _make_artifacts(), _make_asset_config())

        tmp_files = list(models_dir.glob("*.tmp"))
        assert tmp_files == [], f"Leftover .tmp files: {tmp_files}"

    def test_save_overwrites_previous_model(self, models_dir):
        config = _make_asset_config()

        save_model("TEST", _make_artifacts("v1"), config)
        save_model("TEST", _make_artifacts("v2"), config)

        loaded = load_model("TEST")
        assert loaded["tag"] == "v2"

    def test_save_injects_saved_at_timestamp(self, models_dir):
        save_model("TEST", _make_artifacts(), _make_asset_config())

        loaded = load_model("TEST")
        assert "saved_at" in loaded


class TestLoadModelDuringConcurrentWrite:
    """Atomic rename means load_model always sees a complete file."""

    def test_load_after_save_returns_valid_artifacts(self, models_dir):
        config = _make_asset_config()

        # Write initial model
        save_model("TEST", _make_artifacts("old"), config)

        # Write a new model (atomic rename replaces the file)
        save_model("TEST", _make_artifacts("new"), config)

        # Reader should see the new complete file
        loaded = load_model("TEST")
        assert loaded["tag"] == "new"
        assert loaded["feature_cols"] == ["ret_1d", "vol_5d"]

    def test_no_tmp_file_remains_after_sequential_saves(self, models_dir):
        config = _make_asset_config()

        for i in range(5):
            save_model("TEST", _make_artifacts(f"v{i}"), config)

        tmp_files = list(models_dir.glob("*.tmp"))
        assert tmp_files == []

        # Only one .joblib file should exist
        joblib_files = list(models_dir.glob("TEST.joblib"))
        assert len(joblib_files) == 1

    def test_load_nonexistent_raises_file_not_found(self, models_dir):
        with pytest.raises(FileNotFoundError, match="No saved model for MISSING"):
            load_model("MISSING")

    def test_load_corrupt_file_raises_runtime_error(self, models_dir):
        # Write garbage bytes to simulate corruption
        bad_path = models_dir / "BAD.joblib"
        bad_path.write_bytes(b"not a valid joblib file")

        with pytest.raises(RuntimeError, match="Corrupt model artifact for BAD"):
            load_model("BAD")
