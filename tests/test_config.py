"""Tests for config dataclass defaults and invariants."""

from rally.config import AssetConfig, ASSETS, PARAMS, PIPELINE


def test_asset_config_fields():
    ac = AssetConfig(ticker="TEST", asset_class="equity", r_up=0.03, d_dn=0.015)
    assert ac.ticker == "TEST"
    assert ac.asset_class == "equity"
    assert ac.r_up == 0.03
    assert ac.d_dn == 0.015


def test_assets_dict_nonempty():
    assert len(ASSETS) > 0
    for key, ac in ASSETS.items():
        assert ac.asset_class in ("equity", "crypto")
        assert ac.r_up > 0
        assert ac.d_dn > 0
        assert ac.r_up > ac.d_dn  # rally target must exceed drawdown tolerance


def test_params_defaults():
    assert PARAMS.forward_horizon > 0
    assert PARAMS.atr_period > 0
    assert PARAMS.walk_forward_train_years > 0
    assert 0 < PARAMS.p_rally_threshold < 1
    assert 0 < PARAMS.max_risk_frac <= 1


def test_pipeline_defaults():
    assert PIPELINE.n_workers > 0
    assert PIPELINE.hmm_n_iter > 0
    assert PIPELINE.hmm_tol > 0


def test_crypto_assets_use_usd_suffix():
    for key, ac in ASSETS.items():
        if ac.asset_class == "crypto":
            assert ac.ticker.endswith("-USD"), f"{key} crypto ticker should end with -USD"
