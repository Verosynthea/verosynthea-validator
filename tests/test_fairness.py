"""
Tests for verosynthea-validator fairness metrics.

Covers:
- FairnessReport with a fair model (should pass)
- FairnessReport with a biased model (should fail)
- assert_fair CI helper
- Edge cases: small groups, single group, missing columns
- All 8 demographic profiles as protected attribute
"""
import numpy as np
import pandas as pd
import pytest

from verosynthea_validator import FairnessReport, assert_fair
from verosynthea_validator.assertions import FairnessAssertionError


@pytest.fixture
def sample_data():
    """Create a synthetic dataset mimicking AUSynth structure."""
    np.random.seed(42)
    n = 2000
    profiles = np.random.choice(
        ["High-earning professionals", "Young singles and non-workers",
         "Established partnered households", "Trades and technical workers",
         "Retired and semi-retired", "Labourers and operators",
         "Non-earning dependants", "Children"],
        size=n,
        p=[0.35, 0.20, 0.15, 0.10, 0.08, 0.07, 0.03, 0.02],
    )
    return pd.DataFrame({
        "SEXP": np.random.choice(["Male", "Female"], n),
        "BPLP": np.random.choice(
            ["Oceania and Antarctica", "North-West Europe", "South-East Asia"],
            n, p=[0.6, 0.2, 0.2],
        ),
        "profile_name": profiles,
        "profile_id": pd.Categorical(profiles).codes,
        "y_true": np.random.binomial(1, 0.4, n),
    })


@pytest.fixture
def fair_predictions(sample_data):
    """A model with roughly equal accuracy across groups."""
    np.random.seed(42)
    noise = np.random.normal(0, 0.2, len(sample_data))
    sample_data["y_pred"] = (
        (sample_data["y_true"] + noise) > 0.5
    ).astype(int)
    return sample_data


@pytest.fixture
def biased_predictions(sample_data):
    """A model that performs worse for South-East Asian birthplace."""
    np.random.seed(42)
    noise = np.random.normal(0, 0.2, len(sample_data))
    bias = np.where(sample_data["BPLP"] == "South-East Asia", 0.4, 0)
    sample_data["y_pred"] = (
        (sample_data["y_true"] + noise + bias) > 0.5
    ).astype(int)
    return sample_data


class TestFairnessReport:
    def test_fair_model_passes(self, fair_predictions):
        report = FairnessReport(
            fair_predictions, "y_true", "y_pred",
            protected_columns=["SEXP"],
        )
        results = report.run()
        assert results.n_total == 2000
        assert results.overall_accuracy > 0.5
        assert len(results.results) == 1
        assert results.results[0].accuracy_gap < 0.10

    def test_biased_model_detected(self, biased_predictions):
        report = FairnessReport(
            biased_predictions, "y_true", "y_pred",
            protected_columns=["BPLP"],
        )
        results = report.run()
        bplp = results.results[0]
        assert bplp.accuracy_gap > 0.05
        assert bplp.column == "BPLP"

    def test_multiple_protected_columns(self, fair_predictions):
        report = FairnessReport(
            fair_predictions, "y_true", "y_pred",
            protected_columns=["SEXP", "BPLP", "profile_name"],
        )
        results = report.run()
        assert len(results.results) == 3
        columns = {r.column for r in results.results}
        assert columns == {"SEXP", "BPLP", "profile_name"}

    def test_profiles_as_protected(self, fair_predictions):
        report = FairnessReport(
            fair_predictions, "y_true", "y_pred",
            protected_columns=["profile_name"],
            min_group_size=20,
        )
        results = report.run()
        profile_result = results.results[0]
        assert profile_result.column == "profile_name"
        assert len(profile_result.groups) >= 5

    def test_summary_output(self, fair_predictions):
        report = FairnessReport(
            fair_predictions, "y_true", "y_pred",
            protected_columns=["SEXP"],
        )
        results = report.run()
        summary = results.summary()
        assert "Fairness Report" in summary
        assert "accuracy gap" in summary.lower()

    def test_to_dataframe(self, fair_predictions):
        report = FairnessReport(
            fair_predictions, "y_true", "y_pred",
            protected_columns=["SEXP", "BPLP"],
        )
        df = report.run().to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "protected_column" in df.columns
        assert "accuracy" in df.columns
        assert len(df) >= 4

    def test_min_group_size_filter(self, sample_data):
        sample_data["y_pred"] = sample_data["y_true"]
        sample_data["RARE"] = "common"
        sample_data.loc[:4, "RARE"] = "rare"
        report = FairnessReport(
            sample_data, "y_true", "y_pred",
            protected_columns=["RARE"],
            min_group_size=30,
        )
        results = report.run()
        assert len(results.results) == 0

    def test_missing_column_raises(self, sample_data):
        sample_data["y_pred"] = 0
        with pytest.raises(ValueError, match="not found"):
            FairnessReport(
                sample_data, "y_true", "y_pred",
                protected_columns=["NONEXISTENT"],
            )


class TestAssertFair:
    def test_fair_model_passes(self, fair_predictions):
        assert_fair(
            fair_predictions, "y_true", "y_pred",
            protected_columns=["SEXP"],
            max_accuracy_gap=0.10,
        )

    def test_biased_model_fails(self, biased_predictions):
        with pytest.raises(FairnessAssertionError) as exc_info:
            assert_fair(
                biased_predictions, "y_true", "y_pred",
                protected_columns=["BPLP"],
                max_accuracy_gap=0.05,
            )
        assert "failed fairness check" in str(exc_info.value)
        assert hasattr(exc_info.value, "details")
        assert len(exc_info.value.details["failures"]) > 0

    def test_custom_thresholds(self, biased_predictions):
        with pytest.raises(FairnessAssertionError):
            assert_fair(
                biased_predictions, "y_true", "y_pred",
                protected_columns=["BPLP"],
                max_accuracy_gap=0.01,
                max_demographic_parity_gap=0.01,
            )

    def test_relaxed_thresholds_pass(self, biased_predictions):
        assert_fair(
            biased_predictions, "y_true", "y_pred",
            protected_columns=["BPLP"],
            max_accuracy_gap=0.50,
            max_demographic_parity_gap=0.50,
            max_equalised_odds_gap=0.50,
        )
