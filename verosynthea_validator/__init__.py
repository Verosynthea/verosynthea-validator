"""
verosynthea-validator — Fairness testing for ML models using Australian demographic data.

Quick start:
    from verosynthea_validator import FairnessReport, assert_fair

    report = FairnessReport(data, y_true="label", y_pred="prediction",
                            protected_columns=["SEXP", "BPLP"])
    results = report.run()
    print(results.summary())

    # CI/CD gate:
    assert_fair(data, "label", "prediction", max_accuracy_gap=0.05)
"""

from verosynthea_validator.fairness import FairnessReport, FairnessResults
from verosynthea_validator.assertions import assert_fair
from verosynthea_validator.data import load_ausynth_sample

__version__ = "0.1.0"
__all__ = ["FairnessReport", "FairnessResults", "assert_fair", "load_ausynth_sample"]
