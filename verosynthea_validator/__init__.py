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
from .demos import load_us_adult_baseline, load_ausynth_test_set

# Pro tier (paid API). pro.py imports only the stdlib at module level —
# ``requests`` is imported lazily inside the methods that need it — so
# importing the package never forces the optional [pro] dependency on
# free-tier users.
from .pro import (
    ProValidation,
    submit_pro_validation,
    check_api_key,
    show,
    render_report,
    VerosyntheaAPIError,
    InvalidAPIKeyError,
    InsufficientCreditsError,
    ValidationJobError,
)

__version__ = "0.2.1"
__all__ = [
    # free tier
    "FairnessReport",
    "FairnessResults",
    "assert_fair",
    "load_ausynth_sample",
    # pro tier
    "ProValidation",
    "submit_pro_validation",
    "check_api_key",
    "show",
    "render_report",
    "VerosyntheaAPIError",
    "InvalidAPIKeyError",
    "InsufficientCreditsError",
    "ValidationJobError",
]
