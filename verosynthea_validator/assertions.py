"""
CI/CD assertion helpers for fairness gating.

Usage in pytest or CI:
    from verosynthea_validator import assert_fair
    assert_fair(test_data, "y_true", "y_pred", max_accuracy_gap=0.05)
"""
from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from verosynthea_validator.fairness import FairnessReport


class FairnessAssertionError(AssertionError):
    """Raised when a model fails a fairness check."""

    def __init__(self, message: str, details: dict):
        super().__init__(message)
        self.details = details


def assert_fair(
    data: pd.DataFrame,
    y_true: str,
    y_pred: str,
    protected_columns: Optional[Sequence[str]] = None,
    max_accuracy_gap: float = 0.05,
    max_demographic_parity_gap: float = 0.10,
    max_equalised_odds_gap: float = 0.10,
    min_group_size: int = 30,
) -> None:
    """Assert that a model's predictions are fair across demographic groups.

    Raises FairnessAssertionError if any threshold is exceeded. Designed
    for use in pytest tests and CI/CD pipelines.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with predictions and AUSynth demographic columns.
    y_true : str
        Ground-truth binary label column.
    y_pred : str
        Predicted binary label column.
    protected_columns : list[str], optional
        Columns to check. Default: ["SEXP", "BPLP", "profile_name"].
    max_accuracy_gap : float
        Maximum allowed accuracy difference between any two groups (default 0.05).
    max_demographic_parity_gap : float
        Maximum allowed selection rate difference (default 0.10).
    max_equalised_odds_gap : float
        Maximum allowed TPR or FPR difference (default 0.10).
    min_group_size : int
        Minimum group size to include in analysis (default 30).

    Raises
    ------
    FairnessAssertionError
        If any fairness threshold is exceeded. The error's `.details` dict
        contains the full results for debugging.
    """
    report = FairnessReport(
        data=data,
        y_true=y_true,
        y_pred=y_pred,
        protected_columns=protected_columns,
        min_group_size=min_group_size,
    )
    results = report.run()

    failures = []
    for r in results.results:
        if r.accuracy_gap > max_accuracy_gap:
            failures.append(
                f"{r.column}: accuracy gap {r.accuracy_gap:.3f} > {max_accuracy_gap}"
            )
        if r.demographic_parity_gap > max_demographic_parity_gap:
            failures.append(
                f"{r.column}: demographic parity gap {r.demographic_parity_gap:.3f} "
                f"> {max_demographic_parity_gap}"
            )
        if r.equalised_odds_gap > max_equalised_odds_gap:
            failures.append(
                f"{r.column}: equalised odds gap {r.equalised_odds_gap:.3f} "
                f"> {max_equalised_odds_gap}"
            )

    if failures:
        msg = (
            f"Model failed fairness check on {len(failures)} metric(s):\n"
            + "\n".join(f"  - {f}" for f in failures)
            + "\n\nFull report:\n" + results.summary()
        )
        raise FairnessAssertionError(msg, {
            "failures": failures,
            "results": [r.to_dict() for r in results.results],
        })
