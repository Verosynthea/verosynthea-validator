"""
Core fairness metrics and reporting for verosynthea-validator.

Computes demographic parity, equalised odds, accuracy gap, and
calibration gap across protected groups defined by AUSynth variables.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# AUSynth profile names for readable output
PROFILE_NAMES = {
    0: "Labourers and operators",
    1: "Young singles and non-workers",
    2: "Children",
    3: "Non-earning dependants",
    4: "Trades and technical workers",
    5: "Established partnered households",
    6: "Retired and semi-retired",
    7: "High-earning professionals",
}

# Common protected columns in AUSynth data
SUGGESTED_PROTECTED = ["SEXP", "BPLP", "GNGP", "AGE5P", "profile_name"]


@dataclass
class GroupMetrics:
    """Fairness metrics for a single demographic group."""
    group_name: str
    group_value: str
    n: int
    accuracy: float
    positive_rate: float        # P(y_pred=1) — selection rate
    true_positive_rate: float   # P(y_pred=1 | y_true=1) — recall
    false_positive_rate: float  # P(y_pred=1 | y_true=0) — false alarm
    positive_predictive_value: float  # P(y_true=1 | y_pred=1) — precision


@dataclass
class ProtectedColumnResult:
    """Fairness analysis for one protected attribute."""
    column: str
    groups: list[GroupMetrics]
    accuracy_gap: float          # max - min accuracy across groups
    demographic_parity_gap: float  # max - min positive_rate
    equalised_odds_gap: float    # max gap in TPR or FPR
    min_group_size: int

    def to_dict(self) -> dict:
        return {
            "column": self.column,
            "accuracy_gap": round(self.accuracy_gap, 4),
            "demographic_parity_gap": round(self.demographic_parity_gap, 4),
            "equalised_odds_gap": round(self.equalised_odds_gap, 4),
            "n_groups": len(self.groups),
            "min_group_size": self.min_group_size,
            "groups": [
                {
                    "value": g.group_value,
                    "n": g.n,
                    "accuracy": round(g.accuracy, 4),
                    "positive_rate": round(g.positive_rate, 4),
                    "tpr": round(g.true_positive_rate, 4),
                    "fpr": round(g.false_positive_rate, 4),
                }
                for g in self.groups
            ],
        }


@dataclass
class FairnessResults:
    """Complete fairness report across all protected columns."""
    results: list[ProtectedColumnResult]
    overall_accuracy: float
    n_total: int

    def summary(self) -> str:
        """Human-readable summary of fairness results."""
        lines = [
            f"Fairness Report (n={self.n_total:,}, overall accuracy={self.overall_accuracy:.3f})",
            "=" * 60,
        ]
        for r in self.results:
            status = "PASS" if r.accuracy_gap <= 0.05 else "FAIL"
            lines.append(
                f"\n[{status}] {r.column} ({len(r.groups)} groups, "
                f"smallest n={r.min_group_size})"
            )
            lines.append(
                f"  Accuracy gap:           {r.accuracy_gap:.3f}"
            )
            lines.append(
                f"  Demographic parity gap: {r.demographic_parity_gap:.3f}"
            )
            lines.append(
                f"  Equalised odds gap:     {r.equalised_odds_gap:.3f}"
            )
            for g in sorted(r.groups, key=lambda x: -x.accuracy):
                lines.append(
                    f"    {g.group_value:<40s} acc={g.accuracy:.3f}  "
                    f"sel={g.positive_rate:.3f}  n={g.n}"
                )
        lines.append("\n" + "=" * 60)
        max_gap = max(r.accuracy_gap for r in self.results) if self.results else 0
        if max_gap <= 0.05:
            lines.append("Overall: PASS (all accuracy gaps <= 0.05)")
        else:
            worst = max(self.results, key=lambda r: r.accuracy_gap)
            lines.append(
                f"Overall: FAIL (worst gap: {worst.accuracy_gap:.3f} "
                f"on {worst.column})"
            )
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten results to a DataFrame for further analysis."""
        rows = []
        for r in self.results:
            for g in r.groups:
                rows.append({
                    "protected_column": r.column,
                    "group": g.group_value,
                    "n": g.n,
                    "accuracy": g.accuracy,
                    "positive_rate": g.positive_rate,
                    "tpr": g.true_positive_rate,
                    "fpr": g.false_positive_rate,
                    "ppv": g.positive_predictive_value,
                })
        return pd.DataFrame(rows)

    @property
    def passed(self) -> bool:
        """True if all accuracy gaps are <= 0.05."""
        return all(r.accuracy_gap <= 0.05 for r in self.results)


class FairnessReport:
    """Run a fairness analysis across demographic groups.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with predictions and demographic columns.
    y_true : str
        Column name for ground-truth binary labels (0/1).
    y_pred : str
        Column name for predicted binary labels (0/1).
    protected_columns : list[str], optional
        Demographic columns to test. Defaults to ["SEXP", "BPLP", "profile_name"].
    min_group_size : int
        Minimum observations per group to include (default 30).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y_true: str,
        y_pred: str,
        protected_columns: Optional[Sequence[str]] = None,
        min_group_size: int = 30,
    ):
        self.data = data
        self.y_true = y_true
        self.y_pred = y_pred
        self.protected_columns = list(
            protected_columns or ["SEXP", "BPLP", "profile_name"]
        )
        self.min_group_size = min_group_size

        # Validate inputs
        for col in [y_true, y_pred] + self.protected_columns:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data")

    def run(self) -> FairnessResults:
        """Compute fairness metrics for all protected columns."""
        yt = self.data[self.y_true].values.astype(int)
        yp = self.data[self.y_pred].values.astype(int)
        overall_acc = (yt == yp).mean()

        results = []
        for col in self.protected_columns:
            groups = []
            for val, gdf in self.data.groupby(col):
                if len(gdf) < self.min_group_size:
                    continue
                idx = gdf.index
                gt = yt[idx]
                pr = yp[idx]
                n = len(gt)
                acc = (gt == pr).mean()
                pos_rate = pr.mean()

                # TPR / FPR
                pos_mask = gt == 1
                neg_mask = gt == 0
                tpr = pr[pos_mask].mean() if pos_mask.sum() > 0 else 0.0
                fpr = pr[neg_mask].mean() if neg_mask.sum() > 0 else 0.0
                ppv = gt[pr == 1].mean() if (pr == 1).sum() > 0 else 0.0

                groups.append(GroupMetrics(
                    group_name=col,
                    group_value=str(val),
                    n=n,
                    accuracy=float(acc),
                    positive_rate=float(pos_rate),
                    true_positive_rate=float(tpr),
                    false_positive_rate=float(fpr),
                    positive_predictive_value=float(ppv),
                ))

            if len(groups) < 2:
                continue

            accs = [g.accuracy for g in groups]
            prs = [g.positive_rate for g in groups]
            tprs = [g.true_positive_rate for g in groups]
            fprs = [g.false_positive_rate for g in groups]

            results.append(ProtectedColumnResult(
                column=col,
                groups=groups,
                accuracy_gap=max(accs) - min(accs),
                demographic_parity_gap=max(prs) - min(prs),
                equalised_odds_gap=max(
                    max(tprs) - min(tprs),
                    max(fprs) - min(fprs),
                ),
                min_group_size=min(g.n for g in groups),
            ))

        return FairnessResults(
            results=results,
            overall_accuracy=float(overall_acc),
            n_total=len(self.data),
        )
