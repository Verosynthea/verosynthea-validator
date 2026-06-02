# verosynthea-validator

Fairness testing for ML models using real Australian demographic data. One line to check whether your model treats demographic groups equally.

```bash
pip install verosynthea-validator
```

```python
from verosynthea_validator import FairnessReport

report = FairnessReport(
    data=test_data,
    y_true="label",
    y_pred="prediction",
    protected_columns=["SEXP", "BPLP", "profile_name"],
)
results = report.run()
print(results.summary())
```

Output:

```
Fairness Report (n=5,000, overall accuracy=0.847)
============================================================

[PASS] SEXP (2 groups, smallest n=2,451)
  Accuracy gap:           0.012
  Demographic parity gap: 0.008
  Equalised odds gap:     0.015

[FAIL] BPLP (3 groups, smallest n=312)
  Accuracy gap:           0.073
  Demographic parity gap: 0.091
  Equalised odds gap:     0.064

============================================================
Overall: FAIL (worst gap: 0.073 on BPLP)
```

## CI/CD gate

```python
from verosynthea_validator import assert_fair

# Fails the build if any group accuracy gap > 5%
assert_fair(test_data, "label", "prediction", max_accuracy_gap=0.05)
```

In pytest:

```python
def test_model_fairness():
    predictions = model.predict(test_data)
    test_data["y_pred"] = predictions
    assert_fair(
        test_data, "y_true", "y_pred",
        protected_columns=["SEXP", "BPLP", "profile_name"],
        max_accuracy_gap=0.05,
        max_demographic_parity_gap=0.10,
    )
```

## What it measures

For each protected column (e.g. sex, birthplace, demographic profile), the validator computes:

| Metric | What it checks |
|--------|---------------|
| **Accuracy gap** | Max accuracy difference between any two groups |
| **Demographic parity gap** | Max difference in selection rate (P(y_pred=1)) |
| **Equalised odds gap** | Max difference in true positive rate or false positive rate |

Groups smaller than 30 observations are excluded (configurable via `min_group_size`).

## Why this instead of fairlearn or aif360?

Those are general-purpose fairness frameworks. This package is purpose-built for Australian demographics:

- **Pre-loaded demographic data.** The free tier includes 5,000 synthetic individuals from [AUSynth](https://huggingface.co/datasets/vero-synthea/ausynth-sample) with 25 Census-calibrated variables. No need to source your own protected attributes.
- **8 demographic profiles.** AUSynth clusters every person into one of 8 profiles (High-earning professionals, Young singles, Retired, etc.) — a richer protected attribute than just age or sex.
- **Australia-specific calibration.** Variables match ABS Census 2021 categories exactly. Income brackets, occupation codes, education levels, birthplace regions — all in Australian standard classifications.
- **One-line CI gate.** `assert_fair()` drops into pytest with zero configuration.

## Data tiers

| Tier | Data | Cost |
|------|------|------|
| **Free** | 5,000-row Paddington 4064 sample from [Hugging Face](https://huggingface.co/datasets/vero-synthea/ausynth-sample) | $0 |
| **Paid** | Full national dataset (32M individuals, 15,352 suburbs) via API | [verosynthea.com](https://verosynthea.com) |

```python
from verosynthea_validator import load_ausynth_sample

# Free tier (downloads from HF on first call)
df = load_ausynth_sample()

# Paid tier
df = load_ausynth_sample(api_key="vero_...", geography="bondi-2026-nsw")
```

## The 8 demographic profiles

| ID | Name | Typical characteristics |
|----|------|------------------------|
| 0 | Labourers and operators | Blue-collar, lower income |
| 1 | Young singles and non-workers | Under 25, students, NILF |
| 2 | Children | Under 15 |
| 3 | Non-earning dependants | Adults not in workforce |
| 4 | Trades and technical workers | Certificate-qualified, mid income |
| 5 | Established partnered households | Married, mid-career |
| 6 | Retired and semi-retired | Over 60, pension income |
| 7 | High-earning professionals | Degree-qualified, professional occupations |

## Installation

```bash
pip install verosynthea-validator          # core (pandas + numpy)
pip install verosynthea-validator[hf]     # + Hugging Face datasets loader
pip install verosynthea-validator[paid]   # + httpx for API access
pip install verosynthea-validator[dev]    # + pytest + sklearn for development
```

## Links

- **Dataset:** [vero-synthea/ausynth-sample](https://huggingface.co/datasets/vero-synthea/ausynth-sample) on Hugging Face
- **Full product:** [verosynthea.com](https://verosynthea.com)
- **Methodology:** [verosynthea.com/about](https://verosynthea.com/about)

## Citation

```
Verosynthea AUSynth (2026). Synthetic Australian Census Data.
https://verosynthea.com
```

## License

MIT
