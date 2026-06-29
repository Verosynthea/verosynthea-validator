# verosynthea-validator

## Quick Start: Test a US-trained Model on Australian Data

This is a real-world fairness scenario. We take a standard classifier trained on US Census data and test how it performs on Australian demographics.

```bash
pip install verosynthea-validator
```

```python
from verosynthea_validator import FairnessReport
from verosynthea_validator.demos import (
    load_us_adult_baseline,
    load_ausynth_test_set,
)

# Load a US-trained income classifier and Australian test data
model = load_us_adult_baseline()
au_data = load_ausynth_test_set()

# Run fairness audit
report = FairnessReport(
    model=model,
    target_column="income_above_threshold",
    protected_attributes=["SEXP", "BPLP", "AGE5P"],
)
report.run(test_data=au_data)
report.show()
```

### What you'll see

- **Country-of-birth bias gap (~30%):** The US-trained model handles US-typical birth countries well, others poorly
- **Income threshold miscalibration:** $50K USD doesn't map cleanly to Australian income distributions
- **Occupation bias (~18%):** Australian occupation categories don't align with UCI Adult codes

This is the standard fairness-testing scenario for Australian deployments: models trained on US data need to be validated against Australian populations before production use.

### What just happened?

The [UCI Adult Income dataset](https://archive.ics.uci.edu/dataset/2/adult) is the canonical fairness benchmark in ML, but it's US Census data from 1994. When you run a model trained on it against Australian population data, the validator surfaces the bias gaps that come from the distribution mismatch.

For your own models, replace `load_us_adult_baseline()` with your model and `load_ausynth_test_set()` with your test data or an [AUSynth subset](https://huggingface.co/datasets/vero-synthea/ausynth-sample).

---


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
```

## Pro tier: validate against the full national dataset

The free tier scores your predictions against a 5,000-row sample. The **pro
tier** uploads your fitted model and scores it against the full national
synthetic dataset (~32M individuals) server-side, returning a weighted fairness
report. Get an API key at [verosynthea.com/account/api](https://verosynthea.com/account/api).

```python
import os
from verosynthea_validator import ProValidation, show, check_api_key

# Optional: confirm the key + see your credit balance (no charge)
check_api_key(os.environ["VEROSYNTHEA_API_KEY"])

pro = ProValidation(
    model=your_model,                       # any fitted estimator with .predict()
    target_column="income_above_threshold",
    protected_attributes=["sex", "country_of_birth", "age_group"],
    api_key=os.environ["VEROSYNTHEA_API_KEY"],
)

job_id = pro.submit()                       # costs 50 credits; refunded if it fails
report = pro.wait_for_completion(job_id)    # polls until done
pro.show(report)                            # pretty-prints the fairness report
```

One-liner:

```python
from verosynthea_validator import submit_pro_validation, show
report = submit_pro_validation(model, "income_above_threshold",
                               ["sex", "country_of_birth"], wait=True)
show(report)
```

Set `VEROSYNTHEA_API_KEY` to avoid passing `api_key=` everywhere. To target a
non-production deployment, set `VEROSYNTHEA_API_BASE_URL`.

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
pip install verosynthea-validator[pro]    # + requests for the pro-tier API client
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
