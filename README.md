# verosynthea-validator

Fairness testing for machine learning models, using synthetic Australian
Census data.

```bash
pip install verosynthea-validator
```

## Quick start: a US-trained model on Australian data

A common fairness question: a classifier trained on US data is about to be
used in Australia. Does it treat Australians of different sexes, birthplaces
and ages alike?

```python
from verosynthea_validator import FairnessReport
from verosynthea_validator.demos import (
    UCI_FEATURES,
    load_ausynth_test_set,
    load_us_adult_baseline,
)

model = load_us_adult_baseline()      # a classifier trained on UCI Adult (US)
au = load_ausynth_test_set()          # the free AUSynth sample, in UCI Adult columns

au["prediction"] = model.predict(au[UCI_FEATURES])

report = FairnessReport(
    au,
    y_true="income_above_threshold",
    y_pred="prediction",
    protected_columns=["SEXP", "BPLP", "AGE5P"],
)
print(report.run().summary())
```

The [UCI Adult Income dataset](https://archive.ics.uci.edu/dataset/2/adult)
is a standard fairness benchmark, built from the 1994 US Census.
`load_ausynth_test_set()` maps each Australian person onto its columns
(age, education, occupation, hours worked, country of birth and so on), so a
model trained on it can score Australians. The report then compares the
model's accuracy and selection rate across Australian groups.

For your own model, replace the baseline with your model and the test set
with your own data, or with the free sample below.

## The free sample

`load_ausynth_sample()` downloads the free sample from
[Hugging Face](https://huggingface.co/datasets/vero-synthea/ausynth-sample)
on first use and caches it in `~/.cache/verosynthea/`.

```python
from verosynthea_validator import load_ausynth_sample

df = load_ausynth_sample()
```

- **5,000 synthetic people from the AUSynth national population (v2.2).** They
  come from all eight states and territories and 2,545 suburbs, drawn in
  proportion to how many people live in each suburb.
- **25 columns:** `suburb_id`, `state` and 23 ABS Census 2021 variables, as
  labels: `AGE5P`, `SEXP`, `BPLP`, `CITP`, `INGP`, `ENGLP`, `LANP`, `YARP`,
  `MSTP`, `RLHP`, `LFSP`, `HRWRP`, `OCCP`, `INDP`, `INCP`, `HEAP`, `QALLP`,
  `STUP`, `ASSNP`, `HLTHP`, `LTHP`, `MTWP`, `TISRP`.
- **Synthetic.** No row describes a real person.
- **Licence:** derived from Australian Bureau of Statistics, Census of
  Population and Housing 2021, used under CC BY 4.0.

## CI/CD gate

```python
from verosynthea_validator import assert_fair

# Fails if any group's accuracy differs from another's by more than 5 points
assert_fair(test_data, "label", "prediction", max_accuracy_gap=0.05)
```

In pytest:

```python
def test_model_fairness():
    test_data["y_pred"] = model.predict(test_data[FEATURES])
    assert_fair(
        test_data, "y_true", "y_pred",
        protected_columns=["SEXP", "BPLP", "AGE5P"],
        max_accuracy_gap=0.05,
        max_demographic_parity_gap=0.10,
    )
```

When `protected_columns` is not given, the validator uses whichever of `SEXP`,
`BPLP` and `profile_name` your data has.

## What it measures

For each protected column (for example sex or birthplace), the validator
computes:

| Metric | What it checks |
|--------|---------------|
| **Accuracy gap** | The largest accuracy difference between any two groups |
| **Demographic parity gap** | The largest difference in selection rate, P(y_pred = 1) |
| **Equalised odds gap** | The largest difference in true positive rate or false positive rate |

Groups with fewer than 30 people are left out. Change this with
`min_group_size`.

## Why this instead of fairlearn or aif360?

Those are general-purpose fairness frameworks. This package is built for
Australian populations:

- **Australian protected attributes, ready to use.** The free sample carries
  sex, age, birthplace, citizenship, Indigenous status, language and English
  proficiency, so you do not need to source them yourself.
- **ABS Census 2021 categories.** Income bands, occupations, education levels
  and birthplace regions follow the Australian standard classifications.
- **A one-line CI gate.** `assert_fair()` drops into pytest with no setup.

## Data tiers

| Tier | Data |
|------|------|
| **Free** | The 5,000-person national sample on [Hugging Face](https://huggingface.co/datasets/vero-synthea/ausynth-sample) |
| **Pro** | The full national population (v2.2) through the API: 27,664,147 synthetic person records across 15,342 suburbs. See [verosynthea.com](https://verosynthea.com) |

## Pro tier: validate against the full national population

The free tier scores your predictions against the 5,000-person sample. The
**pro tier** uploads your fitted model and scores it against the full
national synthetic population on our servers, then returns a weighted
fairness report. Get an API key at
[verosynthea.com/account/api](https://verosynthea.com/account/api).

```python
import os
from verosynthea_validator import ProValidation, show, check_api_key

# Optional: confirm the key and see your credit balance (no charge)
check_api_key(os.environ["VEROSYNTHEA_API_KEY"])

pro = ProValidation(
    model=your_model,                       # any fitted estimator with .predict()
    target_column="income_above_threshold",
    protected_attributes=["sex", "country_of_birth", "age_group"],
    api_key=os.environ["VEROSYNTHEA_API_KEY"],
)

job_id = pro.submit()                       # costs 50 credits; refunded if it fails
report = pro.wait_for_completion(job_id)    # polls until done
pro.show(report)                            # prints the fairness report
```

One call:

```python
from verosynthea_validator import submit_pro_validation, show
report = submit_pro_validation(model, "income_above_threshold",
                               ["sex", "country_of_birth"], wait=True)
show(report)
```

Set `VEROSYNTHEA_API_KEY` to avoid passing `api_key=` every time. To target a
different deployment, set `VEROSYNTHEA_API_BASE_URL`.

## Installation

```bash
pip install verosynthea-validator          # core, including the Hugging Face loader
pip install verosynthea-validator[pro]    # adds requests, for the pro-tier API client
pip install verosynthea-validator[dev]    # adds pytest and scikit-learn, for development
```

## Changes

See [CHANGELOG.md](CHANGELOG.md).

## Links

- **Dataset:** [vero-synthea/ausynth-sample](https://huggingface.co/datasets/vero-synthea/ausynth-sample) on Hugging Face
- **Full product:** [verosynthea.com](https://verosynthea.com)
- **Methodology:** [verosynthea.com/about](https://verosynthea.com/about)

## Citation

```
Verosynthea AUSynth (2026). Synthetic Australian Census Data.
https://verosynthea.com
```

## Licence

MIT. The AUSynth sample is CC BY 4.0, derived from the ABS Census of
Population and Housing 2021.
