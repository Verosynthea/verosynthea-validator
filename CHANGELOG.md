# Changelog

## 0.3.0

- **The free sample is now national.** `load_ausynth_sample()` and
  `demos.load_ausynth_test_set()` read the 5,000-person AUSynth v2.2 sample
  (`ausynth_sample.parquet`). It covers all eight states and territories and
  2,545 suburbs. The earlier sample covered one suburb (Paddington 4064).
- **New cache file name.** The sample is cached as
  `~/.cache/verosynthea/ausynth_sample_v2.2.parquet`, so an older cached
  sample is never read in its place.
- **The UCI Adult mapping reads the v2.2 labels.** Education (`HEAP`), hours
  worked (`HRWRP`) and birthplace (`BPLP`) use new category names in v2.2.
  Every label in the sample now has its own entry. Before, those labels fell
  back to the defaults ("Some-college", 40 hours). `demos.V22_LABELS` lists
  the labels, and a test checks each one is mapped.
- **`demos.UCI_FEATURES`** names the baseline model's input columns, in
  order.
- **Default protected columns.** When `protected_columns` is not given,
  `FairnessReport` and `assert_fair` use whichever of `SEXP`, `BPLP` and
  `profile_name` the data has. The v2.2 sample has no `profile_name` column,
  so the old default raised an error on it.
- **Fixed:** `load_ausynth_test_set(n)` with `n` below 5,000 resets the row
  index after sampling. Before, the report could read the wrong rows.
- **README:** the quick start now uses the package's real API. Figures the
  README did not measure have been removed.
- Build files (`dist/`, `*.egg-info`, `__pycache__`) are no longer kept in
  the repository.

## Earlier versions

- 0.2.1, 0.2.0 and 0.1.0: see the git history.
