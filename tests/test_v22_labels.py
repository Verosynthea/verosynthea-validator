"""The v2.2 free sample: its labels map explicitly, and the loader reads it.

These tests need no network: the loader is pointed at a local copy.
"""
import pandas as pd
import pytest

from verosynthea_validator import data as data_module
from verosynthea_validator import demos
from verosynthea_validator.demos import V22_LABELS, map_ausynth_to_uci_adult

MAPS = {
    "HEAP": demos._EDUCATION_MAP,
    "HRWRP": demos._HOURS_MAP,
    "MSTP": demos._MARITAL_MAP,
    "OCCP": demos._OCCUPATION_MAP,
    "BPLP": demos._COUNTRY_MAP,
    "SEXP": demos._SEX_MAP,
    "AGE5P": demos._AGE_MAP,
}


@pytest.mark.parametrize("column", sorted(MAPS))
def test_every_v22_label_has_an_explicit_entry(column):
    missing = [label for label in V22_LABELS[column] if label not in MAPS[column]]
    assert missing == []


def test_every_v22_income_band_has_a_midpoint():
    target = demos._map_income_target(pd.Series(V22_LABELS["INCP"]))
    assert len(target) == len(V22_LABELS["INCP"])
    # The top bands are above the US$50,000 line, the bottom ones below it.
    by_label = dict(zip(V22_LABELS["INCP"], target))
    assert by_label["$3,500 or more ($182,000 or more)"] == 1
    assert by_label["$150-$299 ($7,800-$15,599)"] == 0


def _v22_frame(n=60):
    rows = []
    for i in range(n):
        rows.append({col: labels[i % len(labels)] for col, labels in V22_LABELS.items()})
    return pd.DataFrame(rows)


def test_v22_records_do_not_fall_back_to_defaults():
    uci = map_ausynth_to_uci_adult(_v22_frame())
    degrees = _v22_frame()["HEAP"].isin(["Bachelor Degree Level", "Postgraduate Degree Level"])
    assert set(uci.loc[degrees, "education"]) <= {"Bachelors", "Doctorate"}
    hours = _v22_frame()["HRWRP"] == "70 hours or more"
    assert (uci.loc[hours, "hours-per-week"] == 75).all()


def test_the_loader_reads_the_v22_cache(tmp_path, monkeypatch):
    sample = _v22_frame(10)
    monkeypatch.setattr(data_module, "_CACHE_DIR", tmp_path)
    sample.to_parquet(tmp_path / data_module._CACHE_FILE, index=False)
    got = data_module.load_ausynth_sample()
    assert got.equals(sample)
    assert data_module._HF_FILE == "ausynth_sample.parquet"
    assert data_module.SAMPLE_VERSION in data_module._CACHE_FILE


def test_an_old_paddington_cache_is_not_read(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, "_CACHE_DIR", tmp_path)
    pd.DataFrame({"old": [1]}).to_parquet(tmp_path / "ausynth_sample_paddington_4064.parquet")
    assert not (tmp_path / data_module._CACHE_FILE).exists()
    assert data_module._CACHE_FILE != "ausynth_sample_paddington_4064.parquet"


def test_default_protected_columns_are_the_ones_present():
    from verosynthea_validator import FairnessReport

    df = _v22_frame(200)
    df["y_true"] = [i % 2 for i in range(len(df))]
    df["y_pred"] = df["y_true"]
    report = FairnessReport(df, "y_true", "y_pred", min_group_size=5)
    assert report.protected_columns == ["SEXP", "BPLP"]
    assert {r.column for r in report.run().results} == {"SEXP", "BPLP"}
    with pytest.raises(ValueError, match="No protected columns"):
        FairnessReport(df[["y_true", "y_pred"]], "y_true", "y_pred")
