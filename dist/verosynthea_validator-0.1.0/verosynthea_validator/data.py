"""
Data loading utilities for verosynthea-validator.

Free tier: loads the 5,000-row AUSynth sample from Hugging Face.
Paid tier: connects to verosynthea.com API for full national data.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd


_HF_DATASET = "vero-synthea/ausynth-sample"
_HF_FILE = "ausynth_sample_paddington_4064.parquet"
_CACHE_DIR = Path.home() / ".cache" / "verosynthea"


def load_ausynth_sample(
    api_key: Optional[str] = None,
    geography: Optional[str] = None,
) -> pd.DataFrame:
    """Load AUSynth demographic data for fairness testing.

    Parameters
    ----------
    api_key : str, optional
        Verosynthea API key for the full national dataset. If None, loads
        the free 5,000-row Paddington sample from Hugging Face.
    geography : str, optional
        Suburb slug (e.g. "paddington-4064-qld") for paid-tier queries.
        Ignored when using the free sample.

    Returns
    -------
    pd.DataFrame
        Person-level demographic data with 25+ variables including
        profile_id and profile_name.
    """
    if api_key:
        return _load_paid(api_key, geography)
    return _load_free_sample()


def _load_free_sample() -> pd.DataFrame:
    """Load the free HF sample. Downloads on first call, caches locally."""
    cache_path = _CACHE_DIR / _HF_FILE
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    # Try loading from HF datasets library
    try:
        from datasets import load_dataset
        ds = load_dataset(_HF_DATASET, split="train")
        df = ds.to_pandas()
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        return df
    except ImportError:
        pass

    # Fallback: direct parquet download
    try:
        url = f"https://huggingface.co/datasets/{_HF_DATASET}/resolve/main/{_HF_FILE}"
        df = pd.read_parquet(url)
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        return df
    except Exception as e:
        raise RuntimeError(
            f"Could not load AUSynth sample. Install the `datasets` library "
            f"(`pip install datasets`) or check your internet connection. "
            f"Original error: {e}"
        ) from e


def _load_paid(api_key: str, geography: Optional[str] = None) -> pd.DataFrame:
    """Load data from the Verosynthea API (paid tier)."""
    import httpx

    base = os.environ.get("VEROSYNTHEA_API_URL", "https://api.verosynthea.com/v1")
    params = {}
    if geography:
        params["geography"] = geography

    resp = httpx.get(
        f"{base}/data/persons",
        headers={"Authorization": f"Bearer {api_key}"},
        params=params,
        timeout=60.0,
    )
    resp.raise_for_status()
    import io
    return pd.read_parquet(io.BytesIO(resp.content))
