"""
Pro tier support for verosynthea-validator.

Provides asynchronous full-national-dataset validation via the
Verosynthea API. The free tier tests against a 5,000-row sample
from one suburb; the pro tier validates against the full national
synthetic dataset (~32M records) for production-ready fairness
audits.

Usage
-----
    import os
    from verosynthea_validator import ProValidation, show

    pro = ProValidation(
        model=your_model,                      # fitted sklearn Pipeline
        target_column="income_above_threshold",
        protected_attributes=["sex", "country_of_birth", "age_group"],
        api_key=os.environ["VEROSYNTHEA_API_KEY"],
    )

    job_id = pro.submit()
    report = pro.wait_for_completion(job_id)
    pro.show(report)                           # or: show(report)

One-liner:
    from verosynthea_validator import submit_pro_validation, show
    report = submit_pro_validation(model, "income_above_threshold",
                                   ["sex", "country_of_birth"], wait=True)
    show(report)

Configuration
-------------
- ``VEROSYNTHEA_API_KEY``        API key (alternative to api_key=).
- ``VEROSYNTHEA_API_BASE_URL``   Override the API base (for staging
                                 / self-hosted). Defaults to production.
- ``VERCEL_PROTECTION_BYPASS``   Sent as x-vercel-protection-bypass so
                                 the client can reach a protected Vercel
                                 preview deployment during testing.

Note: ``requests`` is imported inside methods so it remains an
optional dependency. Free-tier users never need it installed.
"""

import base64
import io
import os
import pickle
import time


# ── Constants ──────────────────────────────────────────────────────

# Base URL is configurable so the same client can target production,
# a Vercel preview, or a self-hosted deployment without code changes.
DEFAULT_API_BASE_URL = "https://verosynthea.com/api/validator"
API_BASE_URL = os.environ.get("VEROSYNTHEA_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")

MAX_MODEL_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
DEFAULT_POLL_INTERVAL = 30  # seconds
DEFAULT_TIMEOUT = 1800  # 30 minutes

KEYS_URL = "https://verosynthea.com/account/api"


# ── Exceptions ─────────────────────────────────────────────────────

class VerosyntheaAPIError(Exception):
    """Base exception for Verosynthea API errors."""
    pass


class InvalidAPIKeyError(VerosyntheaAPIError):
    """Raised when the API key is invalid or revoked (HTTP 401)."""
    pass


class InsufficientCreditsError(VerosyntheaAPIError):
    """Raised when the account has insufficient credits (HTTP 402)."""
    pass


class ValidationJobError(VerosyntheaAPIError):
    """Raised when a validation job fails server-side."""
    pass


# ── Shared helpers ─────────────────────────────────────────────────

def _api_base_url():
    """Resolve the API base URL at call time.

    Reads the env var on each call (not just at import) so a test can
    set VEROSYNTHEA_API_BASE_URL after importing the module.
    """
    return os.environ.get("VEROSYNTHEA_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


def _auth_headers(api_key):
    """Authorization headers, plus the Vercel protection-bypass header
    when VERCEL_PROTECTION_BYPASS is set (needed to reach a protected
    preview deployment)."""
    h = {"Authorization": f"Bearer {api_key}"}
    bypass = os.environ.get("VERCEL_PROTECTION_BYPASS")
    if bypass:
        h["x-vercel-protection-bypass"] = bypass
    return h


def _import_requests():
    """Lazily import ``requests`` with a helpful error if it's missing."""
    try:
        import requests
        return requests
    except ImportError:
        raise ImportError(
            "The 'requests' library is required for pro-tier validation. "
            "Install it with:\n"
            "  pip install requests\n"
            "Or install the pro extras:\n"
            "  pip install verosynthea-validator[pro]"
        )


# ── Pro Validation Class ───────────────────────────────────────────

class ProValidation:
    """
    Client for submitting and tracking pro-tier fairness validation
    jobs against the Verosynthea API.

    Parameters
    ----------
    model : object
        A fitted model (typically an sklearn Pipeline) with a
        ``.predict()`` method. The model is serialised via pickle
        and uploaded to the server for evaluation.
    target_column : str
        Name of the target variable in the test dataset
        (e.g. ``"income_above_threshold"``).
    protected_attributes : list of str
        Demographic columns to audit for fairness gaps
        (e.g. ``["sex", "country_of_birth", "age_group"]``).
    api_key : str, optional
        Verosynthea API key. If not provided, falls back to the
        ``VEROSYNTHEA_API_KEY`` environment variable. Required for
        pro-tier validation.
    schema : str, optional
        Data schema identifier. Defaults to ``"uci_adult_compatible"``.

    Raises
    ------
    ValueError
        If no API key is provided via constructor or environment.

    Examples
    --------
    >>> pro = ProValidation(
    ...     model=my_pipeline,
    ...     target_column="income_above_threshold",
    ...     protected_attributes=["sex", "country_of_birth"],
    ...     api_key="ask_live_abc123...",
    ... )
    >>> job_id = pro.submit()
    >>> report = pro.wait_for_completion(job_id)
    >>> pro.show(report)
    """

    def __init__(
        self,
        model,
        target_column,
        protected_attributes,
        api_key=None,
        schema="uci_adult_compatible",
    ):
        self.model = model
        self.target_column = target_column
        self.protected_attributes = list(protected_attributes)
        self.schema = schema

        # Resolve API key: explicit param > env var
        self.api_key = api_key or os.environ.get("VEROSYNTHEA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Pro tier requires an API key. "
                "Pass api_key= to the constructor or set the "
                "VEROSYNTHEA_API_KEY environment variable.\n"
                f"Get a key at {KEYS_URL}"
            )

    # ── Model serialisation ────────────────────────────────────

    def _serialize_model(self):
        """
        Serialise the model to a base64-encoded pickle string.

        Returns
        -------
        str
            Base64-encoded string of the pickled model.

        Raises
        ------
        ValueError
            If the serialised model exceeds the 50 MB size limit.
        TypeError
            If the model cannot be pickled.
        """
        try:
            buf = io.BytesIO()
            pickle.dump(self.model, buf)
            raw_bytes = buf.getvalue()
        except (pickle.PicklingError, TypeError) as exc:
            raise TypeError(
                f"Could not serialise the model via pickle. "
                f"Ensure your model object is picklable. "
                f"Original error: {exc}"
            ) from exc

        if len(raw_bytes) > MAX_MODEL_SIZE_BYTES:
            size_mb = len(raw_bytes) / (1024 * 1024)
            raise ValueError(
                f"Serialised model is {size_mb:.1f} MB, which exceeds "
                f"the {MAX_MODEL_SIZE_BYTES // (1024 * 1024)} MB limit. "
                f"For large models, consider uploading to HuggingFace "
                f"and using the model_repo_id parameter (coming in v0.4.0)."
            )

        return base64.b64encode(raw_bytes).decode("ascii")

    # ── API helpers ────────────────────────────────────────────

    def _get_headers(self):
        """Return authorization headers (+ Vercel bypass if set)."""
        return _auth_headers(self.api_key)

    def _handle_error_response(self, response):
        """
        Inspect an HTTP response and raise the appropriate
        domain-specific exception for known error codes.

        Raises
        ------
        InvalidAPIKeyError
            On HTTP 401.
        InsufficientCreditsError
            On HTTP 402.
        PermissionError
            On HTTP 403 (job belongs to a different account).
        VerosyntheaAPIError
            On HTTP 400/404 or any other non-2xx status.
        ValueError
            On HTTP 413 (model exceeds server size limit).
        """
        if response.status_code == 401:
            raise InvalidAPIKeyError(
                f"Invalid API key. Check your key at {KEYS_URL}"
            )
        if response.status_code == 402:
            raise InsufficientCreditsError(
                "Insufficient credits. Each validation run costs "
                "50 credits. Top up at https://verosynthea.com/products"
            )
        if response.status_code == 403:
            raise PermissionError(
                "Access denied. This job belongs to a different account."
            )
        if response.status_code == 400:
            detail = ""
            try:
                body = response.json()
                detail = body.get("detail") or body.get("message") or response.text
            except Exception:
                detail = response.text
            raise VerosyntheaAPIError(f"Bad request: {detail}")
        if response.status_code == 404:
            raise VerosyntheaAPIError(
                "Job not found. The job ID may be invalid or expired."
            )
        if response.status_code == 413:
            raise ValueError(
                "Model artifact exceeds the server-side size limit "
                f"({MAX_MODEL_SIZE_BYTES // (1024 * 1024)} MB). "
                "Consider uploading to HuggingFace and using "
                "the model_repo_id parameter (coming in v0.4.0)."
            )
        # Catch-all for any other non-success status
        if not response.ok:
            raise VerosyntheaAPIError(
                f"API error (HTTP {response.status_code}): {response.text}"
            )

    @staticmethod
    def _import_requests():
        return _import_requests()

    # ── Core API methods ───────────────────────────────────────

    def submit(self):
        """
        Submit an asynchronous validation job to the Verosynthea API.

        The model is serialised, uploaded, and queued for evaluation
        against the full national synthetic dataset. Each submission
        costs 50 credits.

        Returns
        -------
        str
            The job ID for tracking the validation run.
        """
        requests = self._import_requests()

        print("Serialising model...")
        model_artifact = self._serialize_model()

        print("Submitting validation job...")
        response = requests.post(
            f"{_api_base_url()}/run",
            headers=self._get_headers(),
            json={
                "model_artifact": model_artifact,
                "target_column": self.target_column,
                "protected_attributes": self.protected_attributes,
                "schema": self.schema,
            },
            timeout=60,
        )

        self._handle_error_response(response)

        result = response.json()
        job_id = result["job_id"]

        remaining = result.get("credits_remaining")
        print(f"Validation submitted. Job ID: {job_id}")
        print(f"Credits charged:  {result.get('credits_charged', 50)}"
              + (f"  (remaining: {remaining})" if remaining is not None else ""))
        print(f"Sample:           full national synthetic dataset")
        print(f"Check status:     pro.check_status('{job_id}')")
        print(f"Wait & retrieve:  pro.wait_for_completion('{job_id}')")

        return job_id

    def check_status(self, job_id):
        """
        Check the status of a validation job.

        Returns
        -------
        dict
            Status dictionary with at minimum a ``'state'`` key
            (``'queued'`` | ``'running'`` | ``'complete'`` |
            ``'failed'``). When complete, also contains ``'report'``;
            when failed, contains ``'error'``.
        """
        requests = self._import_requests()

        response = requests.get(
            f"{_api_base_url()}/status/{job_id}",
            headers=self._get_headers(),
            timeout=30,
        )

        self._handle_error_response(response)
        return response.json()

    def wait_for_completion(
        self,
        job_id,
        poll_interval=DEFAULT_POLL_INTERVAL,
        timeout=DEFAULT_TIMEOUT,
    ):
        """
        Poll until the validation job completes or times out.

        Returns
        -------
        dict
            The fairness ``report`` dict from the completed job.

        Raises
        ------
        ValidationJobError
            If the job fails server-side.
        TimeoutError
            If the job does not complete within ``timeout`` seconds.
        """
        elapsed = 0
        last_state = None

        while elapsed < timeout:
            status = self.check_status(job_id)
            state = status.get("state", "unknown")
            pct = status.get("progress_pct")

            if state != last_state:
                suffix = f" ({pct}%)" if pct is not None else ""
                print(f"[{elapsed}s] Job {job_id}: {state}{suffix}")
                last_state = state

            if state == "complete":
                print(f"Validation complete in {elapsed}s.")
                return status["report"]

            if state == "failed":
                error_detail = status.get("error", "Unknown error")
                raise ValidationJobError(
                    f"Validation job failed: {error_detail}"
                )

            time.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(
            f"Validation job {job_id} did not complete within "
            f"{timeout} seconds. You can continue polling with "
            f"pro.check_status('{job_id}')"
        )

    # ── Rendering ──────────────────────────────────────────────

    def show(self, report):
        """Pretty-print a completed fairness ``report`` dict to stdout."""
        show(report)


# ── Report rendering ───────────────────────────────────────────────

def render_report(report):
    """
    Render a pro fairness ``report`` dict as a human-readable string.

    Mirrors the free-tier ``FairnessResults.summary()`` style so output
    feels consistent across tiers. Accepts the report shape emitted by
    the worker: ``summary``, ``fairness_metrics``, ``max_gaps``,
    ``flagged_groups``.
    """
    if not isinstance(report, dict):
        raise TypeError(
            "render_report expects the report dict returned by "
            "wait_for_completion(). Got: " + type(report).__name__
        )

    s = report.get("summary", {})
    width = 64
    lines = ["Verosynthea Pro Fairness Report", "=" * width]

    lines.append(f"Sample: {report.get('sample_size', 'national')}")
    if s.get("total_records") is not None:
        lines.append(f"Records tested:          {s['total_records']:,}")
    if s.get("model_accuracy") is not None:
        lines.append(f"Model accuracy:          {s['model_accuracy']:.3f}")
    if s.get("overall_positive_rate") is not None:
        lines.append(f"Predicted positive rate: {s['overall_positive_rate']:.3f}")
    if s.get("baseline_positive_rate") is not None:
        lines.append(f"Actual positive rate:    {s['baseline_positive_rate']:.3f}")

    fm = report.get("fairness_metrics", {})
    for attr, m in fm.items():
        status = "FAIL" if m.get("flagged") else "PASS"
        lines.append("")
        lines.append(
            f"[{status}] {attr}  (max gap {m.get('max_gap', 0):.3f}, "
            f"parity ratio {m.get('demographic_parity_ratio', 0):.3f})"
        )
        groups = m.get("groups", {})
        for label, g in sorted(
            groups.items(), key=lambda kv: -kv[1].get("positive_rate", 0)
        ):
            rate = g.get("positive_rate", 0)
            cnt = g.get("count", 0)
            lines.append(f"    {label:<36s} rate={rate:.3f}  n={cnt:,}")

    fg = report.get("flagged_groups", [])
    lines.append("")
    lines.append("-" * width)
    if fg:
        lines.append(f"Flagged groups ({len(fg)}), worst gap first:")
        for f in fg:
            sev = str(f.get("severity", "")).upper()
            lines.append(
                f"  [{sev:<6s}] {f.get('attribute')} = {f.get('group')}  "
                f"rate={f.get('positive_rate', 0):.3f}  "
                f"gap={f.get('gap_from_highest', 0):.3f}"
            )
    else:
        lines.append("No groups flagged (no demographic parity gap >= 0.05).")

    max_gaps = report.get("max_gaps", {})
    if max_gaps:
        worst_attr = max(max_gaps, key=max_gaps.get)
        worst = max_gaps[worst_attr]
        lines.append("")
        verdict = "PASS" if worst < 0.05 else "FAIL"
        lines.append(
            f"Overall: {verdict} (worst gap {worst:.3f} on {worst_attr})"
        )

    return "\n".join(lines)


def show(report):
    """Print a rendered fairness report to stdout."""
    print(render_report(report))


# ── Convenience functions ──────────────────────────────────────────

def submit_pro_validation(
    model,
    target_column,
    protected_attributes,
    api_key=None,
    schema="uci_adult_compatible",
    wait=False,
    poll_interval=DEFAULT_POLL_INTERVAL,
    timeout=DEFAULT_TIMEOUT,
):
    """
    Submit a pro-tier validation in one call.

    Returns
    -------
    str or dict
        Job ID (if ``wait=False``) or the fairness report dict
        (if ``wait=True``).

    Examples
    --------
    >>> job_id = submit_pro_validation(model, "target", ["sex"])
    >>> report = submit_pro_validation(model, "target", ["sex"], wait=True)
    """
    pro = ProValidation(
        model=model,
        target_column=target_column,
        protected_attributes=protected_attributes,
        api_key=api_key,
        schema=schema,
    )

    job_id = pro.submit()

    if wait:
        return pro.wait_for_completion(
            job_id,
            poll_interval=poll_interval,
            timeout=timeout,
        )

    return job_id


def check_api_key(api_key=None):
    """
    Verify that an API key is valid without submitting a job.

    Returns
    -------
    dict
        Account info including remaining credits.

    Raises
    ------
    InvalidAPIKeyError
        If the key is invalid.
    ValueError
        If no key is provided.
    """
    key = api_key or os.environ.get("VEROSYNTHEA_API_KEY")
    if not key:
        raise ValueError(
            "No API key provided. Pass api_key= or set the "
            "VEROSYNTHEA_API_KEY environment variable.\n"
            f"Get a key at {KEYS_URL}"
        )

    requests = _import_requests()

    response = requests.get(
        f"{_api_base_url()}/account",
        headers=_auth_headers(key),
        timeout=15,
    )

    if response.status_code == 401:
        raise InvalidAPIKeyError(
            f"Invalid API key. Check your key at {KEYS_URL}"
        )
    if not response.ok:
        raise VerosyntheaAPIError(
            f"API error (HTTP {response.status_code}): {response.text}"
        )

    account = response.json()
    print(f"API key valid. Credits remaining: {account.get('credits', '?')}")
    return account
