"""Minimal checks for the CI-API bucket mapping used by vibe-checks.yaml."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "ci_api_check",
    Path(__file__).resolve().parents[1] / "tools" / "ci_api_check.py",
)
ci_api_check = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ci_api_check)


def test_check_run_buckets():
    bucket_of = ci_api_check.bucket_of
    assert bucket_of({"status": "completed", "conclusion": "success"}) == "pass"
    assert bucket_of({"status": "completed", "conclusion": "neutral"}) == "pass"
    assert bucket_of({"status": "completed", "conclusion": "skipped"}) == "skipping"
    assert bucket_of({"status": "completed", "conclusion": "failure"}) == "fail"
    assert bucket_of({"status": "completed", "conclusion": "cancelled"}) == "cancel"
    assert bucket_of({"status": "in_progress", "conclusion": None}) == "pending"


def test_status_buckets():
    bucket_of = ci_api_check.bucket_of
    assert bucket_of({"state": "success"}) == "pass"
    assert bucket_of({"state": "failure"}) == "fail"
    assert bucket_of({"state": "error"}) == "cancel"
    assert bucket_of({"state": "pending"}) == "pending"
