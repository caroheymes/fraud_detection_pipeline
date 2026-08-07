# tests/test_drift.py
from src.audit.drift_analysis import run_evidently_drift_check


def test_drift_analysis_execution():
    drift_detected, report = run_evidently_drift_check()
    assert isinstance(drift_detected, bool)
    assert "dataset_drift" in report
    assert "metrics" in report
