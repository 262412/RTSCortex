"""Experiment execution and report generation."""

from rtscortex.evaluation.cortex import (
    CORTEX_EVENT_TYPES,
    CortexObservabilityMetrics,
    compute_cortex_observability,
)
from rtscortex.evaluation.engineering import (
    ENGINEERING_GATES_FILENAME,
    REQUIRED_ENGINEERING_GATES,
    build_engineering_gate_report,
)
from rtscortex.evaluation.metrics import EpisodeMetrics, compute_episode_metrics
from rtscortex.evaluation.report import (
    ReportError,
    RunReportArtifacts,
    render_timeline,
    write_run_reports,
    write_timeline_report,
)
from rtscortex.evaluation.runner import run_mock_episode, run_mock_suite

__all__ = [
    "CORTEX_EVENT_TYPES",
    "CortexObservabilityMetrics",
    "EpisodeMetrics",
    "ENGINEERING_GATES_FILENAME",
    "REQUIRED_ENGINEERING_GATES",
    "ReportError",
    "RunReportArtifacts",
    "compute_cortex_observability",
    "compute_episode_metrics",
    "build_engineering_gate_report",
    "render_timeline",
    "run_mock_episode",
    "run_mock_suite",
    "write_run_reports",
    "write_timeline_report",
]
