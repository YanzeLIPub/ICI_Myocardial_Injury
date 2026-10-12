"""Retrospective workload accounting for one request at the first alert.

Reports and specimen collection times answer different questions. These
calculations estimate additional requests and draws; they do not establish that
unrecorded tests were absent or that an earlier test would have been positive.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Sequence


@dataclass(frozen=True)
class TestEpisode:
    report_time: float
    collection_time: float | None = None

    def __post_init__(self):
        if not isfinite(self.report_time):
            raise ValueError("Report time must be finite")
        if self.collection_time is not None and (
                not isfinite(self.collection_time) or self.collection_time > self.report_time):
            raise ValueError("Collection must be finite and no later than its report")

    @property
    def clinical_time(self):
        return self.report_time if self.collection_time is None else self.collection_time


@dataclass(frozen=True)
class ObservationWindow:
    horizon_end: float
    followup_end: float
    recorded_event: float | None = None

    def __post_init__(self):
        values = (self.horizon_end, self.followup_end)
        if not all(isfinite(x) for x in values):
            raise ValueError("Observation limits must be finite")
        if self.recorded_event is not None and not isfinite(self.recorded_event):
            raise ValueError("Event time must be finite or None")


@dataclass(frozen=True)
class RetestEstimate:
    """Bounds refer to one proposed first-alert request, not clinical outcomes."""

    first_alert: float | None
    request: int = 0
    report_lower: int = 0
    report_proxy: int = 0
    report_upper: int = 0
    sampling_lower: int = 0
    sampling_proxy: int = 0
    sampling_upper: int = 0
    qualifying_report: bool = False
    verified_collection: bool = False
    unknown_collection_proxy: bool = False
    pending_result_after_deadline: bool = False
    short_observation: bool = False
    deadline_exceeds_horizon: bool = False
    excluded_prealert_collections: int = 0
    earlier_check_opportunity: bool = False
    hours_before_next_check: float | None = None


def first_alert_workload(first_alert: float | None, episodes: Sequence[TestEpisode],
                         window: ObservationWindow, *, hours: float = 48.0) -> RetestEstimate:
    """Count first-request workload with collection and report uncertainty.

    Both a verified new collection and an unknown-collection report must be
    strictly after the alert. A delayed report from a pre-alert specimen cannot
    substitute. Report-based counts require a result by the deadline; sampling
    counts also recognize a specimen collected in time whose result is pending.
    Only reports documented by the outcome horizon enter either calculation.
    A recorded event is used only when assessing follow-up coverage, never to
    infer an unobserved laboratory value.
    """
    if not isfinite(hours) or hours <= 0:
        raise ValueError("Retest interval must be finite and positive")
    if first_alert is None:
        return RetestEstimate(None)
    if not isfinite(first_alert) or first_alert >= window.horizon_end:
        raise ValueError("First alert must precede the analysis horizon")
    deadline = first_alert + hours * 3600.0
    available = [t for t in episodes if t.report_time <= window.horizon_end]
    post = [t for t in available if t.report_time > first_alert
            and (t.collection_time is None or t.collection_time > first_alert)]
    eligible = [t for t in post if t.report_time <= deadline]
    confirmed = [t for t in eligible if t.collection_time is not None]
    proxy = [t for t in eligible if t.collection_time is None]
    collected = [t for t in post if t.collection_time is not None and t.collection_time <= deadline]
    sampling_or_proxy = bool(collected or proxy)
    followed = (window.followup_end >= deadline or bool(eligible)
                or (window.recorded_event is not None and window.recorded_event <= deadline))
    next_check = min(post, key=lambda t: t.clinical_time) if post else None
    opportunity = next_check is not None and deadline < next_check.clinical_time
    return RetestEstimate(
        first_alert=first_alert, request=1,
        report_lower=int(not eligible and followed), report_proxy=int(not eligible),
        report_upper=int(not confirmed), sampling_lower=int(not sampling_or_proxy and followed),
        sampling_proxy=int(not sampling_or_proxy), sampling_upper=int(not collected),
        qualifying_report=bool(eligible), verified_collection=bool(collected),
        unknown_collection_proxy=bool(proxy) and not bool(collected),
        pending_result_after_deadline=any(t.report_time > deadline for t in collected),
        short_observation=not followed and not eligible,
        deadline_exceeds_horizon=deadline > window.horizon_end,
        excluded_prealert_collections=sum(t.report_time > first_alert and t.collection_time is not None
                                         and t.collection_time <= first_alert for t in available),
        earlier_check_opportunity=opportunity,
        hours_before_next_check=(next_check.clinical_time - deadline) / 3600.0 if opportunity else None,
    )


def paired_sampling_difference(current: RetestEstimate, comparator: RetestEstimate):
    """Patient-matched uncertainty bounds for alternative first-alert workflows.

    Identical first alerts share the same substitution uncertainty, which
    cancels. Changed windows use cross-bounds; subtracting two marginal lower
    estimates is not a valid lower bound for their difference.
    """
    same = current.first_alert == comparator.first_alert
    if same and current != comparator:
        raise ValueError("Identical alert times must use the same observations and window")
    return {
        "proxy": current.sampling_proxy - comparator.sampling_proxy,
        "lower": 0 if same else current.sampling_lower - comparator.sampling_upper,
        "upper": 0 if same else current.sampling_upper - comparator.sampling_lower,
    }
