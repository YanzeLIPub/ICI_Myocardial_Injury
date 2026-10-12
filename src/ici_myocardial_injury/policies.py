"""Whole-score alert policies, outcome accounting and capacity matching.

The first threshold crossing counts once per subject. Unknown outcome is a
distinct state. No study-specific operating threshold is distributed.
"""

from dataclasses import dataclass
from math import inf, isfinite, isnan, nextafter
from typing import Callable, Mapping, Sequence

from .timing import DAY, PatientWindow
from .workload import RetestEstimate


@dataclass(frozen=True)
class Assessment:
    subject: str
    time: float
    score: float | None
    latest_source_time: float


def validate_assessments(assessments: Sequence[Assessment], windows: Mapping[str, PatientWindow]):
    """Reject duplicate rows, out-of-window predictions and future information."""
    seen = set()
    for row in assessments:
        key = (row.subject, row.time)
        if key in seen:
            raise ValueError("Duplicate subject-assessment rows")
        seen.add(key)
        if row.subject not in windows or not windows[row.subject].eligible(row.time):
            raise ValueError("Every score must precede its strict stopping boundaries")
        if not isfinite(row.latest_source_time) or row.latest_source_time > row.time:
            raise ValueError("A feature report is unavailable at assessment time")


def first_alerts(assessments: Sequence[Assessment], windows: Mapping[str, PatientWindow],
                 threshold: float) -> dict[str, float | None]:
    """Alert iff score > threshold; all equal scores enter or leave together."""
    if isnan(threshold):
        raise ValueError("Threshold cannot be NaN")
    validate_assessments(assessments, windows)
    first = dict.fromkeys(windows, None)
    for row in assessments:
        if row.score is not None and isfinite(row.score) and row.score > threshold:
            old = first[row.subject]
            if old is None or row.time < old:
                first[row.subject] = row.time
    return first


def strict_thresholds(assessments: Sequence[Assessment]) -> list[float]:
    """Enumerate every distinct operating state, including no/all finite scores.

    Use all assessment scores, not just subject maxima: a lower threshold can
    move a first alert earlier without changing the number of alerted subjects.
    """
    values = sorted({r.score for r in assessments if r.score is not None and isfinite(r.score)}, reverse=True)
    return [inf, *values, nextafter(values[-1], -inf)] if values else [inf]


def outcome_metrics(alerts: Mapping[str, float | None], outcomes: Mapping[str, str],
                    *, stopping_times: Mapping[str, float | None] | None = None):
    """Observed reference alerts and unresolved alerts use separate denominators."""
    if set(alerts) != set(outcomes) or not alerts:
        raise ValueError("Outcome and alert mappings must cover the same nonempty cohort")
    if any(y not in {"event", "reference", "unresolved"} for y in outcomes.values()):
        raise ValueError("Outcome must be event, reference or unresolved")
    n = len(alerts)
    counts = {label: sum(y == label for y in outcomes.values())
              for label in ("event", "reference", "unresolved")}
    hit = {label: sum(t is not None and outcomes[k] == label for k, t in alerts.items()) for label in counts}
    total = sum(t is not None for t in alerts.values())
    ge14 = 0
    if stopping_times is not None:
        if set(stopping_times) != set(alerts):
            raise ValueError("Stopping times must cover the complete cohort")
        for key, time in alerts.items():
            if time is not None and outcomes[key] == "event":
                stop = stopping_times[key]
                if stop is None or not isfinite(stop) or time >= stop:
                    raise ValueError("Captured events require a strictly later stopping boundary")
                ge14 += (stop - time) / DAY >= 14
    return {
        "subjects": n, "events": counts["event"], "references": counts["reference"],
        "unresolved": counts["unresolved"], "alerts": total, "captured": hit["event"],
        "reference_alerts": hit["reference"], "unresolved_alerts": hit["unresolved"],
        "capture": hit["event"] / counts["event"] if counts["event"] else None,
        "reference_alerts_per_all": hit["reference"] / n,
        "reference_false_positive_rate": hit["reference"] / counts["reference"] if counts["reference"] else None,
        "false_alerts_per_all_upper": (hit["reference"] + hit["unresolved"]) / n,
        "precision_lower": hit["event"] / total if total else None,
        "precision_upper": (hit["event"] + hit["unresolved"]) / total if total else None,
        "captured_ge14": int(ge14),
    }


def retrospective_threshold(assessments: Sequence[Assessment], windows: Mapping[str, PatientWindow],
                            outcomes: Mapping[str, str], *, false_alert_fraction: float):
    """Outcome-assisted attainable bound; this is not independent validation.

    Maximize event capture, then captures at least 14 days early, then minimize
    reference alerts, unresolved alerts and total alerts; finally maximize the
    strict cutoff. Reference alerts/all subjects must be strictly below the
    supplied fraction, including when the cohort size makes 1% an integer.
    """
    if not isfinite(false_alert_fraction) or not 0 < false_alert_fraction <= 1:
        raise ValueError("False-alert fraction must lie in (0, 1]")
    stops = {key: window.prediction_stop for key, window in windows.items()}
    feasible = []
    for threshold in strict_thresholds(assessments):
        metrics = outcome_metrics(first_alerts(assessments, windows, threshold), outcomes, stopping_times=stops)
        if metrics["reference_alerts_per_all"] < false_alert_fraction:
            feasible.append({"threshold": threshold, **metrics})
    return max(feasible, key=lambda r: (r["captured"], r["captured_ge14"], -r["reference_alerts"],
                                       -r["unresolved_alerts"], -r["alerts"], r["threshold"]))


def match_testing_budget(assessments: Sequence[Assessment], windows: Mapping[str, PatientWindow],
                        workload: Callable[[str, float | None], RetestEstimate], *,
                        proxy_budget: int, verified_budget: int,
                        review_budget: int | None = None):
    """Match first-test capacity using scores and routine-test timing only.

    ``workload(subject, first_alert)`` supplies the same observations and retest
    interval for every candidate. Selection accesses only sampling-proxy and
    verified-collection counts plus alert coverage. It accepts no outcomes.
    Both bounds must fit the target budget; ties are never split. Within those
    constraints maximize alerted subjects, then use the lowest feasible cutoff.
    This is retrospective resource standardization, not prospective validation.
    """
    if proxy_budget < 0 or verified_budget < 0 or (review_budget is not None and review_budget < 0):
        raise ValueError("Budgets must be nonnegative")
    cache = {}
    feasible = []
    for threshold in strict_thresholds(assessments):
        alerts = first_alerts(assessments, windows, threshold)
        proxy, upper = 0, 0
        for subject, time in alerts.items():
            key = (subject, time)
            if key not in cache:
                cache[key] = workload(subject, time)
            estimate = cache[key]
            if estimate.first_alert != time or estimate.request != int(time is not None):
                raise ValueError("Workload must describe exactly this subject's first alert")
            proxy += estimate.sampling_proxy
            upper += estimate.sampling_upper
        count = sum(t is not None for t in alerts.values())
        if (proxy <= proxy_budget and upper <= verified_budget
                and (review_budget is None or count <= review_budget)):
            feasible.append({"threshold": threshold, "alerts": count,
                             "additional_tests_proxy": proxy, "additional_tests_upper": upper})
    if not feasible:
        raise ValueError("No feasible threshold; check the no-alert workload")
    return max(feasible, key=lambda row: (row["alerts"], -row["threshold"]))
