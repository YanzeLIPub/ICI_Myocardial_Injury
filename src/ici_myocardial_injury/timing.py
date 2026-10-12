"""Report-available assessment times and strictly historical model inputs.

Times use seconds on one consistent clock. Clinical endpoint adjudication and
unit harmonization are upstream responsibilities; missing outcomes stay missing.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

import numpy as np

DAY = 86400.0
FIXED_DAYS = (0, 7, 14, 21, 30, 45, 60)


@dataclass(frozen=True)
class PatientWindow:
    """Separate eligibility boundaries from the time of prediction.

    ``prediction_stop`` is the first disqualifying marker/clinical boundary.
    It may precede baseline, in which case no assessment is eligible. A future
    stopping boundary limits follow-up; it is never a feature of the model.
    """

    anchor: float
    followup_end: float
    prediction_stop: float | None = None

    def __post_init__(self):
        values = (self.anchor, self.followup_end)
        if not all(isfinite(x) for x in values):
            raise ValueError("Anchor and follow-up times must be finite")
        if self.prediction_stop is not None and not isfinite(self.prediction_stop):
            raise ValueError("Prediction stopping time must be finite or None")

    def eligible(self, time: float) -> bool:
        return (isfinite(time) and self.anchor <= time < self.followup_end
                and (self.prediction_stop is None or time < self.prediction_stop))


@dataclass(frozen=True)
class LabReport:
    """One report time with harmonized, nonnegative numeric measurements.

    Use the name ``tni`` for troponin I. Only exact numeric measurements belong
    here; results expressed as inequalities require separate handling upstream.
    Report time is availability time, regardless of specimen collection time.
    """

    time: float
    values: Mapping[str, float]

    def __post_init__(self):
        if not isfinite(self.time):
            raise ValueError("Report time must be finite")
        if any(not isfinite(v) or v < 0 for v in self.values.values()):
            raise ValueError("Measurements must be finite and nonnegative")


def resolve_reports(reports: Sequence[LabReport]) -> list[LabReport]:
    """Collapse identical measurements; omit conflicts at the same time/channel."""
    groups: dict[float, dict[str, set[float]]] = {}
    for report in reports:
        channels = groups.setdefault(report.time, {})
        for channel, value in report.values.items():
            channels.setdefault(channel, set()).add(value)
    output = []
    for time, channels in sorted(groups.items()):
        values = {c: next(iter(v)) for c, v in channels.items() if len(v) == 1}
        if values:
            output.append(LabReport(time, values))
    return output


def assessment_times(window: PatientWindow, report_times: Sequence[float], *,
                     fixed_days=FIXED_DAYS, last_day: float = 60.0) -> list[float]:
    """Union fixed assessments and every available report update.

    The cap is baseline plus ``last_day``, not the last fixed assessment before
    a future event. Equality with a stopping or follow-up boundary is excluded.
    """
    if not isfinite(last_day) or last_day < 0:
        raise ValueError("The assessment horizon must be finite and nonnegative")
    if any(not isfinite(t) for t in report_times):
        raise ValueError("Report times must be finite")
    if any(not isfinite(d) or d < 0 for d in fixed_days):
        raise ValueError("Fixed assessment days must be finite and nonnegative")
    times = {window.anchor + d * DAY for d in fixed_days} | set(report_times)
    return sorted(t for t in times if window.eligible(t)
                  and t <= window.anchor + last_day * DAY)


def minimal_features(window: PatientWindow, assessment: float,
                     reports: Sequence[LabReport], *,
                     supplemental: Sequence[LabReport] = (),
                     lookback_days: float = 62.0,
                     records_per_source: int = 64) -> np.ndarray:
    """Latest log(1+TnI), measurement age and elapsed baseline days.

    Mirrors the source history construction: retain up to 64 primary report
    times and up to 64 additional supplemental times in the 62-day window.
    Supplemental values may fill a channel but cannot replace a primary value.
    Relative times and transformed measurements are stored at float32 precision
    as in the analysis tensors. No outcome value enters the feature vector.
    An empty eligible history raises; a history without TnI returns missing TnI
    and age, for fitting-only imputation downstream.
    """
    if not window.eligible(assessment):
        raise ValueError("Assessment is outside the strict pre-event window")
    if not isfinite(lookback_days) or lookback_days < 0 or records_per_source < 1:
        raise ValueError("Invalid history limits")
    cutoff = assessment - lookback_days * DAY
    def past(source):
        return [r for r in resolve_reports(source) if cutoff <= r.time <= assessment]
    def encoded(report):
        relative = float(np.float32((report.time - assessment) / DAY))
        values = {c: np.float32(np.log1p(v)) for c, v in report.values.items()}
        return relative, values
    records = dict(encoded(r) for r in past(reports)[-records_per_source:])
    pending = dict(encoded(r) for r in past(supplemental))
    extra_times = set(sorted(t for t in pending if t not in records)[-records_per_source:])
    for time, values in pending.items():
        if time in records or time in extra_times:
            current = records.setdefault(time, {})
            for channel, value in values.items():
                if channel in current and current[channel] != value:
                    raise ValueError("Supplemental measurements conflict with primary reports")
                current[channel] = value
    if not records:
        raise ValueError("No report is available in the historical window")
    last, age = np.nan, np.nan
    for relative, values in sorted(records.items()):
        if "tni" in values:
            last, age = float(values["tni"]), -relative
    return np.array([last, age, (assessment - window.anchor) / DAY], dtype=float)


def assert_disjoint_subject_sets(*groups: Sequence[str]) -> None:
    """Check unique subject membership before splitting repeated assessments."""
    seen = set()
    for group in groups:
        unique = set(group)
        if len(unique) != len(group) or seen & unique:
            raise ValueError("Subject memberships must be unique and disjoint")
        seen.update(unique)
