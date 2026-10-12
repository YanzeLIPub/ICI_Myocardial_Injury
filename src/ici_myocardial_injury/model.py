"""Fitting-only preprocessing, exponential survival risk and empirical ranks.

This module specifies an algorithm, not the study's trained model. Every fitted
quantity is learned from arrays supplied to ``fit``. It provides no checkpoint,
study cutoff, reference distribution, inference service or serialization helper.
"""

from collections import Counter
from dataclasses import dataclass
from math import isfinite
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

FEATURE_NAMES = ("tni_latest_log", "tni_age_days", "elapsed_days")


def _matrix(values, columns=None, *, allow_missing=True):
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or (columns is not None and x.shape[1] != columns):
        raise ValueError("Expected a two-dimensional feature matrix with matching columns")
    if not allow_missing and not np.isfinite(x).all():
        raise ValueError("Model features must be finite after preprocessing")
    return x


def subject_weights(subjects: Sequence[str]) -> np.ndarray:
    """Give each subject total weight one across repeated assessments."""
    counts = Counter(subjects)
    return np.array([1.0 / counts[key] for key in subjects], dtype=float)


class FeaturePreprocessor:
    """Fit medians, 1st/99th percentiles and scaling on fitting rows only.

    TnI concentration is not winsorized. Every feature receives a missingness
    indicator. The standard deviation has a 0.05 floor. Missing includes all
    nonfinite values; medians use finite fitting measurements only.
    """

    def fit(self, features, names=FEATURE_NAMES):
        x = _matrix(features, len(names))
        if not len(x):
            raise ValueError("At least one fitting row is required")
        self.names_ = tuple(names)
        self.median_ = np.zeros(x.shape[1])
        self.lower_ = np.full(x.shape[1], -np.inf)
        self.upper_ = np.full(x.shape[1], np.inf)
        for j, name in enumerate(names):
            values = x[np.isfinite(x[:, j]), j]
            if len(values):
                self.median_[j] = np.median(values)
                if name != "tni_latest_log":
                    self.lower_[j], self.upper_[j] = np.quantile(values, [0.01, 0.99])
        filled = np.where(np.isfinite(x), np.clip(x, self.lower_, self.upper_), self.median_)
        self.mean_ = filled.mean(axis=0)
        self.scale_ = np.maximum(filled.std(axis=0), 0.05)
        return self

    def transform(self, features):
        if not hasattr(self, "names_"):
            raise RuntimeError("Fit preprocessing on authorized training data first")
        x = _matrix(features, len(self.names_))
        missing = ~np.isfinite(x)
        filled = np.where(missing, self.median_, np.clip(x, self.lower_, self.upper_))
        return np.column_stack(((filled - self.mean_) / self.scale_, missing.astype(float)))


def exponential_objective(parameters, features, duration, event, weight, penalty):
    """Weighted censored exponential loss and its analytic gradient.

    The exponent is clipped for numerical stability, matching the research
    implementation. The gradient differentiates that same clipped expression,
    including its zero exponential derivative outside the clipping interval.
    The unpenalized intercept and ridge penalty on slopes are kept separate.
    """
    eta = parameters[0] + features @ parameters[1:]
    rate = np.exp(np.clip(eta, -40.0, 30.0))
    expected = rate * duration
    active = (eta > -40.0) & (eta < 30.0)
    residual = weight * (expected * active - event)
    loss = np.sum(weight * (expected - event * eta))
    loss += 0.5 * penalty * np.sum(parameters[1:] ** 2)
    gradient = np.r_[residual.sum(), features.T @ residual + penalty * parameters[1:]]
    return float(loss), gradient


class ExponentialRidge:
    """Patient-weighted constant-hazard model for censored landmark intervals."""

    def __init__(self, penalty: float = 0.005):
        if not isfinite(penalty) or penalty < 0:
            raise ValueError("Penalty must be finite and nonnegative")
        self.penalty = penalty

    def fit(self, features, duration, event, subjects: Sequence[str]):
        x = _matrix(features, allow_missing=False)
        duration, event = np.asarray(duration, float), np.asarray(event, float)
        if (not len(x) or duration.shape != (len(x),) or event.shape != (len(x),)
                or len(subjects) != len(x)):
            raise ValueError("Fitting arrays must have the same nonzero row count")
        if not np.isfinite(duration).all() or np.any(duration <= 0):
            raise ValueError("Observed survival exposure must be finite and positive")
        if not np.isin(event, [0.0, 1.0]).all():
            raise ValueError("Event indicators must be 0 or 1; censoring contributes exposure")
        weight = subject_weights(subjects)
        weight /= weight.sum()
        intercept = np.log((np.sum(weight * event) + 1e-7)
                           / (np.sum(weight * duration) + 1e-7))
        result = minimize(
            exponential_objective, np.r_[intercept, np.zeros(x.shape[1])],
            args=(x, duration, event, weight, self.penalty), jac=True,
            method="L-BFGS-B", bounds=[(-20.0, 0.0)] + [(-4.0, 4.0)] * x.shape[1],
            options={"maxiter": 700, "ftol": 1e-11, "gtol": 1e-7},
        )
        if not result.success:
            raise RuntimeError(f"Survival optimization failed: {result.message}")
        eta = result.x[0] + x @ result.x[1:]
        if np.any((eta <= -40.0) | (eta >= 30.0)):
            raise RuntimeError("Fitted hazards reach numerical clipping; inspect input scaling")
        self.intercept_, self.coef_ = float(result.x[0]), result.x[1:].copy()
        self.iterations_ = int(result.nit)
        return self

    def rate(self, features):
        if not hasattr(self, "coef_"):
            raise RuntimeError("Fit the survival model first")
        x = _matrix(features, len(self.coef_), allow_missing=False)
        return np.exp(np.clip(self.intercept_ + x @ self.coef_, -40.0, 30.0))

    def risk(self, features, remaining_days):
        rates = self.rate(features)
        remaining = np.asarray(remaining_days, dtype=float)
        if remaining.shape != rates.shape or not np.isfinite(remaining).all() or np.any(remaining < 0):
            raise ValueError("Each row needs a finite, nonnegative remaining horizon")
        return -np.expm1(-rates * remaining)


@dataclass(frozen=True)
class EmpiricalCDF:
    """Right-inclusive ranks from finite fitting observations only."""

    reference: np.ndarray

    @classmethod
    def fit(cls, values):
        x = np.asarray(values, dtype=float)
        if x.ndim != 1:
            raise ValueError("CDF input must be one-dimensional")
        reference = np.sort(x[np.isfinite(x)])
        if not len(reference):
            raise ValueError("A reference distribution requires observed fitting values")
        reference.setflags(write=False)
        return cls(reference)

    def transform(self, values):
        x = np.asarray(values, dtype=float)
        if x.ndim != 1:
            raise ValueError("CDF input must be one-dimensional")
        rank = np.searchsorted(self.reference, np.where(np.isfinite(x), x, -np.inf), side="right")
        return np.where(np.isfinite(x), rank / len(self.reference), 0.0)


class RankFusionModel:
    """Combine 75% concentration rank with 25% landmark survival-risk rank.

    The output is a rank-based warning score, not a calibrated event probability.
    Both empirical distributions use fitting assessments only. Calibration or
    evaluation data cannot be passed as a second data source to ``fit``.
    Repeated fitting assessments retain the source scheme's unweighted ECDF;
    subject weighting applies to the survival likelihood only.
    """

    def __init__(self, concentration_weight=0.75, penalty=0.005):
        if not isfinite(concentration_weight) or not 0 <= concentration_weight <= 1:
            raise ValueError("Concentration weight must lie in [0, 1]")
        self.concentration_weight = concentration_weight
        self.penalty = penalty

    def fit(self, features, duration, event, remaining_days, subjects):
        x = _matrix(features, len(FEATURE_NAMES))
        remaining = np.asarray(remaining_days, float)
        duration = np.asarray(duration, float)
        if (remaining.shape != (len(x),) or duration.shape != remaining.shape
                or not np.isfinite(remaining).all() or np.any(remaining <= 0)
                or np.any(duration > remaining + 1e-6)):
            raise ValueError("Exposure must fit within a positive remaining outcome horizon")
        prep = FeaturePreprocessor().fit(x)
        transformed = prep.transform(x)
        model = ExponentialRidge(self.penalty).fit(transformed, duration, event, subjects)
        risk_reference = EmpiricalCDF.fit(model.risk(transformed, remaining))
        concentration_reference = EmpiricalCDF.fit(x[:, 0])
        self.preprocessor_ = prep
        self.survival_ = model
        self.risk_reference_ = risk_reference
        self.concentration_reference_ = concentration_reference
        return self

    def predict(self, features, remaining_days):
        if not hasattr(self, "survival_"):
            raise RuntimeError("No trained model is included; fit authorized data first")
        x = _matrix(features, len(FEATURE_NAMES))
        risk = self.survival_.risk(self.preprocessor_.transform(x), remaining_days)
        weight = self.concentration_weight
        return (weight * self.concentration_reference_.transform(x[:, 0])
                + (1 - weight) * self.risk_reference_.transform(risk))
