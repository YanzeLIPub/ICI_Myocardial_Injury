"""Small hand-constructed examples; no study records or teacher predictions.

These tests verify algorithmic boundaries. They do not reproduce study metrics
or validate clinical performance. Fitted toy objects stay in memory only.
"""

from pathlib import Path
import math
import sys
import unittest

# Allow direct unittest discovery from a clean source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from ici_myocardial_injury.model import (
    EmpiricalCDF, FeaturePreprocessor, RankFusionModel,
    exponential_objective, subject_weights,
)
from ici_myocardial_injury.policies import (
    Assessment, first_alerts, match_testing_budget, outcome_metrics,
    retrospective_threshold, strict_thresholds,
)
from ici_myocardial_injury.timing import (
    DAY, LabReport, PatientWindow, assessment_times,
    assert_disjoint_subject_sets, minimal_features, resolve_reports,
)
from ici_myocardial_injury.workload import (
    ObservationWindow, TestEpisode, first_alert_workload, paired_sampling_difference,
)


class TimingTests(unittest.TestCase):
    def test_updates_before_next_fixed_day_and_strict_stops(self):
        early = assessment_times(PatientWindow(0, 90 * DAY, 6 * DAY), [3 * DAY, 6 * DAY])
        later = assessment_times(PatientWindow(0, 90 * DAY, 8 * DAY), [3 * DAY, 6 * DAY])
        self.assertEqual(early, [0, 3 * DAY])
        self.assertTrue(set(early) <= set(later))
        self.assertNotIn(70 * DAY, assessment_times(PatientWindow(0, 90 * DAY), [70 * DAY]))
        self.assertEqual(assessment_times(PatientWindow(0, 90 * DAY, 0), [DAY]), [])

    def test_future_reports_cannot_change_past_features(self):
        window = PatientWindow(0, 90 * DAY, 10 * DAY)
        past = [LabReport(-DAY, {"tni": 2.0}), LabReport(DAY, {"tni": 3.0})]
        a = minimal_features(window, DAY, past)
        b = minimal_features(window, DAY, past + [LabReport(2 * DAY, {"tni": 900.0})])
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a[0], float(np.float32(math.log1p(3.0))))
        self.assertEqual(a[1:].tolist(), [0.0, 1.0])
        with self.assertRaises(ValueError):
            minimal_features(window, 10 * DAY, past)

    def test_duplicates_conflicts_and_supplemental_overwrite(self):
        reports = [LabReport(0, {"tni": 2.0}), LabReport(0, {"tni": 2.0}),
                   LabReport(DAY, {"tni": 3.0}), LabReport(DAY, {"tni": 4.0})]
        self.assertEqual(len(resolve_reports(reports)), 1)
        with self.assertRaises(ValueError):
            minimal_features(PatientWindow(0, 5 * DAY), 0, reports,
                             supplemental=[LabReport(0, {"tni": 8.0})])

    def test_disjoint_subject_membership(self):
        assert_disjoint_subject_sets(["toy-a"], ["toy-b"])
        with self.assertRaises(ValueError):
            assert_disjoint_subject_sets(["toy-a"], ["toy-a"])


class ModelTests(unittest.TestCase):
    def test_weights_and_right_inclusive_ranks(self):
        np.testing.assert_array_equal(subject_weights(["toy-a", "toy-a", "toy-b"]), [0.5, 0.5, 1.0])
        cdf = EmpiricalCDF.fit([1.0, 2.0, 2.0, np.nan])
        np.testing.assert_allclose(cdf.transform([0.0, 1.0, 2.0, np.nan]), [0.0, 1/3, 1.0, 0.0])
        with self.assertRaises(ValueError):
            EmpiricalCDF.fit([np.nan])

    def test_preprocessing_is_fitting_only_and_preserves_tni_tail(self):
        fit = np.array([[0.0, 1.0, 0.0], [1.0, 3.0, 2.0], [2.0, np.nan, 4.0]])
        prep = FeaturePreprocessor().fit(fit)
        median, mean = prep.median_.copy(), prep.mean_.copy()
        transformed = prep.transform([[50.0, np.nan, 100.0]])
        np.testing.assert_array_equal(prep.median_, median)
        np.testing.assert_array_equal(prep.mean_, mean)
        self.assertGreater(transformed[0, 0], 10.0)
        self.assertEqual(transformed[0, 4], 1.0)
        self.assertTrue(np.isfinite(transformed).all())

    def test_objective_gradient_interior_and_clipped_exponent(self):
        x = np.array([[1.0], [-1.0]])
        duration, events, weights = np.array([2.0, 5.0]), np.array([1.0, 0.0]), np.array([0.5, 0.5])
        for parameters in (np.array([-2.0, 0.3]), np.array([-45.0, 0.3])):
            _, gradient = exponential_objective(parameters, x, duration, events, weights, 0.005)
            numerical = []
            for j in range(len(parameters)):
                shift = np.zeros_like(parameters); shift[j] = 1e-5
                hi = exponential_objective(parameters + shift, x, duration, events, weights, 0.005)[0]
                lo = exponential_objective(parameters - shift, x, duration, events, weights, 0.005)[0]
                numerical.append((hi - lo) / (2e-5))
            np.testing.assert_allclose(gradient, numerical, atol=1e-8)

    def test_toy_fit_and_rank_fusion_without_saved_outputs(self):
        x = np.array([[0.1, 1, 0], [0.3, 0, 2], [0.5, 2, 0], [0.7, 0, 2],
                      [0.2, 1, 0], [0.4, 1, 2]], dtype=float)
        duration = np.array([6, 4, 8, 6, 10, 8], dtype=float)
        event = [1, 1, 0, 0, 0, 0]
        remaining = [10, 8, 10, 8, 10, 8]
        subjects = ["toy-a", "toy-a", "toy-b", "toy-b", "toy-c", "toy-c"]
        model = RankFusionModel().fit(x, duration, event, remaining, subjects)
        before = model.risk_reference_.reference.copy()
        scores = model.predict(x, remaining)
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(np.all((scores >= 0) & (scores <= 1)))
        np.testing.assert_array_equal(before, model.risk_reference_.reference)
        expected = (0.75 * model.concentration_reference_.transform(x[:, 0])
                    + 0.25 * model.risk_reference_.transform(model.survival_.risk(
                        model.preprocessor_.transform(x), remaining)))
        np.testing.assert_array_equal(scores, expected)
        with self.assertRaises(RuntimeError):
            RankFusionModel().predict(x, remaining)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.windows = {key: PatientWindow(0, 90 * DAY, 30 * DAY)
                        for key in ("toy-a", "toy-b", "toy-c")}
        self.rows = [Assessment("toy-a", DAY, 0.8, DAY),
                     Assessment("toy-a", 2 * DAY, 0.9, DAY),
                     Assessment("toy-b", DAY, 0.8, DAY)]

    def test_strict_ties_and_earliest_threshold_crossing(self):
        self.assertEqual(first_alerts(self.rows, self.windows, 0.8),
                         {"toy-a": 2 * DAY, "toy-b": None, "toy-c": None})
        below = first_alerts(self.rows, self.windows, math.nextafter(0.8, -math.inf))
        self.assertEqual(below["toy-a"], DAY)
        self.assertEqual(below["toy-b"], DAY)
        self.assertIn(0.8, strict_thresholds(self.rows))

    def test_invalid_timing_is_rejected(self):
        for row in (Assessment("toy-a", 30 * DAY, 0.8, DAY),
                    Assessment("toy-a", DAY, 0.8, 2 * DAY)):
            with self.assertRaises(ValueError):
                first_alerts([row], self.windows, 0.5)

    def test_unknown_is_never_silently_negative(self):
        result = outcome_metrics({"toy-a": 1.0, "toy-b": 1.0, "toy-c": 1.0},
                                 {"toy-a": "event", "toy-b": "reference", "toy-c": "unresolved"})
        self.assertEqual(result["reference_alerts_per_all"], 1/3)
        self.assertEqual(result["reference_false_positive_rate"], 1.0)
        self.assertEqual(result["unresolved_alerts"], 1)

    def test_whole_ties_cannot_exceed_review_capacity(self):
        window = ObservationWindow(90 * DAY, 90 * DAY)
        work = lambda key, time: first_alert_workload(time, [], window)
        result = match_testing_budget(self.rows, self.windows, work,
                                      proxy_budget=1, verified_budget=1, review_budget=1)
        self.assertEqual(result["threshold"], 0.8)
        self.assertEqual(result["alerts"], 1)

    def test_strict_false_alert_limit_at_integer_boundary(self):
        rows = [Assessment("toy-a", DAY, 0.6, DAY), Assessment("toy-b", DAY, 0.5, DAY)]
        out = retrospective_threshold(rows, self.windows,
                                      {"toy-a": "event", "toy-b": "reference", "toy-c": "unresolved"},
                                      false_alert_fraction=1/3)
        self.assertEqual(out["captured"], 1)
        self.assertEqual(out["reference_alerts"], 0)


class WorkloadTests(unittest.TestCase):
    def setUp(self):
        self.window = ObservationWindow(20 * DAY, 20 * DAY)

    def test_prealert_and_equal_time_collections_cannot_substitute(self):
        tests = [TestEpisode(2 * DAY, DAY), TestEpisode(2 * DAY, 0)]
        result = first_alert_workload(DAY, tests, self.window)
        self.assertEqual(result.sampling_proxy, 1)
        self.assertEqual(result.excluded_prealert_collections, 2)

    def test_collection_in_window_with_pending_result_is_not_second_draw(self):
        result = first_alert_workload(DAY, [TestEpisode(4 * DAY, 2 * DAY)], self.window)
        self.assertEqual(result.report_proxy, 1)
        self.assertEqual(result.sampling_upper, 0)
        self.assertTrue(result.pending_result_after_deadline)

    def test_unknown_collection_keeps_uncertainty_interval(self):
        result = first_alert_workload(DAY, [TestEpisode(2 * DAY)], self.window)
        self.assertEqual((result.sampling_lower, result.sampling_proxy, result.sampling_upper), (0, 0, 1))
        self.assertTrue(result.unknown_collection_proxy)
        self.assertEqual(paired_sampling_difference(result, result), {"proxy": 0, "lower": 0, "upper": 0})

    def test_short_followup_preserves_lower_upper_bounds(self):
        result = first_alert_workload(DAY, [], ObservationWindow(20 * DAY, 1.5 * DAY))
        self.assertEqual((result.sampling_lower, result.sampling_proxy, result.sampling_upper), (0, 1, 1))
        self.assertTrue(result.short_observation)

    def test_report_beyond_horizon_does_not_substitute(self):
        result = first_alert_workload(DAY, [TestEpisode(21 * DAY, 2 * DAY)], self.window)
        self.assertEqual(result.sampling_upper, 1)
        self.assertEqual(first_alert_workload(None, [], self.window).request, 0)


if __name__ == "__main__":
    unittest.main()
