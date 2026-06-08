import itertools
import importlib
import unittest

import numpy as np

from verianet.activations import (
    certify_gelu_envelope,
    certify_gelu_line,
    gelu,
    gelu_bounds,
    sound_gelu_envelope,
)
from verianet.bounds import ibp_affine_keras
from verianet.counterfactual import find_minimal_relaxed_counterfactual, solve_target_margin
from verianet.network import NetworkWeights
from verianet.objectives import (
    build_network_polytope,
    class_margin,
    input_box,
    maximize_margin_pattern,
    verify_robust,
)
from verianet.paths import WEIGHTS_PATH
from verianet.refinement import class_margin_with_splits, split_input_box, verify_robust_with_splits
from verianet.stats import clipped_error_interval, hoeffding_samples


class ActivationTests(unittest.TestCase):
    def test_gelu_bounds_enclose_samples(self):
        intervals = [
            (-3.0, -0.1),
            (-2.0, 0.0),
            (-1.0, 1.0),
            (-0.5, 2.0),
            (0.0, 2.0),
            (0.5, 2.0),
        ]

        for L, U in intervals:
            with self.subTest(interval=(L, U)):
                lb, ub = gelu_bounds(np.array([L]), np.array([U]))
                xs = np.linspace(L, U, 5001)
                ys = gelu(xs)
                self.assertLessEqual(float(lb[0]), float(ys.min()) + 1e-10)
                self.assertGreaterEqual(float(ub[0]), float(ys.max()) - 1e-10)

    def test_gelu_envelope_encloses_samples(self):
        intervals = [
            (-3.0, 3.0),
            (-2.0, 2.0),
            (-1.0, 1.0),
            (-1.0, 2.0),
            (-3.0, -0.1),
            (0.0, 1.0),
            (0.5, 2.0),
        ]

        for L, U in intervals:
            with self.subTest(interval=(L, U)):
                lower_lines, upper_lines = sound_gelu_envelope(L, U)
                xs = np.linspace(L, U, 5001)
                ys = gelu(xs)
                lower = np.maximum.reduce([m * xs + c for m, c, _, _ in lower_lines])
                upper = np.minimum.reduce([m * xs + c for m, c, _, _ in upper_lines])
                self.assertLessEqual(float(np.max(lower - ys)), 1e-10)
                self.assertLessEqual(float(np.max(ys - upper)), 1e-10)

    def test_gelu_envelope_lines_are_analytically_certified(self):
        intervals = [
            (-4.0, -2.0),
            (-3.0, 0.5),
            (-2.0, 2.0),
            (-0.25, 3.0),
            (0.0, 2.0),
        ]

        for L, U in intervals:
            with self.subTest(interval=(L, U)):
                lower_lines, upper_lines = sound_gelu_envelope(L, U)
                certificates = certify_gelu_envelope(lower_lines, upper_lines, L, U)
                self.assertEqual(len(certificates), len(lower_lines) + len(upper_lines))
                self.assertTrue(all(cert.min_residual >= -1e-10 for cert in certificates))

    def test_gelu_line_certificate_rejects_bad_bounds(self):
        self.assertIsNone(certify_gelu_line(0.0, 1.0, 0.0, 1.0, "lower"))
        self.assertIsNone(certify_gelu_line(0.0, 0.0, 0.1, 1.0, "upper"))

    def test_gelu_envelope_tightens_positive_interval(self):
        lower_lines, upper_lines = sound_gelu_envelope(0.0, 2.0)
        x = 1.0
        lower_at_x = max(m * x + c for m, c, _, _ in lower_lines)
        upper_at_x = min(m * x + c for m, c, _, _ in upper_lines)

        self.assertGreater(lower_at_x, 0.4)
        self.assertLessEqual(lower_at_x, float(gelu(x)) + 1e-10)
        self.assertGreaterEqual(upper_at_x, float(gelu(x)) - 1e-10)

    def test_gelu_envelope_tightens_mixed_interval_upper_bound(self):
        lower_lines, upper_lines = sound_gelu_envelope(-3.0, 3.0)
        x = 0.0
        upper_at_x = min(m * x + c for m, c, _, _ in upper_lines)

        self.assertLess(upper_at_x, 2.0)
        self.assertGreaterEqual(upper_at_x, float(gelu(x)) - 1e-10)


class BoundTests(unittest.TestCase):
    def test_keras_affine_ibp_matches_corners(self):
        L = np.array([-1.0, 0.0, 2.0])
        U = np.array([0.5, 1.0, 3.0])
        W = np.array([[1.0, -2.0], [0.5, 4.0], [-3.0, 0.25]])
        b = np.array([0.1, -0.2])

        lb, ub = ibp_affine_keras(L, U, W, b)
        corners = np.array(list(itertools.product(*zip(L, U))))
        vals = corners @ W + b
        np.testing.assert_allclose(lb, vals.min(axis=0))
        np.testing.assert_allclose(ub, vals.max(axis=0))


class StatsTests(unittest.TestCase):
    def test_two_sided_hoeffding_sample_count(self):
        self.assertEqual(hoeffding_samples(0.95, 0.05), 738)

    def test_clipped_error_interval(self):
        self.assertEqual(clipped_error_interval(0.03, 0.05), (0.0, 0.08))
        self.assertEqual(clipped_error_interval(0.98, 0.05), (0.9299999999999999, 1.0))


class RefinementTests(unittest.TestCase):
    def test_split_input_box_covers_root_volume(self):
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 2.0])
        leaves = split_input_box(lb, ub, max_leaves=4)

        self.assertEqual(len(leaves), 4)
        self.assertAlmostEqual(sum(box.volume for box in leaves), 2.0)
        for box in leaves:
            self.assertTrue(np.all(box.lb >= lb))
            self.assertTrue(np.all(box.ub <= ub))

    def test_split_margin_refines_root_margin_bound(self):
        weights = NetworkWeights.load()
        x0 = np.full(weights.input_dim, 0.5)
        label = int(weights.predict(x0))
        competitor = (label + 1) % weights.num_classes

        root = class_margin(weights, x0, epsilon=0.05, label=label, competitor=competitor)
        split = class_margin_with_splits(
            weights,
            x0,
            epsilon=0.05,
            label=label,
            competitor=competitor,
            max_leaves=4,
        )

        self.assertIsNotNone(root.value)
        self.assertIsNotNone(split.value)
        assert root.value is not None and split.value is not None
        self.assertLessEqual(split.value, root.value + 1e-8)

    def test_split_robustness_implies_margin_certificates(self):
        weights = NetworkWeights.load()
        x0 = np.zeros(weights.input_dim)
        label = int(weights.predict(x0))

        result = verify_robust_with_splits(weights, x0, epsilon=0.0, label=label, max_leaves=2)

        self.assertTrue(result.robust)
        self.assertEqual(result.solver_failures, 0)
        self.assertEqual(len(result.margins), weights.num_classes - 1)
        self.assertTrue(all(margin is not None and margin <= 1e-8 for margin in result.margins.values()))


class PolytopeTests(unittest.TestCase):
    def test_project_weights_build_expected_polytope(self):
        weights = NetworkWeights.load()
        self.assertEqual(WEIGHTS_PATH.name, "verysmallnn_weights.npz")
        x0 = np.zeros(weights.input_dim)
        lb, ub = input_box(x0, epsilon=0.01)
        build = build_network_polytope(weights, lb, ub)

        self.assertEqual(build.analyzer.nvars, 71)
        self.assertEqual(build.analyzer.var_slices["x0"], slice(0, 49))
        self.assertEqual(build.analyzer.var_slices["a3"].stop, 71)
        self.assertTrue(np.all(np.isfinite(build.analyzer.A)))
        self.assertTrue(np.all(np.isfinite(build.analyzer.b)))

        c = build.analyzer.objective("a3", np.eye(weights.num_classes)[0])
        result = build.analyzer.optimize(c, sense="max")
        self.assertIsNotNone(result.value)

    def test_concrete_forward_trace_satisfies_polytope_constraints(self):
        weights = NetworkWeights.load()
        rng = np.random.default_rng(123)
        center = rng.uniform(0.0, 1.0, size=weights.input_dim)
        lb, ub = input_box(center, epsilon=0.05)
        build = build_network_polytope(weights, lb, ub)

        x0 = rng.uniform(lb, ub)
        a1 = x0 @ weights.W1 + weights.b1
        z1 = gelu(a1)
        a2 = z1 @ weights.W2 + weights.b2
        z2 = gelu(a2)
        a3 = z2 @ weights.W3 + weights.b3

        trace = np.zeros(build.analyzer.nvars, dtype=np.float64)
        trace[build.analyzer.var_slices["x0"]] = x0
        trace[build.analyzer.var_slices["a1"]] = a1
        trace[build.analyzer.var_slices["z1"]] = z1
        trace[build.analyzer.var_slices["a2"]] = a2
        trace[build.analyzer.var_slices["z2"]] = z2
        trace[build.analyzer.var_slices["a3"]] = a3

        max_violation = float(np.max(build.analyzer.A @ trace - build.analyzer.b))
        self.assertLessEqual(max_violation, 1e-9)

    def test_class_margin_objective_upper_bounds_concrete_samples(self):
        weights = NetworkWeights.load()
        rng = np.random.default_rng(456)
        center = rng.uniform(0.0, 1.0, size=weights.input_dim)
        label = int(weights.predict(center))
        competitor = (label + 1) % weights.num_classes

        result = class_margin(weights, center, epsilon=0.05, label=label, competitor=competitor)
        self.assertIsNotNone(result.value)
        assert result.value is not None

        lb, ub = input_box(center, epsilon=0.05)
        for _ in range(50):
            x = rng.uniform(lb, ub)
            logits = weights.forward_logits(x)
            concrete_margin = float(logits[competitor] - logits[label])
            self.assertLessEqual(concrete_margin, result.value + 1e-8)

    def test_margin_optimizer_auxiliary_tracks_max_other_logit(self):
        weights = NetworkWeights.load()
        x0 = np.zeros(weights.input_dim)
        result, analyzer = maximize_margin_pattern(
            weights,
            x0,
            target_class=0,
            epsilon=0.01,
            l1_penalty=0.0,
        )

        self.assertIsNotNone(result.value)
        t = float(analyzer.values(result, "max_other_logit")[0])
        relaxed_logits = analyzer.values(result, "a3")
        max_other = float(np.max(np.delete(relaxed_logits, 0)))
        self.assertAlmostEqual(t, max_other, places=7)

    def test_margin_pattern_optimizer_returns_input(self):
        weights = NetworkWeights.load()
        x0 = np.zeros(weights.input_dim)
        result, analyzer = maximize_margin_pattern(
            weights,
            x0,
            target_class=0,
            epsilon=0.01,
            l1_penalty=0.01,
        )

        self.assertIsNotNone(result.value)
        x_opt = analyzer.values(result, "x0")
        self.assertEqual(x_opt.shape, (weights.input_dim,))
        self.assertTrue(np.all(x_opt >= -1e-8))
        self.assertTrue(np.all(x_opt <= 0.01000001))

    def test_zero_epsilon_robustness_matches_prediction_margin(self):
        weights = NetworkWeights.load()
        rng = np.random.default_rng(789)
        x0 = rng.uniform(0.0, 1.0, size=weights.input_dim)
        label = int(weights.predict(x0))
        robust, margins = verify_robust(weights, x0, epsilon=0.0, label=label)

        logits = weights.forward_logits(x0)
        expected_robust = all(
            float(logits[k] - logits[label]) <= 1e-8
            for k in range(weights.num_classes)
            if k != label
        )
        self.assertEqual(robust, expected_robust)
        for k, margin in margins.items():
            self.assertIsNotNone(margin)
            assert margin is not None
            self.assertAlmostEqual(margin, float(logits[k] - logits[label]), places=7)

    def test_counterfactual_candidate_shape_and_bounds(self):
        weights = NetworkWeights.load()
        x0 = np.zeros(weights.input_dim)
        original = weights.predict(x0)
        target = (original + 1) % weights.num_classes

        candidate = solve_target_margin(
            weights,
            x0,
            epsilon=0.01,
            original_class=original,
            target_class=target,
            margin=0.0,
            require_target_prediction=False,
        )

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.image.shape, (weights.input_dim,))
        self.assertEqual(candidate.delta.shape, (weights.input_dim,))
        self.assertTrue(np.all(candidate.image >= -1e-8))
        self.assertTrue(np.all(candidate.image <= 0.01000001))
        self.assertEqual(candidate.candidate_logits.shape, (weights.num_classes,))
        self.assertEqual(candidate.relaxed_feasible, candidate.relaxed_margin >= 0.0)
        self.assertEqual(candidate.concrete_valid, candidate.concrete_margin >= 0.0)
        self.assertEqual(candidate.accepted, candidate.concrete_valid)

    def test_relaxed_counterfactual_search_reports_relaxed_feasibility(self):
        weights = NetworkWeights.load()
        x0 = np.zeros(weights.input_dim)
        original = weights.predict(x0)
        target = (original + 1) % weights.num_classes

        candidate = find_minimal_relaxed_counterfactual(
            weights,
            x0,
            original_class=original,
            target_class=target,
            margin=-1e6,
            max_epsilon=0.01,
            binary_steps=2,
            require_target_prediction=False,
        )

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.epsilon, 0.0)
        self.assertTrue(candidate.relaxed_feasible)


class LegacyScriptTests(unittest.TestCase):
    def test_finding_import_is_dependency_light(self):
        module = importlib.import_module("scripts.experiments.finding")
        self.assertTrue(hasattr(module, "find_cf_for_pair"))

    def test_ideal_digits_import_is_dependency_light(self):
        module = importlib.import_module("scripts.experiments.ideal_digits")
        self.assertTrue(hasattr(module, "optimize_digit_patterns"))

    def test_robustness_script_import_is_dependency_light(self):
        module = importlib.import_module("scripts.experiments.epsilon_robustness_test")
        self.assertTrue(hasattr(module, "run_sweep"))

    def test_robustness_check_import_is_dependency_light(self):
        module = importlib.import_module("scripts.experiments.robustness_check")
        self.assertTrue(hasattr(module, "run_digit_zero_smoke"))


if __name__ == "__main__":
    unittest.main()
