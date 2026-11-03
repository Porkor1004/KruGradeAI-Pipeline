"""Hand-calculated metrics, including fractional scores and undefined kappa."""
import unittest
from decimal import Decimal
from pydantic import ValidationError
from krugradeai.metrics import ScorePair, calculate_metrics


def pair(ai, ta, maximum="2"):
    return ScorePair(predicted=ai, reference=ta, maximum=maximum)


class MetricsTests(unittest.TestCase):
    def test_mae_exact_match_and_fractional_scores(self):
        metrics = calculate_metrics([pair("1.5", "1"), pair("0.2", "0.2")], expected_count=2)
        self.assertEqual(metrics.mae, Decimal("0.25"))
        self.assertEqual(metrics.exact_match, Decimal("0.5"))
        self.assertEqual(metrics.normalized_mae, Decimal("0.125"))
        self.assertEqual(metrics.sample_count, 2)
        self.assertEqual(metrics.coverage, Decimal(1))

    def test_normalized_mae_for_different_maxima(self):
        metrics = calculate_metrics([pair("1", "0", "2"), pair("5", "0", "10")], expected_count=2)
        self.assertEqual(metrics.mae, Decimal(3))
        self.assertEqual(metrics.normalized_mae, Decimal("0.5"))
        self.assertIsNone(metrics.qwk)
        self.assertTrue(metrics.qwk_undefined_reason)

    def test_zero_is_a_score_but_missing_is_not(self):
        metrics = calculate_metrics([pair("0", "1")], expected_count=2)
        self.assertEqual(metrics.mae, Decimal(1))
        self.assertEqual(metrics.coverage, Decimal("0.5"))
        with self.assertRaises(ValidationError):
            pair(None, "1")

    def test_empty_evaluation_is_explicitly_undefined(self):
        metrics = calculate_metrics([], expected_count=2)
        self.assertIsNone(metrics.mae)
        self.assertIsNone(metrics.exact_match)
        self.assertIsNone(metrics.qwk)
        self.assertEqual(metrics.coverage, 0)
        self.assertEqual(metrics.sample_count, 0)

    def test_qwk_agreement_and_disagreement(self):
        cats = (Decimal(0), Decimal(1), Decimal(2))
        perfect = calculate_metrics([pair(0, 0), pair(1, 1), pair(2, 2)], expected_count=3, qwk_categories=cats)
        self.assertEqual(perfect.qwk, 1)
        inverse = calculate_metrics([pair(0, 2), pair(2, 0)], expected_count=2, qwk_categories=cats)
        self.assertEqual(inverse.qwk, -1)

    def test_qwk_fractional_categories_without_rounding(self):
        metrics = calculate_metrics([pair("0", "0", "1"), pair("0.5", "0.5", "1"), pair("1", "1", "1")], expected_count=3,
                                    qwk_categories=(Decimal(0), Decimal("0.5"), Decimal(1)))
        self.assertEqual(metrics.qwk, 1)

    def test_qwk_undefined_constant_or_unmapped_scores(self):
        for samples, categories in (([pair(1, 1)], (Decimal(0), Decimal(1))),
                                    ([pair("0.5", "1")], (Decimal(0), Decimal(1))),
                                    ([pair(1, 1)], None)):
            with self.subTest(categories=categories):
                metrics = calculate_metrics(samples, expected_count=1, qwk_categories=categories)
                self.assertIsNone(metrics.qwk)
                self.assertTrue(metrics.qwk_undefined_reason)

    def test_invalid_bounds_population_and_categories(self):
        for args in (("-1", "0", "2"), ("3", "0", "2"), ("0", "NaN", "2"), ("0", "0", "0")):
            with self.subTest(args=args), self.assertRaises(ValidationError):
                pair(*args)
        with self.assertRaises(ValueError):
            calculate_metrics([pair(1, 1)], expected_count=0)
        with self.assertRaises(ValueError):
            calculate_metrics([pair(1, 1)], expected_count=1, qwk_categories=(Decimal(1), Decimal(0)))

class AdditionalMetricsTests(unittest.TestCase):
    def test_qwk_uses_empirical_marginals(self):
        metrics = calculate_metrics([pair(0, 0), pair(1, 0), pair(2, 2)], expected_count=3,
                                    qwk_categories=(Decimal(0), Decimal(1), Decimal(2)))
        self.assertEqual(metrics.qwk, Decimal("0.8"))

    def test_decimal_results_do_not_depend_on_callers_rounding_context(self):
        from decimal import localcontext, ROUND_UP
        samples = [pair(0, 1, 3), pair(0, 0, 3), pair(0, 0, 3)]
        ordinary = calculate_metrics(samples, expected_count=3)
        with localcontext() as context:
            context.prec = 2
            context.rounding = ROUND_UP
            changed_context = calculate_metrics(samples, expected_count=3)
        self.assertEqual(ordinary, changed_context)
