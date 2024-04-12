import unittest
import math
from itertools import combinations

from qfa_toolkit.recognition_strategy import IsolatedCutPoint
from qfa_toolkit.recognition_strategy import NegativeOneSidedBoundedError

from .utils import get_arbitrary_mmqfa
from qfa_toolkit.quantum_finite_state_automaton_language import (
    MeasureManyQuantumFiniteStateAutomatonLanguage as Mmqfl)


class TestMeasureManyQuantumFiniteStateAutomatonLanguage(unittest.TestCase):
    def test_measure_many_quantum_finite_state_automaton_language(self):
        # M(a^n) = (1/2)^n
        mmqfa = get_arbitrary_mmqfa(1/2)

        isolated_cut_point = IsolatedCutPoint(3/16, 1/32)
        neg_one_sided = NegativeOneSidedBoundedError(3/4)
        invalid_neg_one_sided = NegativeOneSidedBoundedError(1/16)

        # { a^n }, n <= 2
        lang_isolated_cut_point = Mmqfl(mmqfa, isolated_cut_point)
        # { a^n }, n <= 0
        lang_neg_one_sided = Mmqfl(mmqfa, neg_one_sided)
        # Invalid for { a^n } 0 < n < 4,
        # since 1/16 < p < 1 for probabilities p.
        lang_invalid = Mmqfl(mmqfa, invalid_neg_one_sided)

        self.assertIn([], lang_isolated_cut_point)
        self.assertIn([1], lang_isolated_cut_point)
        self.assertIn([1, 1], lang_isolated_cut_point)
        self.assertNotIn([1, 1, 1], lang_isolated_cut_point)
        target_isolated_cut_point = [[], [1], [1, 1]]
        self.assertEqual(
            list(lang_isolated_cut_point.enumerate_length_less_than_n(10)),
            target_isolated_cut_point,
        )

        self.assertIn([], lang_neg_one_sided)
        self.assertNotIn([1], lang_neg_one_sided)
        self.assertNotIn([1, 1], lang_neg_one_sided)
        self.assertNotIn([1, 1, 1], lang_neg_one_sided)
        target_neg_one_sided = [[]]
        self.assertEqual(
            list(lang_neg_one_sided.enumerate_length_less_than_n(10)),
            target_neg_one_sided,
        )

        self.assertIn([], lang_invalid)
        with self.assertRaises(ValueError):
            [1] in lang_invalid
        with self.assertRaises(ValueError):
            [1, 1] in lang_invalid
        with self.assertRaises(ValueError):
            [1, 1, 1] in lang_invalid

    def test_isolated_cut_point_complement(self):
        m1 = get_arbitrary_mmqfa(1/2)
        isolated_cut_point_1 = IsolatedCutPoint(3/16, 1/32)

        l1 = Mmqfl(m1, isolated_cut_point_1)
        l2 = ~l1

        n = 100
        self.assertTrue(all(map(
            lambda w: w not in l2, l1.enumerate_length_less_than_n(n))))
        self.assertTrue(all(map(
            lambda w: w not in l1, l2.enumerate_length_less_than_n(n))))

    def test_from_unary_singleton(self):
        n = 10
        for k in range(n):
            for theta in map(lambda e: math.pi / 8 * e, range(1, 4)):
                for phi in map(lambda e: math.pi / 8 * e, range(1, 4)):
                    qfl = Mmqfl.from_unary_singleton(k, (theta, phi))
                    language = list(qfl.enumerate_length_less_than_n(2 * n))
                    target = [[1] * k]
                    self.assertEqual(language, target)

    def test_from_unary_singleton_without_params(self):
        n = 10
        for k in range(n):
            qfl = Mmqfl.from_unary_singleton(k)
            language = list(qfl.enumerate_length_less_than_n(2 * n))
            target = [[1] * k]
            self.assertEqual(language, target)

    def test_from_unary_finite(self):
        n = 5
        for ks in combinations(range(n), 3):
            qfl = Mmqfl.from_unary_finite(ks)
            language = list(qfl.enumerate_length_less_than_n(2 * n))
            target = [[1] * k for k in ks]
            self.assertEqual(language, target)


if __name__ == '__main__':
    unittest.main()
