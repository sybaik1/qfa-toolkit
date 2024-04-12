import unittest
import math
from itertools import combinations
from itertools import product

from qfa_toolkit.quantum_finite_state_automaton_language import (
    MeasureManyQuantumFiniteStateAutomatonLanguage as Mmqfl)


class TestMeasureManyQuantumFiniteStateAutomatonLanguage(unittest.TestCase):
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
        n = 12
        for k in range(n):
            qfl = Mmqfl.from_unary_singleton(k)
            language = list(qfl.enumerate_length_less_than_n(2 * n))
            target = [[1] * k]
            self.assertEqual(language, target)

    def test_union(self):
        n = 5
        for k1, k2 in product(range(n), repeat=2):
            qfl_1 = Mmqfl.from_unary_singleton(k1)
            qfl_2 = Mmqfl.from_unary_singleton(k2)
            qfl = qfl_1 | qfl_2
            language = list(qfl.enumerate_length_less_than_n(2 * n))
            if k1 == k2:
                target = [[1] * k1]
            else:
                target = sorted([[1] * k1, [1] * k2], key=lambda x: len(x))
            self.assertEqual(language, target)

    def test_from_unary_finite(self):
        n = 5
        for ks in combinations(range(n), 3):
            qfl = Mmqfl.from_unary_finite(ks)
            language = list(qfl.enumerate_length_less_than_n(2 * n))
            target = [[1] * k for k in ks]
            self.assertEqual(language, target)

    def test_intersection(self):
        n = 5
        comb = list(combinations(range(n), 3))
        for ks, ls in zip(comb, comb):
            ms = sorted(list(set(ks) & set(ls)))
            qfl_1 = Mmqfl.from_unary_finite(ks)
            qfl_2 = Mmqfl.from_unary_finite(ls)
            qfl = qfl_1 & qfl_2
            language = list(qfl.enumerate_length_less_than_n(2 * n))
            target = [[1] * m for m in ms]
            self.assertEqual(language, target)


if __name__ == '__main__':
    unittest.main()
