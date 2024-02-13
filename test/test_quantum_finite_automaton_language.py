import unittest

import math
import numpy as np

import src.quantum_finite_automaton as qfa
import src.quantum_finite_automaton_language as qfal

from src.recognition_strategy import IsolatedCutPoint
from src.recognition_strategy import NegativeOneSidedBoundedError

Mmqfa = qfa.MeasureManyQuantumFiniteAutomaton
MmqfaLang = qfal.MeasureManyQuantumFiniteAutomatonLanguage


class TestMeasureManyQuantumFiniteAutomatonLanguage(unittest.TestCase):
    def test_measure_many_quantum_finite_automaton_language(self):
        # M(a^n) = (sin^2(theta))^n, theta = pi / 4

        theta = math.pi / 4
        a, b = (math.cos(theta), math.sin(theta))
        acceptings = {2, }
        rejectings = {1, }
        transition = np.array([
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            [
                [a, b, 0],
                [-b, a, 0],
                [0, 0, 1],
            ],
            [
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
            ],
        ], dtype=np.cfloat)
        qfa = Mmqfa(transition, acceptings, rejectings)
        isolated_cut_point = IsolatedCutPoint(0.325, 0.0625)
        neg_one_sided = NegativeOneSidedBoundedError(0.75)
        invalid_neg_one_sided = NegativeOneSidedBoundedError(0.1875)

        # { a^n }, n <= 2
        lang_isolated_cut_point = MmqfaLang(qfa, isolated_cut_point)
        # { a^n }, n <= 0
        lang_neg_one_sided = MmqfaLang(qfa, neg_one_sided)
        # invalid for { a, aa } (whose probabilities are 0.5, 0.25)
        lang_invalid = MmqfaLang(qfa, invalid_neg_one_sided)

        self.assertIn([], lang_isolated_cut_point)
        self.assertIn([1], lang_isolated_cut_point)
        self.assertNotIn([1, 1], lang_isolated_cut_point)
        self.assertNotIn([1, 1, 1], lang_isolated_cut_point)

        self.assertIn([], lang_neg_one_sided)
        self.assertNotIn([1], lang_neg_one_sided)
        self.assertNotIn([1, 1], lang_neg_one_sided)
        self.assertNotIn([1, 1, 1], lang_neg_one_sided)

        self.assertIn([], lang_invalid)
        self.assertRaises(ValueError, lambda: [1] in lang_invalid)
        self.assertRaises(ValueError, lambda: [1, 1] in lang_invalid)
        self.assertNotIn([1, 1, 1], lang_invalid)


if __name__ == '__main__':
    unittest.main()
