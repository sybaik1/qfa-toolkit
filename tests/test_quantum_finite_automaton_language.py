import unittest

from qfa_toolkit.recognition_strategy import IsolatedCutPoint
from qfa_toolkit.recognition_strategy import NegativeOneSidedBoundedError
from qfa_toolkit.recognition_strategy import PositiveOneSidedBoundedError

from .utils import get_measure_once_quantum_finite_automaton
from .utils import get_measure_many_quantum_finite_automaton
from qfa_toolkit.quantum_finite_automaton_language import (
    QuantumFiniteAutomatonLanguage as QfaLanguage)


class TestMeasureManyQuantumFiniteAutomatonLanguage(unittest.TestCase):
    def test_measure_once_quantum_finite_automaton_language(self):
        # M(a^{4p+q}) is 1 if q = 0; 1/2 if q = 1, 3; 0 if q = 2
        moqfa = get_measure_once_quantum_finite_automaton(4)

        isolated_cut_point = IsolatedCutPoint(1/4, 1/8)
        neg_one_sided = NegativeOneSidedBoundedError(3/4)
        pos_one_sided = PositiveOneSidedBoundedError(3/4)
        invalid_isolated_cut_point = IsolatedCutPoint(1/2, 1/8)

        # { a^{4p+q} }, q = 0, 1, 3
        lang_isolated_cut_point = QfaLanguage(moqfa, isolated_cut_point)
        # { a^{4p+q} }, q = 0
        lang_neg_one_sided = QfaLanguage(moqfa, neg_one_sided)
        # { a^{4p+q} }, q = 0, 1, 3
        lang_pos_one_sided = QfaLanguage(moqfa, pos_one_sided)
        # Invalid for { a^{4p+q} }, q = 1, 3
        lang_invalid = QfaLanguage(moqfa, invalid_isolated_cut_point)

        self.assertIn([], lang_isolated_cut_point)
        self.assertIn([1], lang_isolated_cut_point)
        self.assertNotIn([1, 1], lang_isolated_cut_point)
        self.assertIn([1, 1, 1], lang_isolated_cut_point)
        self.assertIn([1, 1, 1, 1], lang_isolated_cut_point)

        self.assertIn([], lang_neg_one_sided)
        self.assertNotIn([1], lang_neg_one_sided)
        self.assertNotIn([1, 1], lang_neg_one_sided)
        self.assertNotIn([1, 1, 1], lang_neg_one_sided)
        self.assertIn([1, 1, 1, 1], lang_neg_one_sided)

        self.assertIn([], lang_pos_one_sided)
        self.assertIn([1], lang_pos_one_sided)
        self.assertNotIn([1, 1], lang_pos_one_sided)
        self.assertIn([1, 1, 1], lang_pos_one_sided)
        self.assertIn([1, 1, 1, 1], lang_pos_one_sided)

        self.assertIn([], lang_invalid)
        self.assertRaises(ValueError, lambda: [1] in lang_invalid)
        self.assertNotIn([1, 1], lang_invalid)
        self.assertRaises(ValueError, lambda: [1, 1, 1] in lang_invalid)
        self.assertIn([1, 1, 1, 1], lang_invalid)

    def test_measure_many_quantum_finite_automaton_language(self):
        # M(a^n) = (1/2)^n
        mmqfa = get_measure_many_quantum_finite_automaton(4)

        isolated_cut_point = IsolatedCutPoint(3/16, 1/32)
        neg_one_sided = NegativeOneSidedBoundedError(3/4)
        invalid_neg_one_sided = NegativeOneSidedBoundedError(1/16)

        # { a^n }, n <= 2
        lang_isolated_cut_point = QfaLanguage(mmqfa, isolated_cut_point)
        # { a^n }, n <= 0
        lang_neg_one_sided = QfaLanguage(mmqfa, neg_one_sided)
        # Invalid for { a^n } 0 < n < 4,
        # since 1/16 < p < 1 for probabilities p.
        lang_invalid = QfaLanguage(mmqfa, invalid_neg_one_sided)

        self.assertIn([], lang_isolated_cut_point)
        self.assertIn([1], lang_isolated_cut_point)
        self.assertIn([1, 1], lang_isolated_cut_point)
        self.assertNotIn([1, 1, 1], lang_isolated_cut_point)

        self.assertIn([], lang_neg_one_sided)
        self.assertNotIn([1], lang_neg_one_sided)
        self.assertNotIn([1, 1], lang_neg_one_sided)
        self.assertNotIn([1, 1, 1], lang_neg_one_sided)

        self.assertIn([], lang_invalid)
        self.assertRaises(ValueError, lambda: [1] in lang_invalid)
        self.assertRaises(ValueError, lambda: [1, 1] in lang_invalid)
        self.assertRaises(ValueError, lambda: [1, 1, 1] in lang_invalid)


if __name__ == '__main__':
    unittest.main()
