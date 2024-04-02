import unittest

from qfa_toolkit.recognition_strategy import IsolatedCutPoint
from qfa_toolkit.recognition_strategy import NegativeOneSidedBoundedError
from qfa_toolkit.recognition_strategy import PositiveOneSidedBoundedError

from .utils import get_arbitrary_moqfa
from qfa_toolkit.quantum_finite_automaton_language import (
    QuantumFiniteAutomatonLanguageBase as Qfl)
from qfa_toolkit.quantum_finite_automaton_language import (
    MeasureOnceQuantumFiniteAutomatonLanguage as Moqfl)


class TestMeasureOnceuantumFiniteAutomatonLanguage(unittest.TestCase):
    def test_measure_once_quantum_finite_automaton_language(self):
        # M(a^{4p+q}) is 1 if q = 0; 1/2 if q = 1, 3; 0 if q = 2
        moqfa = get_arbitrary_moqfa(4)

        isolated_cut_point = IsolatedCutPoint(1/4, 1/8)
        neg_one_sided = NegativeOneSidedBoundedError(3/4)
        pos_one_sided = PositiveOneSidedBoundedError(3/4)
        invalid_isolated_cut_point = IsolatedCutPoint(1/2, 1/8)

        # { a^{4p+q} }, q = 0, 1, 3
        lang_isolated_cut_point = Qfl(moqfa, isolated_cut_point)
        # { a^{4p+q} }, q = 0
        lang_neg_one_sided = Qfl(moqfa, neg_one_sided)
        # { a^{4p+q} }, q = 0, 1, 3
        lang_pos_one_sided = Qfl(moqfa, pos_one_sided)
        # Invalid for { a^{4p+q} }, q = 1, 3
        lang_invalid = Qfl(moqfa, invalid_isolated_cut_point)

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

    def test_modulo_prime(self):
        for p in [23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]:
            moqfl = Moqfl.from_modulo_prime(p)
            for j in range(2*p):
                w = [1] * j
                if j % p == 0:
                    self.assertIn(w, moqfl)
                else:
                    self.assertNotIn(w, moqfl)


if __name__ == '__main__':
    unittest.main()
