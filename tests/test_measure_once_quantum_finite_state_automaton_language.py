import unittest
from qfa_toolkit.quantum_finite_state_automaton_language import (
    MeasureOnceQuantumFiniteStateAutomatonLanguage as Moqfl,
    MeasureManyQuantumFiniteStateAutomatonLanguage as Mmqfl,
)


class TestMeasureOnceQuantumFiniteStateAutomatonLanguage(unittest.TestCase):

    def test_modulo(self):
        for n in range(2, 10):
            moqfl = Moqfl.from_modulo(n)
            for j in range(10 * n):
                w = [1] * j
                self.assertEqual(len(w) % n == 0, w in moqfl, f"n={n}, j={j}")

    def test_modulo_prime(self):
        for p in [3, 5, 7, 11, 13, 17, 19]:
            moqfl = Moqfl.from_modulo_prime(p)
            for j in range(10 * p):
                w = [1] * j
                self.assertEqual(len(w) % p == 0, w in moqfl)

    def test_intersection_and_union(self):
        mod_3 = Moqfl.from_modulo(3)
        mod_5 = Moqfl.from_modulo(5)
        mod_15_lin = mod_3 & mod_5
        mod_15_had = mod_3.intersection(mod_5, use_linear_combination=False)
        mod_3_or_5 = mod_3 | mod_5
        for j in range(31):
            w = [1] * j
            self.assertEqual(j % 15 == 0, w in mod_15_lin, f"j={j}")
            self.assertEqual(j % 15 == 0, w in mod_15_had, f"j={j}")
            self.assertEqual(
                (j % 3 == 0) or (j % 5 == 0), w in mod_3_or_5, f"j={j}"
            )

    def test_composed_modulo_and_singleton(self):
        mod_3 = Moqfl.from_modulo(3)
        mod_5 = Moqfl.from_modulo(5)
        mod_15 = mod_3.intersection(mod_5, use_linear_combination=False)
        mm_mod_15 = mod_15.to_measure_many_quantum_finite_state_automaton_language()
        equ_4 = Mmqfl.from_unary_singleton(4)
        composed = mm_mod_15 | equ_4
        for j in range(31):
            w = [1] * j
            self.assertEqual(
                (j % 15 == 0) or (j == 4), w in composed, f"j={j}"
            )


if __name__ == '__main__':
    unittest.main()
