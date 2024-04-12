import unittest
from qfa_toolkit.quantum_finite_state_automaton_language import (
    MeasureOnceQuantumFiniteStateAutomatonLanguage as Moqfl)


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


if __name__ == '__main__':
    unittest.main()
