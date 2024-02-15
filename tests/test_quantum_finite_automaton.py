import math
import unittest
from itertools import product

from .utils import get_measure_once_quantum_finite_automaton
from .utils import get_measure_many_quantum_finite_automaton


class TestQuantumFiniteAutomaton(unittest.TestCase):
    def test_measure_once_quantum_finite_automaton(self):
        for k, n in product(range(1, 8), range(8)):
            moqfa = get_measure_once_quantum_finite_automaton(k)
            with self.subTest(k=k, n=n):
                w = [1] * n
                theta = math.pi / k
                target = math.cos(theta * n) ** 2
                self.assertAlmostEqual(moqfa(w), target)

    def test_measure_many_quantum_finite_automaton(self):
        for k, n in product(range(1, 8), range(8)):
            mmqfa = get_measure_many_quantum_finite_automaton(k)
            with self.subTest(k=k, n=n):
                w = [1] * n
                theta = math.pi / k
                target = (math.cos(theta) ** 2) ** n
                self.assertAlmostEqual(mmqfa(w), target)


if __name__ == '__main__':
    unittest.main()
