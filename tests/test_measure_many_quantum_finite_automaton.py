import math
import unittest
from itertools import product

from .utils import get_measure_many_quantum_finite_automaton
from qfa_toolkit.quantum_finite_automaton import (
    NotClosedUnderOperationException)


class TestMeasureManyQuantumFiniteAutomaton(unittest.TestCase):
    def test_apply(self):
        for k in range(1, 8):
            with self.subTest(k=k):
                mmqfa = get_measure_many_quantum_finite_automaton(k)
                for n in range(8):
                    w = [1] * n
                    theta = math.pi / k
                    target = (math.cos(theta) ** 2) ** n
                    self.assertAlmostEqual(mmqfa(w), target)

    def test_complement(self):
        for k in range(1, 8):
            with self.subTest(k=k):
                mmqfa = get_measure_many_quantum_finite_automaton(k)
                complement = ~mmqfa
                for n in range(8):
                    w = [1] * n
                    self.assertAlmostEqual(complement(w), 1-mmqfa(w))

    def test_linear_combination(self):
        for k, l, i in product(range(1, 4), range(1, 4), range(4)):
            m1 = get_measure_many_quantum_finite_automaton(k)
            m2 = get_measure_many_quantum_finite_automaton(l)
            c = i / 4
            combination = m1.linear_combination(m2, c)
            with self.subTest(k=k, l=l):
                for n in range(8):
                    w = [1] * n
                    target = c * m1(w) + (1 - c) * m2(w)
                    self.assertAlmostEqual(combination(w), target)

    def test_intersection(self):
        m1 = get_measure_many_quantum_finite_automaton(1)
        m2 = get_measure_many_quantum_finite_automaton(1)
        with self.assertRaises(NotClosedUnderOperationException):
            m1 & m2

    def test_union(self):
        m1 = get_measure_many_quantum_finite_automaton(1)
        m2 = get_measure_many_quantum_finite_automaton(1)
        with self.assertRaises(NotClosedUnderOperationException):
            m1 | m2


if __name__ == '__main__':
    unittest.main()
