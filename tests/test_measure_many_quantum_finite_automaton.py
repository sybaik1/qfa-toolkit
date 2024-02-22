import math
import unittest
from itertools import product

from .utils import get_measure_many_quantum_finite_automaton
from qfa_toolkit.quantum_finite_automaton import TotalState
from qfa_toolkit.quantum_finite_automaton import (
    NotClosedUnderOperationException)


class TestMeasureManyQuantumFiniteAutomaton(unittest.TestCase):
    def test_apply(self):
        for k in range(1, 8):
            r = 1 / k
            with self.subTest(k=k):
                mmqfa = get_measure_many_quantum_finite_automaton(r)
                for n in range(8):
                    w = [1] * n
                    target = r ** n
                    self.assertAlmostEqual(mmqfa(w), target)

    def test_complement(self):
        for k in range(1, 8):
            r = 1 / k
            with self.subTest(k=k):
                mmqfa = get_measure_many_quantum_finite_automaton(r)
                complement = ~mmqfa
                for n in range(8):
                    w = [1] * n
                    self.assertAlmostEqual(complement(w), 1-mmqfa(w))

    def test_linear_combination(self):
        for k, l, i in product(range(1, 4), range(1, 4), range(4)):
            r1 = 1 / k
            r2 = 1 / l
            m1 = get_measure_many_quantum_finite_automaton(r1)
            m2 = get_measure_many_quantum_finite_automaton(r2)
            c = i / 4
            combination = m1.linear_combination(m2, c)
            with self.subTest(k=k, l=l):
                for n in range(8):
                    w = [1] * n
                    target = c * m1(w) + (1 - c) * m2(w)
                    self.assertAlmostEqual(combination(w), target)

    def test_intersection(self):
        for k, l in product(range(1, 4), repeat=2):
            r1 = 1 / k
            r2 = 1 / l
            m1 = get_measure_many_quantum_finite_automaton(r1)
            m2 = get_measure_many_quantum_finite_automaton(r2)
            intersection = m1 & m2
            with self.subTest(k=k, l=l):
                for n in range(8):
                    w = [1] * n
                    target = m1(w) * m2(w)
                    self.assertAlmostEqual(intersection(w), target)

    def test_intersection_exception(self):
        r = 1 / 4
        m1 = get_measure_many_quantum_finite_automaton(r)
        m2 = get_measure_many_quantum_finite_automaton(r)
        with self.assertRaises(NotClosedUnderOperationException):
            ~m1 & m2

    def test_union(self):
        for k, l in product(range(1, 4), repeat=2):
            r1 = 1 / k
            r2 = 1 / l
            m1 = ~get_measure_many_quantum_finite_automaton(r1)
            m2 = ~get_measure_many_quantum_finite_automaton(r2)
            union = m1 | m2
            with self.subTest(k=k, l=l):
                for n in range(8):
                    w = [1] * n
                    target = 1 - (1 - m1(w)) * (1 - m2(w))
                    self.assertAlmostEqual(union(w), target)

    def test_union_exception(self):
        r = 1 / 4
        m1 = get_measure_many_quantum_finite_automaton(r)
        m2 = get_measure_many_quantum_finite_automaton(r)
        with self.assertRaises(NotClosedUnderOperationException):
            ~m1 | m2

    def test_end_decisive(self):
        r = 1 / 4
        m = get_measure_many_quantum_finite_automaton(r)
        total_state = TotalState.initial(m.states).apply(m.transition[0])
        self.assertAlmostEqual(total_state.acceptance, 0)
        for _ in range(8):
            total_state = TotalState.initial(m.states).apply(m.transition[1])
            self.assertAlmostEqual(total_state.acceptance, 0)
        self.assertTrue(m.is_end_decisive())
        self.assertFalse((~m).is_end_decisive())

    def test_co_end_decisive(self):
        r = 1 / 4
        m = ~get_measure_many_quantum_finite_automaton(r)
        total_state = TotalState.initial(m.states).apply(m.transition[0])
        self.assertAlmostEqual(total_state.rejection, 0)
        for _ in range(8):
            total_state = TotalState.initial(m.states).apply(m.transition[1])
            self.assertAlmostEqual(total_state.rejection, 0)
        self.assertTrue(m.is_co_end_decisive())
        self.assertFalse((~m).is_co_end_decisive())


if __name__ == '__main__':
    unittest.main()
