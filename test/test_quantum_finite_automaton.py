import unittest

import math
import numpy as np

import src.quantum_finite_automaton as qfa

Mmqfa = qfa.MeasureManyQuantumFiniteAutomaton


class TestMeasureManyQuantumFiniteAutomaton(unittest.TestCase):
    def test_measure_many_quantum_finite_automaton(self):
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
        M = Mmqfa(transition, acceptings, rejectings)
        self.assertAlmostEqual(M([]), 1)
        self.assertAlmostEqual(M([1]), 1/2)
        self.assertAlmostEqual(M([1, 1]), 1/4)
        self.assertAlmostEqual(M([1, 1, 1]), 1/8)
        self.assertAlmostEqual(M([1, 1, 1, 1]), 1/16)
        self.assertAlmostEqual(M([1, 1, 1, 1, 1]), 1/32)
        self.assertAlmostEqual(M([1, 1, 1, 1, 1, 1]), 1/64)


if __name__ == '__main__':
    unittest.main()
