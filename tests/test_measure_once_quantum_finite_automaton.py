import math
import unittest
from functools import reduce
from itertools import product

from .utils import get_measure_once_quantum_finite_automaton


class TestMeasureOnceQuantumFiniteAutomaton(unittest.TestCase):
    def test_apply(self):
        for k in range(1, 8):
            with self.subTest(k=k):
                moqfa = get_measure_once_quantum_finite_automaton(k)
                for n in range(8):
                    w = [1] * n
                    theta = math.pi / k
                    target = math.cos(theta * n) ** 2
                    self.assertAlmostEqual(moqfa(w), target)

    def test_complement(self):
        for k in range(1, 8):
            with self.subTest(k=k):
                moqfa = get_measure_once_quantum_finite_automaton(k)
                complement = ~moqfa
                for n in range(8):
                    w = [1] * n
                    self.assertAlmostEqual(complement(w), 1-moqfa(w))

    def test_intersection(self):
        for k, l in product(range(1, 4), repeat=2):
            m1 = get_measure_once_quantum_finite_automaton(k)
            m2 = get_measure_once_quantum_finite_automaton(l)
            intersection = m1 & m2
            with self.subTest(k=k, l=l):
                for n in range(8):
                    w = [1] * n
                    target = m1(w) * m2(w)
                    self.assertAlmostEqual(intersection(w), target)

    def test_union(self):
        for k, l in product(range(1, 4), repeat=2):
            m1 = get_measure_once_quantum_finite_automaton(k)
            m2 = get_measure_once_quantum_finite_automaton(l)
            union = m1 | m2
            with self.subTest(k=k, l=l):
                for n in range(8):
                    w = [1] * n
                    target = (1 - m1(w)) * (1 - m2(w))
                    self.assertAlmostEqual(1 - union(w), target)

    def test_linear_combination(self):
        for k, l, i in product(range(1, 4), range(1, 4), range(4)):
            m1 = get_measure_once_quantum_finite_automaton(k)
            m2 = get_measure_once_quantum_finite_automaton(l)
            c = i / 4
            combination = m1.linear_combination(m2, c)
            with self.subTest(k=k, l=l):
                for n in range(8):
                    w = [1] * n
                    target = c * m1(w) + (1 - c) * m2(w)
                    self.assertAlmostEqual(combination(w), target)

    def test_word_quotient(self):
        for k in range(1, 8):
            moqfa = get_measure_once_quantum_finite_automaton(k)
            for m in range(8):
                u = [1] * m
                quotient = moqfa.word_quotient(u)
                with self.subTest(k=k, m=m):
                    for n in range(8):
                        w = [1] * n
                        target = moqfa(u + w)
                        self.assertAlmostEqual(quotient(w), target)

    def test_inverse_homomorphism(self):
        for k in range(1, 8):
            moqfa = get_measure_once_quantum_finite_automaton(k)
            for m in range(8):
                u = [1] * m
                phi = [[0], u]
                inverse_homomorphism = moqfa.inverse_homomorphism(phi)
                with self.subTest(k=k, m=m):
                    for n in range(8):
                        w = [1] * n
                        # phi: w |-> phi(w) = v
                        v = reduce(lambda v, c: v + phi[c], w, [])
                        target = moqfa(v)
                        self.assertAlmostEqual(inverse_homomorphism(w), target)


if __name__ == '__main__':
    unittest.main()