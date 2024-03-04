import math
import functools
import unittest

from qfa_toolkit.quantum_finite_automaton import (
    MeasureOnceQuantumFiniteAutomaton as Moqfa)

from .utils import get_arbitrary_moqfa
from .utils import test_qfa
from .utils import test_unary_operation
from .utils import test_binary_operation


class TestMeasureOnceQuantumFiniteAutomaton(unittest.TestCase):

    def setUp(self):
        self.get_moqfa = get_arbitrary_moqfa
        self.qfa_parameters = list(range(1, 8))
        self.max_string_len = 8

    def test_apply(self):
        test_qfa(
            self, self.get_moqfa,
            lambda k, w: math.cos(math.pi / k * len(w)) ** 2,
            self.qfa_parameters, self.max_string_len
        )

    def test_complement(self):
        test_unary_operation(
            self, Moqfa.__invert__, lambda x: 1-x,
            self.get_moqfa, self.qfa_parameters, self.max_string_len
        )

    def test_intersection(self):
        test_binary_operation(
            self, Moqfa.__and__, lambda x, y: x*y,
            self.get_moqfa, self.get_moqfa,
            self.qfa_parameters, self.qfa_parameters, self.max_string_len
        )

    def test_union(self):
        test_binary_operation(
            self, Moqfa.__or__, lambda x, y: 1 - (1-x)*(1-y),
            self.get_moqfa, self.get_moqfa,
            self.qfa_parameters, self.qfa_parameters, self.max_string_len
        )

    def test_linear_combination(self):
        for c in [i / 4 for i in range(4)]:
            def lin_comb(x, y):
                return x.linear_combination(y, c)
            test_binary_operation(
                self, lin_comb, lambda x, y: c*x + (1-c)*y,
                self.get_moqfa, self.get_moqfa,
                self.qfa_parameters, self.qfa_parameters, self.max_string_len
            )

    def test_word_quotient(self):
        for u in [[1] * n for n in range(8)]:
            def get_quotient(m: Moqfa) -> Moqfa:
                return m.word_quotient(u)
            test_unary_operation(
                self, get_quotient, lambda x: x,
                self.get_moqfa, self.qfa_parameters, self.max_string_len,
                lambda w: u + w
            )

    def test_inverse_homomorphism(self):
        # phi: [0] |-> [0] and [1] |-> [1] * n
        for phi in [[[0], [1] * n] for n in range(8)]:
            def get_inverse_homomorphism(m: Moqfa) -> Moqfa:
                return m.inverse_homomorphism(phi)

            def get_preimage_str(w: list[int]) -> list[int]:
                u: list[int] = functools.reduce(lambda v, c: v + phi[c], w, [])
                return u

            test_unary_operation(
                self, get_inverse_homomorphism, lambda x: x,
                self.get_moqfa, self.qfa_parameters, self.max_string_len,
                get_preimage_str
            )


if __name__ == '__main__':
    unittest.main()
