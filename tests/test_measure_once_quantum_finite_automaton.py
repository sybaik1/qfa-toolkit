import math
import functools
import itertools
import unittest

import numpy as np

from qfa_toolkit.quantum_finite_automaton import (
    MeasureOnceQuantumFiniteAutomaton as Moqfa)

from .utils import get_arbitrary_moqfa
from .utils import test_qfa
from .utils import test_unary_operation
from .utils import test_binary_operation
from .utils import test_total_state_during_process
from .utils import multiply_arbitrary_global_phase


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

    def test_to_real_valued(self) -> None:
        def constraint(m: Moqfa) -> bool:
            return np.allclose(m.transitions.imag, 0)
        test_unary_operation(
            self, Moqfa.to_real_valued, lambda x: x,
            self.get_moqfa, self.qfa_parameters, self.max_string_len,
            constraint=constraint
        )

    def test_to_bilinear(self) -> None:
        def constraint(tape, i, total_state) -> bool:
            return np.allclose(total_state.superposition.imag, 0)
        test_total_state_during_process(
            self, lambda e: self.get_moqfa(e).to_bilinear(),
            self.qfa_parameters, self.max_string_len, constraint
        )
        test_unary_operation(
            self, Moqfa.to_bilinear, lambda x: x ** 2,
            self.get_moqfa, self.qfa_parameters, self.max_string_len
        )

    def test_to_without_initial_transition(self):
        def constraint(m: Moqfa) -> bool:
            return np.allclose(m.initial_transition, np.eye(m.states))
        test_unary_operation(
            self, Moqfa.to_without_initial_transition, lambda x: x,
            self.get_moqfa, self.qfa_parameters, self.max_string_len
        )

    def test_to_without_final_transition(self):
        def constraint(m: Moqfa) -> bool:
            return np.allclose(m.final_transition, np.eye(m.states))
        test_unary_operation(
            self, Moqfa.to_without_final_transition, lambda x: x,
            self.get_moqfa, self.qfa_parameters, self.max_string_len
        )

    def test_word_quotient(self):
        for u in [[1] * n for n in range(8)]:
            def get_quotient(m: Moqfa) -> Moqfa:
                return m.word_quotient(u)
            test_unary_operation(
                self, get_quotient, lambda x: x,
                self.get_moqfa, self.qfa_parameters, self.max_string_len,
                get_preimage_str=lambda w: u + w
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
                get_preimage_str=get_preimage_str
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

    def test_counter_example(self):
        for k1, k2 in itertools.product(self.qfa_parameters, repeat=2):
            m1 = self.get_moqfa(k1)
            m2 = self.get_moqfa(k2).to_bilinear()
            m2.transitions = multiply_arbitrary_global_phase(m2.transitions)
            counter_examples = [m1.counter_example(m2), m2.counter_example(m1)]
            for counter_example in counter_examples:
                if counter_example is not None:
                    self.assertNotAlmostEqual(
                        m1(counter_example), m2(counter_example))

    def test_equivalence(self):
        return
        for k1, k2 in itertools.product(self.qfa_parameters, repeat=2):
            m1 = self.get_moqfa(k1)
            m2 = self.get_moqfa(k2)
            m2.transitions = multiply_arbitrary_global_phase(m2.transitions)
            self.assertEqual(m1 == m2, k1 == k2)
            self.assertEqual(m2 == m1, k2 == k1)


if __name__ == '__main__':
    unittest.main()
