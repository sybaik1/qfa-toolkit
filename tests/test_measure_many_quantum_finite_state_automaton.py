import functools
import math
import unittest

import numpy as np

from qfa_toolkit.quantum_finite_state_automaton import (
    NotClosedUnderOperationException)
from qfa_toolkit.quantum_finite_state_automaton import (
    MeasureManyQuantumFiniteStateAutomaton as Mmqfa)

from .utils import get_arbitrary_mmqfa
from .utils import test_qfa
from .utils import test_unary_operation
from .utils import test_binary_operation
from .utils import test_total_state_during_process


class TestMeasureManyQuantumFiniteStateAutomaton(unittest.TestCase):

    def setUp(self) -> None:
        self.get_mmqfa = get_arbitrary_mmqfa
        self.get_end_decisive = self.get_mmqfa
        self.get_co_end_decisive = (
            lambda r: ~get_arbitrary_mmqfa(r))

        self.qfa_parameters = [1 / k for k in range(1, 8)]
        self.max_string_len = 8

    def test_apply(self) -> None:
        test_qfa(
            self, self.get_mmqfa, lambda r, w: r ** len(w),
            self.qfa_parameters, self.max_string_len
        )

    def test_complement(self) -> None:
        test_unary_operation(
            self, Mmqfa.__invert__, lambda x: 1-x,
            self.get_mmqfa, self.qfa_parameters, self.max_string_len
        )

    def test_real_valued(self) -> None:
        def constraint(m: Mmqfa) -> bool:
            return np.allclose(m.transitions.imag, 0)
        test_unary_operation(
            self, Mmqfa.to_real_valued, lambda x: x,
            self.get_mmqfa, self.qfa_parameters, self.max_string_len
        )

    def test_intersection(self) -> None:
        test_binary_operation(
            self, Mmqfa.__and__, lambda x, y: x*y,
            self.get_end_decisive, self.get_end_decisive,
            self.qfa_parameters, self.qfa_parameters, self.max_string_len,
            constraint=Mmqfa.is_end_decisive
        )

    def test_union(self) -> None:
        test_binary_operation(
            self, Mmqfa.__or__, lambda x, y: 1 - (1-x)*(1-y),
            self.get_co_end_decisive, self.get_co_end_decisive,
            self.qfa_parameters, self.qfa_parameters, self.max_string_len,
            constraint=Mmqfa.is_co_end_decisive
        )

    def test_linear_combination(self) -> None:
        for c in [i / 4 for i in range(4)]:
            def lin_comb(x: Mmqfa, y: Mmqfa) -> Mmqfa:
                return Mmqfa.linear_combination(x, y, coefficients=[c, 1-c])
            test_binary_operation(
                self, lin_comb, lambda x, y: c*x + (1-c)*y,
                self.get_mmqfa, self.get_co_end_decisive,
                self.qfa_parameters, self.qfa_parameters, self.max_string_len
            )

    def test_end_decisive(self) -> None:
        def constraint(tape, i, total_state) -> bool:
            if i < len(tape) - 1:
                return math.isclose(total_state.acceptance, 0, abs_tol=1e-6)
            return True
        test_total_state_during_process(
            self, self.get_end_decisive, self.qfa_parameters,
            self.max_string_len, constraint
        )
        m = self.get_end_decisive(1/4)
        self.assertTrue(m.is_end_decisive())
        self.assertFalse((~m).is_end_decisive())

    def test_co_end_decisive(self) -> None:
        def constraint(tape, i, total_state) -> bool:
            if i < len(tape) - 1:
                return math.isclose(total_state.rejection, 0, abs_tol=1e-6)
            return True
        test_total_state_during_process(
            self, self.get_co_end_decisive, self.qfa_parameters,
            self.max_string_len, constraint
        )
        m = self.get_co_end_decisive(1/4)
        self.assertTrue(m.is_co_end_decisive())
        self.assertFalse((~m).is_co_end_decisive())

    def test_intersection_exception(self) -> None:
        with self.assertRaises(NotClosedUnderOperationException):
            test_binary_operation(
                self, Mmqfa.__and__, lambda x, y: x*y,
                self.get_co_end_decisive, self.get_mmqfa,
                self.qfa_parameters, self.qfa_parameters, self.max_string_len
            )

    def test_union_exception(self) -> None:
        with self.assertRaises(NotClosedUnderOperationException):
            test_binary_operation(
                self, Mmqfa.__or__, lambda x, y: 1 - (1-x) * (1-y),
                self.get_end_decisive, self.get_mmqfa,
                self.qfa_parameters, self.qfa_parameters, self.max_string_len
            )

    def test_word_quotient(self) -> None:
        for u in [[1] * n for n in range(4)]:
            def get_quotient(m: Mmqfa) -> Mmqfa:
                q = m.word_quotient(u)
                expected_states = (
                    m.states + len(u) * int(np.sum(m.halting_states))
                )
                self.assertEqual(q.states, expected_states)
                return q

            def get_preimage_str(
                w: list[int], prefix: list[int] = u
            ) -> list[int]:
                return prefix + w

            test_unary_operation(
                self, get_quotient, lambda x: x,
                self.get_end_decisive, [1/2, 1/3], 6,
                get_preimage_str=get_preimage_str,
                constraint=Mmqfa.is_end_decisive
            )
            test_unary_operation(
                self, get_quotient, lambda x: x,
                self.get_co_end_decisive, [1/2, 1/3], 6,
                get_preimage_str=get_preimage_str,
                constraint=Mmqfa.is_co_end_decisive
            )

    def test_inverse_homomorphism(self) -> None:
        for phi in [[[0], []], [[0], [1]], [[0], [1, 1]], [[0], [1, 1, 1]]]:
            def get_inv_hom(m: Mmqfa) -> Mmqfa:
                h_m = m.inverse_homomorphism(phi)
                c_h = max(len(img) for img in phi[1:])
                expected_max = m.states + c_h * int(np.sum(m.halting_states))
                self.assertLessEqual(h_m.states, expected_max)
                return h_m

            def get_preimage_str(
                w: list[int], mapping: list[list[int]] = phi
            ) -> list[int]:
                u: list[int] = functools.reduce(
                    lambda v, c: v + mapping[c], w, []
                )
                return u

            test_unary_operation(
                self, get_inv_hom, lambda x: x,
                self.get_end_decisive, [1/2, 1/3], 5,
                get_preimage_str=get_preimage_str,
                constraint=Mmqfa.is_end_decisive
            )
            test_unary_operation(
                self, get_inv_hom, lambda x: x,
                self.get_co_end_decisive, [1/2, 1/3], 5,
                get_preimage_str=get_preimage_str,
                constraint=Mmqfa.is_co_end_decisive
            )

    def test_semantic_end_decisive_halted_rotation(self) -> None:
        # Non-halting state 0 moves on symbol a to rejecting state 1, while
        # already-halted state 1 rotates to accepting state 2 (which is never
        # reached under T_{#w} because P_non zeroes state 1 immediately).
        u_sos = np.eye(3, dtype=np.cdouble)
        u_a = np.array(
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.cdouble
        )
        u_eos = np.array(
            [[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.cdouble
        )
        mm = Mmqfa(
            np.stack([u_sos, u_a, u_eos]),
            np.array([False, False, True]),
            np.array([False, True, False]),
        )
        self.assertTrue(mm.is_end_decisive())
        self.assertFalse(mm.is_co_end_decisive())
        self.assertTrue((~mm).is_co_end_decisive())
        self.assertAlmostEqual(mm([]), 1.0)
        self.assertAlmostEqual(mm([1]), 0.0)
        self.assertAlmostEqual(mm([1, 1]), 0.0)


if __name__ == '__main__':
    unittest.main()
