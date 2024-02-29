import unittest

from qfa_toolkit.quantum_finite_automaton import TotalState
from qfa_toolkit.quantum_finite_automaton import (
    NotClosedUnderOperationException)
from qfa_toolkit.quantum_finite_automaton import (
    MeasureManyQuantumFiniteAutomaton as Mmqfa)

from .utils import get_arbitrary_mmqfa
from .utils import test_qfa
from .utils import test_unary_operation
from .utils import test_binary_operation


class TestMeasureManyQuantumFiniteAutomaton(unittest.TestCase):

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
        test_unary_operation(
            self, Mmqfa.to_real_valued, lambda x: x,
            self.get_mmqfa, self.qfa_parameters, self.max_string_len
        )

    def test_intersection(self) -> None:
        test_binary_operation(
            self, Mmqfa.__and__, lambda x, y: x*y,
            self.get_end_decisive, self.get_end_decisive,
            self.qfa_parameters, self.qfa_parameters, self.max_string_len
        )

    def test_union(self) -> None:
        test_binary_operation(
            self, Mmqfa.__or__, lambda x, y: 1 - (1-x)*(1-y),
            self.get_co_end_decisive, self.get_co_end_decisive,
            self.qfa_parameters, self.qfa_parameters, self.max_string_len
        )

    def test_linear_combination(self) -> None:
        for i in range(4):
            c = i / 4

            def lin_comb(x: Mmqfa, y: Mmqfa) -> Mmqfa:
                return x.linear_combination(y, c)
            test_binary_operation(
                self, lin_comb, lambda x, y: c*x + (1-c)*y,
                self.get_mmqfa, self.get_mmqfa,
                self.qfa_parameters, self.qfa_parameters, self.max_string_len
            )

    def test_end_decisive(self) -> None:
        r = 1 / 4
        m = get_arbitrary_mmqfa(r)
        total_state = TotalState.initial(m.states).apply(m.transitions[0])
        self.assertAlmostEqual(total_state.acceptance, 0)
        for _ in range(8):
            total_state = TotalState.initial(m.states).apply(m.transitions[1])
            self.assertAlmostEqual(total_state.acceptance, 0)
        self.assertTrue(m.is_end_decisive())
        self.assertFalse((~m).is_end_decisive())

    def test_co_end_decisive(self) -> None:
        r = 1 / 4
        m = ~get_arbitrary_mmqfa(r)
        total_state = TotalState.initial(m.states).apply(m.transitions[0])
        self.assertAlmostEqual(total_state.rejection, 0)
        for _ in range(8):
            total_state = TotalState.initial(m.states).apply(m.transitions[1])
            self.assertAlmostEqual(total_state.rejection, 0)
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


if __name__ == '__main__':
    unittest.main()
