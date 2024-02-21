from functools import reduce

import numpy as np
import numpy.typing as npt

from .quantum_finite_automaton_base import TotalState
from .quantum_finite_automaton_base import Observable
from .quantum_finite_automaton_base import QuantumFiniteAutomatonBase
from .quantum_finite_automaton_base import NotClosedUnderOperationException

Mmqfa = 'MeasureManyQuantumFiniteAutomaton'


class MeasureManyQuantumFiniteAutomaton(QuantumFiniteAutomatonBase):
    def __init__(
        self,
        transition: npt.NDArray[np.cdouble],
        accepting_states: set[int],
        rejecting_states: set[int],
    ) -> None:
        super().__init__(transition)

        if not all(
            state < self.states for state
            in accepting_states | rejecting_states
        ):
            raise ValueError("Halting states must be a subset of states")
        if len(accepting_states & rejecting_states) != 0:
            raise ValueError("Accepting and rejecting states must be disjoint")

        self.accepting_states = accepting_states
        self.rejecting_states = rejecting_states

    @property
    def halting_states(self) -> set[int]:
        return self.accepting_states | self.rejecting_states

    @property
    def non_halting_states(self) -> set[int]:
        return set(range(self.states)) - self.halting_states

    @property
    def observable(self) -> Observable:
        return (
            self.accepting_states,
            self.rejecting_states,
            self.non_halting_states
        )

    def process(self, total_state: TotalState, w: list[int]) -> TotalState:
        return reduce(self.step, w, total_state)

    def step(self, total_state: TotalState, c: int) -> TotalState:
        total_state = total_state.apply(self.transition[c])
        total_state = total_state.measure_by(self.observable)
        return total_state

    def concatenation(self, other: Mmqfa) -> Mmqfa:
        raise NotImplementedError()

    def union(self, other: Mmqfa) -> Mmqfa:
        """Generally, MMQFA is not closed under the intersection.

        Maria Paola Bianchi and Beatrice Palano. 2010. Behaviours of Unary
        Quantum Automata. Fundamenta Informaticae.

        Raises: NotClosedUnderOperationError
        """
        raise NotClosedUnderOperationException()

    def intersection(self, other: Mmqfa) -> Mmqfa:
        """ Returns the intersection of two MMQFA.

        Generally, MMQFA is not closed under the intersection. However,
        end-decisive MMQFA with pure states are closed under the intersection.

        Maria Paola Bianchi and Beatrice Palano. 2010. Behaviours of Unary
        Quantum Automata. Fundamenta Informaticae.

        Raises: NotClosedUnderOperationError
        """
        # TODO: Implement the intersection of MMQFA for special cases
        raise NotClosedUnderOperationException()

    def complement(self: Mmqfa) -> Mmqfa:
        """Returns the complement of the quantum finite automaton.

        For a quantum finite automaton M, the complement is defined as the
        quantum finite automaton M' such that M'(w) = 1 - M(w) for all w.

        Alberto Bertoni, Carlo Mereghetti, and Beatrice Palano. 2003. Quantum
        Computing: 1-Way Quantum Automata. In Proceedings of the 8th
        International Conference on Developments in Language Theory (DLT'04).
        """
        return self.__class__(
            self.transition, self.rejecting_states, self.accepting_states)

    def linear_combination(self, other: Mmqfa, c: float) -> Mmqfa:
        """Returns the linear combination of two MMQFA.

        For a quantum finite automaton M, N and 0 <= c <= 1, the linear
        combination M' is an MMQFA such that M'(w) = c * M(w) + (1 - c) * N(w)
        for all w.

        Alberto Bertoni, Carlo Mereghetti, and Beatrice Palano. 2003. Quantum
        Computing: 1-Way Quantum Automata. In Proceedings of the 8th
        International Conference on Developments in Language Theory (DLT'04).
        """
        if not 0 <= c and c <= 1:
            raise ValueError("c must be in [0, 1]")

        # (U, V) -> [U 0; 0 V]
        def direct_sum(
            u: npt.NDArray[np.cdouble],
            v: npt.NDArray[np.cdouble]
        ) -> npt.NDArray[np.cdouble]:
            w1 = np.concatenate(
                (u, np.zeros((u.shape[0], v.shape[1]))), axis=1)
            w2 = np.concatenate(
                (np.zeros((v.shape[0], u.shape[1])), v), axis=1)
            w = np.concatenate((w1, w2), axis=0)
            return w

        f = np.sqrt(1 - c)
        d = np.sqrt(c)

        states = self.states + other.states
        initial_transition = np.eye(states)
        initial_transition[0][0] = d
        initial_transition[0][self.states] = f
        initial_transition[self.states][0] = -f
        initial_transition[self.states][self.states] = d
        transition = np.stack([
            direct_sum(u, v) for u, v in zip(self.transition, other.transition)
        ])
        transition[0] = transition[0] @ initial_transition
        accepting_states = (
            self.accepting_states
            | set(state + self.states for state in other.accepting_states)
        )
        rejecting_states = (
            self.rejecting_states
            | set(state + self.states for state in other.rejecting_states)
        )
        return self.__class__(transition, accepting_states, rejecting_states)

    def difference(self, other: Mmqfa) -> Mmqfa:
        raise NotImplementedError()

    def equivalence(self, other: Mmqfa) -> bool:
        raise NotImplementedError()

    def minimize(self: Mmqfa) -> Mmqfa:
        raise NotImplementedError()

    def symmetric_difference(self, other: Mmqfa) -> Mmqfa:
        raise NotImplementedError()

    def kleene_star(self: Mmqfa) -> Mmqfa:
        raise NotImplementedError()

    def kleene_plus(self: Mmqfa) -> Mmqfa:
        raise NotImplementedError()

    def reverse(self: Mmqfa) -> Mmqfa:
        raise NotImplementedError()

    def is_empty(self) -> bool:
        raise NotImplementedError()
