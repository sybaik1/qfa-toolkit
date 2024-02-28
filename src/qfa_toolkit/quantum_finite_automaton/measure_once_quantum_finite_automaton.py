from functools import reduce
from itertools import product

import numpy as np
import numpy.typing as npt

from .quantum_finite_automaton_base import TotalState
from .quantum_finite_automaton_base import Observable
from .quantum_finite_automaton_base import QuantumFiniteAutomatonBase
from ..quantum_finite_automaton.measure_many_quantum_finite_automaton import (
    MeasureManyQuantumFiniteAutomaton as Mmqfa)

Moqfa = 'MeasureOnceQuantumFiniteAutomaton'


class MeasureOnceQuantumFiniteAutomaton(QuantumFiniteAutomatonBase):
    def __init__(
        self,
        transitions: npt.NDArray[np.cdouble],
        accepting_states: set[int]
    ) -> None:
        super().__init__(transitions)

        if not all(state < self.states for state in accepting_states):
            raise ValueError("Accepting states must be a subset of states")

        self.accepting_states = accepting_states

    @property
    def rejecting_states(self) -> set[int]:
        return set(range(self.states)) - self.accepting_states

    @property
    def observable(self) -> Observable:
        return (self.accepting_states, self.rejecting_states, set())

    def process(self, total_state: TotalState, w: list[int]) -> TotalState:
        total_state = reduce(self.step, w, total_state)
        return total_state.measure_by(self.observable)

    def step(self, total_state: TotalState, c: int) -> TotalState:
        return total_state.apply(self.transitions[c])

    def word_transition(self, w: list[int]) -> npt.NDArray[np.cfloat]:
        transition = reduce(
            lambda transition, c: self.transitions[c] @ transition,
            w, np.eye(self.states)
        )
        return transition

    def concatenation(self, other: Moqfa) -> Moqfa:
        raise NotImplementedError()

    def union(self, other: Moqfa) -> Moqfa:
        """Returns the union of the measure-many quantum finite automaton.

        For a quantum finite automaton M and N, the union is defined as the
        quantum finite automaton M' such that 1 - M'(w) = (1 - M(w)) * (1 -
        N(w)) for all w.
        """
        return ~(~self & ~other)

    def intersection(self, other: Moqfa) -> Moqfa:
        """Returns the intersection of the measure-many quantum finite
        automaton.

        For a quantum finite automaton M and N, the intersection, also known as
        Hadamard product, is defined as the quantum finite automaton M' such
        that M'(w) = M(w) * N(w) for all w.

        Alberto Bertoni, Carlo Mereghetti, and Beatrice Palano. 2003. Quantum
        Computing: 1-Way Quantum Automata. In Proceedings of the 8th
        International Conference on Developments in Language Theory (DLT'04).
        """
        transitions = np.stack([
            np.kron(u, v) for u, v
            in zip(self.transitions, other.transitions)
        ])
        accepting_states = set(
            i * other.states + j for i, j
            in product(self.accepting_states, other.accepting_states)
        )

        return self.__class__(transitions, accepting_states)

    def complement(self: Moqfa) -> Moqfa:
        """Returns the complement of the measure-many quantum finite automaton.

        For a quantum finite automaton M, the complement is defined as the
        quantum finite automaton M' such that M'(w) = 1 - M(w) for all w.

        Alberto Bertoni, Carlo Mereghetti, and Beatrice Palano. 2003. Quantum
        Computing: 1-Way Quantum Automata. In Proceedings of the 8th
        International Conference on Developments in Language Theory (DLT'04).
        """
        return self.__class__(self.transitions, self.rejecting_states)

    def linear_combination(self, other: Moqfa, c: float) -> Moqfa:
        """Returns the linear combination of two measure-many quantum finite
        automata.

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

        states = self.states + other.states
        f = np.sqrt(1 - c)
        d = np.sqrt(c)

        initial_transition = np.eye(states)
        initial_transition[0][0] = d
        initial_transition[0][self.states] = f
        initial_transition[self.states][0] = -f
        initial_transition[self.states][self.states] = d

        transitions = np.stack([
            direct_sum(u, v)
            for u, v in zip(self.transitions, other.transitions)
        ])
        transitions[self.start_of_string] = (
            transitions[self.start_of_string] @ initial_transition)
        accepting_states = (
            self.accepting_states
            | set(state + self.states for state in other.accepting_states)
        )
        return self.__class__(transitions, accepting_states)

    def word_quotient(self, w: list[int]) -> Moqfa:
        """Returns the word quotient of the measure-many quantum finite
        automaton.

        For a quantum finite automaton M and a word w, the word quotient M' of
        M with respect to u is an MMQFA M' such that M'(w) = M(uw) for all w.

        Alex Brodsky, and Nicholas Pippenger. 2002. Characterazations of 1-Way
        Quantum Finite Automata. SIAM Jornal on Computing 31.5.
        """
        if not all(c <= self.alphabet for c in w):
            raise ValueError(
                "w must be a string over the alphabet of the MOQFA")

        transitions = self.transitions.copy()
        transitions[self.start_of_string] = (
            self.word_transition(w) @ transitions[self.start_of_string])
        return self.__class__(transitions, self.accepting_states)

    def inverse_homomorphism(self, phi: list[list[int]]) -> Moqfa:
        """Returns the inverse homomorphism of the measure-many quantum finite
        automaton.

        For a quantum finite automaton M and a homomorphism phi, the inverse
        homomorphism M' of M with respect to phi is an MMQFA M' such that M'(w)
        = M(phi(w)).

        Cristopher Moore and James P. Crutchfield. 2000. Quantum Automata and
        Quantum Grammars. Theoretical Computer Science (TCS'00).
        """
        if len(phi) == 0:
            raise ValueError("phi must be non-empty")
        if phi[self.start_of_string] != [self.start_of_string]:
            raise ValueError("phi[start_of_string] must be [start_of_string]")

        alphabet = len(phi) - 1
        image = max(map(lambda v: max(v, default=0), phi))
        if not image <= self.alphabet:
            raise ValueError(
                "The co-domain of phi must be the set of words over the "
                "alphabet of the automaton"
            )
        transitions = np.stack(
            [self.word_transition(phi[c]) for c in range(alphabet+1)]
            + [self.transitions[self.end_of_string]]
        )
        return self.__class__(transitions, self.accepting_states)

    def difference(self, other: Moqfa) -> Moqfa:
        raise NotImplementedError()

    def equivalence(self, other: Moqfa) -> bool:
        """Returns whether the measure-many quantum finite automaton is equal.

        For quantum finite automata M and M', the equivalence is defined as
        whether M(w) = M'(w) for all w.

        Alex Brodsky, and Nicholas Pippenger. 2002. Characterazations of 1-Way
        Quantum Finite Automata. SIAM Jornal on Computing 31.5.
        """
        raise NotImplementedError()

    def minimize(self: Moqfa) -> Moqfa:
        raise NotImplementedError()

    def symmetric_difference(self, other: Moqfa) -> Moqfa:
        raise NotImplementedError()

    def kleene_star(self: Moqfa) -> Moqfa:
        raise NotImplementedError()

    def kleene_plus(self: Moqfa) -> Moqfa:
        raise NotImplementedError()

    def reverse(self: Moqfa) -> Moqfa:
        raise NotImplementedError()

    def is_empty(self) -> bool:
        raise NotImplementedError()

    def to_measure_many_quantum_finite_automaton(self) -> Mmqfa:
        raise NotImplementedError()
