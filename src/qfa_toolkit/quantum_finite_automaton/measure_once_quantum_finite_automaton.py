from functools import reduce
from itertools import product
from typing import (TypeVar, )

import numpy as np
import numpy.typing as npt

from .quantum_finite_automaton_base import TotalState
from .quantum_finite_automaton_base import QuantumFiniteAutomatonBase
from .quantum_finite_automaton_base import (Observable, Transition, States, )
from ..quantum_finite_automaton.measure_many_quantum_finite_automaton import (
    MeasureManyQuantumFiniteAutomaton as Mmqfa)

TMoqfa = TypeVar('TMoqfa', bound='MeasureOnceQuantumFiniteAutomaton')


def _get_binlinear_form(moqfa: TMoqfa) -> tuple[Transition, set[int]]:
    """Returns the (n^2) x (n^2) size binlinear form of the quantum finite
    automaton.

    Cristopher Moore and James P. Crutchfield. 2000. Quantum Automata and
    Quantum Grammars. Theoretical Computer Science (TCS'00).
    """

    transitions = np.stack([
        np.kron(transition.T.conj(), transition)
        for transition in moqfa.transitions
    ])
    accepting_states = set(
        i * moqfa.states + j for i, j
        in product(moqfa.accepting_states, moqfa.accepting_states)
    )

    return transitions, accepting_states


def _get_real_valued_form(transition: Transition) -> npt.NDArray[np.double]:
    raise NotImplementedError()


def _get_stochastic_form(
    qfa: TMoqfa
) -> tuple[npt.NDArray[np.double], set[int]]:
    """Returns the 2(n^2) x 2(n^2) size stochastic form of the quantum finite
    automaton.

    Roughly speaking, it represents the "quantum" transition matrix as a
    "stochastic" transition matrix.

    Cristopher Moore and James P. Crutchfield. 2000. Quantum Automata and
    Quantum Grammars. Theoretical Computer Science (TCS'00).
    """
    raise NotImplementedError()


class MeasureOnceQuantumFiniteAutomaton(QuantumFiniteAutomatonBase):
    def __init__(
        self,
        transitions: Transition,
        accepting_states: States
    ) -> None:
        super().__init__(transitions)

        if len(accepting_states) != self.states:
            raise ValueError(
                "accepting_states must have the same size as states")
        self.accepting_states = accepting_states

    @property
    def rejecting_states(self) -> States:
        return ~self.accepting_states

    @property
    def observable(self) -> Observable:
        return np.stack([
            self.accepting_states,
            self.rejecting_states,
            np.zeros((self.states), dtype=bool)
        ])

    def process(self, total_state: TotalState, w: list[int]) -> TotalState:
        total_state = reduce(self.step, w, total_state)
        return total_state.measure_by(self.observable)

    def step(self, total_state: TotalState, c: int) -> TotalState:
        return total_state.apply(self.transitions[c])

    def word_transition(self, w: list[int]) -> Transition:
        transition = reduce(
            lambda transition, c: self.transitions[c] @ transition,
            w, np.eye(self.states, dtype=complex)
        )
        return transition

    def concatenation(self, other: TMoqfa) -> TMoqfa:
        raise NotImplementedError()

    def union(self: TMoqfa, other: TMoqfa) -> TMoqfa:
        """Returns the union of the measure-many quantum finite automaton.

        For a quantum finite automaton M and N, the union is defined as the
        quantum finite automaton M' such that 1 - M'(w) = (1 - M(w)) * (1 -
        N(w)) for all w.
        """
        return ~(~self & ~other)

    def intersection(self: TMoqfa, other: TMoqfa) -> TMoqfa:
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
        accepting_states = np.kron(
            self.accepting_states,
            other.accepting_states
        )

        return self.__class__(transitions, accepting_states)

    def complement(self: TMoqfa) -> TMoqfa:
        """Returns the complement of the measure-many quantum finite automaton.

        For a quantum finite automaton M, the complement is defined as the
        quantum finite automaton M' such that M'(w) = 1 - M(w) for all w.

        Alberto Bertoni, Carlo Mereghetti, and Beatrice Palano. 2003. Quantum
        Computing: 1-Way Quantum Automata. In Proceedings of the 8th
        International Conference on Developments in Language Theory (DLT'04).
        """
        return self.__class__(self.transitions, self.rejecting_states)

    def linear_combination(self: TMoqfa, other: TMoqfa, c: float) -> TMoqfa:
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
        def direct_sum(u: Transition, v: Transition) -> Transition:
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
        accepting_states = np.concatenate(
            (self.accepting_states, other.accepting_states))
        return self.__class__(transitions, accepting_states)

    def word_quotient(self: TMoqfa, w: list[int]) -> TMoqfa:
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

    def inverse_homomorphism(self: TMoqfa, phi: list[list[int]]) -> TMoqfa:
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

    def difference(self, other: TMoqfa) -> TMoqfa:
        raise NotImplementedError()

    def equivalence(self, other: TMoqfa) -> bool:
        """Returns whether the measure-many quantum finite automaton is equal.

        For quantum finite automata M and M', the equivalence is defined as
        whether M(w) = M'(w) for all w.

        """
        raise NotImplementedError()

    def minimize(self: TMoqfa) -> TMoqfa:
        raise NotImplementedError()

    def symmetric_difference(self, other: TMoqfa) -> TMoqfa:
        raise NotImplementedError()

    def kleene_star(self: TMoqfa) -> TMoqfa:
        raise NotImplementedError()

    def kleene_plus(self: TMoqfa) -> TMoqfa:
        raise NotImplementedError()

    def reverse(self: TMoqfa) -> TMoqfa:
        raise NotImplementedError()

    def is_empty(self) -> bool:
        raise NotImplementedError()

    def to_measure_many_quantum_finite_automaton(self) -> Mmqfa:
        raise NotImplementedError()
