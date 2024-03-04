from functools import reduce
from typing import (TypeVar, Optional, )

import numpy as np
import numpy.typing as npt

from .quantum_finite_automaton_base import Observable
from .quantum_finite_automaton_base import QuantumFiniteAutomatonBase
from .quantum_finite_automaton_base import States
from .quantum_finite_automaton_base import TotalState
from .quantum_finite_automaton_base import Transition
from .quantum_finite_automaton_base import Transitions
from ..quantum_finite_automaton.measure_many_quantum_finite_automaton import (
    MeasureManyQuantumFiniteAutomaton as Mmqfa)
from .utils import direct_sum
from .utils import get_real_valued_transition

TMoqfa = TypeVar('TMoqfa', bound='MeasureOnceQuantumFiniteAutomaton')


def _get_bilinear_form(moqfa: TMoqfa) -> tuple[Transitions, States]:
    """Returns the (n^2) x (n^2) size bilinear form of the quantum finite
    automaton.

    In bilinear form ({U'(c)}_c, F') of ({U(c)}_c, F), |F' * U'(w)|^2 = F *
    U(w).

    Cristopher Moore and James P. Crutchfield. 2000. Quantum Automata and
    Quantum Grammars. Theoretical Computer Science (TCS'00).
    """

    transitions = np.stack([
        np.kron(transition.T.conj(), transition)
        for transition in moqfa.transitions
    ])
    accepting_states = np.kron(moqfa.accepting_states, moqfa.accepting_states)
    return transitions, accepting_states


def _get_stochastic_form(
    qfa: TMoqfa
) -> tuple[npt.NDArray[np.double], States]:
    """Returns the 2(n^2) x 2(n^2) size stochastic form of the quantum finite
    automaton.

    Roughly speaking, it represents the "quantum" transition matrix as a
    "stochastic" transition matrix.

    Cristopher Moore and James P. Crutchfield. 2000. Quantum Automata and
    Quantum Grammars. Theoretical Computer Science (TCS'00).
    """
    transitions, accepting_states = _get_bilinear_form(qfa)
    stochastic_transitions = np.stack([
        get_real_valued_transition(transition) for transition in transitions])
    states = len(accepting_states)
    stacked = np.stack([accepting_states] * 2)
    accpeting_states = stacked.T.reshape(2 * states)
    return stochastic_transitions, accpeting_states


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

        states = self.states + other.states
        f = np.sqrt(1 - c)
        d = np.sqrt(c)

        initial_transition = np.eye(states, dtype=np.cdouble)
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

        Alex Brodsky, and Nicholas Pippenger. 2002. Characterizations of 1-Way
        Quantum Finite Automata. SIAM Jornal on Computing 31.5.
        """
        if not all(c <= self.alphabet for c in w):
            raise ValueError(
                "w must be a string over the alphabet of the MOQFA")

        transitions = self.transitions.copy()
        transitions[self.start_of_string] = (
            self.word_transition(w) @ self.initial_transition)
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

    def to_real_valued(self: TMoqfa) -> TMoqfa:
        transitions = np.stack([
            get_real_valued_transition(transition)
            for transition in self.transitions
        ]).astype(complex)

        stacked_accepting = np.stack([self.accepting_states] * 2)
        accepting_states = stacked_accepting.T.reshape(2 * self.states)
        return self.__class__(transitions, accepting_states)

    def equivalence(self, other: TMoqfa) -> bool:
        """Returns whether the measure-many quantum finite automaton is equal.

        For quantum finite automata M and M', the equivalence is defined as
        whether M(w) = M'(w) for all w.

        See also counter_example().

        """
        return self.counter_example(other) is None

    def counter_example(self, other: TMoqfa) -> Optional[list[int]]:
        """Returns a counter example of the equivalence of the measure-many
        quantum finite automaton.

        For quantum finite automata M and M', the counter example is defined as
        a word w such that M(w) != M'(w).
        """
        if self.alphabet != other.alphabet:
            raise ValueError("Alphabets must be the same")
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

    def to_without_final_transition(self: TMoqfa) -> TMoqfa:
        """Returns the quantum finite automaton without the final transition.

        Alex Brodsky, and Nicholas Pippenger. 2002. Characterizations of 1-Way
        Quantum Finite Automata. SIAM Jornal on Computing 31.5.
        """

        initial_transition = self.final_transition @ self.initial_transition
        final_transition = np.eye(self.states, dtype=np.cdouble)
        self_final_inverse = self.final_transition.T.conj()

        def update_transition(transition: Transition) -> Transition:
            return self.final_transition @ transition @ self_final_inverse

        updated_transitions = [
            update_transition(transition)
            for transition in self.transitions[1:-1]
        ]

        transitions = np.stack(
            [initial_transition] + updated_transitions + [final_transition])
        return self.__class__(transitions, self.accepting_states)

    def to_without_initial_transition(self: TMoqfa) -> TMoqfa:
        """Returns the quantum finite automaton without the initial transition.

        Alex Brodsky, and Nicholas Pippenger. 2002. Characterizations of 1-Way
        Quantum Finite Automata. SIAM Jornal on Computing 31.5.
        """

        initial_transition = np.eye(self.states, dtype=np.cdouble)
        final_transition = self.final_transition @ self.initial_transition
        self_initial_inverse = self.initial_transition.T.conj()

        def update_transition(transition: Transition) -> Transition:
            return self_initial_inverse @ transition @ self.initial_transition

        updated_transitions = [
            update_transition(transition)
            for transition in self.transitions[1:-1]
        ]

        transitions = np.stack(
            [initial_transition] + updated_transitions + [final_transition])
        return self.__class__(transitions, self.accepting_states)

    def to_bilinear(self: TMoqfa) -> TMoqfa:
        """Returns the (n^2) x (n^2) size bilinear form of the quantum finite
        automaton.

        For a quantum finite automaton M' and a word w, the bilinear form M' of
        M is an MMQFA such that M'(w) = M(w)^2 for all w. Furthermore, the
        superposition of the M' is always real-valued.

        Cristopher Moore and James P. Crutchfield. 2000. Quantum Automata and
        Quantum Grammars. Theoretical Computer Science (TCS'00).
        """

        bilinear_transitions = np.stack([
            np.kron(transition.T.conj(), transition)
            for transition in self.transitions
        ])
        accepting_states = np.kron(
            self.accepting_states, self.accepting_states)
        return self.__class__(bilinear_transitions, accepting_states)

    def to_stochastic_form(self: TMoqfa) -> TMoqfa:
        """Returns the 2(n^2) x 2(n^2) size stochastic form of the quantum
        finite automaton.

        For a quantum finite automaton M' and a word w, the stochastic form M'
        of M is an MMQFA such that M'(w) = M(w)^2 for all w. Furthermore, the
        transitions of the M' is real-valued.

        Cristopher Moore and James P. Crutchfield. 2000. Quantum Automata and
        Quantum Grammars. Theoretical Computer Science (TCS'00). """

        return self.to_bilinear().to_real_valued()

