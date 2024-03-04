import math
from functools import reduce
from typing import (TypeVar, Optional, )

import numpy as np

from .quantum_finite_automaton_base import Observable
from .quantum_finite_automaton_base import QuantumFiniteAutomatonBase
from .quantum_finite_automaton_base import States
from .quantum_finite_automaton_base import TotalState
from .quantum_finite_automaton_base import Transition
from ..quantum_finite_automaton.measure_many_quantum_finite_automaton import (
    MeasureManyQuantumFiniteAutomaton as Mmqfa)
from .utils import direct_sum
from .utils import get_real_valued_transition

TMoqfa = TypeVar('TMoqfa', bound='MeasureOnceQuantumFiniteAutomaton')


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
        N(w)) for all w. Note that |M'| = |M||N|.

        See also intersection() and complement().
        """
        return ~(~self & ~other)

    def intersection(self: TMoqfa, other: TMoqfa) -> TMoqfa:
        """Returns the intersection of the measure-many quantum finite
        automaton.

        For a quantum finite automaton M and N, the intersection, also known as
        Hadamard product, is defined as the quantum finite automaton M' such
        that M'(w) = M(w) * N(w) for all w. Note that |M'| = |M||N|.

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
        quantum finite automaton M' such that M'(w) = 1 - M(w) for all w. Note
        that |M'| = |M|.

        Alberto Bertoni, Carlo Mereghetti, and Beatrice Palano. 2003. Quantum
        Computing: 1-Way Quantum Automata. In Proceedings of the 8th
        International Conference on Developments in Language Theory (DLT'04).
        """
        return self.__class__(self.transitions, self.rejecting_states)

    def linear_combination(
        self: TMoqfa, other: TMoqfa, c: float = 0.5
    ) -> TMoqfa:
        """Returns the linear combination of two measure-many quantum finite
        automata.

        For a quantum finite automaton M, N and 0 <= c <= 1, the linear
        combination M' is an MMQFA such that M'(w) = c * M(w) + (1 - c) * N(w)
        for all w. Note that |M'| = |M| + |N|.

        Alberto Bertoni, Carlo Mereghetti, and Beatrice Palano. 2003. Quantum
        Computing: 1-Way Quantum Automata. In Proceedings of the 8th
        International Conference on Developments in Language Theory (DLT'04).
        """
        if not 0 <= c and c <= 1:
            raise ValueError("c must be in [0, 1]")

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
        M with respect to u is an MMQFA such that M'(w) = M(uw) for all w. Note
        that |M'| = |M|.

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
        = M(phi(w)). Note that |M'| = |M|.

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
        """Returns a counter example of the equality of measure-once quantum
        finite automata.

        For quantum finite automata M and M', the counter example is a word w
        such that M(w) != M'(w). If there is no such word, it returns None.

        Lvzhou Li and Daowen Qiu. 2009. A note on quantum sequential machines.
        Theoretical Computer Science (TCS'09).
        """
        if self.alphabet != other.alphabet:
            raise ValueError("Alphabets must be the same")
        alphabet = self.alphabet

        # TODO: Storing every w is inefficient.
        # ws = [w_0, ..., w_N]
        ws: list[list[int]] = [[0]]
        # basis[i] = [U_{w_i}; V_{w_i}]; 1 x (n^2 + m^2)
        basis = [np.array([self.transitions[0], self.transitions[0]])]
        queue = [(w, basis_vector) for w, basis_vector in zip(ws, basis)]
        while len(queue) > 0:
            w, basis_vector = queue.pop()
            for c in range(1, alphabet+1):
                wc = w + [c]
                basis_vector_candidate = np.array([
                    self.transitions[c] @ basis_vector[0],
                    other.transitions[c] @ basis_vector[1]
                ])
                new_basis = np.concatenate([basis, [basis_vector_candidate]])
                S = np.linalg.svd(
                    new_basis.reshape((len(new_basis), -1)),
                    compute_uv=False
                )
                # basis_vector_candidate is not linearly independent
                if np.allclose(S[-1], 0):
                    continue
                queue.append((wc, basis_vector_candidate))
                basis = new_basis
                ws.append(wc)

        initial_total_state = TotalState.initial(self.states)

        def transition_to_acceptance(transition: Transition) -> float:
            total_state = initial_total_state.apply(transition)
            measured = total_state.measure_by(self.observable)
            return measured.acceptance

        self_transitions = (self.transitions[-1] @ vec[0] for vec in basis)
        other_transitions = (other.transitions[-1] @ vec[1] for vec in basis)
        xs = map(transition_to_acceptance, self_transitions)
        ys = map(transition_to_acceptance, other_transitions)

        for w, x, y in zip(ws, xs, ys):
            is_equal = math.isclose(x, y, abs_tol=1e-9)
            if not is_equal:
                return w[1:]
        return None

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
