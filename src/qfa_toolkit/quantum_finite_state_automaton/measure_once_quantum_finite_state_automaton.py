from collections import deque
import math
from functools import reduce
from typing import (TypeVar, Optional, Union, overload, )

import numpy as np
import numpy.typing as npt
from .quantum_finite_state_automaton_base import (
    NotClosedUnderOperationException)
from .quantum_finite_state_automaton_base import Observable
from .quantum_finite_state_automaton_base import (
    QuantumFiniteStateAutomatonBase)
from .quantum_finite_state_automaton_base import States
from .quantum_finite_state_automaton_base import TotalState
from .quantum_finite_state_automaton_base import Transition
from ..quantum_finite_state_automaton.measure_many_quantum_finite_state_automaton import (
    MeasureManyQuantumFiniteStateAutomaton as Mmqfa)
from .utils import direct_sum
from .utils import get_real_valued_transition
from .utils import get_transition_from_initial_to_superposition
from .utils import mapping_to_transition

MoqfaT = TypeVar('MoqfaT', bound='MeasureOnceQuantumFiniteStateAutomaton')


class MeasureOnceQuantumFiniteStateAutomaton(QuantumFiniteStateAutomatonBase):
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

    def step(self, total_state: TotalState, c: int) -> TotalState:
        return total_state.apply(self.transitions[c])

    def __call__(self, w):
        tape = self.string_to_tape(w)
        return self.process(tape).measure_by(self.observable).acceptance

    def bilinearized_call(self, w: list[int]) -> float:
        tape = self.string_to_tape(w)
        last_superposition = self.process(tape).superposition
        return float(np.sum(self.observable[0] * last_superposition).real)

    def word_transition(self, w: list[int]) -> Transition:
        transition = reduce(
            lambda transition, c: self.transitions[c] @ transition,
            w, np.eye(self.states, dtype=np.cdouble)
        )
        return transition

    @overload
    def union(self: MoqfaT, other: MoqfaT) -> MoqfaT: ...

    @overload
    def union(self, other: Mmqfa) -> Mmqfa: ...

    def union(
        self: MoqfaT, other: Union[MoqfaT, Mmqfa]
    ) -> Union[MoqfaT, Mmqfa]:
        """Returns the mn-size union of two quantum finite automata.

        For a quantum finite automaton M and N, the union is defined as the
        quantum finite automaton M' such that 1 - M'(w) = (1 - M(w)) * (1 -
        N(w)) for all w.

        Supports MOQFA | MOQFA (returning an MOQFA) and MOQFA | MMQFA for
        co-end-decisive MMQFAs (returning an mn-state co-end-decisive MMQFA).

        See also intersection().
        """
        if isinstance(other, Mmqfa):
            if self.alphabet != other.alphabet:
                raise ValueError("Alphabets must be the same")
            if not other.is_co_end_decisive():
                raise NotClosedUnderOperationException()
            return ~((~self).intersection(~other))
        return ~((~self).intersection(~other))

    @overload
    def __or__(self: MoqfaT, other: MoqfaT) -> MoqfaT: ...

    @overload
    def __or__(self, other: Mmqfa) -> Mmqfa: ...

    def __or__(
        self: MoqfaT, other: Union[MoqfaT, Mmqfa]
    ) -> Union[MoqfaT, Mmqfa]:
        """Returns the union of the quantum finite automaton.

        See union() for details.
        """
        if isinstance(other, Mmqfa):
            return self.union(other)
        return self.union(other)

    @overload
    def intersection(self: MoqfaT, other: MoqfaT) -> MoqfaT: ...

    @overload
    def intersection(self, other: Mmqfa) -> Mmqfa: ...

    def intersection(
        self: MoqfaT, other: Union[MoqfaT, Mmqfa]
    ) -> Union[MoqfaT, Mmqfa]:
        """Returns the mn-size intersection of two quantum finite automata.

        For a quantum finite automaton M and N, the intersection, also known as
        Hadamard product, is defined as the quantum finite automaton M' such
        that M'(w) = M(w) * N(w) for all w.

        Supports MOQFA & MOQFA (returning an MOQFA) and MOQFA & MMQFA for
        end-decisive MMQFAs (returning an mn-state end-decisive MMQFA).

        Alberto Bertoni, Carlo Mereghetti, and Beatrice Palano. 2003. Quantum
        Computing: 1-Way Quantum Automata. In Proceedings of the 8th
        International Conference on Developments in Language Theory (DLT'04).
        """
        if isinstance(other, Mmqfa):
            if self.alphabet != other.alphabet:
                raise ValueError("Alphabets must be the same")
            if not other.is_end_decisive():
                raise NotClosedUnderOperationException()
            transitions = np.stack([
                np.kron(u, v) for u, v
                in zip(self.transitions, other.transitions)
            ])
            non_halting_states = np.kron(
                np.ones(self.states, dtype=bool),
                other.non_halting_states
            )
            halting_states = ~non_halting_states
            accepting_states = np.kron(
                self.accepting_states,
                other.accepting_states
            )
            rejecting_states = halting_states & ~accepting_states
            return Mmqfa(transitions, accepting_states, rejecting_states)

        transitions = np.stack([
            np.kron(u, v) for u, v
            in zip(self.transitions, other.transitions)
        ])
        accepting_states = np.kron(
            self.accepting_states,
            other.accepting_states
        )

        return self.__class__(transitions, accepting_states)

    @overload
    def __and__(self: MoqfaT, other: MoqfaT) -> MoqfaT: ...

    @overload
    def __and__(self, other: Mmqfa) -> Mmqfa: ...

    def __and__(
        self: MoqfaT, other: Union[MoqfaT, Mmqfa]
    ) -> Union[MoqfaT, Mmqfa]:
        """Returns the intersection of the quantum finite automaton.

        See intersection() for details.
        """
        if isinstance(other, Mmqfa):
            return self.intersection(other)
        return self.intersection(other)

    def complement(self: MoqfaT) -> MoqfaT:
        """Returns the complement of the measure-once quantum finite automaton.

        For a quantum finite automaton M, the complement is defined as the
        quantum finite automaton M' such that M'(w) = 1 - M(w) for all w.

        Alberto Bertoni, Carlo Mereghetti, and Beatrice Palano. 2003. Quantum
        Computing: 1-Way Quantum Automata. In Proceedings of the 8th
        International Conference on Developments in Language Theory (DLT'04).
        """
        return self.__class__(self.transitions, self.rejecting_states)

    def __invert__(self: MoqfaT) -> MoqfaT:
        """Returns the complement of the quantum finite automaton.

        See complement() for details.
        """
        return self.complement()

    @classmethod
    def linear_combination(
        cls: type[MoqfaT],
        *moqfas: MoqfaT,
        coefficients: Optional[list[float]] = None
    ) -> MoqfaT:
        """Returns the linear combination of the measure-once quantum finite
        automata.

        For quantum finite automata M, N and 0 <= c <= 1, the linear
        combination M' is an MOQFA such that M'(w) = c * M(w) + (1 - c) * N(w)
        for all w.

        Alberto Bertoni, Carlo Mereghetti, and Beatrice Palano. 2003. Quantum
        Computing: 1-Way Quantum Automata. In Proceedings of the 8th
        International Conference on Developments in Language Theory (DLT'04).
        """
        if len(moqfas) == 0:
            raise ValueError("moqfas must be non-empty")

        alphabet = moqfas[0].alphabet
        if not all(moqfa.alphabet == alphabet for moqfa in moqfas):
            raise ValueError("moqfas must have the same alphabet")

        if coefficients is None:
            coefficients = [1/len(moqfas)] * len(moqfas)

        if len(moqfas) != len(coefficients):
            raise ValueError("moqfas and coefficients must have the same size")

        if not math.isclose(sum(coefficients), 1):
            raise ValueError("coefficients must sum to 1")

        initial_superposition = np.concatenate([
            [math.sqrt(coefficient)] + [0] * (moqfa.states - 1)
            for coefficient, moqfa in zip(coefficients, moqfas)
        ])
        initial_transition = get_transition_from_initial_to_superposition(
            initial_superposition)
        transitions = np.stack([
            reduce(direct_sum, sigma_transitions)
            for sigma_transitions
            in zip(*[moqfa.transitions for moqfa in moqfas])
        ])
        transitions[cls.start_of_string] = (
            transitions[cls.start_of_string] @ initial_transition)
        accepting_states = np.concatenate(
            [moqfa.accepting_states for moqfa in moqfas])
        return cls(transitions, accepting_states)

    def word_quotient(self: MoqfaT, w: list[int]) -> MoqfaT:
        """Returns the word quotient of the measure-once quantum finite
        automaton.

        For a quantum finite automaton M and a word w, the word quotient M' of
        M with respect to u is an MOQFA M' such that M'(w) = M(uw) for all w.
        """
        if not all(1 <= c <= self.alphabet for c in w):
            raise ValueError(
                "w must be a string over the alphabet of the MOQFA")

        transitions = self.transitions.copy()
        transitions[self.start_of_string] = (
            self.word_transition(w) @ self.initial_transition)
        return self.__class__(transitions, self.accepting_states)

    def inverse_homomorphism(self: MoqfaT, phi: list[list[int]]) -> MoqfaT:
        """Returns the inverse homomorphism of the measure-once quantum finite
        automaton.

        For a quantum finite automaton M and a homomorphism phi, the inverse
        homomorphism M' of M with respect to phi is an MOQFA M' such that M'(w)
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

    def to_measure_many_quantum_finite_state_automaton(self) -> Mmqfa:
        moqfa = self.to_without_final_transition()
        transitions = np.stack([
            direct_sum(transition, transition)
            for transition in moqfa.transitions
        ])
        mapping = {state: state + self.states for state in range(self.states)}
        mapping.update(
            {state + self.states: state for state in range(self.states)})
        final_transition = mapping_to_transition(mapping)
        transitions[moqfa.end_of_string] = final_transition
        accepting_states = np.concatenate(
            [np.zeros(self.states, dtype=bool), moqfa.accepting_states]
        )
        rejecting_states = np.concatenate(
            [np.zeros(self.states, dtype=bool), moqfa.rejecting_states]
        )
        return Mmqfa(transitions, accepting_states, rejecting_states)

    def to_without_final_transition(self: MoqfaT) -> MoqfaT:
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

    def to_without_initial_transition(self: MoqfaT) -> MoqfaT:
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

    def to_real_valued(self: MoqfaT) -> MoqfaT:
        """Returns the 2n-size real-valued form of the quantum finite
        automaton.

        Cristopher Moore and James P. Crutchfield. 2000. Quantum Automata and
        Quantum Grammars. Theoretical Computer Science (TCS'00).
        """
        transitions = np.stack([
            get_real_valued_transition(transition)
            for transition in self.transitions
        ]).astype(np.cdouble)

        stacked_accepting = np.stack([self.accepting_states] * 2)
        accepting_states = stacked_accepting.T.reshape(2 * self.states)
        return self.__class__(transitions, accepting_states)

    def bilinearize(self: MoqfaT) -> MoqfaT:
        return self.to_bilinear()

    def to_bilinear(self: MoqfaT) -> MoqfaT:
        """Returns the (n^2)-size bilinear form of the quantum finite
        automaton.

        For a quantum finite automaton M, the bilinear form M' of M is an
        automaton such that M(w) is the sum of amplitude of the accepting
        states at the end of the computation of M'.

        Cristopher Moore and James P. Crutchfield. 2000. Quantum Automata and
        Quantum Grammars. Theoretical Computer Science (TCS'00).
        """

        bilinear_transitions = np.stack([
            np.kron(transition.conj(), transition)
            for transition in self.transitions
        ])
        accepting_states = np.zeros(self.states ** 2, dtype=bool)
        indices = self.accepting_states.nonzero()[0]
        accepting_states[[i * self.states + i for i in indices]] = True
        return self.__class__(bilinear_transitions, accepting_states)

    def to_stochastic(self: MoqfaT) -> MoqfaT:
        """Returns the 2(n^2)-size stochastic form of the quantum
        finite automaton.

        For a quantum finite automaton M, the bilinear form M' of M is an
        automaton such that M(w) is the sum of amplitude of the accepting
        states at the end of the computation of M'. Furthermore, the
        transitions of the M' is real-valued.

        Cristopher Moore and James P. Crutchfield. 2000. Quantum Automata and
        Quantum Grammars. Theoretical Computer Science (TCS'00). """

        return self.bilinearize().to_real_valued()

    def counter_example(
        self: MoqfaT, other: 'MeasureOnceQuantumFiniteStateAutomaton'
    ) -> Optional[list[int]]:
        """Returns a counter example of the equivalence of the measure-once
        quantum finite automaton.

        For quantum finite automata M and M', the counter example is defined as
        a word w such that M(w) != M'(w).

        Lvzhou Li and Daowen Qiu. 2009. A note on quantum sequential machines.
        Theoretical Computer Science (TCS'09).
        """
        if self.alphabet != other.alphabet:
            raise ValueError("Alphabets must be the same")

        b1 = self.to_bilinear()
        b2 = other.to_bilinear()
        d1 = b1.states

        def diff(v: npt.NDArray[np.cdouble]) -> complex:
            end1 = b1.final_transition @ v[:d1]
            end2 = b2.final_transition @ v[d1:]
            val1 = np.sum(b1.accepting_states * end1)
            val2 = np.sum(b2.accepting_states * end2)
            return complex(val1 - val2)

        v0 = np.concatenate(
            [b1.initial_transition[:, 0], b2.initial_transition[:, 0]]
        ).astype(np.cdouble)
        if abs(diff(v0)) > 1e-9:
            return []

        basis: list[npt.NDArray[np.cdouble]] = [
            v0 / float(np.linalg.norm(v0))
        ]
        queue: deque[tuple[list[int], npt.NDArray[np.cdouble]]] = deque(
            [([], v0)]
        )
        while queue:
            w, v = queue.popleft()
            for c in range(1, self.alphabet + 1):
                wc = w + [c]
                v_next = np.concatenate([
                    b1.transitions[c] @ v[:d1],
                    b2.transitions[c] @ v[d1:]
                ]).astype(np.cdouble)
                r = v_next
                for _ in range(2):
                    for b in basis:
                        r = r - np.vdot(b, r) * b
                norm_r = float(np.linalg.norm(r))
                if norm_r <= 1e-9:
                    continue
                if abs(diff(v_next)) > 1e-9:
                    return wc
                basis.append(r / norm_r)
                queue.append((wc, v_next))
        return None

    def equivalence(
        self: MoqfaT, other: 'MeasureOnceQuantumFiniteStateAutomaton'
    ) -> bool:
        """Returns whether the two measure-once quantum finite automata are
        equal.

        For quantum finite automata M and M', the equivalence is defined as
        whether M(w) = M'(w) for all w.

        See also counter_example().
        """
        return self.counter_example(other) is None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MeasureOnceQuantumFiniteStateAutomaton):
            return False
        if self.alphabet != other.alphabet:
            return False
        return self.equivalence(other)
