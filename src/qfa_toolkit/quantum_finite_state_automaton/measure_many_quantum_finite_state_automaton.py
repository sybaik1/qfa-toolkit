from collections import deque
import math
from typing import TYPE_CHECKING, TypeVar, Optional, Union, overload
from functools import reduce

import numpy as np
import numpy.typing as npt

from .quantum_finite_state_automaton_base import NotClosedUnderOperationException
from .quantum_finite_state_automaton_base import Observable
from .quantum_finite_state_automaton_base import QuantumFiniteStateAutomatonBase
from .quantum_finite_state_automaton_base import States
from .quantum_finite_state_automaton_base import TotalState
from .quantum_finite_state_automaton_base import Transition
from .utils import direct_sum
from .utils import get_real_valued_transition
from .utils import get_transition_from_initial_to_superposition

if TYPE_CHECKING:
    from .measure_once_quantum_finite_state_automaton import (
        MeasureOnceQuantumFiniteStateAutomaton)

MmqfaT = TypeVar('MmqfaT', bound='MeasureManyQuantumFiniteStateAutomaton')


class MeasureManyQuantumFiniteStateAutomaton(QuantumFiniteStateAutomatonBase):
    def __init__(
        self,
        transition: Transition,
        accepting_states: States,
        rejecting_states: States,
    ) -> None:
        super().__init__(transition)

        if not len(accepting_states) == self.states:
            raise ValueError(
                "accepting_states must be the same size as states")
        if not len(rejecting_states) == self.states:
            raise ValueError(
                "rejecting_states must be the same size as states")
        if np.any(accepting_states & rejecting_states):
            raise ValueError("Accepting and rejecting states must be disjoint")

        self.accepting_states = accepting_states
        self.rejecting_states = rejecting_states

    @property
    def halting_states(self) -> States:
        return self.accepting_states | self.rejecting_states

    @property
    def non_halting_states(self) -> States:
        return ~self.halting_states

    @property
    def observable(self) -> Observable:
        return np.stack([
            self.accepting_states,
            self.rejecting_states,
            self.non_halting_states
        ])

    def step(self, total_state: TotalState, c: int) -> TotalState:
        total_state = total_state.apply(self.transitions[c])
        total_state = total_state.measure_by(self.observable)
        return total_state

    def __call__(self, w):
        tape = self.string_to_tape(w)
        return self.process(tape).acceptance

    @overload
    def union(self: MmqfaT, other: MmqfaT) -> MmqfaT: ...

    @overload
    def union(
        self: MmqfaT, other: 'MeasureOnceQuantumFiniteStateAutomaton'
    ) -> MmqfaT: ...

    def union(
        self: MmqfaT,
        other: Union[MmqfaT, 'MeasureOnceQuantumFiniteStateAutomaton']
    ) -> MmqfaT:
        """Returns the union of two quantum finite automata.

        For a quantum finite automaton M and N, the union is defined as the
        quantum finite automaton M' such that 1 - M'(w) = (1 - M(w)) * (1 -
        N(w)) for all w.

        Generally, MMQFA is not closed under the union. However, co-end-decisive
        MMQFAs are closed under union, including mixed union with an MOQFA.

        Maria Paola Bianchi and Beatrice Palano. 2010. Behaviours of Unary
        Quantum Automata. Fundamenta Informaticae.

        Raises: NotClosedUnderOperationException
        """
        if self.alphabet != other.alphabet:
            raise ValueError("Alphabets must be the same")

        if isinstance(other, MeasureManyQuantumFiniteStateAutomaton):
            if not self.is_co_end_decisive() or not other.is_co_end_decisive():
                raise NotClosedUnderOperationException()
        else:
            if not self.is_co_end_decisive():
                raise NotClosedUnderOperationException()

        return ~((~self) & (~other))

    @overload
    def __or__(self: MmqfaT, other: MmqfaT) -> MmqfaT: ...

    @overload
    def __or__(
        self: MmqfaT, other: 'MeasureOnceQuantumFiniteStateAutomaton'
    ) -> MmqfaT: ...

    def __or__(
        self: MmqfaT,
        other: Union[MmqfaT, 'MeasureOnceQuantumFiniteStateAutomaton']
    ) -> MmqfaT:
        """Returns the union of the quantum finite automaton.

        See union() for details.
        """
        return self.union(other)

    @overload
    def intersection(self: MmqfaT, other: MmqfaT) -> MmqfaT: ...

    @overload
    def intersection(
        self: MmqfaT, other: 'MeasureOnceQuantumFiniteStateAutomaton'
    ) -> MmqfaT: ...

    def intersection(
        self: MmqfaT,
        other: Union[MmqfaT, 'MeasureOnceQuantumFiniteStateAutomaton']
    ) -> MmqfaT:
        """Returns the intersection of two quantum finite automata.

        For a quantum finite automaton M and N, the intersection is defined as
        the quantum finite automaton M' such that M'(w) = M(w) * N(w) for all
        w.

        Generally, MMQFA is not closed under the intersection. However,
        end-decisive MMQFAs are closed under intersection (including mixed
        intersection with an MOQFA, which halts only on end-of-string).

        Maria Paola Bianchi and Beatrice Palano. 2010. Behaviours of Unary
        Quantum Automata. Fundamenta Informaticae.

        Raises: NotClosedUnderOperationException
        """
        if self.alphabet != other.alphabet:
            raise ValueError("Alphabets must be the same")

        if isinstance(other, MeasureManyQuantumFiniteStateAutomaton):
            if not self.is_end_decisive() or not other.is_end_decisive():
                raise NotClosedUnderOperationException()
            other_non_halting = other.non_halting_states
        else:
            if not self.is_end_decisive():
                raise NotClosedUnderOperationException()
            other_non_halting = np.ones(other.states, dtype=bool)

        transition = np.stack([
            np.kron(u, v) for u, v
            in zip(self.transitions, other.transitions)
        ])
        non_halting_states = np.kron(
            self.non_halting_states, other_non_halting)
        halting_states = ~non_halting_states
        accepting_states = np.kron(
            self.accepting_states, other.accepting_states)
        rejecting_states = halting_states & ~accepting_states
        return self.__class__(transition, accepting_states, rejecting_states)

    @overload
    def __and__(self: MmqfaT, other: MmqfaT) -> MmqfaT: ...

    @overload
    def __and__(
        self: MmqfaT, other: 'MeasureOnceQuantumFiniteStateAutomaton'
    ) -> MmqfaT: ...

    def __and__(
        self: MmqfaT,
        other: Union[MmqfaT, 'MeasureOnceQuantumFiniteStateAutomaton']
    ) -> MmqfaT:
        """Returns the intersection of the quantum finite automaton.

        See intersection() for details.
        """
        return self.intersection(other)

    def complement(self: MmqfaT) -> MmqfaT:
        """Returns the complement of the quantum finite automaton.

        For a quantum finite automaton M, the complement is defined as the
        quantum finite automaton M' such that M'(w) = 1 - M(w) for all w.

        Alberto Bertoni, Carlo Mereghetti, and Beatrice Palano. 2003. Quantum
        Computing: 1-Way Quantum Automata. In Proceedings of the 8th
        International Conference on Developments in Language Theory (DLT'04).
        """
        return self.__class__(
            self.transitions, self.rejecting_states, self.accepting_states)

    def __invert__(self: MmqfaT) -> MmqfaT:
        """Returns the complement of the quantum finite automaton.

        See complement() for details.
        """
        return self.complement()

    @classmethod
    def linear_combination(
        cls: type[MmqfaT],
        *mmqfas: MmqfaT,
        coefficients: Optional[list[float]] = None
    ) -> MmqfaT:
        """Returns the linear combination of the measure-once quantum finite
        automata.

        For quantum finite automata M, N and 0 <= c <= 1, the linear
        combination M' is an mmqfa such that M'(w) = c * M(w) + (1 - c) * N(w)
        for all w.

        Maria Paola Bianchi and Beatrice Palano. 2010. Behaviours of Unary
        Quantum Automata. Fundamenta Informaticae.
        """
        if len(mmqfas) == 0:
            raise ValueError("mmqfas must be non-empty")

        alphabet = mmqfas[0].alphabet
        if not all(mmqfa.alphabet == alphabet for mmqfa in mmqfas):
            raise ValueError("mmqfas must have the same alphabet")

        if coefficients is None:
            coefficients = [1/len(mmqfas)] * len(mmqfas)

        if len(mmqfas) != len(coefficients):
            raise ValueError("mmqfas and coefficients must have the same size")

        if not math.isclose(sum(coefficients), 1):
            raise ValueError("coefficients must sum to 1")

        initial_superposition = np.concatenate([
            [math.sqrt(coefficient)] + [0] * (mmqfa.states - 1)
            for coefficient, mmqfa in zip(coefficients, mmqfas)
        ])
        initial_transition = get_transition_from_initial_to_superposition(
            initial_superposition)
        transitions = np.stack([
            reduce(direct_sum, sigma_transitions)
            for sigma_transitions
            in zip(*[mmqfa.transitions for mmqfa in mmqfas])
        ])
        transitions[cls.start_of_string] = (
            transitions[cls.start_of_string] @ initial_transition)
        accepting_states = np.concatenate(
            [mmqfa.accepting_states for mmqfa in mmqfas])
        rejecting_states = np.concatenate(
            [mmqfa.rejecting_states for mmqfa in mmqfas])
        return cls(transitions, accepting_states, rejecting_states)

    def _build_macro_step_unitary(
        self, seq: list[int], num_aux_copies: int
    ) -> Transition:
        n = self.states
        halt_indices = np.flatnonzero(self.halting_states)
        n_halt = len(halt_indices)
        total_states = n + num_aux_copies * n_halt
        if len(seq) == 0:
            return np.eye(total_states, dtype=np.cdouble)

        def embed(u: Transition) -> Transition:
            out = np.eye(total_states, dtype=np.cdouble)
            out[:n, :n] = u
            return out

        def swap_copy(copy_idx: int) -> Transition:
            perm = np.eye(total_states, dtype=np.cdouble)
            base = n + copy_idx * n_halt
            for j, h in enumerate(halt_indices):
                aux = base + j
                perm[h, h] = 0
                perm[aux, aux] = 0
                perm[h, aux] = 1
                perm[aux, h] = 1
            return perm

        acc = embed(self.transitions[seq[0]])
        for step_idx, sym in enumerate(seq[1:]):
            acc = embed(self.transitions[sym]) @ swap_copy(step_idx) @ acc
        return acc

    def _extended_halting_masks(
        self, num_aux_copies: int
    ) -> tuple[States, States]:
        halt_indices = np.flatnonzero(self.halting_states)
        aux_acc = self.accepting_states[halt_indices]
        aux_rej = self.rejecting_states[halt_indices]
        ext_acc = np.concatenate(
            [self.accepting_states] + [aux_acc] * num_aux_copies
        ).astype(bool)
        ext_rej = np.concatenate(
            [self.rejecting_states] + [aux_rej] * num_aux_copies
        ).astype(bool)
        return ext_acc, ext_rej

    def word_quotient(self: MmqfaT, w: list[int]) -> MmqfaT:
        """Returns the word quotient of the measure-many quantum finite
        automaton.

        For a quantum finite automaton M and a word w, the word quotient M' of
        M with respect to w is an MMQFA M' such that f_{M'}(x) = f_M(wx) for
        all x, using n + |w| * n_halt states.
        """
        if not all(1 <= c <= self.alphabet for c in w):
            raise ValueError(
                "w must be a string over the alphabet of the MMQFA")

        num_aux = len(w)
        ext_acc, ext_rej = self._extended_halting_masks(num_aux)
        w_sos = self._build_macro_step_unitary(
            [self.start_of_string] + w, num_aux)
        other_transitions = [
            self._build_macro_step_unitary([c], num_aux)
            for c in range(1, self.alphabet + 2)
        ]
        transitions = np.stack([w_sos] + other_transitions)
        return self.__class__(transitions, ext_acc, ext_rej)

    def inverse_homomorphism(self: MmqfaT, phi: list[list[int]]) -> MmqfaT:
        """Returns the inverse homomorphism of the measure-many quantum finite
        automaton.

        For a quantum finite automaton M and a homomorphism phi, the inverse
        homomorphism M' of M with respect to phi is an MMQFA M' such that
        f_{M'}(w) = f_M(phi(w)), using at most n + c(h) * n_halt states.

        Alex Brodsky and Nicholas Pippenger 2002. Characterizations of 1-way
        Quantum Finite Automata. SIAM Journal on Computing.
        """
        if len(phi) < 2:
            raise ValueError("phi must define at least one alphabet symbol")
        if phi[self.start_of_string] != [self.start_of_string]:
            raise ValueError("phi[start_of_string] must be [start_of_string]")

        alphabet = len(phi) - 1
        for c in range(1, alphabet + 1):
            if not all(1 <= sym <= self.alphabet for sym in phi[c]):
                raise ValueError(
                    "The co-domain of phi must be the set of words over the "
                    "alphabet of the automaton"
                )

        c_h = max((len(phi[c]) for c in range(1, alphabet + 1)), default=0)
        num_aux = max(0, c_h - 1)
        ext_acc, ext_rej = self._extended_halting_masks(num_aux)
        transitions = np.stack(
            [self._build_macro_step_unitary(phi[c], num_aux)
             for c in range(alphabet + 1)]
            + [self._build_macro_step_unitary([self.end_of_string], num_aux)]
        )
        return self.__class__(transitions, ext_acc, ext_rej)

    def equivalence(self, other: MmqfaT) -> bool:
        """Returns whether the measure-many quantum finite automaton is equal.

        For quantum finite automata M and M', the equivalence is defined as
        whether M(w) = M'(w) for all w.

        See also counter_example().

        """
        return self.counter_example(other) is None

    def counter_example(self, other: MmqfaT) -> Optional[list[int]]:
        """Returns a counter example of the equivalence of the measure-many
        quantum finite automaton.

        For quantum finite automata M and M', the counter example is defined as
        a word w such that M(w) != M'(w).
        """
        if self.alphabet != other.alphabet:
            raise ValueError("Alphabets must be the same")
        raise NotImplementedError()

    def _never_halts_before_eos_in(
        self, target_halting_states: States, atol: float = 1e-9
    ) -> bool:
        v_0 = self.initial_transition[:, 0].astype(np.cdouble)
        if not np.allclose(v_0[target_halting_states], 0, atol=atol):
            return False

        psi_0 = v_0 * self.non_halting_states
        norm_0 = float(np.linalg.norm(psi_0))
        if norm_0 <= atol:
            return True

        basis: list[npt.NDArray[np.cdouble]] = [psi_0 / norm_0]
        queue: deque[npt.NDArray[np.cdouble]] = deque([basis[0]])
        while queue:
            v = queue.popleft()
            for u in self.transitions[1:-1]:
                y = u @ v
                if not np.allclose(y[target_halting_states], 0, atol=atol):
                    return False
                r = y * self.non_halting_states
                for _ in range(2):
                    for b in basis:
                        r = r - np.vdot(b, r) * b
                norm_r = float(np.linalg.norm(r))
                if norm_r > atol:
                    b_new = r / norm_r
                    basis.append(b_new)
                    queue.append(b_new)
        return True

    def is_end_decisive(self) -> bool:
        """Returns whether the quantum finite automaton is end-decisive.

        A quantum finite automaton is end-decisive if p_acc = 0 for every
        T_{#w}(|q_init>, 0, 0) (w in Sigma*), i.e., it accepts only on
        end-of-string.

        Alex Brodsky and Nicholas Pippenger. 2002. Characterizations of 1-way
        Quantum Finite Automata. SIAM Journal on Computing.
        """
        return self._never_halts_before_eos_in(self.accepting_states)

    def is_co_end_decisive(self) -> bool:
        """Returns whether the quantum finite automaton is co-end-decisive.

        A quantum finite automaton is co-end-decisive if p_rej = 0 for every
        T_{#w}(|q_init>, 0, 0) (w in Sigma*), i.e., it rejects only on
        end-of-string.

        Alex Brodsky and Nicholas Pippenger. 2002. Characterizations of 1-way
        Quantum Finite Automata. SIAM Journal on Computing.
        """
        return self._never_halts_before_eos_in(self.rejecting_states)

    def to_real_valued(self: MmqfaT) -> MmqfaT:
        transitions = np.stack([
            get_real_valued_transition(transition)
            for transition in self.transitions
        ]).astype(complex)

        stacked_accepting = np.stack([self.accepting_states] * 2)
        accepting_states = stacked_accepting.T.reshape(2 * self.states)
        stacked_rejecting = np.stack([self.rejecting_states] * 2)
        rejecting_states = stacked_rejecting.T.reshape(2 * self.states)
        return self.__class__(transitions, accepting_states, rejecting_states)
