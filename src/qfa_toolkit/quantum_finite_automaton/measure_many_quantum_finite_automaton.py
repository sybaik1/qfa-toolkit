from functools import reduce
from typing import (TypeVar, )

import numpy as np
import numpy.typing as npt

from .quantum_finite_automaton_base import TotalState
from .quantum_finite_automaton_base import Observable
from .quantum_finite_automaton_base import QuantumFiniteAutomatonBase

T = TypeVar('T', bound='MeasureManyQuantumFiniteAutomaton')


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

    def concatenation(self, other: T) -> T:
        raise NotImplementedError()

    def union(self, other: T) -> T:
        raise NotImplementedError()

    def intersection(self, other: T) -> T:
        raise NotImplementedError()

    def complement(self: T) -> T:
        raise NotImplementedError()

    def difference(self, other: T) -> T:
        raise NotImplementedError()

    def equivalence(self, other: T) -> bool:
        raise NotImplementedError()

    def minimize(self: T) -> T:
        raise NotImplementedError()

    def symmetric_difference(self, other: T) -> T:
        raise NotImplementedError()

    def kleene_star(self: T) -> T:
        raise NotImplementedError()

    def kleene_plus(self: T) -> T:
        raise NotImplementedError()

    def reverse(self: T) -> T:
        raise NotImplementedError()

    def is_empty(self) -> bool:
        raise NotImplementedError()
