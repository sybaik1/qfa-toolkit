from functools import reduce
from typing import (TypeVar, )

import numpy as np
import numpy.typing as npt

from .quantum_finite_automaton_base import TotalState
from .quantum_finite_automaton_base import Observable
from .quantum_finite_automaton_base import QuantumFiniteAutomatonBase

T = TypeVar('T', bound='MeasureOnceQuantumFiniteAutomaton')


class MeasureOnceQuantumFiniteAutomaton(QuantumFiniteAutomatonBase):
    def __init__(
        self,
        transition: npt.NDArray[np.cdouble],
        accepting_states: set[int]
    ) -> None:
        super().__init__(transition)

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
        return total_state.apply(self.transition[c])

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
