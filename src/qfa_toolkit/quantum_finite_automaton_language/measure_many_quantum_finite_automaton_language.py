import math
from typing import (TypeVar, Generic, )

import numpy as np

from .quantum_finite_automaton_language_base import (
    QuantumFiniteAutomatonLanguageBase as Qfl)
from ..quantum_finite_automaton import (
    MeasureManyQuantumFiniteAutomaton as Mmqfa)
from ..recognition_strategy import RecognitionStrategy
from ..recognition_strategy import NegativeOneSidedBoundedError as NegOneSided
from ..recognition_strategy import IsolatedCutPoint
from ..recognition_strategy import NegativeOneSidedBoundedError
from ..recognition_strategy import PositiveOneSidedBoundedError

TMmqfl = TypeVar('TMmqfl', bound='MeasureManyQuantumFiniteAutomatonLanguage')
TRecognitionStrategy = TypeVar(
    'TRecognitionStrategy', bound=RecognitionStrategy)


class MeasureManyQuantumFiniteAutomatonLanguage(
    Qfl[Mmqfa, TRecognitionStrategy],
    Generic[TRecognitionStrategy]
):

    def complement(self: TMmqfl) -> TMmqfl:
        if not isinstance(self.strategy, IsolatedCutPoint):
            raise ValueError(
                "Complement is only defined for IsolatedCutPoint strategies")
        threshold = 1.0 - self.strategy.threshold
        epsilon = self.strategy.epsilon
        isolated_cut_point = IsolatedCutPoint(threshold, epsilon)
        return self.__class__(
            ~self.quantum_finite_automaton, isolated_cut_point)

    def intersection(self: TMmqfl, other: TMmqfl) -> TMmqfl:
        qfa = self.quantum_finite_automaton & other.quantum_finite_automaton
        strategy: RecognitionStrategy
        if (
            isinstance(self.strategy, PositiveOneSidedBoundedError)
            and isinstance(other.strategy, PositiveOneSidedBoundedError)
        ):
            epsilon = self.strategy.epsilon * other.strategy.epsilon
            strategy = PositiveOneSidedBoundedError(epsilon)
        else:
            raise NotImplementedError()
        return self.__class__(qfa, strategy)

    def __or__(self: TMmqfl, other: TMmqfl) -> TMmqfl:
        return self.union(other)

    def union(self: TMmqfl, other: TMmqfl) -> TMmqfl:
        qfa = self.quantum_finite_automaton | other.quantum_finite_automaton
        strategy = self.strategy | other.strategy
        if (
            isinstance(self.strategy, NegativeOneSidedBoundedError)
            and isinstance(other.strategy, NegativeOneSidedBoundedError)
        ):
            one_minus_epsilon = (
                (1 - self.strategy.epsilon) * (1 - other.strategy.epsilon))
            epsilon = 1 - one_minus_epsilon
            strategy = NegativeOneSidedBoundedError(epsilon)
        else:
            raise NotImplementedError()
        return self.__class__(qfa, strategy)

    @classmethod
    def from_singleton(
        cls,
        n: int
    ) -> "MeasureManyQuantumFiniteAutomatonLanguage[NegOneSided]":
        acceptings = np.array([0, 0, 1, 0], dtype=bool)
        rejectings = np.array([0, 0, 0, 1], dtype=bool)

        # TODO: make it as a parameter
        theta_initial = math.pi / 4
        theta_sigma = math.pi / 3

        c = math.sqrt(math.pow(2, 2*n + 1) / (math.pow(2, 2*n) + 1))
        a = c * math.sqrt(2) / math.pow(2, n+1)
        b = c * math.sqrt(2) / 2

        initial_transition = np.array([
            [math.cos(theta_initial), -math.sin(theta_initial), 0, 0],
            [math.sin(theta_initial), math.cos(theta_initial), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.cdouble)
        sigma_transition = np.array([
            [math.cos(theta_sigma), 0, -math.sin(theta_sigma), 0],
            [0, 1, 0, 0],
            [math.sin(theta_sigma), 0, math.cos(theta_sigma), 0],
            [0, 0, 0, 1]
        ], dtype=np.cdouble)
        final_transition = np.array([
            [0, 0, a, -b],
            [0, 0, b, a],
            [a, b, 0, 0],
            [-b, a, 0, 0]
        ], dtype=np.cdouble)
        transitions = np.stack([
            initial_transition,
            sigma_transition,
            final_transition
        ])
        mmqfa = Mmqfa(transitions, acceptings, rejectings)

        # for m != n, mmqfa([1] * m) = 1 - 1/N
        N = math.pow(2, 2*n + 3) + 8
        epsilon = 1 - 1/(N+2)
        strategy = NegOneSided(epsilon)

        return cls(mmqfa, strategy)  # type: ignore
