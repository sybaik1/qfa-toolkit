import math
from typing import (TypeVar, Generic, Optional, )

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
    def from_unary_singleton(
        cls,
        k: int,
        theta: Optional[float] = None,
        phi: Optional[float] = None,
    ) -> "MeasureManyQuantumFiniteAutomatonLanguage[NegOneSided]":
        acceptings = np.array([0, 0, 1, 0], dtype=bool)
        rejectings = np.array([0, 0, 0, 1], dtype=bool)

        if theta is None:
            theta = math.pi / 4
        if phi is None:
            phi = math.pi / 3

        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)

        c = (
            math.pow(cos_phi, 2*k) * math.pow(cos_theta, 2)
            + math.pow(sin_theta, 2)
        )
        a = math.pow(cos_phi, k) * cos_theta / math.sqrt(c)
        b = sin_theta / math.sqrt(c)

        initial_transition = np.array([
            [cos_theta, -sin_theta, 0, 0],
            [sin_theta, cos_theta, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.cdouble)
        sigma_transition = np.array([
            [cos_phi, 0, -sin_phi, 0],
            [0, 1, 0, 0],
            [sin_phi, 0, cos_phi, 0],
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

        # for m != n, mmqfa([1] * m) = 1 - 1/N
        mmqfa = Mmqfa(transitions, acceptings, rejectings)
        c_reject_probability = math.pow(
            cos_theta * sin_theta * math.pow(cos_phi, k) * (1-cos_phi), 2)
        reject_probability = c_reject_probability / c
        epsilon = 1 - reject_probability / 2
        strategy = NegOneSided(epsilon)

        return cls(mmqfa, strategy)  # type: ignore
