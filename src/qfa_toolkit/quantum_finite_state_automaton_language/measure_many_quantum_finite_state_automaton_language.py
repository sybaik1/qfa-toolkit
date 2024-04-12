import math
from scipy.optimize import newton  # type: ignore
from typing import (TypeVar, Generic, Optional, )
from functools import reduce

import numpy as np

from .quantum_finite_state_automaton_language_base import (
    QuantumFiniteStateAutomatonLanguageBase as Qfl)
from ..quantum_finite_state_automaton import (
    MeasureManyQuantumFiniteStateAutomaton as Mmqfa)
from ..recognition_strategy import PositiveOneSidedBoundedError as PosOneSided
from ..recognition_strategy import RecognitionStrategy
from ..recognition_strategy import NegativeOneSidedBoundedError as NegOneSided
from ..recognition_strategy import IsolatedCutPoint

MmqflT = TypeVar('MmqflT', bound='MeasureManyQuantumFiniteStateAutomatonLanguage')
RecognitionStrategyT = TypeVar(
    'RecognitionStrategyT', bound=RecognitionStrategy)


class MeasureManyQuantumFiniteStateAutomatonLanguage(
    Qfl[Mmqfa, RecognitionStrategyT],
    Generic[RecognitionStrategyT]
):

    def complement(self: MmqflT) -> MmqflT:
        if not isinstance(self.strategy, IsolatedCutPoint):
            raise ValueError(
                "Complement is only defined for IsolatedCutPoint strategies")
        threshold = 1.0 - self.strategy.threshold
        epsilon = self.strategy.epsilon
        isolated_cut_point = IsolatedCutPoint(threshold, epsilon)
        return self.__class__(
            ~self.quantum_finite_state_automaton, isolated_cut_point)

    def intersection(self: MmqflT, other: MmqflT) -> MmqflT:
        qfa = self.quantum_finite_state_automaton & other.quantum_finite_state_automaton
        strategy: RecognitionStrategy
        if (
            isinstance(self.strategy, PosOneSided)
            and isinstance(other.strategy, PosOneSided)
        ):
            epsilon = min(self.strategy.epsilon, other.strategy.epsilon)
            strategy = PosOneSided(epsilon)
        else:
            raise NotImplementedError()
        return self.__class__(qfa, strategy)

    def __or__(self: MmqflT, other: MmqflT) -> MmqflT:
        return self.union(other)

    def union(self: MmqflT, other: MmqflT) -> MmqflT:
        qfa = self.quantum_finite_state_automaton | other.quantum_finite_state_automaton
        if (
            isinstance(self.strategy, NegOneSided)
            and isinstance(other.strategy, NegOneSided)
        ):
            one_minus_epsilon = (
                (1 - self.strategy.epsilon) * (1 - other.strategy.epsilon))
            epsilon = 1 - one_minus_epsilon
            strategy = NegOneSided(epsilon)
        else:
            raise NotImplementedError()
        return self.__class__(qfa, strategy)

    @classmethod
    def from_unary_finite(
        cls,
        ks: list[int],
        params: Optional[tuple[float, float]] = None
    ) -> "MeasureManyQuantumFiniteStateAutomatonLanguage[NegOneSided]":
        return reduce(
            cls.union,
            map(lambda k: cls.from_unary_singleton(k, params), ks)
        )

    @classmethod
    def _find_unary_singleton_parameters(cls, k: int) -> tuple[float, float]:
        omega = newton(
            lambda x: math.pow(x, k+1)+(k+1)*x-k,
            x0=0.5,
            tol=1e-10
        )
        phi = math.acos(omega)
        theta = math.atan(math.sqrt(math.pow(omega, k)))
        return theta, phi

    @classmethod
    def from_unary_singleton(
        cls,
        k: int,
        params: Optional[tuple[float, float]] = None
    ) -> "MeasureManyQuantumFiniteStateAutomatonLanguage[NegOneSided]":
        if params is None:
            params = cls._find_unary_singleton_parameters(k)

        acceptings = np.array([0, 0, 1, 0], dtype=bool)
        rejectings = np.array([0, 0, 0, 1], dtype=bool)

        theta, phi = params
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

        return MeasureManyQuantumFiniteStateAutomatonLanguage(mmqfa, strategy)
