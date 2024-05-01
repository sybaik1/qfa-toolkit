import math
from scipy.optimize import newton  # type: ignore
from typing import (TypeVar, Optional, )
from functools import reduce

import numpy as np

from .quantum_finite_state_automaton_language_base import (
    QuantumFiniteStateAutomatonLanguageBase as Qfl)
from ..quantum_finite_state_automaton import (
    MeasureManyQuantumFiniteStateAutomaton as Mmqfa)
from ..recognition_strategy import NegativeOneSidedBoundedError as NegOneSided

MmqflT = TypeVar(
    'MmqflT', bound='MeasureManyQuantumFiniteStateAutomatonLanguage')


class MeasureManyQuantumFiniteStateAutomatonLanguage(Qfl[Mmqfa, NegOneSided]):

    def intersection(self: MmqflT, other: MmqflT) -> MmqflT:
        epsilon_1 = self.strategy.epsilon
        epsilon_2 = other.strategy.epsilon
        c = epsilon_2 / (epsilon_1 + epsilon_2)
        qfa_1 = self.quantum_finite_state_automaton
        qfa_2 = other.quantum_finite_state_automaton
        qfa = Mmqfa.linear_combination(qfa_1, qfa_2, coefficients=[c, 1-c])
        epsilon = (epsilon_1 * epsilon_2) / (epsilon_1 + epsilon_2)
        return self.__class__(qfa, NegOneSided(epsilon))

    def __and__(self: MmqflT, other: MmqflT) -> MmqflT:
        return self.intersection(other)

    def union(self: MmqflT, other: MmqflT) -> MmqflT:
        qfa_1 = self.quantum_finite_state_automaton
        qfa_2 = other.quantum_finite_state_automaton

        if not (qfa_1.is_co_end_decisive() and qfa_2.is_co_end_decisive()):
            raise ValueError("QFAs must be co-end-decisive")

        qfa = qfa_1 | qfa_2
        epsilon = self.strategy.epsilon * other.strategy.epsilon
        return self.__class__(qfa, NegOneSided(epsilon))

    def __or__(self: MmqflT, other: MmqflT) -> MmqflT:
        return self.union(other)

    @classmethod
    def from_unary_finite(
        cls,
        ks: list[int],
        params: Optional[tuple[float, float]] = None
    ) -> "MeasureManyQuantumFiniteStateAutomatonLanguage[NegOneSided]":
        qfls = map(lambda k: cls.from_unary_singleton(k, params), ks)
        return reduce(cls.union, qfls)

    @classmethod
    def _find_unary_constant_margin(
            cls,
            k: int,
            params: Optional[tuple[float, float]] = None) -> int:

        if params is None:
            theta, phi = cls._find_unary_singleton_parameters(k)
        N = math.ceil(
            1 / ((1-math.cos(theta)**2) * (1 - math.cos(phi))**2) +
            1 / (math.cos(theta)**2*math.cos(phi)**2*k*(1 - math.cos(phi))**2)
        ) - 1
        return N

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
        epsilon = reject_probability / 2
        strategy = NegOneSided(epsilon)

        return MeasureManyQuantumFiniteStateAutomatonLanguage(mmqfa, strategy)
