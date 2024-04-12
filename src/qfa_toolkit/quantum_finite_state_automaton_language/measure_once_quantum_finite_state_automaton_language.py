import numpy as np
from functools import reduce
from typing import (Generic, TypeVar, )

from .quantum_finite_state_automaton_language_base import (
    QuantumFiniteStateAutomatonLanguageBase as Qfl)
from ..quantum_finite_state_automaton import (
    MeasureOnceQuantumFiniteStateAutomaton as Moqfa)
from ..quantum_finite_state_automaton.utils import (
    direct_sum, get_transition_from_initial_to_superposition)
from ..recognition_strategy import RecognitionStrategy
from ..recognition_strategy import NegativeOneSidedBoundedError as NegOneSided


MoqflT = TypeVar('MoqflT', bound='MeasureOnceQuantumFiniteStateAutomatonLanguage')
RecognitionStrategyT = TypeVar(
    'RecognitionStrategyT', bound=RecognitionStrategy)


class MeasureOnceQuantumFiniteStateAutomatonLanguage(
    Qfl[Moqfa, RecognitionStrategyT],
    Generic[RecognitionStrategyT]
):

    def concatenation(self, other: MoqflT) -> MoqflT:
        raise NotImplementedError()

    def union(self, other: MoqflT) -> MoqflT:
        raise NotImplementedError()

    def intersection(self, other: MoqflT) -> MoqflT:
        raise NotImplementedError()

    @classmethod
    def from_modulo(cls, n: int):
        raise NotImplementedError()

    @classmethod
    def _params_for_modulo_prime(
            cls, p: int, length: int, seed: int) -> list[float]:
        generator = np.random.default_rng(seed)
        max_iter = 100
        candidate_list = [
            [
                k for k in range(1, p)
                if p/8 <= j*k % p <= p*3/8
                or p*5/8 <= j*k % p <= p*7/8
            ]
            for j in range(1, p)
        ]

        for _ in range(max_iter):
            ks = generator.choice(range(1, p), size=length, replace=True)
            if all(len([True for k in ks if k in candidate]) >= length/4
                   for candidate in candidate_list):
                break
        else:
            raise ValueError("Cannot find suitable parameters.")

        phi = [2 * np.pi * k / p for k in ks]
        return phi

    @classmethod
    def from_modulo_prime(cls, p: int, seed: int = 42):
        # select ks
        length = int(np.ceil(8 * np.log(p)))
        phi = cls._params_for_modulo_prime(p, length, seed)

        # uniform distribution to selected
        f = np.sqrt(1/length)
        initial_transition = get_transition_from_initial_to_superposition(
            np.array([f, 0]*length))
        # k automaton
        acceptings = np.array([1, 0]*length, dtype=bool)
        sigma_k_transitions = [np.array([
            [np.cos(k), 1j*np.sin(k)],
            [1j*np.sin(k), np.cos(k)],
        ], dtype=np.cdouble) for k in phi]
        sigma_transition = reduce(direct_sum, sigma_k_transitions)
        final_transition = np.eye(2*length)

        transitions = np.stack([
            initial_transition,
            sigma_transition,
            final_transition
        ])

        # for j != 0 mod p, mmqfa([1] * j) = 1 - 1/8
        moqfa = Moqfa(transitions, acceptings)
        epsilon = 1/8
        strategy = NegOneSided(1 - epsilon)

        return MeasureOnceQuantumFiniteStateAutomatonLanguage(moqfa, strategy)
