import numpy as np
from functools import reduce
from typing import (TypeVar, )

from .quantum_finite_state_automaton_language_base import (
    QuantumFiniteStateAutomatonLanguageBase as Qfl)
from ..quantum_finite_state_automaton import (
    MeasureOnceQuantumFiniteStateAutomaton as Moqfa)
from ..quantum_finite_state_automaton.utils import (
    direct_sum, get_transition_from_initial_to_superposition)
from ..recognition_strategy import NegativeOneSidedBoundedError as NegOneSided

MoqflT = TypeVar(
    'MoqflT', bound='MeasureOnceQuantumFiniteStateAutomatonLanguage')


class MeasureOnceQuantumFiniteStateAutomatonLanguage(Qfl[Moqfa, NegOneSided]):

    def union(self, other: MoqflT) -> MoqflT:
        qfa = (
            self.quantum_finite_state_automaton
            | other.quantum_finite_state_automaton
        )
        epsilon = self.strategy.epsilon * other.strategy.epsilon
        strategy = NegOneSided(epsilon)
        return self.__class__(qfa, strategy)

    def __or__(self, other: MoqflT) -> MoqflT:
        return self.union(other)

    def intersection(self, other: MoqflT) -> MoqflT:
        qfa = (
            self.quantum_finite_state_automaton
            & other.quantum_finite_state_automaton
        )
        epsilon = (
            1 - (1 - self.strategy.epsilon) * (1 - other.strategy.epsilon))
        strategy = NegOneSided(epsilon)
        return self.__class__(qfa, strategy)

    def __and__(self, other: MoqflT) -> MoqflT:
        return self.intersection(other)

    def word_quotient(self, w: list[int]) -> MoqflT:
        qfa = self.quantum_finite_state_automaton.word_quotient(w)
        epsilon = self.strategy.epsilon
        strategy = NegOneSided(epsilon)
        return self.__class__(qfa, strategy)

    def inverse_homomorphism(self, phi: list[list[int]]) -> MoqflT:
        qfa = self.quantum_finite_state_automaton.inverse_homomorphism(phi)
        epsilon = self.strategy.epsilon
        strategy = NegOneSided(epsilon)
        return self.__class__(qfa, strategy)

    @classmethod
    def from_modulo(cls, n: int) -> MoqflT:
        """Create a quantum finite state automaton that recognizes the language
        of strings whose length is divisible by n.
        """
        states = 2
        initial_transition = np.eye(states, dtype=np.cdouble)
        final_transition = np.eye(states, dtype=np.cdouble)
        phi = np.pi / n
        sigma_transition = np.array([
            [np.cos(phi), 1j*np.sin(phi)],
            [1j*np.sin(phi), np.cos(phi)],
        ], dtype=np.cdouble)
        transitions = np.stack([
            initial_transition, sigma_transition, final_transition])
        acceptings = np.array([True, False])
        moqfa = Moqfa(transitions, acceptings)
        epsilon = np.sin(phi) ** 2 / 2
        return cls(moqfa, NegOneSided(epsilon))

    @classmethod
    def from_modulo_prime(
            cls,
            p: int,
            length: int = 0,
            seed: int = 42) -> MoqflT:
        """Create a quantum finite state automaton that recognizes the language
        of strings whose length is divisible by p.

        TODO: Add references.
        """

        def is_prime(n: int) -> bool:
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True

        if not is_prime(p) or p <= 2:
            raise ValueError("p must be a prime number larger than 2.")

        # select ks
        if length == 0:
            length = int(np.ceil(8 * np.log(p)))
        phis = cls._params_for_modulo_prime(p, length, seed)

        # uniform distribution to selected
        f = np.sqrt(1/length)
        initial_transition = get_transition_from_initial_to_superposition(
            np.array([f, 0]*length))
        # k automaton
        acceptings = np.array([1, 0]*length, dtype=bool)
        sigma_k_transitions = [np.array([
            [np.cos(phi), 1j*np.sin(phi)],
            [1j*np.sin(phi), np.cos(phi)],
        ], dtype=np.cdouble) for phi in phis]
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
        strategy = NegOneSided(epsilon)

        return MeasureOnceQuantumFiniteStateAutomatonLanguage(moqfa, strategy)

    @staticmethod
    def _params_for_modulo_prime(
        p: int, length: int, seed: int, max_iter: int = 1000
    ) -> list[float]:
        generator = np.random.default_rng(seed)
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

        phis = [2 * np.pi * k / p for k in ks]
        return phis
