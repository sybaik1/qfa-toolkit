import math
import numpy as np
from functools import reduce
from typing import (TypeVar, Union, overload, )

from .quantum_finite_state_automaton_language_base import (
    QuantumFiniteStateAutomatonLanguageBase as Qfl)
from .measure_many_quantum_finite_state_automaton_language import (
    MeasureManyQuantumFiniteStateAutomatonLanguage as Mmqfl)
from ..quantum_finite_state_automaton import (
    MeasureOnceQuantumFiniteStateAutomaton as Moqfa)
from ..quantum_finite_state_automaton.utils import (
    direct_sum, get_transition_from_initial_to_superposition)
from ..recognition_strategy import NegativeOneSidedBoundedError as NegOneSided

MoqflT = TypeVar(
    'MoqflT', bound='MeasureOnceQuantumFiniteStateAutomatonLanguage')


class MeasureOnceQuantumFiniteStateAutomatonLanguage(Qfl[Moqfa, NegOneSided]):

    @overload
    def union(self: MoqflT, other: MoqflT) -> MoqflT: ...

    @overload
    def union(self, other: Mmqfl) -> Mmqfl: ...

    def union(
        self: MoqflT, other: Union[MoqflT, Mmqfl]
    ) -> Union[MoqflT, Mmqfl]:
        if isinstance(other, Mmqfl):
            if not other.quantum_finite_state_automaton.is_co_end_decisive():
                raise ValueError("QFAs must be co-end-decisive")
            mm_qfa = (
                self.quantum_finite_state_automaton
                | other.quantum_finite_state_automaton
            )
            epsilon = self.strategy.epsilon * other.strategy.epsilon
            return Mmqfl(mm_qfa, NegOneSided(epsilon))
        mo_qfa = (
            self.quantum_finite_state_automaton
            | other.quantum_finite_state_automaton
        )
        epsilon = self.strategy.epsilon * other.strategy.epsilon
        strategy = NegOneSided(epsilon)
        return self.__class__(mo_qfa, strategy)

    @overload
    def __or__(self: MoqflT, other: MoqflT) -> MoqflT: ...

    @overload
    def __or__(self, other: Mmqfl) -> Mmqfl: ...

    def __or__(
        self: MoqflT, other: Union[MoqflT, Mmqfl]
    ) -> Union[MoqflT, Mmqfl]:
        if isinstance(other, Mmqfl):
            return self.union(other)
        return self.union(other)

    @overload
    def intersection(
        self: MoqflT,
        other: MoqflT,
        use_linear_combination: bool = True
    ) -> MoqflT: ...

    @overload
    def intersection(
        self,
        other: Mmqfl,
        use_linear_combination: bool = True
    ) -> Mmqfl: ...

    def intersection(
        self: MoqflT,
        other: Union[MoqflT, Mmqfl],
        use_linear_combination: bool = True
    ) -> Union[MoqflT, Mmqfl]:
        if isinstance(other, Mmqfl):
            if use_linear_combination:
                return (
                    self.to_measure_many_quantum_finite_state_automaton_language()
                    .intersection(other)
                )
            mm_qfa = (
                self.quantum_finite_state_automaton
                & other.quantum_finite_state_automaton
            )
            epsilon = min(self.strategy.epsilon, other.strategy.epsilon)
            return Mmqfl(mm_qfa, NegOneSided(epsilon))
        if use_linear_combination:
            epsilon_1 = self.strategy.epsilon
            epsilon_2 = other.strategy.epsilon
            c = epsilon_2 / (epsilon_1 + epsilon_2)
            qfa_1 = self.quantum_finite_state_automaton
            qfa_2 = other.quantum_finite_state_automaton
            mo_qfa = Moqfa.linear_combination(
                qfa_1, qfa_2, coefficients=[c, 1 - c]
            )
            epsilon = (epsilon_1 * epsilon_2) / (epsilon_1 + epsilon_2)
        else:
            mo_qfa = (
                self.quantum_finite_state_automaton
                & other.quantum_finite_state_automaton
            )
            epsilon = min(self.strategy.epsilon, other.strategy.epsilon)
        strategy = NegOneSided(epsilon)
        return self.__class__(mo_qfa, strategy)

    @overload
    def __and__(self: MoqflT, other: MoqflT) -> MoqflT: ...

    @overload
    def __and__(self, other: Mmqfl) -> Mmqfl: ...

    def __and__(
        self: MoqflT, other: Union[MoqflT, Mmqfl]
    ) -> Union[MoqflT, Mmqfl]:
        if isinstance(other, Mmqfl):
            return self.intersection(other)
        return self.intersection(other)

    def word_quotient(self: MoqflT, w: list[int]) -> MoqflT:
        qfa = self.quantum_finite_state_automaton.word_quotient(w)
        epsilon = self.strategy.epsilon
        strategy = NegOneSided(epsilon)
        return self.__class__(qfa, strategy)

    def inverse_homomorphism(self: MoqflT, phi: list[list[int]]) -> MoqflT:
        qfa = self.quantum_finite_state_automaton.inverse_homomorphism(phi)
        epsilon = self.strategy.epsilon
        strategy = NegOneSided(epsilon)
        return self.__class__(qfa, strategy)

    def to_measure_many_quantum_finite_state_automaton_language(
        self
    ) -> Mmqfl:
        mmqfa = (
            self.quantum_finite_state_automaton
            .to_measure_many_quantum_finite_state_automaton()
        )
        strategy = NegOneSided(self.strategy.epsilon)
        return Mmqfl(mmqfa, strategy)

    @classmethod
    def from_modulo(cls: type[MoqflT], n: int, d: int = 1) -> MoqflT:
        """Create a quantum finite state automaton that recognizes the language
        of strings whose length is divisible by n.
        """
        if n < 2:
            raise ValueError("n must be at least 2.")
        if math.gcd(n, d) != 1:
            raise ValueError("n and d must be coprime.")
        states = 2
        initial_transition = np.eye(states, dtype=np.cdouble)
        final_transition = np.eye(states, dtype=np.cdouble)
        phi = np.pi * d / n
        sigma_transition = np.array([
            [np.cos(phi), -np.sin(phi)],
            [np.sin(phi), np.cos(phi)],
        ], dtype=np.cdouble)
        transitions = np.stack([
            initial_transition, sigma_transition, final_transition])
        acceptings = np.array([True, False])
        moqfa = Moqfa(transitions, acceptings)
        epsilon = float(np.sin(np.pi / n) ** 2)
        return cls(moqfa, NegOneSided(epsilon))

    @classmethod
    def from_modulo_prime(
            cls: type[MoqflT],
            p: int,
            copy_num: int = 0,
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
        if copy_num == 0:
            copy_num = int(np.ceil(8 * np.log(p)))
        phis = cls._params_for_modulo_prime(p, copy_num, seed)

        # uniform distribution to selected
        f = np.sqrt(1/copy_num)
        initial_transition = get_transition_from_initial_to_superposition(
            np.array([f, 0]*copy_num))
        # k automaton
        acceptings = np.array([1, 0]*copy_num, dtype=bool)
        sigma_k_transitions = [np.array([
            [np.cos(phi), -np.sin(phi)],
            [np.sin(phi), np.cos(phi)],
        ], dtype=np.cdouble) for phi in phis]
        sigma_transition = reduce(direct_sum, sigma_k_transitions)
        final_transition = np.eye(2*copy_num)

        transitions = np.stack([
            initial_transition,
            sigma_transition,
            final_transition
        ])

        # for j != 0 mod p, mmqfa([1] * j) = 1 - 1/8
        moqfa = Moqfa(transitions, acceptings)
        epsilon = 1/8
        strategy = NegOneSided(epsilon)

        return cls(moqfa, strategy)

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
