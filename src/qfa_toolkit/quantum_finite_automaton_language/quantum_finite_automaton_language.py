from ..quantum_finite_automaton import QuantumFiniteAutomatonBase
from ..recognition_strategy import RecognitionStrategy


Qfl = 'QuantumFiniteAutomatonLanguage'


class QuantumFiniteAutomatonLanguage():
    def __init__(
        self,
        qfa: QuantumFiniteAutomatonBase,
        strategy: RecognitionStrategy,
    ) -> None:
        self.qfa = qfa
        self.strategy = strategy

    @property
    def alphabet(self) -> int:
        return self.qfa.alphabet

    @property
    def start_of_string(self) -> int:
        return self.qfa.start_of_string

    @property
    def end_of_string(self) -> int:
        return self.qfa.end_of_string

    def __contains__(self, w: list[int]) -> bool:
        result = self.strategy(self.qfa(w))
        if result == RecognitionStrategy.Result.INVALID:
            raise ValueError("Invalid result from recognition strategy")
        return result == RecognitionStrategy.Result.ACCEPT

    def concatination(self: Qfl, other: Qfl) -> Qfl:
        raise NotImplementedError()

    def union(self: Qfl, other: Qfl) -> Qfl:
        raise NotImplementedError()

    def intersection(self: Qfl, other: Qfl) -> Qfl:
        raise NotImplementedError()

    def complement(self: Qfl) -> Qfl:
        raise NotImplementedError()

    def difference(self: Qfl, other: Qfl) -> Qfl:
        raise NotImplementedError()

    def equivalence(self: Qfl, other: Qfl) -> bool:
        raise NotImplementedError()

    def kleene_star(self: Qfl) -> Qfl:
        raise NotImplementedError()

    def kleene_plus(self: Qfl) -> Qfl:
        raise NotImplementedError()

    def reverse(self: Qfl) -> Qfl:
        raise NotImplementedError()

    def is_empty(self: Qfl) -> bool:
        raise NotImplementedError()
