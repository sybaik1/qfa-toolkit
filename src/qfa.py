from abc import (ABC, abstractmethod, )


class QuantumState:
    def __init__(self) -> None:
        pass


class QuantumFiniteAutomatonBase(ABC):
    _states: int
    _alphabet: int
    # _transition: np.array(dim=(alphabet, states, states))

    @abstractmethod
    def step(self, psi: QuantumState, string: str) -> QuantumState:
        pass

    @abstractmethod
    def process(self, psi: QuantumState, w: str) -> QuantumState:
        pass

    @abstractmethod
    def test(self, w: str) -> bool:
        pass


class DynamicConfiguration:
    def __init__(
        self,
        qfa: QuantumFiniteAutomatonBase,
        q: QuantumState
    ) -> None:
        pass

    def step(self, c: str) -> None:
        if len(c) != 1:
            raise ValueError("c must be a single character")
        pass

    def process(self, w: str) -> None:
        pass


class MeasureManyQuantumFiniteAutomaton(QuantumFiniteAutomatonBase):
    def __init__(self, qfa, measure) -> None:
        self.qfa = qfa
        self.measure = measure


class MeasureOnceQuantumFiniteAutomaton(QuantumFiniteAutomatonBase):
    def __init__(self, qfa, measure) -> None:
        self.qfa = qfa
        self.measure = measure
