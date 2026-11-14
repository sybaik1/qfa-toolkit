import enum
import math


class RecognitionStrategy:

    class Result(enum.IntEnum):
        ACCEPT = enum.auto()
        REJECT = enum.auto()
        INVALID = enum.auto()

    def __init__(
        self,
        reject_upperbound: float,
        accept_lowerbound: float,
        reject_inclusive: bool = False,
        accept_inclusive: bool = False,
    ) -> None:
        self.reject_upperbound = reject_upperbound
        self.accept_lowerbound = accept_lowerbound
        self.reject_inclusive = reject_inclusive
        self.accept_inclusive = accept_inclusive

    def __str__(self) -> str:
        return "rej: {} {}, acc: {} {}".format(
            '<=' if self.reject_inclusive else '<', self.reject_upperbound,
            self.accept_lowerbound, '<=' if self.accept_inclusive else '<'
        )

    def __call__(self, probability: float) -> Result:

        if probability < -1e-9 or probability > 1.0 + 1e-9:
            raise ValueError(
                "Probability must be between 0.0 and 1.0, "
                f"but probability is {probability}."
            )
        probability = min(1.0, max(0.0, probability))

        gap = self.accept_lowerbound - self.reject_upperbound
        tol = min(1e-9, gap / 4.0) if gap > 0 else 1e-9

        if self.reject_inclusive:
            if math.isclose(
                probability, self.reject_upperbound, rel_tol=0.0, abs_tol=tol
            ):
                return RecognitionStrategy.Result.REJECT
        if self.accept_inclusive:
            if math.isclose(
                probability, self.accept_lowerbound, rel_tol=0.0, abs_tol=tol
            ):
                return RecognitionStrategy.Result.ACCEPT

        if probability < self.reject_upperbound:
            return RecognitionStrategy.Result.REJECT
        if self.accept_lowerbound < probability:
            return RecognitionStrategy.Result.ACCEPT

        return RecognitionStrategy.Result.INVALID


class CutPoint(RecognitionStrategy):
    def __init__(self, probability: float) -> None:
        super().__init__(probability, probability, True, False)


class IsolatedCutPoint(RecognitionStrategy):
    """
    Michael O. Rabin,
    Probabilistic automata,
    Information and Control,
    Volume 6, Issue 3,
    1963,
    Pages 230-245,
    ISSN 0019-9958,
    https://doi.org/10.1016/S0019-9958(63)90290-0.
    """

    def __init__(self, threshold: float, epsilon: float) -> None:
        if epsilon < 0.0:
            raise ValueError("epsilon must be positive")
        super().__init__(
            threshold - epsilon, threshold + epsilon)

    @property
    def threshold(self) -> float:
        return (self.accept_lowerbound + self.reject_upperbound) / 2

    @property
    def epsilon(self) -> float:
        return self.accept_lowerbound - self.threshold


class PositiveOneSidedBoundedError(RecognitionStrategy):
    def __init__(self, epsilon: float) -> None:
        if epsilon > 1.0:
            raise ValueError("epsilon must be at most 1.0")
        super().__init__(0.0, epsilon, True, True)

    @property
    def epsilon(self) -> float:
        return self.accept_lowerbound


class NegativeOneSidedBoundedError(RecognitionStrategy):
    def __init__(self, epsilon: float) -> None:
        if epsilon > 1.0:
            raise ValueError("epsilon must be at most 1.0")
        super().__init__(1.0 - epsilon, 1.0, True, True)

    @property
    def epsilon(self) -> float:
        return 1.0 - self.reject_upperbound
