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
        reject_inclusive: bool,
        accept_inclusive: bool,
    ) -> None:
        # TODO: Deal with the floating point error
        if reject_upperbound < 0.0 or reject_upperbound > 1.0:
            raise ValueError("reject_upperbound must be between 0.0 and 1.0")

        if accept_lowerbound < 0.0 or accept_lowerbound > 1.0:
            raise ValueError("accept_lowerbound must be between 0.0 and 1.0")

        if accept_lowerbound < reject_upperbound:
            raise ValueError(
                "accept_lowerbound must be less "
                "than or equal to reject_upperbound"
            )

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

        if probability < 0.0 or probability > 1.0:
            raise ValueError("probability must be between 0.0 and 1.0")

        if self.reject_inclusive:
            if math.isclose(probability, self.reject_upperbound, abs_tol=1e-9):
                return RecognitionStrategy.Result.REJECT
        if self.accept_inclusive:
            if math.isclose(probability, self.accept_lowerbound, abs_tol=1e-9):
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

    def __init__(self, probability: float, epsilon: float) -> None:
        if epsilon < 0.0:
            raise ValueError("epsilon must be positive")
        super().__init__(
            probability - epsilon, probability + epsilon, True, True)


class PositiveOneSidedBoundedError(RecognitionStrategy):
    def __init__(self, epsilon: float) -> None:
        if epsilon > 1.0:
            raise ValueError("epsilon must be at most 1.0")
        super().__init__(0.0, 1.0 - epsilon, True, False)


class NegativeOneSidedBoundedError(RecognitionStrategy):
    def __init__(self, epsilon: float) -> None:
        if epsilon > 1.0:
            raise ValueError("epsilon must be at most 1.0")
        super().__init__(epsilon, 1.0, False, True)
