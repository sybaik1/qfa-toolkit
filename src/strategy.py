import enum


class RecognitionStrategy:
    class Result(enum.Enum):
        ACCEPT = enum.auto
        REJECT = enum.auto
        INVALID = enum.auto

    def __init__(self) -> None:
        pass

    def __call__(self, probability: float) -> Result:
        raise NotImplementedError()
