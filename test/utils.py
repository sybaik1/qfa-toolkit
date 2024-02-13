import math

import numpy as np

from src.quantum_finite_automaton import (
    MeasureManyQuantumFiniteAutomaton as Mmqfa)
from src.quantum_finite_automaton import (
    MeasureOnceQuantumFiniteAutomaton as Moqfa)


def get_measure_once_quantum_finite_automaton(k: int) -> Moqfa:
    """
    M_k(a^n) = cos^2(n * theta), theta = pi / k
    """

    theta = math.pi / k
    a, b = math.cos(theta), math.sin(theta)
    acceptings = {0}
    transition = np.array([
        [
            [1, 0],
            [0, 1],
        ],
        [
            [a, b],
            [-b, a],
        ],
        [
            [1, 0],
            [0, 1],
        ],
    ], dtype=np.cfloat)
    return Moqfa(transition, acceptings)


def get_measure_many_quantum_finite_automaton(k: int) -> Mmqfa:
    """
    M_k(a^n) = (cos^2(theta))^n, theta = pi / k
    """

    theta = math.pi / k
    a, b = (math.cos(theta), math.sin(theta))
    acceptings = {2}
    rejectings = {1}
    transition = np.array([
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        [
            [a, b, 0],
            [-b, a, 0],
            [0, 0, 1],
        ],
        [
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ],
    ], dtype=np.cfloat)
    return Mmqfa(transition, acceptings, rejectings)
