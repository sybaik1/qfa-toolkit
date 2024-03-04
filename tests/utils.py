import cmath
import math
import unittest
import itertools
from typing import (Callable, TypeVar, Iterable, Iterator, )

import numpy as np

from qfa_toolkit.quantum_finite_automaton import (
    QuantumFiniteAutomatonBase as QfaBase)
from qfa_toolkit.quantum_finite_automaton import (
    MeasureManyQuantumFiniteAutomaton as Mmqfa)
from qfa_toolkit.quantum_finite_automaton import (
    MeasureOnceQuantumFiniteAutomaton as Moqfa)
from qfa_toolkit.quantum_finite_automaton import Transitions
from qfa_toolkit.quantum_finite_automaton import TotalState


def iterate_length_n_strings(alphabet: int, n: int) -> Iterator[list[int]]:
    return map(list, itertools.product(range(1, alphabet+1), repeat=n))


def iterate_length_less_than_n_strings(
    alphabet: int, n: int
) -> Iterator[list[int]]:
    count = range(n)
    iterables = map(lambda n: iterate_length_n_strings(alphabet, n), count)
    return itertools.chain.from_iterable(iterables)


def multiply_arbitrary_global_phase(transitions: Transitions) -> Transitions:
    transitions = np.array(transitions)
    global_phases = np.array([
        [cmath.exp((cmath.pi / 2 ** (i+1)) * 1j)]
        for i, _ in enumerate(transitions)
    ])

    transitions *= global_phases[:, np.newaxis]
    return transitions


def get_arbitrary_moqfa(k: int) -> Moqfa:
    """
    M_k(a^n) = cos^2(n * theta), theta = pi / k
    """

    theta = math.pi / k
    a, b = math.cos(theta), math.sin(theta)
    acceptings = np.array([1, 0], dtype=bool)
    transitions = np.array([
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
    transitions = multiply_arbitrary_global_phase(transitions)
    return Moqfa(transitions, acceptings)


def get_arbitrary_mmqfa(r: float) -> Mmqfa:
    """
    M_k(a^n) = r^n
    M_k is end-decisive
    """
    a, b = (math.sqrt(r), math.sqrt(1-r))
    acceptings = np.array([0, 0, 1], dtype=bool)
    rejectings = np.array([0, 1, 0], dtype=bool)
    transitions = np.array([
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
    transitions = multiply_arbitrary_global_phase(transitions)
    return Mmqfa(transitions, acceptings, rejectings)


TQfa = TypeVar('TQfa', bound=QfaBase)


def test_qfa(
    test: unittest.TestCase,
    get_qfa: Callable[[float], TQfa],
    target: Callable[[float, list[int]], float],
    qfa_parameters: Iterable[float],
    max_string_len: int
) -> None:
    for k in qfa_parameters:
        qfa = get_qfa(k)
        ws = iterate_length_less_than_n_strings(
            qfa.alphabet, max_string_len)
        for w in ws:
            test.assertAlmostEqual(qfa(w), target(k, w))


def test_total_state_during_process(
    test: unittest.TestCase,
    get_qfa: Callable[[float], TQfa],
    qfa_parameters: Iterable[float],
    string_len: int,
    constraint: Callable[[list[int], int, TotalState], bool]
    = lambda tape, i, total_state: True,
) -> None:
    for m in map(get_qfa, qfa_parameters):
        ws = iterate_length_n_strings(m.alphabet, string_len)
        for w in ws:
            tape = [m.start_of_string] + w + [m.end_of_string]
            total_state = TotalState.initial(m.states)
            for i, c in enumerate(tape):
                total_state = m.step(total_state, c)
                test.assertTrue(constraint(tape, i, total_state))


def test_unary_operation(
    test: unittest.TestCase,
    operation: Callable[[TQfa], TQfa],
    get_target_prob: Callable[[float], float],
    get_qfa: Callable[[float], TQfa],
    qfa_parameters: Iterable[float],
    max_string_len: int,
    *,
    get_preimage_str: Callable[[list[int]], list[int]] = lambda x: x,
    constraint: Callable[[TQfa], bool] = lambda x: True,

) -> None:
    """Test the unary operation qfa_op on the qfa.

    Check n(w) = get_target_prob(m(u)) for all strings u of length less than n,
    where n = operation(m) and u = get_preimage_str(w).
    """
    for m in map(get_qfa, qfa_parameters):
        ws = iterate_length_less_than_n_strings(m.alphabet, max_string_len)
        n = operation(m)
        test.assertTrue(constraint(n))
        for w in ws:
            u = get_preimage_str(w)
            test.assertAlmostEqual(n(w), get_target_prob(m(u)))


def test_binary_operation(
    test: unittest.TestCase,
    operation: Callable[[TQfa, TQfa], TQfa],
    get_target_prob: Callable[[float, float], float],
    get_qfa_1: Callable[[float], TQfa],
    get_qfa_2: Callable[[float], TQfa],
    qfa_parameters_1: Iterable[float],
    qfa_parameters_2: Iterable[float],
    max_string_len: int,
    *,
    get_preimage_pair:
    Callable[[list[int]], tuple[list[int], list[int]]] = lambda x: (x, x),
    constraint: Callable[[TQfa], bool] = lambda x: True,
) -> None:
    """Test the binary operation qfa_op on the qfa.

    Check n(w) = get_target_prob(m1(u1), m2(u2)) for all strings u of length
    less than n, where n = operation(m1, m2) and (u1, u2) =
    get_preimage_pair(w).
    """
    for k, l in itertools.product(qfa_parameters_1, qfa_parameters_2):
        m1 = get_qfa_1(k)
        m2 = get_qfa_2(l)

        if m1.alphabet != m2.alphabet:
            raise ValueError("Alphabets must be the same")
        alphabet = m1.alphabet
        ws = iterate_length_less_than_n_strings(alphabet, max_string_len)
        n = operation(m1, m2)
        test.assertTrue(constraint(n))
        for w in ws:
            u1, u2 = get_preimage_pair(w)
            test.assertAlmostEqual(n(w), get_target_prob(m1(u1), m2(u2)))
