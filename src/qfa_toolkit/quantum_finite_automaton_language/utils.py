import itertools
from typing import (Iterator, )


def iterate_length_n_strings(alphabet: int, n: int) -> Iterator[list[int]]:
    return map(list, itertools.product(range(1, alphabet+1), repeat=n))


def iterate_length_less_than_n_strings(
    alphabet: int, n: int
) -> Iterator[list[int]]:
    count = range(n)
    iterables = map(lambda n: iterate_length_n_strings(alphabet, n), count)
    return itertools.chain.from_iterable(iterables)


def iterate_every_string(alphabet: int) -> Iterator[list[int]]:
    count = itertools.count()
    iterables = map(lambda n: iterate_length_n_strings(alphabet, n), count)
    return itertools.chain.from_iterable(iterables)
