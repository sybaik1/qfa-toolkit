import numpy as np
import math

from qiskit.circuit.library import UnitaryGate  # type: ignore


def unitary_matrix_to_circuit(unitary_matrix, label=None):
    """
    Use qiskit unitary gate to convert unitary matrix to circuit
    """
    return UnitaryGate(unitary_matrix, label=label)
