import numpy as np
import math

from qiskit.circuit.library import UnitaryGate  # type: ignore


def unitary_matrix_to_circuit(unitary_matrix, label=None):
    """
    Use qiskit unitary gate to convert unitary matrix to circuit
    """
    circuit_size = math.ceil(math.log2(unitary_matrix.shape[0]))
    matrix_size = 2**circuit_size

    # make unitary matrix to be a square matrix of size 2^n
    convert_unitary_matrix = np.eye(matrix_size, dtype=complex)
    convert_unitary_matrix[:unitary_matrix.shape[0],
                           :unitary_matrix.shape[1]] = unitary_matrix
    return UnitaryGate(convert_unitary_matrix, label=label)
