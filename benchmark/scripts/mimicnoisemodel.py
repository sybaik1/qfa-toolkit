from copy import deepcopy
from collections import defaultdict

from qiskit.circuit import Gate, Measure  # type: ignore

from qiskit_aer.noise import (  # type: ignore
    NoiseModel, ReadoutError, depolarizing_error)

_CONTROL_FLOW_OPS = frozenset({
    "for_loop", "switch_case", "while_loop", "if_else",
})
_ALL_QUBIT_GATES = frozenset({
    "ecr", "cz", "id", "rz", "sx", "x", "rx", "rzz",
})


class MimicNoiseModel(NoiseModel):
    """Scaled depolarizing and readout noise model derived from a backend."""

    def __init__(
            self,
            backend,
            gate_error_mimic_rate: float = 0.5,
            readout_error_mimic_rate: float = 0.5,
            all_qubit: bool = False):
        self.backend = backend
        self._all_qubit = all_qubit
        basis_gates = [
            op for op in self.backend.operation_names
            if op not in _CONTROL_FLOW_OPS
        ]
        if all_qubit:
            basis_gates = [op for op in basis_gates if op in _ALL_QUBIT_GATES]
        super().__init__(basis_gates=basis_gates)
        self.gate_error_mimic_rate = gate_error_mimic_rate
        self.readout_error_mimic_rate = readout_error_mimic_rate
        self._get_basic_values()
        self._get_basic_errors()
        self._gen_noise_model()

    def _get_basic_values(self):
        self.basic_gates = list(self.basis_gates)
        self.target = deepcopy(self.backend.target)
        self.coupling_map = self.backend.coupling_map
        self.qubit_prop = self.backend.target.qubit_properties
        self.dt = self.backend.dt
        self.qubit_length = self.target.num_qubits

    def _get_basic_errors(self):
        # Filter disabled/uncalibrated hardware sentinels (e.g., error == 1.0 or <= 0).
        self.qubit_errors = []
        for q in range(self.qubit_length):
            inst_prop = self.target["measure"].get((q,))
            if inst_prop is not None and inst_prop.error is not None:
                if 0 < inst_prop.error < 0.5:
                    self.qubit_errors.append(float(inst_prop.error))

        self.gate_errors = defaultdict(list)
        allowed_gates = set(self.basic_gates)
        for gate_op, inst_prop_dic in self.target.items():
            if gate_op not in allowed_gates:
                continue
            operation = self.target.operation_from_name(gate_op)
            if isinstance(operation, Measure) or not isinstance(operation, Gate):
                continue
            for qubits, inst_prop in inst_prop_dic.items():
                if inst_prop is None or inst_prop.error is None:
                    continue
                if 0 < inst_prop.error < 0.5:
                    num_qubits = len(qubits)
                    dim = 2 ** num_qubits
                    error_param = min(float(inst_prop.error), dim / (dim + 1))
                    depol_param = dim * error_param / (dim - 1)
                    max_param = 4 ** num_qubits / (4 ** num_qubits - 1)
                    depol_param = min(depol_param, max_param)
                    self.gate_errors[gate_op].append((qubits, depol_param))

    def _gen_noise_model(self):
        self.mean_q_error = (
            sum(self.qubit_errors) / len(self.qubit_errors)
            if self.qubit_errors else 0.0
        )
        mimic_q_error = self.mean_q_error * self.readout_error_mimic_rate
        probs = [
            [1 - mimic_q_error, mimic_q_error],
            [mimic_q_error, 1 - mimic_q_error],
        ]
        if self._all_qubit:
            if self.readout_error_mimic_rate > 0 and mimic_q_error > 0:
                self.add_all_qubit_readout_error(ReadoutError(probs))
        else:
            for q in range(self.qubit_length):
                self.add_readout_error(ReadoutError(probs), [q])

        self.mean_depol_params: dict[str, float] = {}
        for gate_op, gate_errors in self.gate_errors.items():
            mean_depol_param = sum(
                depol_param for _, depol_param in gate_errors
            ) / len(gate_errors)
            self.mean_depol_params[gate_op] = mean_depol_param
            mimic_depol_param = mean_depol_param * self.gate_error_mimic_rate
            num_qubits = len(gate_errors[0][0])
            err = depolarizing_error(mimic_depol_param, num_qubits)
            if self._all_qubit:
                self.add_all_qubit_quantum_error(err, gate_op)
            else:
                for qubits, _ in gate_errors:
                    self.add_quantum_error(err, gate_op, qubits)


if __name__ == '__main__':
    from qiskit_ibm_runtime.fake_provider import (  # type: ignore
        FakeBrisbane, FakeFez)

    for backend in (FakeBrisbane(), FakeFez()):
        mimic_noise_model = MimicNoiseModel(backend, all_qubit=True)
        print(backend.name, mimic_noise_model.mean_q_error, mimic_noise_model.mean_depol_params)
