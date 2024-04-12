from copy import deepcopy
from collections import defaultdict

from qiskit.circuit import Gate, Measure  # type: ignore

from qiskit_aer.noise import (  # type: ignore
    NoiseModel, ReadoutError, depolarizing_error)


class MimicNoiseModel(NoiseModel):
    def __init__(self, backend, error_mimic_rate=0.5):
        self.backend = backend
        super().__init__(basis_gates=self.backend.operation_names)
        self.error_mimic_rate = error_mimic_rate
        self._get_basic_values()
        self._get_basic_errors()
        self._gen_noise_model()

    def _get_basic_values(self):
        self.basic_gates = self.backend.operation_names
        self.target = deepcopy(self.backend.target)
        self.qubit_prop = self.backend.target.qubit_properties
        self.dt = self.backend.dt
        self.qubit_length = self.target.num_qubits

    def _get_basic_errors(self):
        # get qubit error for measure
        self.qubit_errors = []
        for q in range(self.qubit_length):
            q_error = self.target["measure"][(q,)].error
            self.qubit_errors.append(q_error)

        # get gate erros on qubits
        self.gate_errors = defaultdict(list)
        for gate_op, inst_prop_dic in self.target.items():
            operation = self.target.operation_from_name(gate_op)
            if isinstance(operation, Measure):
                continue
            for qubits, inst_prop in inst_prop_dic.items():
                if inst_prop is None:
                    continue
                if inst_prop.error and isinstance(operation, Gate):
                    num_qubits = len(qubits)

                    dim = 2**num_qubits
                    error_max = dim / (dim+1)
                    error_param = min(inst_prop.error, error_max)

                    depol_param = dim * error_param / (dim-1)
                    max_param = 4**num_qubits / (4**num_qubits - 1)
                    depol_param = min(depol_param, max_param)
                    self.gate_errors[gate_op].append((qubits, depol_param))

    def _gen_noise_model(self):
        # mean readout error on each qubit
        mean_q_error = sum(self.qubit_errors) / self.qubit_length
        mimic_q_error = mean_q_error * self.error_mimic_rate
        probs = [
            [1-mimic_q_error, mimic_q_error],
            [mimic_q_error, 1-mimic_q_error]]
        for q in range(self.qubit_length):
            self.add_readout_error(ReadoutError(probs), [q])

        # mean gate error on each gate
        for gate_op, gate_errors in self.gate_errors.items():
            mean_depol_param = sum([
                depol_param
                for _, depol_param in gate_errors]) / len(gate_errors)
            mimic_depol_param = mean_depol_param * self.error_mimic_rate
            for qubits, _ in gate_errors:
                self.add_quantum_error(
                    depolarizing_error(mimic_depol_param, len(qubits)),
                    gate_op,
                    qubits)


if __name__ == '__main__':
    from qiskit_ibm_runtime import QiskitRuntimeService  # type: ignore
    provider = QiskitRuntimeService()
    backend = provider.get_backend('ibm_hanoi')
    mimic_noise_model = MimicNoiseModel(backend)
    print(mimic_noise_model)
