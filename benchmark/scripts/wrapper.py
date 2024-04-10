from typing import Union

from qiskit_ibm_runtime import QiskitRuntimeService  # type: ignore
from qiskit.transpiler.preset_passmanagers import (  # type: ignore
        generate_preset_pass_manager)

from qfa_toolkit.quantum_finite_automaton import (
    MeasureOnceQuantumFiniteAutomaton as Moqfa,
    MeasureManyQuantumFiniteAutomaton as Mmqfa)
from qfa_toolkit.quantum_finite_automaton_language import (
    MeasureOnceQuantumFiniteAutomatonLanguage as Moqfl,
    MeasureManyQuantumFiniteAutomatonLanguage as Mmqfl)
from qfa_toolkit.qiskit_converter import (
    QiskitMeasureOnceQuantumFiniteAutomaton as QMoqfa,
    QiskitMeasureManyQuantumFiniteAutomaton as QMmqfa)

from qiskit_aer.noise import (  # type: ignore
    NoiseModel,)
from qiskit_aer import AerSimulator  # type: ignore


# TODO
class CustumNoiseModel(NoiseModel):
    def __init__(self, noise_model):
        self.noise_model = noise_model

    def __call__(self, circuit):
        return self.noise_model.run(circuit)


class ExperimentHandler:
    def __init__(
            self,
            test_subject: Union[Moqfa, Moqfl, Mmqfa, Mmqfl],
            words: list[list[int]],  # canditates for testing on the QFA or QFL
            backend,
            use_entropy_mapping: bool = True,
            simulator: bool = False,
            shots: int = 10000,
            retry_limit: int = 3):
        self.test_subject = test_subject
        self.qfa: Union[Moqfa, Mmqfa]
        self.Qqfa: Union[QMoqfa, QMmqfa]
        if isinstance(test_subject, Moqfl):
            self.qfa = test_subject.quantum_finite_automaton
        elif isinstance(test_subject, Moqfa):
            self.qfa = test_subject
        elif isinstance(test_subject, Mmqfl):
            self.qfa = test_subject.quantum_finite_automaton
        elif isinstance(test_subject, Mmqfa):
            self.qfa = test_subject

        if isinstance(self.qfa, Moqfa):
            self.Qqfa = QMoqfa(self.qfa, use_entropy_mapping)
        elif isinstance(self.qfa, Mmqfa):
            self.Qqfa = QMmqfa(self.qfa, use_entropy_mapping)

        self.words = words
        if simulator:
            self.backend = AerSimulator.from_backend(backend=backend)
        else:
            self.backend = backend
        self.shots = shots

        self.pm = generate_preset_pass_manager(
            optimization_level=1,
            backend=self.backend)

        self.retry_limit = retry_limit

    def make_pool(self):
        self.circuits = list()
        for w in self.words:
            circuit = self.Qqfa.get_circuit_for_string(w)
            self.circuits.append((w, self.pm.run(circuit)))

    def change_noise_model(self, noise_model):
        self.backend = AerSimulator(noise_model=noise_model)

    def _run(self, circuit) -> tuple[float, float]:
        job = self.backend.run(circuit, shots=self.shots)
        result = job.result()

        counts = {int(k, base=2): v for k, v in result.get_counts(0).items()}

        accepting_states = self.Qqfa.accepting_states
        rejection_states = self.Qqfa.rejecting_states

        observed_acceptance = sum(
            counts.get(state, 0) for state in accepting_states)/self.shots
        observed_rejection = sum(
            counts.get(state, 0) for state in rejection_states)/self.shots
        observed = (observed_acceptance, observed_rejection)

        return observed

    def run(self) -> dict[str, dict[str, tuple[float, float]]]:
        self.results = dict()
        for w, circuit in self.circuits:
            retry_count = 0
            while retry_count < self.retry_limit:
                try:
                    observed = self._run(circuit)
                    break
                except Exception as e:
                    print(e)
                    print('retrying...')
                    retry_count += 1

            expected_acceptance = self.qfa(w)
            expected_rejection = 1 - expected_acceptance
            expected = (expected_acceptance, expected_rejection)

            self.results[''.join(map(str, w))] = {
                'observed': observed, 'expected': expected}

        return self.results


# %%
if __name__ == '__main__':
    service = QiskitRuntimeService()
    backend = service.get_backend('ibm_hanoi')
    aer = AerSimulator.from_backend(backend=backend)

    for prime in [3]:
        qfl = Moqfl.from_modulo_prime(prime)
        handler = ExperimentHandler(
            qfl,
            [[1], [1, 1], [1, 1, 1]],
            aer)
        handler.make_pool()
        results = handler.run()
        print(results)
