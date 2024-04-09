import json
from pathlib import Path

from qiskit_ibm_runtime import QiskitRuntimeService, Sampler  # type: ignore
from qiskit.transpiler.preset_passmanagers import (  # type: ignore
        generate_preset_pass_manager)

from qfa_toolkit.quantum_finite_automaton_language import (
    MeasureOnceQuantumFiniteAutomatonLanguage as Moqfl)
from qfa_toolkit.qiskit_converter import (
    QiskitMeasureOnceQuantumFiniteAutomaton as QMoqfa)

from qiskit_aer.noise import NoiseModel  # type: ignore
from qiskit_aer import AerSimulator  # type: ignore



n = 6
shots = 10000
primes = [3, 5, 7, 11]  # 13, 17, 19, 23, 29, 31, 37, 41]

# %%
service = QiskitRuntimeService()
backend = service.get_backend('ibm_cusco')
noise_model = NoiseModel.from_backend(backend)
# add or reduce noise here
aer = AerSimulator(noise_model=noise_model)
print('built noise sim')

# %%
primes = [3, 5, 7, 11]
pm = generate_preset_pass_manager(optimization_level=3, backend=aer)

for prime in primes:
    print(f"prime: {prime}")
    moqfl = Moqfl.from_modulo_prime(prime)
    moqfa = moqfl.quantum_finite_automaton
    qmoqfa = QMoqfa(moqfa)
    qmoqfa.__str__
    for k in range(1, 20):
        w = [1]*k
        print(f"word: {w}")
        circuit = qmoqfa.get_circuit_for_string(w)
        transpiled_circuit = pm.run(circuit)

        job = Sampler(backend=aer).run(transpiled_circuit, shots=shots)
        print(f"job id: {job.job_id()}")

        result = job.result()
        print(result.quasi_dists[0])

        counts = {int(k): v for k, v in result.quasi_dists[0].items()}

        accepting_states = qmoqfa.accepting_states
        rejection_states = qmoqfa.rejecting_states

        observed_acceptance = sum(
            counts.get(state, 0) for state in qmoqfa.accepting_states)
        observed_rejection = sum(
            counts.get(state, 0) for state in qmoqfa.rejecting_states)
        observed = [observed_acceptance, observed_rejection]

        expected_acceptance = moqfa(w)
        expected_rejection = 1.0 - expected_acceptance
        expected = [expected_acceptance, expected_rejection]

        print(f"observed: {observed}")
        print(f"expected: {expected}")
