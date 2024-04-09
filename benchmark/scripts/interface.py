# %%
import json
from pathlib import Path

from qiskit_ibm_runtime import QiskitRuntimeService, Sampler  # type: ignore
from qiskit.transpiler.preset_passmanagers import (  # type: ignore
        generate_preset_pass_manager)
from qiskit import qpy  # type: ignore

from qfa_toolkit.quantum_finite_automaton_language import (
    MeasureOnceQuantumFiniteAutomatonLanguage as Moqfl)
from qfa_toolkit.qiskit_converter import (
    QiskitMeasureOnceQuantumFiniteAutomaton as QMoqfa)

import os
import json


n = 6
shots = 10000
primes = [3, 5, 7, 11]  # 13, 17, 19, 23, 29, 31, 37, 41]

# %%
min_num_qubits = 6
service = QiskitRuntimeService()
backend = service.least_busy(
    min_num_qubits=min_num_qubits, operational=True, simulator=False)
print(f"backend: {backend}")

# %%
results_dir = Path("../results")
circuits_dir = results_dir / "circuits"
circuits_dir.mkdir(parents=True, exist_ok=True)

# %%
primes = [3, 5, 7, 11]
pm = generate_preset_pass_manager(optimization_level=3, backend=backend)

for prime in primes:
    print(f"prime: {prime}")
    moqfl = Moqfl.from_modulo_prime(prime)
    moqfa = moqfl.quantum_finite_automaton
    qmoqfa = QMoqfa(moqfa)
    qmoqfa.__str__
    for k in range(1, 20):
        w = [1]*k
        circuit = qmoqfa.get_circuit_for_string(w)
        transpiled_circuit = pm.run(circuit)

        with open(circuits_dir / f"{prime}_{k}.qpy", 'wb') as f:
            qpy.dump(transpiled_circuit, f)
        job = Sampler(backend).run(transpiled_circuit, shots=shots)
        print(f"job id: {job.job_id()}")

        result = job.result()
        with open(results_dir / f"{prime}_{k}.json", 'w') as f:
            json.dump(result.quasi_dists[0], f)

        counts = {int(k): v for k, v in result.quasi_dists[0].items()}

        accepting_states = qmoqfa.accepting_states
        rejection_states = qmoqfa.rejecting_states

        observed_acceptance = sum(
            counts[state] for state in qmoqfa.accepting_states)
        observed_rejection = sum(
            counts[state] for state in qmoqfa.rejecting_states)
        observed = [observed_acceptance, observed_rejection]

        with open(f"results/{prime}_{k}_observed.json", 'w') as f:
            json.dump(observed, f)

        observed = [observed_rejection, observed_acceptance]

        expected_acceptance = int(moqfa(w) * shots)
        expected_rejection = shots - expected_acceptance
        expected = [expected_rejection, expected_acceptance]

        with open(f"results/{prime}_{k}_observed.json", 'w') as f:
            json.dump(observed, f)
            json.dump(expected, f)
