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

moqfl = Moqfl.from_modulo_prime(23)
qmoqfa = QMoqfa(moqfl.quantum_finite_automaton)

n = 6
primes = [23, 29, 31, 37, 41, 43]

service = QiskitRuntimeService()
backend = service.least_busy(
        min_num_qubits=n,
        operational=True,
        simulator=False)
pm = generate_preset_pass_manager(optimization_level=3, backend=backend)

if not os.path.exists("results/circuits"):
    os.makedirs("results/circuits")

for prime in primes:
    for k in range(1, prime):
        circuit = qmoqfa.get_circuit_for_string([1]*k)
        transpiled_circuit = pm.run(circuit)
        with open(f"results/circuits/{prime}_{k}.qpy", 'wb') as f:
            qpy.dump(transpiled_circuit, f)
        job = Sampler(backend).run(transpiled_circuit, shots=10000)
        print(f"job id: {job.job_id()}")
        result = job.result()
        with open(f"results/{prime}_{k}.json", 'w') as f:
            json.dump(result.quasi_dists[0], f)
