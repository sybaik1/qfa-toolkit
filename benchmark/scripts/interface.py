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
    qmoqfa = QMoqfa(moqfl.quantum_finite_automaton)
    for k in range(0, 2 * prime + 1):
        w = [1] * k
        circuit = qmoqfa.get_circuit_for_string(w)
        transpiled_circuit = pm.run(circuit)

        with open(circuits_dir / f"{prime}_{k}.qpy", 'wb') as f:
            qpy.dump(transpiled_circuit, f)

        job = Sampler(backend).run(transpiled_circuit, shots=10000)
        print(f"job id: {job.job_id()}")

        result = job.result()
        with open(results_dir / f"{prime}_{k}.json", 'w') as f:
            json.dump(result.quasi_dists[0], f)

        counts = {int(k): v for k, v in result.quasi_dists[0].items()}

        accepting_states = qmoqfa.accepting_states
        rejection_states = qmoqfa.rejection_states
        observed_acceptance = sum(
            counts[state] for state in qmoqfa.accepting_states)
        observed_rejection = sum(
            counts[state] for state in qmoqfa.rejection_states)
        observed = [observed_acceptance, observed_rejection]

        with open(f"results/{prime}_{k}_observed.json", 'w') as f:
            json.dump(observed, f)

# %%
