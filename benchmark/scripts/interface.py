import json
from pathlib import Path

from qiskit_ibm_runtime import QiskitRuntimeService, Sampler  # type: ignore
from qiskit.transpiler.preset_passmanagers import (  # type: ignore
        generate_preset_pass_manager)
from qiskit import qpy  # type: ignore

from qfa_toolkit.quantum_finite_state_automaton_language import (
    MeasureOnceQuantumFiniteStateAutomatonLanguage as Moqfl)
from qfa_toolkit.qiskit_converter import (
    QiskitMeasureOnceQuantumFiniteStateAutomaton as QMoqfa)


n = 6
shots = 10000
primes = [11]  # 13, 17, 19, 23, 29, 31, 37, 41]

# %%
min_num_qubits = 6
service = QiskitRuntimeService()
backend = service.get_backend('ibm_torino')
print(f"backend: {backend}")

# %%
results_dir = Path("../resultsWithMapping")
circuits_dir = results_dir / "circuits"
circuits_dir.mkdir(parents=True, exist_ok=True)

# %%
pm = generate_preset_pass_manager(optimization_level=3, backend=backend)

for prime in primes:
    moqfl = Moqfl.from_modulo_prime(prime)
    moqfa = moqfl.quantum_finite_state_automaton
    qmoqfa = QMoqfa(moqfa, True)
    qmoqfa.__str__
    for k in range(0, 2 * prime + 1):
        if k < 18:
            continue
        w = [1]*k
        print(f"prime: {prime}, k: {k}")

        circuit = qmoqfa.get_circuit_for_string(w)
        transpiled_circuit = pm.run(circuit)

        with open(circuits_dir / f"{prime}_{k}.qpy", 'wb') as f:
            qpy.dump(transpiled_circuit, f)

        try_limit = 10
        while try_limit > 0:
            try:
                job = Sampler(backend).run(transpiled_circuit, shots=shots)
                print(f"job id: {job.job_id()}")

                result = job.result()
                break
            except KeyboardInterrupt:
                print("KeyboardInterrupt")
                exit(1)
            except Exception as e:
                print(f"Exception: {e}")
                try_limit -= 1

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

        expected_acceptance = int(moqfa(w) * shots)
        expected_rejection = shots - expected_acceptance
        expected = [expected_rejection, expected_acceptance]

        with open(f"results/{prime}_{k}_observed.json", 'w') as f:
            json.dump(observed, f)
            json.dump(expected, f)
