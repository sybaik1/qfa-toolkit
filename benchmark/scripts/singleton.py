# %%
import json
from pathlib import Path

from qiskit_ibm_runtime import QiskitRuntimeService, Sampler  # type: ignore
from qiskit.transpiler.preset_passmanagers import (  # type: ignore
        generate_preset_pass_manager)
from qiskit import qpy  # type: ignore

from qfa_toolkit.quantum_finite_automaton_language import (
    MeasureManyQuantumFiniteStateAutomatonLanguage as Mmqfl)
from qfa_toolkit.qiskit_converter import (
    QiskitMeasureManyQuantumFiniteStateAutomaton as QMmqfa)

import os
import json

# %%
min_num_qubits = 3
service = QiskitRuntimeService()
backend = service.get_backend('ibm_sherbrooke')
print(f"backend: {backend}")

# %%
results_dir = Path("../results/singleton")
circuits_dir = results_dir / "circuits"
results_dir.mkdir(parents=True, exist_ok=True)
circuits_dir.mkdir(parents=True, exist_ok=True)

# %%
shots = 10000
ns = list(range(1, 12, 2))
pm = generate_preset_pass_manager(optimization_level=3, backend=backend)

for n in ns:
    print(f"n: {n}")
    mmqfl = Mmqfl.from_unary_singleton(n)
    mmqfa = mmqfl.quantum_finite_automaton
    qmmqfa = QMmqfa(mmqfa)
    for k in range(max(n-2, 0), n+3):
        w = [1]*k
        circuit = qmmqfa.get_circuit_for_string(w)
        transpiled_circuit = pm.run(circuit)

        with open(circuits_dir / f"{n}_{k}.qpy", 'wb') as f:
            qpy.dump(transpiled_circuit, f)
        job = Sampler(backend).run(transpiled_circuit, shots=shots)
        print(f"job id: {job.job_id()}")

        result = job.result()
        with open(results_dir / f"{n}_{k}.json", 'w') as f:
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
        expected = [expected_acceptance, expected_rejection]

        with open(f"results/{n}_{k}_observed.json", 'w') as f:
            json.dump(observed, f)
            json.dump(expected, f)
