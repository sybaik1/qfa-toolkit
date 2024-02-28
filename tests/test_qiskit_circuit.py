import unittest
import sys

from scipy.stats import chisquare  # type: ignore
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler  # type: ignore

import qfa_toolkit.qiskit_converter as qc
import qfa_toolkit.recognition_strategy as rs

from .utils import get_measure_many_quantum_finite_automaton

Result = rs.RecognitionStrategy.Result

# QiskitRuntimeService.save_account(
#         channel='ibm_quantum',
#         token='<IBM_TOKEN>')

service = QiskitRuntimeService()


class TestQiskitCircuit(unittest.TestCase):

    def test_qiskit_circuit(self):
        backend = service.backend('ibmq_qasm_simulator')
        shots = 10000

        for k in range(1, 4):
            r = 1 / k
            mmqfa = get_measure_many_quantum_finite_automaton(r)
            qc_mmqfa = qc.QiskitMeasureManyQuantumFiniteAutomaton(mmqfa)

            for n in range(5):
                w = [1] * n
                circuit = qc_mmqfa.get_circuit_for_string(w)
                job = Sampler(backend).run(circuit, shots=shots)
                result = job.result()
                accept, reject = 0, 0
                for key in result.quasi_dists[0].keys():
                    if key in qc_mmqfa.accepting_states:
                        accept += result.quasi_dists[0][key]
                    elif key in qc_mmqfa.rejecting_states:
                        reject += result.quasi_dists[0][key]
                    else:
                        print(f'Error key: {key}', file=sys.stderr)
                expected_acceptance = mmqfa(w) * shots
                if expected_acceptance == shots:
                    self.assertEqual(accept, 1.0)
                elif expected_acceptance == 0:
                    self.assertEqual(reject, 1.0)
                else:
                    chi_value = chisquare(
                        [accept*shots, reject*shots],
                        [expected_acceptance, shots - expected_acceptance])
                    self.assertGreater(chi_value.pvalue, 0.05)


if __name__ == '__main__':
    unittest.main()
