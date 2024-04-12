import unittest

from qiskit.providers.basic_provider import BasicSimulator  # type: ignore
from scipy.stats import chisquare  # type: ignore

import qfa_toolkit.qiskit_converter as qc
import qfa_toolkit.recognition_strategy as rs

from .utils import get_arbitrary_moqfa, get_arbitrary_mmqfa

Result = rs.RecognitionStrategy.Result


class TestQiskitCircuit(unittest.TestCase):

    def setUp(self):
        self.simulator = BasicSimulator()

    def test_qiskit_measure_once_quantum_finite_state_automaton_circuit(self):
        shots = 10000

        for k in range(1, 4):
            moqfa = get_arbitrary_moqfa(k)
            qc_moqfa = qc.QiskitMeasureOnceQuantumFiniteStateAutomaton(moqfa)

            for n in range(5):
                w = [1] * n
                circuit = qc_moqfa.get_circuit_for_string(w)
                job = self.simulator.run(circuit, shots=shots)
                result = job.result()
                counts = {
                    int(k, base=2): v
                    for k, v in result.get_counts().items()
                }

                observed_rejection = sum(
                    counts[state] for state in counts
                    if state in qc_moqfa.rejecting_states
                )
                observed_acceptance = sum(
                    counts[state] for state in counts
                    if state in qc_moqfa.accepting_states
                )
                observed = [observed_rejection, observed_acceptance]

                expected_acceptance = int(moqfa(w) * shots)
                expected_rejection = shots - expected_acceptance
                expected = [expected_rejection, expected_acceptance]

                if expected_acceptance == 0:
                    self.assertEqual(expected_acceptance, observed_acceptance)
                    return
                if expected_rejection == 0:
                    self.assertEqual(expected_rejection, observed_rejection)
                    return

                chi_value = chisquare(observed, expected)
                self.assertGreater(chi_value.pvalue, 0.05)

    def test_qiskit_measure_many_quantum_finite_state_automaton_circuit(self):
        shots = 10000

        for k in range(1, 4):
            r = 1 / k
            mmqfa = get_arbitrary_mmqfa(r)
            qc_mmqfa = qc.QiskitMeasureManyQuantumFiniteStateAutomaton(mmqfa)

            for n in range(5):
                w = [1] * n
                circuit = qc_mmqfa.get_circuit_for_string(w)
                job = self.simulator.run(circuit, shots=shots)
                result = job.result()
                counts = {
                    int(k, base=2): v
                    for k, v in result.get_counts().items()
                }

                observed_rejection = sum(
                    counts[state] for state in counts
                    if state in qc_mmqfa.rejecting_states
                )
                observed_acceptance = sum(
                    counts[state] for state in counts
                    if state in qc_mmqfa.accepting_states
                )
                observed = [observed_rejection, observed_acceptance]

                expected_acceptance = int(mmqfa(w) * shots)
                expected_rejection = shots - expected_acceptance
                expected = [expected_rejection, expected_acceptance]

                if expected_acceptance == 0:
                    self.assertEqual(expected_acceptance, observed_acceptance)
                    return
                if expected_rejection == 0:
                    self.assertEqual(expected_rejection, observed_rejection)
                    return

                chi_value = chisquare(observed, expected)
                self.assertGreater(chi_value.pvalue, 0.05)


if __name__ == '__main__':
    unittest.main()
