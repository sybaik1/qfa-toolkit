# QFA-toolkit
QFA-toolkit is for constructing a quantum finite automaton by using operations between quantum finite languages,
and a quick way to transpile them into quantum circuits with qiskit.
The toolkit consists of three main parts.
## 1-way Quantum Finite-state Automata
- Measure-Many one-way QFA (MM-QFA)
- Measure-Once one-way QFA (MO-QFA)
## 1-way Quantum Finite-state Language
- Measure-Many one-way QFL (MM-QFL)
- Measure-Once one-way QFL (MO-QFL)
## Qiskit Converter
- Qiskit Converter for MM-QFA
- Qiskit Converter for MO-QFA

# 1. Dependency and Installation
Our dependencies is managed by `pyproject.toml`.
The main dependencies are numpy and qiskit.

You can install our package by the following commands.
```bash
git clone git@github.com:sybaik1/qfa-toolkit.git
cd qfa-toolkit
pip install .
```

# 2. Quick tour
You can import the qfa-toolkit as an python package and use each of the classes to construct and test your QFA or QFL.
Building a QFA can be done by defining the transitions and accepting (rejecting) states of the QFA.
```python
import numpy as np
import math
from qfa_toolkit.quantum_finite_state_automaton import (
    MeasureOnceQuantumFiniteStateAutomaton as Moqfa)

# Moqfa construction
theta = math.pi / 3
a, b = math.cos(theta), math.sin(theta)
acceptings = np.array([1,0], dtype=bool)
transitions = np.array([
    [
        [1, 0],
        [0, 1],
    ],
    [
        [a, b],
        [-b, a],
    ],
    [
        [1, 0],
        [0, 1],
    ],
], dtype=np.cfloat)
moqfa = Moqfa(transitions, acceptings)
```

If you want to build a QFL, you need to additionally give the accepting strategy for the language.

```python
from qfa_toolkit.quantum_finite_state_automaton_language import (
    MeasureOnceQuantumFiniteStateAutomatonLanguage as Moqfl)
from qfa_toolkit.recognition_startegy import (
    NegativeOneSidedBoundedError as NegOneSided)

# Moqfl construction
strategy = NegOneSided(0.5)
moqfl = Moqfl(moqfa, strategy)
```

Now you can make a quantum circuit for simulation.
Also, you can simulate the circuit with the qiskit simulator.

```python
from qiskit.providers.basic_provider import BasicSimulator
from qfa_toolkit.qiskit_converter import (
    QiskitMeasureOnceQuantumFiniteStateAutomaton as QMoqfa)

# Circuit initialation
qiskit_moqfa = QMoqfa(moqfa)
word = [1] * 3
circuit = qiskit_moqfa.get_circuit_for_string(word)

# Circuit simulation on Qiskit
simulator = BasicSimulator()
job = simulator.run(circuit, shots=10000)
result = job.result()
counts = {
    int(k, base=2): v
    for k, v in result.get_counts().items()
}
observed_acceptance = sum(
    counts.get(state, 0) for state in counts in qiskit_moqfa.accepting_states)
observed_rejectance = sum(
    counts.get(state, 0) for state in counts in qiskit_moqfa.rejecting_states)
print(f'acceptance: {observed_acceptance}\n'
      f'rejectance: {observed_rejectance}')
```
A similar process can be done with MMQFAs. Also, you can do operations on QFLs to get different languages you desire.

# 3. Functionalities
Go to the document.pdf to see the details of each classes Functionalities.

# Reading Materials for QFA
If your not familiar with QFAs, and want to learn more about QFAs
you can find the basics in the following material.

https://is.muni.cz/th/dy49n/b-thesis-QFA.pdf

# Test
For testing there are additional dependencies with scipy.
The testing is done by unittest module by python.
Within the installed environment you can run the following test
```bash
python3 -m unittest
```
