# Description: This file contains the main function to run the experiments.
import json
from itertools import product

from qiskit_ibm_runtime import QiskitRuntimeService  # type: ignore

from qiskit_aer import AerSimulator  # type: ignore

from wrapper import ExperimentHandler
from mimicnoisemodel import MimicNoiseModel

from qfa_toolkit.quantum_finite_state_automaton_language import (
    MeasureOnceQuantumFiniteStateAutomatonLanguage as Moqfl,
    MeasureManyQuantumFiniteStateAutomatonLanguage as Mmqfl)


experiments = [
    {
        'language':
        [
            Moqfl.from_modulo_prime(5, length) for length in range(5, 13)
        ],
        'strings':
        [
            [1] * length for length in range(2, 21)
        ],
        'settings':
        {
            'name': 'mapping effect, QFA state number effect',
            'shots': 100000,
            'simulator': True,
            'mimic_rate': 1,
            'use_entropy_mapping': entropy_mapping,
            'use_mapping_noise_correction': False
        }
    } for entropy_mapping in [False, True]
] + [
    {
        'language':
        [
            Moqfl.from_modulo_prime(prime) for prime in [3, 5, 7, 11]
        ] + [
            Mmqfl.from_unary_singleton(length) for length in range(2, 13)
        ],
        'strings':
        [
            [1] * length for length in range(2, 21)
        ],
        'settings':
        {
            'name': 'mapping effect, mimic rate {mimic_rate}',
            'shots': 100000,
            'simulator': True,
            'mimic_rate': mimic_rate,
            'use_entropy_mapping': entropy_mapping,
            'use_mapping_noise_correction': False
        }
    } for entropy_mapping, mimic_rate in product(
        [False, True], [0.01, 0.005, 0.001, 0.0005, 0])
]

numbered_experiment = []
index = 0
for experiment in experiments:
    for language in experiment['language']:
        numbered_experiment.append(
            (
                index,
                {
                    'language': language,
                    'strings': experiment['strings'],
                    'settings': experiment['settings']
                }
            )
        )
        index += 1


def main():
    file_path = r'benchmark/results/'

    service = QiskitRuntimeService()
    backend = service.get_backend('ibm_hanoi')

    for index, experiment in numbered_experiment:
        print(f'Running experiment {index}')
        noise_model = MimicNoiseModel(
            backend,
            experiment['settings']['mimic_rate'])
        aer = AerSimulator(noise_model=noise_model)
        qfl = experiment['language']
        handler = ExperimentHandler(
            qfl,
            experiment['strings'],
            aer,
            experiment['settings']['use_entropy_mapping'],
            shots=experiment['settings']['shots'])
        handler.make_pool()
        results = handler.run(True)
        with open(f'{file_path}experiment_{index}.json', 'w') as f:
            json.dump(results, f, indent=4)
        with open(f'{file_path}experiment_{index}_metadata.json', 'w') as f:
            metadata = handler.metadata
            metadata.update(experiment['settings'])
            json.dump(metadata, f, indent=4)


if __name__ == '__main__':
    main()
