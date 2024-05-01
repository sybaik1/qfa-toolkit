# Description: This file contains the main function to run the experiments.
import json
import math
from functools import reduce

from qiskit_ibm_runtime import QiskitRuntimeService  # type: ignore
from qiskit_aer import AerSimulator  # type: ignore

from wrapper import ExperimentHandler
from mimicnoisemodel import MimicNoiseModel

from qfa_toolkit.quantum_finite_state_automaton_language import (
    MeasureOnceQuantumFiniteStateAutomatonLanguage as Moqfl,
    MeasureManyQuantumFiniteStateAutomatonLanguage as Mmqfl)


singleton_mimic_rate = [1, 0.5, 0.1, 0.01, 0.005, 0]
mod_mimic_rate = [1, 0.5, 0.1, 0.01, 0.005, 0]
mod_entropy_mapping = [False, True]


def experiments(start: int = 0, shots: int = 100000):
    index = 0
    if start < 64:
        yield from [
            {
                'language': Moqfl.from_modulo_prime(prime, copy_num),
                'strings':
                [
                    [1] * length for length in range(2, 41)
                ],
                'settings':
                {
                    'language': f'mod_{prime}',
                    'name': 'mapping effect, change QFA state number',
                    'shots': shots,
                    'simulator': True,
                    'mimic_rate_gate': 1,
                    'mimic_rate_readout': 1,
                    'use_entropy_mapping': entropy_mapping,
                    'use_mapping_noise_correction': False
                }
            }
            for prime in [3, 5, 7, 11]
            for copy_num in range(5, 13)
            for entropy_mapping in [False, True]
        ]
    index = 64
    if start < 64 + 48:
        yield from [
            {
                'language': Moqfl.from_modulo_prime(prime),
                'strings':
                [
                    [1] * length for length in range(2, 41)
                ],
                'settings':
                {
                    'language': f'mod_{prime}',
                    'name': 'mapping effect, mimic rate change',
                    'shots': shots,
                    'simulator': True,
                    'mimic_rate_gate': mimic_rate,
                    'mimic_rate_readout': mimic_rate,
                    'use_entropy_mapping': entropy_mapping,
                    'use_mapping_noise_correction': False
                }
            }
            for prime in [3, 5, 7, 11]
            for entropy_mapping in [False, True]
            for mimic_rate in [1, 0.5, 0.1, 0.01, 0.005, 0]
        ]
    index = 112
    if start < 112 + 66:
        yield from [
            {
                'language': Mmqfl.from_unary_singleton(k),
                'strings':
                [
                    [1] * length for length in range(2, 41)
                ],
                'settings':
                {
                    'language': f'singleton_{k}',
                    'name': 'mapping effect, mimic rate change',
                    'shots': shots,
                    'simulator': True,
                    'mimic_rate_gate': mimic_rate,
                    'mimic_rate_readout': mimic_rate,
                    'use_entropy_mapping': True,
                    'use_mapping_noise_correction': False
                }
            }
            for k in range(2, 13)
            for mimic_rate in [1, 0.5, 0.1, 0.01, 0.005, 0]
        ]
    index = 178
    if start < 178 + 16*8:
        yield from [
            {
                'language': reduce(
                    Mmqfl.intersection,
                    [Mmqfl.from_unary_singleton(k) for _ in range(repeat)]),
                'strings':
                [
                    [1] * length for length in range(0, 41)
                ],
                'settings':
                {
                    'language': f'singleton_{k}',
                    'name': 'mapping effect, mimic rate change, intersection',
                    'shots': shots/100,
                    'simulator': True,
                    'mimic_rate_gate': mimic_rate_gate,
                    'mimic_rate_readout': mimic_rate_readout,
                    'use_entropy_mapping': True,
                    'use_mapping_noise_correction': False
                }
            }
            for k in range(1, 5)
            for repeat in range(3, 5)
            for mimic_rate_gate in [0.001, 0.0005, 0.0001, 0.00005]
            for mimic_rate_readout in [0.001, 0.0005, 0.0001, 0.00005]
        ]
    index = 306
    if start < 306 + 100:
        yield from [
            {
                'language': Mmqfl.from_unary_singleton(k),
                'strings':
                [
                    [1] * length for length in range(0, 41)
                ],
                'settings':
                {
                    'language': f'singleton_{k}',
                    'name': 'mapping effect, mimic rate change',
                    'shots': shots/100,
                    'simulator': True,
                    'mimic_rate_gate': mimic_rate_gate,
                    'mimic_rate_readout': mimic_rate_readout,
                    'use_entropy_mapping': True,
                    'use_mapping_noise_correction': False
                }
            }
            for k in [3, 5, 7, 11]
            for mimic_rate_gate in [0.01, 0.005, 0.001, 0.0005, 0]
            for mimic_rate_readout in [0.01, 0.005, 0.001, 0.0005, 0]
        ]
    index = 406
    if start < 406 + 18:
        yield from [
            {
                'language': Moqfl.from_modulo_prime(p),
                'strings':
                [
                    [1] * length for length in range(0, 41)
                ],
                'settings':
                {
                    'language': f'mod_{p}',
                    'name': 'mapping effect, mimic rate change',
                    'shots': shots/100,
                    'simulator': True,
                    'mimic_rate_gate': mimic_rate_gate,
                    'mimic_rate_readout': mimic_rate_readout,
                    'use_entropy_mapping': mapping,
                    'use_mapping_noise_correction': False
                }
            }
            for p in [3]
            for mimic_rate_gate in [0.01, 0.005, 0.001]
            for mimic_rate_readout in [0.01, 0.005, 0.001]
            for mapping in [False, True]
        ]
    index = 424
    if start < 424 + 50:
        keys = [
            (k, repeat, mimic_rate_gate, mimic_rate_readout)
            for k in [3]
            for repeat in range(2, 3)
            for mimic_rate_gate in [0.01, 0.005, 0.001, 0.0005, 0]
            for mimic_rate_readout in [0.01, 0.005, 0.001, 0.0005, 0]
        ]
        for k, repeat, mimic_rate_gate, mimic_rate_readout in keys:
            if index < start:
                index += 1
                continue
            yield {
                'language': reduce(
                    Mmqfl.union,
                    [Mmqfl.from_unary_singleton(k) for _ in range(repeat)]),
                'strings':
                [
                    [1] * length for length in range(0, 41)
                ],
                'settings':
                {
                    'language': f'singleton_{k}',
                    'name': 'mapping effect, mimic rate change, union',
                    'shots': shots/10,
                    'simulator': True,
                    'mimic_rate_gate': mimic_rate_gate,
                    'mimic_rate_readout': mimic_rate_readout,
                    'use_entropy_mapping': True,
                    'use_mapping_noise_correction': False
                }
            }
    index = 474
    if start < 474 + 32:
        yield from [
            {
                'language': Mmqfl.from_unary_singleton(3),
                'strings':
                [
                    [1] * length for length in range(1, 41)
                ],
                'settings':
                {
                    'language': 'singleton_3',
                    'name': 'mapping effect, mimic rate change',
                    'shots': shots/10,
                    'simulator': True,
                    'mimic_rate_gate': mimic_rate_gate,
                    'mimic_rate_readout': mimic_rate_readout,
                    'use_entropy_mapping': entropy_mapping,
                    'use_mapping_noise_correction': False
                }
            }
            for entropy_mapping in [False, True]
            for mimic_rate_gate in [0.01, 0.005, 0.001, 0]
            for mimic_rate_readout in [0.01, 0.005, 0.001, 0]
        ]
    if start < 506 + 144:
        yield from [
            {
                'language': Moqfl.from_modulo_prime(prime, copy_num),
                'strings':
                [
                    [1] * length for length in range(2, 41)
                ],
                'settings':
                {
                    'language': f'mod_{prime}',
                    'name': 'mapping effect, change QFA state number',
                    'shots': shots,
                    'simulator': True,
                    'mimic_rate_gate': mimic_rate_gate,
                    'mimic_rate_readout': mimic_rate_readout,
                    'use_entropy_mapping': entropy_mapping,
                    'use_mapping_noise_correction': False
                }
            }
            for prime in [3]
            for copy_num in range(5, 13)
            for mimic_rate_gate in [0.01, 0.005, 0.001]
            for mimic_rate_readout in [0.01, 0.005, 0.001]
            for entropy_mapping in [False, True]
        ]
    if start < 650 + 36:
        yield from [
            {
                'language': Moqfl.from_modulo_prime(prime),
                'strings':
                [
                    [1] * length for length in range(2, 41)
                ],
                'settings':
                {
                    'language': f'mod_{prime}',
                    'name': 'mapping random',
                    'shots': shots,
                    'simulator': True,
                    'mimic_rate_gate': mimic_rate_gate,
                    'mimic_rate_readout': mimic_rate_readout,
                    'use_entropy_mapping': False,
                    'use_mapping_noise_correction': False
                }
            }
            for prime in [3, 5, 7, 11]
            for mimic_rate_gate in [0.01, 0.005, 0.001]
            for mimic_rate_readout in [0.01, 0.005, 0.001]
        ]


def main():
    file_path = r'benchmark/results/sim/'

    service = QiskitRuntimeService()
    mimic_backend = service.get_backend('ibm_cusco')
    backend = service.get_backend('ibmq_qasm_simulator')

    start_experiment = 650
    total_experiments = 686

    for index, experiment in enumerate(experiments(start_experiment)):
        index = index + start_experiment
        print(f'\
Running experiment {index}/{total_experiments}: \
{experiment["settings"]["name"]} \
on {experiment["settings"]["language"]}\
        ')
        noise_model = MimicNoiseModel(
            mimic_backend,
            experiment['settings']['mimic_rate_gate'],
            experiment['settings']['mimic_rate_readout'])

        backend.options.update_options(noise_model=noise_model)

        qfl = experiment['language']
        handler = ExperimentHandler(
            qfl,
            experiment['strings'],
            backend,
            experiment['settings']['use_entropy_mapping'],
            shots=experiment['settings']['shots'])
        handler.make_pool(True)
        results = handler.run(True)
        with open(f'{file_path}experiment_{index}.json', 'w') as f:
            json.dump(results, f, indent=4)
        with open(f'{file_path}experiment_{index}_metadata.json', 'w') as f:
            metadata = handler.metadata
            metadata.update(experiment['settings'])
            json.dump(metadata, f, indent=4)


if __name__ == '__main__':
    main()
