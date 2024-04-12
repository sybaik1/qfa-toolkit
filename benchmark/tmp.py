from qiskit_ibm_runtime import QiskitRuntimeService  # type: ignore
# from qiskit_aer.noise import NoiseModel  # type: ignore
from qiskit_aer import AerSimulator  # type: ignore


service = QiskitRuntimeService()
backend = service.get_backend('ibm_hanoi')
print(backend.properties().to_dict())
aer = AerSimulator.from_backend(backend=backend)
