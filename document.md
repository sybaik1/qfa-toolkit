# Toolkit Documentation

## qfa\_toolkit.quantum\_finite\_state\_automaton

### Superposition

A type-alias of `ndarray[Any, cdouble]` for superposition.

Implicitly, it represents an `n`-dimensional vector whose norm is 1, where `n`
is the number of states.

### States

A type-alias of `ndarray[Any, bool]` for a set of states.

Implicitly, it represents a set of states with one-hot encoding.

### Transitions

A type-alias of `ndarray[Any, cdouble]` for a set of transition matrices.

Implicitly, it is an `(m, n, n)`-shaped array and each `(n, n)`-shaped subarray
is [Transition](#transition).

### Transition

A type-alias of `ndarray[Any, cdouble]` for a transition matrix.

Implicitly, its shape is `(n, n)` and it denotes a unitary transition matrix.

### Observable

A type-alias of `ndarray[Any, bool]` for an observable.

Implicitly, it is a `(3, n)`-shaped array and each `n`-shaped subarray is
[States](#states) which denotes accepting, rejecting, or non-halting states,
respectively.

### TotalState

Class for total state in quantum finite-state automaton.

* Constructor
  ```
  TotalState(
    superposition_or_list: Superposition | list[complex],
    acceptance: float = 0,
    rejection: float = 0
  )
  ```
* Properties
  ```
  superposition: Superposition
  acceptance: float
  rejection: float
  ```
* Methods
  ```
  initial(states: int) -> TotalState
  measure_by(observable: Observable) -> TotalState
  apply(unitary: Transition) -> TotalState
  to_tuple() -> tuple[Superposition, float, float]

  normalized() -> TotalState
  # Normalize the total state so that its norm is 1.
  ```

### QuantumFiniteStateAutomatonBase

Abstract class for quantum finite-state automaton.

* Properties
  ```
  alphabet: int
  states: int
  start_of_string: int
  end_of_string: int
  initial_transition: Transition
  final_transition: Transition
  observable: Observable
  ```
* Methods
  ```
  __call__(w: list[int]) -> float
  # Return the acceptance probability f_M(w).

  process(
    w: list[int],
    total_state: TotalState | None = None
  ) -> TotalState

  step(total_state: TotalState, c: int) -> TotalState
  string_to_tape(string: list[int]) -> list[int]
  ```

### MeasureOnceQuantumFiniteStateAutomaton

Class for Measure-once quantum finite-state automaton.

Subclass of [QuantumFiniteStateAutomatonBase](#quantumfinitestateautomatonbase).

Abbreviation: MOQFA

* Constructor
  ```
  MeasureOnceQuantumFiniteStateAutomaton(
    transitions: Transitions,
    accepting_states: States
  )
  ```
* Properties
  ```
  accepting_states: States
  rejecting_states: States
  observable: Observable
  ```
* Methods
  ```
  word_transition(w: list[int]) -> Transition

  union(other: MOQFA | MMQFA) -> MOQFA | MMQFA
  __or__(other: MOQFA | MMQFA) -> MOQFA | MMQFA
  # Complement of Hadamard product of complements.
  # Supports MOQFA | MMQFA for co-end-decisive MMQFAs, returning an
  # mn-state co-end-decisive MMQFA.

  intersection(other: MOQFA | MMQFA) -> MOQFA | MMQFA
  __and__(other: MOQFA | MMQFA) -> MOQFA | MMQFA
  # Hadamard product.
  # Supports MOQFA & MMQFA for end-decisive MMQFAs, returning an
  # mn-state end-decisive MMQFA.

  complement() -> MOQFA
  __invert__() -> MOQFA

  linear_combination(
    *moqfas: MOQFA,
    coefficients: list[float] | None = None
  ) -> MOQFA
  # Class method.
  # Default coefficients are 1/N for N MOQFAs.

  word_quotient(w: list[int]) -> MOQFA
  inverse_homomorphism(phi: list[list[int]]) -> MOQFA
  to_measure_many_quantum_finite_state_automaton() -> MMQFA
  to_without_final_transition() -> MOQFA
  to_without_initial_transition() -> MOQFA
  to_real_valued() -> MOQFA
  to_bilinear() -> MOQFA
  to_stochastic() -> MOQFA

  counter_example(other: MOQFA) -> list[int] | None
  # Return a string w such that fM(w) != fN(w).
  # If there is no such string, then return None.

  equivalence(other: MOQFA) -> bool
  __eq__(other: object) -> bool
  ```

### MeasureManyQuantumFiniteStateAutomaton

Class for Measure-many quantum finite-state automaton.

Subclass of [QuantumFiniteStateAutomatonBase](#quantumfinitestateautomatonbase).

Abbreviation: MMQFA

* Constructor
  ```
  MeasureManyQuantumFiniteStateAutomaton(
    transition: Transitions,
    accepting_states: States,
    rejecting_states: States
  )
  ```
* Properties
  ```
  accepting_states: States
  rejecting_states: States
  halting_states: States
  non_halting_states: States
  observable: Observable
  ```
* Methods
  ```
  union(other: MMQFA | MOQFA) -> MMQFA
  __or__(other: MMQFA | MOQFA) -> MMQFA
  # Complement of Hadamard product of complements (requires co-end-decisive
  # MMQFA operands). Supports mixed MMQFA | MOQFA.

  intersection(other: MMQFA | MOQFA) -> MMQFA
  __and__(other: MMQFA | MOQFA) -> MMQFA
  # Hadamard product (requires end-decisive MMQFA operands).
  # Supports mixed MMQFA & MOQFA.

  complement() -> MMQFA
  __invert__() -> MMQFA

  linear_combination(
    *mmqfas: MMQFA,
    coefficients: list[float] | None = None
  ) -> MMQFA
  # Class method.
  # Default coefficients are 1/N for N MMQFAs.

  word_quotient(w: list[int]) -> MMQFA
  # Return word quotient MMQFA of size n + |w| * n_halt.

  inverse_homomorphism(phi: list[list[int]]) -> MMQFA
  # Return inverse homomorphism MMQFA of size at most n + c(h) * n_halt.

  is_end_decisive() -> bool
  is_co_end_decisive() -> bool
  to_real_valued() -> MMQFA
  ```

## qfa\_toolkit.quantum\_finite\_state\_automaton\_language

### QuantumFiniteStateAutomatonLanguageBase

Abstract class for quantum finite-state automaton language.

* Constructor
  ```
  QuantumFiniteStateAutomatonLanguageBase(
    quantum_finite_state_automaton: QuantumFiniteStateAutomatonBase,
    strategy: RecognitionStrategy
  ) -> QuantumFiniteStateAutomatonLanguageBase
  ```
* Properties
  ```
  alphabet: int
  start_of_string: int
  end_of_string: int
  ```
* Methods
  ```
  __contains__(w: list[int]) -> bool

  enumerate() -> Iterator[list[int]]
  # Enumerate all strings in the language

  enumerate_length_less_than_n(n: int) -> Iterator[list[int]]
  enumerate_length_n(n: int) -> Iterator[list[int]]
  ```

### MeasureOnceQuantumFiniteStateAutomatonLanguage

Class for MOQFL.

Subclass of
[QuantumFiniteStateAutomatonLanguageBase](#quantumfinitestateautomatonlanguagebase).

Abbreviation: MOQFL

* Methods
  ```
  intersection(
    other: MOQFL | MMQFL,
    use_linear_combination: bool = True
  ) -> MOQFL | MMQFL
  __and__(other: MOQFL | MMQFL) -> MOQFL | MMQFL

  union(other: MOQFL | MMQFL) -> MOQFL | MMQFL
  __or__(other: MOQFL | MMQFL) -> MOQFL | MMQFL
  # Supports direct MOQFL | MMQFL for co-end-decisive MMQFLs (producing an
  # mn-state MMQFL).

  word_quotient(w: list[int]) -> MOQFL
  inverse_homomorphism(phi: list[list[int]]) -> MOQFL
  to_measure_many_quantum_finite_state_automaton_language() -> MMQFL

  from_modulo(n: int, d: int = 1) -> MOQFL
  # Class method. Margin is sin^2(pi / n).

  from_modulo_prime(
    p: int,
    copy_num: int = 0,
    seed: int = 42,
  ) -> MOQFL
  # Class method. Margin is 1/8.
  # If copy_num is 0, use ceil(8 ln p) instead.
  ```

### MeasureManyQuantumFiniteStateAutomatonLanguage

Class for MMQFL.

Subclass of
[QuantumFiniteStateAutomatonLanguageBase](#quantumfinitestateautomatonlanguagebase).

Abbreviation: MMQFL

* Methods
  ```
  intersection(other: MMQFL | MOQFL) -> MMQFL
  __and__(other: MMQFL | MOQFL) -> MMQFL

  union(other: MMQFL | MOQFL) -> MMQFL
  __or__(other: MMQFL | MOQFL) -> MMQFL
  # Requires co-end-decisive MMQFL operands. Supports mixed MMQFL | MOQFL.

  word_quotient(w: list[int]) -> MMQFL
  inverse_homomorphism(phi: list[list[int]]) -> MMQFL

  from_unary_finite(
    ks: list[int],
    params: Optional[tuple[float, float]] = None
  ) -> MMQFL
  # Class method.

  from_unary_singleton(
    k: int,
    params: Optional[tuple[float, float]] = None
  ) -> MMQFL
  # Class method. Uses optimal margin parameters by default when params is None.
  ```

## qfa\_toolkit.recognition\_strategy

### RecognitionStrategy

Base class mapping acceptance probability $f_M(w) \in [0, 1]$ to
`RecognitionStrategy.Result` (`ACCEPT`, `REJECT`, `INVALID`).

* Enum `RecognitionStrategy.Result`: `ACCEPT`, `REJECT`, `INVALID`
* Constructor
  ```
  RecognitionStrategy(
    reject_upperbound: float,
    accept_lowerbound: float,
    reject_inclusive: bool = False,
    accept_inclusive: bool = False
  )
  ```
* Methods
  ```
  __call__(probability: float) -> RecognitionStrategy.Result
  __str__() -> str
  ```

### CutPoint

Cut-point acceptance policy: rejects $f_M(w) \le \lambda$ and accepts
$f_M(w) > \lambda$.

Subclass of [RecognitionStrategy](#recognitionstrategy).

* Constructor: `CutPoint(probability: float)`

### IsolatedCutPoint

Isolated cut-point acceptance policy with threshold $\lambda$ and isolation
margin $\varepsilon > 0$.

Subclass of [RecognitionStrategy](#recognitionstrategy).

* Constructor: `IsolatedCutPoint(threshold: float, epsilon: float)`
* Properties: `threshold: float`, `epsilon: float`

### PositiveOneSidedBoundedError

Positive one-sided bounded-error policy: rejects $f_M(w) = 0$ and accepts
$f_M(w) \ge \varepsilon$.

Subclass of [RecognitionStrategy](#recognitionstrategy).

* Constructor: `PositiveOneSidedBoundedError(epsilon: float)`
* Properties: `epsilon: float`

### NegativeOneSidedBoundedError

Negative one-sided bounded-error policy: accepts $f_M(w) = 1$ and rejects
$f_M(w) \le 1 - \varepsilon$.

Subclass of [RecognitionStrategy](#recognitionstrategy).

* Constructor: `NegativeOneSidedBoundedError(epsilon: float)`
* Properties: `epsilon: float`

## qfa\_toolkit.qiskit\_converter

### QiskitQuantumFiniteStateAutomaton

Abstract class for Qiskit interface.

* Properties
  ```
  qfa: QuantumFiniteStateAutomatonBase

  size: int
  # The number of qubits

  mapping: dict[int, int]
  # Mapping from QFA states to qubit basis

  reverse_mapping: dict[int, int]
  # Mapping from qubit basis to QFA states

  alphabet: int
  states: int
  defined_states: set[int]
  undefined_states: set[int]
  ```
* Methods
  ```
  get_circuit_for_string(w: list[int]) -> QuantumCircuit
  ```

### QiskitMeasureManyQuantumFiniteStateAutomaton

Class for generating Qiskit circuits from MMQFAs (preparing $|\phi(q_{\text{init}})\rangle$
when $\phi(0) \ne |0\cdots 0\rangle$ and using mid-circuit halting-flag readout with `if_test`).

Subclass of
[QiskitQuantumFiniteStateAutomaton](#qiskitquantumfinitestateautomaton).

* Constructor
  ```
  QiskitMeasureManyQuantumFiniteStateAutomaton(
    qfa: MMQFA,
    use_density_mapping: bool = True
  )
  ```
* Properties
  ```
  accepting_states: set[int]
  rejecting_states: set[int]
  non_halting_states: set[int]
  halting_states: set[int]
  ```
* Methods
  ```
  get_density_mapping() -> None
  get_mapping() -> None
  get_circuit_for_string(w: list[int]) -> QuantumCircuit
  ```

### QiskitMeasureOnceQuantumFiniteStateAutomaton

Class for generating Qiskit circuits from MOQFAs (preparing $|\phi(q_{\text{init}})\rangle$
when $\phi(0) \ne |0\cdots 0\rangle$).

Subclass of
[QiskitQuantumFiniteStateAutomaton](#qiskitquantumfinitestateautomaton).

* Constructor
  ```
  QiskitMeasureOnceQuantumFiniteStateAutomaton(
    qfa: MOQFA,
    use_density_mapping: bool = True
  )
  ```
* Properties
  ```
  accepting_states: set[int]
  rejecting_states: set[int]
  ```
* Methods
  ```
  get_density_mapping() -> None
  get_random_mapping(seed: int = 42) -> None
  get_mapping() -> None
  get_circuit_for_string(w: list[int]) -> QuantumCircuit
  ```
