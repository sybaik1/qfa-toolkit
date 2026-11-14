"""Reproduce the Eagle r3 and Heron r2 noisy Qiskit Aer benchmark suite.

Synthesizes elementary state-preparation, transition, and end-of-string blocks
using uniform Quantum Shannon Decomposition (opt_a1=True, opt_a2=True) for
MO-QFAs and unconditioned mid-circuit halting-flag projections for EQU_k
MM-QFAs, then evaluates exact noisy channel superoperators across gate and
readout mirroring factors eta_g, eta_r in {0.1%, 0.5%, 1.0%}.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import warnings
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile  # type: ignore
from qiskit.quantum_info import Kraus, Operator, SuperOp  # type: ignore
from qiskit.synthesis.unitary.qsd import qs_decomposition  # type: ignore
from qiskit_aer import AerSimulator  # type: ignore
from qiskit_ibm_runtime.fake_provider import FakeBrisbane, FakeFez  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mimicnoisemodel import MimicNoiseModel  # noqa: E402
from qfa_toolkit.qiskit_converter import (  # noqa: E402
    QiskitMeasureManyQuantumFiniteStateAutomaton as QMmqfa,
    QiskitMeasureOnceQuantumFiniteStateAutomaton as QMoqfa,
)
from qfa_toolkit.quantum_finite_state_automaton_language import (  # noqa: E402
    MeasureManyQuantumFiniteStateAutomatonLanguage as Mmqfl,
    MeasureOnceQuantumFiniteStateAutomatonLanguage as Moqfl,
)


PROCESSORS: tuple[tuple[str, Any], ...] = (
    ("eagle", FakeBrisbane),
    ("heron", FakeFez),
)


@dataclass(frozen=True)
class LanguageSpec:
    """Benchmark automaton specification."""

    language: str
    circuit_mode: str
    param: int
    copy_num: int | None
    experiment_name: str


LANGUAGE_SPECS: tuple[LanguageSpec, ...] = (
    LanguageSpec("mod_3", "moqfa", 3, 5, "mapping effect, change QFA state number"),
    LanguageSpec("mod_3", "moqfa", 3, 8, "mapping effect, change QFA state number"),
    LanguageSpec("mod_3", "moqfa", 3, 9, "mapping effect, mimic rate change"),
    LanguageSpec("mod_5", "moqfa", 5, 9, "mapping effect, mimic rate change"),
    LanguageSpec("mod_5", "moqfa", 5, 0, "mapping effect, mimic rate change"),
    LanguageSpec("mod_7", "moqfa", 7, 9, "mapping effect, mimic rate change"),
    LanguageSpec("mod_7", "moqfa", 7, 0, "mapping effect, mimic rate change"),
    LanguageSpec("singleton_3", "mmqfa_unconditioned", 3, None, "mapping effect, mimic rate change"),
    LanguageSpec("singleton_5", "mmqfa_unconditioned", 5, None, "mapping effect, mimic rate change"),
    LanguageSpec("singleton_7", "mmqfa_unconditioned", 7, None, "mapping effect, mimic rate change"),
)

MIRROR_RATES: tuple[float, ...] = (0.001, 0.005, 0.01)
LENGTH_RANGE: range = range(2, 41)
SHOTS: int = 10000


def _build_qfl(spec: LanguageSpec) -> tuple[Any, Any]:
    """Instantiate the language wrapper and underlying QFA."""
    if spec.circuit_mode == "moqfa":
        copy_num = 0 if spec.copy_num is None else spec.copy_num
        qfl = Moqfl.from_modulo_prime(spec.param, copy_num=copy_num)
    elif spec.circuit_mode == "mmqfa_unconditioned":
        qfl = Mmqfl.from_unary_singleton(spec.param)
    else:
        raise ValueError(f"Unsupported circuit_mode: {spec.circuit_mode}")
    return qfl, qfl.quantum_finite_state_automaton


def _compile_block(
    qqfa: QMoqfa | QMmqfa,
    char_idx: int,
    basis_gates: list[str],
    use_qsd: bool,
) -> tuple[QuantumCircuit, dict[str, int]]:
    """Synthesize and transpile a single elementary QFA block with equivalence check."""
    n_q = qqfa.size
    qfa = qqfa.qfa
    prepend_init = char_idx == qfa.start_of_string

    qc_ref = QuantumCircuit(n_q)
    if prepend_init:
        qqfa._prepare_initial_state(qc_ref, qc_ref.qubits)
    qc_ref.append(qqfa.circuit[char_idx], qc_ref.qubits)

    qc_src = QuantumCircuit(n_q)
    if prepend_init:
        qqfa._prepare_initial_state(qc_src, qc_src.qubits)

    unitary = qqfa.circuit[char_idx].to_matrix()
    is_identity = np.allclose(unitary, np.eye(2 ** n_q))
    if not is_identity:
        if use_qsd and n_q >= 3:
            # Qiskit 2.5 UnitaryGate uses an opt_a2=False block shortcut only for
            # naive mapping; uniform opt_a1=True, opt_a2=True preserves fair comparison.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                decomposed = qs_decomposition(unitary, opt_a1=True, opt_a2=True)
            qc_src.compose(decomposed, range(n_q), inplace=True)
        else:
            qc_src.append(qqfa.circuit[char_idx], qc_src.qubits)

    tc = transpile(
        qc_src,
        basis_gates=basis_gates,
        optimization_level=1,
        seed_transpiler=42,
    )
    if not Operator(tc).equiv(Operator(qc_ref)):
        raise RuntimeError(
            f"Transpiled block failed unitary equivalence for char={char_idx}"
        )

    ops = {str(k): int(v) for k, v in tc.count_ops().items()}
    tc.save_superop()
    return tc, ops


def _halting_flag_projection_superop(n_q: int) -> np.ndarray:
    """Return the projective measurement superoperator on MSB halting flag q_{n_q-1}."""
    half = 2 ** (n_q - 1)
    dim = 2 ** n_q
    proj0 = np.diag([1.0] * half + [0.0] * half)
    proj1 = np.diag([0.0] * half + [1.0] * half)
    return np.asarray(SuperOp(Kraus([proj0, proj1])).data)


def _readout_confusion_matrix(base_meas_err: float, eta_r: float, n_q: int) -> np.ndarray:
    """Construct the symmetric n-qubit readout confusion matrix."""
    mq = base_meas_err * eta_r
    single = np.array([[1.0 - mq, mq], [mq, 1.0 - mq]], dtype=float)
    mat = single
    for _ in range(n_q - 1):
        mat = np.kron(mat, single)
    return mat


def _evolve_premeasurement_probabilities(
    superops: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_q: int,
    circuit_mode: str,
) -> dict[int, np.ndarray]:
    """Evolve the initial state over input lengths 2..40 using block superoperators."""
    s0, s1, s2 = superops
    if circuit_mode == "mmqfa_unconditioned":
        s_meas = _halting_flag_projection_superop(n_q)
        s1 = s1 @ s_meas
        s2 = s2 @ s_meas

    dim = 2 ** n_q
    rho0 = np.zeros(dim * dim, dtype=complex)
    rho0[0] = 1.0

    # Length 1 step, then step through 2..40 matching [start] + a^L + [end].
    cur = s1 @ (s0 @ rho0)
    pre_meas: dict[int, np.ndarray] = {}
    for length in LENGTH_RANGE:
        cur = s1 @ cur
        rho_end = (s2 @ cur).reshape(dim, dim)
        probs = np.clip(np.real(np.diag(rho_end)), 0.0, None)
        total = float(probs.sum())
        if total <= 0.0:
            raise RuntimeError(f"Zero total probability at length={length}")
        pre_meas[length] = probs / total
    return pre_meas


def _string_gate_counts(
    block_ops: tuple[dict[str, int], dict[str, int], dict[str, int]],
    length: int,
    n_q: int,
    circuit_mode: str,
) -> dict[str, int]:
    """Aggregate compiled gate counts for a unary string of length `length`."""
    ops0, ops1, ops2 = block_ops
    total: Counter[str] = Counter(ops0)
    for op_name, count in ops1.items():
        total[op_name] += count * length
    total.update(ops2)
    if circuit_mode == "moqfa":
        total["measure"] += n_q
        total["barrier"] += 1
    else:
        total["measure"] += (length + 1) + n_q
        total["barrier"] += 1
    return dict(total)


def _prepare_compiled_blocks(
    backend_by_proc: dict[str, Any],
    basis_by_proc: dict[str, list[str]],
) -> dict[tuple[str, LanguageSpec, bool], dict[str, Any]]:
    """Compile the 40 unique (processor, spec, mapping) circuit block triples."""
    compiled: dict[tuple[str, LanguageSpec, bool], dict[str, Any]] = {}
    for proc_name, _ in PROCESSORS:
        basis_gates = basis_by_proc[proc_name]
        for spec in LANGUAGE_SPECS:
            _, qfa = _build_qfl(spec)
            use_qsd = spec.circuit_mode == "moqfa"
            ideal_probs = {
                length: float(qfa([1] * length))
                for length in LENGTH_RANGE
            }
            for use_d in (False, True):
                qqfa: QMoqfa | QMmqfa
                if spec.circuit_mode == "moqfa":
                    qqfa = QMoqfa(qfa, use_density_mapping=use_d)
                else:
                    qqfa = QMmqfa(qfa, use_density_mapping=use_d)

                tc0, ops0 = _compile_block(qqfa, qfa.start_of_string, basis_gates, use_qsd)
                tc1, ops1 = _compile_block(qqfa, 1, basis_gates, use_qsd)
                tc2, ops2 = _compile_block(qqfa, qfa.end_of_string, basis_gates, use_qsd)
                compiled[(proc_name, spec, use_d)] = {
                    "qfa_states": int(qfa.states),
                    "n_q": int(qqfa.size),
                    "accepting_states": sorted(int(s) for s in qqfa.accepting_states),
                    "rejecting_states": sorted(int(s) for s in qqfa.rejecting_states),
                    "ideal_probs": ideal_probs,
                    "tcs": (tc0, tc1, tc2),
                    "ops": (ops0, ops1, ops2),
                }
    return compiled


def run_all_simulations(output_dir: Path, overwrite: bool = True) -> list[dict[str, Any]]:
    """Run all 360 Eagle r3 and Heron r2 noisy simulations and save JSON artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for existing in output_dir.glob("experiment_*.json"):
            existing.unlink()

    backends: dict[str, Any] = {name: cls() for name, cls in PROCESSORS}
    basis_by_proc: dict[str, list[str]] = {}
    base_meas_err: dict[str, float] = {}
    for proc_name, backend in backends.items():
        probe_nm = MimicNoiseModel(backend, 1.0, 1.0, all_qubit=True)
        basis_by_proc[proc_name] = list(probe_nm.basis_gates)
        base_meas_err[proc_name] = float(probe_nm.mean_q_error)

    print("Compiling 40 elementary (processor, language, mapping) block triples...")
    compiled = _prepare_compiled_blocks(backends, basis_by_proc)

    # Batch-evaluate Aer superoperators per (processor, eta_g): only 6 C++ Aer calls total.
    pre_meas_cache: dict[tuple[str, LanguageSpec, bool, float], dict[int, np.ndarray]] = {}
    for proc_name, backend in backends.items():
        triple_keys = [
            (proc_name, spec, use_d)
            for spec in LANGUAGE_SPECS
            for use_d in (False, True)
        ]
        circuits_batch = []
        for key in triple_keys:
            circuits_batch.extend(compiled[key]["tcs"])

        for eta_g in MIRROR_RATES:
            print(
                f"Evaluating Aer superoperators: processor={proc_name}, "
                f"eta_g={eta_g * 100:.1f}% ({len(circuits_batch)} blocks)..."
            )
            nm_gate = MimicNoiseModel(backend, eta_g, 0.0, all_qubit=True)
            sim = AerSimulator(
                method="superop",
                noise_model=nm_gate,
                max_parallel_threads=16,
                max_parallel_experiments=16,
            )
            result = sim.run(circuits_batch).result()
            for idx, key in enumerate(triple_keys):
                s0 = np.asarray(result.data(3 * idx)["superop"])
                s1 = np.asarray(result.data(3 * idx + 1)["superop"])
                s2 = np.asarray(result.data(3 * idx + 2)["superop"])
                pre_meas_cache[(key[0], key[1], key[2], eta_g)] = (
                    _evolve_premeasurement_probabilities(
                        (s0, s1, s2),
                        compiled[key]["n_q"],
                        key[1].circuit_mode,
                    )
                )

    summary_entries: list[dict[str, Any]] = []
    exp_idx = 0
    for proc_name, backend in backends.items():
        for spec in LANGUAGE_SPECS:
            for eta_g in MIRROR_RATES:
                for eta_r in MIRROR_RATES:
                    for use_d in (False, True):
                        info = compiled[(proc_name, spec, use_d)]
                        n_q = info["n_q"]
                        acc_states = info["accepting_states"]
                        rej_states = info["rejecting_states"]
                        ideal_probs = info["ideal_probs"]
                        block_ops = info["ops"]
                        pre_meas = pre_meas_cache[(proc_name, spec, use_d, eta_g)]
                        m_all = _readout_confusion_matrix(
                            base_meas_err[proc_name], eta_r, n_q
                        )

                        rng = np.random.default_rng(10000 + exp_idx)
                        results: dict[str, Any] = {}
                        ys: list[float] = []
                        zs: list[float] = []
                        invalid_shots: list[int] = []

                        for length in LENGTH_RANGE:
                            p_meas = np.clip(m_all @ pre_meas[length], 0.0, None)
                            p_meas /= float(p_meas.sum())
                            counts = rng.multinomial(SHOTS, p_meas)
                            acc_cnt = int(sum(counts[s] for s in acc_states))
                            rej_cnt = int(sum(counts[s] for s in rej_states))
                            obs_acc = acc_cnt / SHOTS
                            obs_rej = rej_cnt / SHOTS
                            valid_cnt = acc_cnt + rej_cnt
                            ys.append(acc_cnt / valid_cnt if valid_cnt > 0 else 0.0)
                            exp_acc = ideal_probs[length]
                            zs.append(exp_acc)
                            invalid_shots.append(SHOTS - valid_cnt)

                            results["1" * length] = {
                                "observed": [obs_acc, obs_rej],
                                "expected": [exp_acc, 1.0 - exp_acc],
                                "circuit": _string_gate_counts(
                                    block_ops, length, n_q, spec.circuit_mode
                                ),
                            }

                        ys_arr = np.asarray(ys, dtype=float)
                        zs_arr = np.asarray(zs, dtype=float)
                        mape_norm = float(np.mean(np.abs((zs_arr - ys_arr) / zs_arr)) * 100.0)
                        mean_inv = float(np.mean(invalid_shots))

                        exp_path = output_dir / f"experiment_{exp_idx}.json"
                        meta_path = output_dir / f"experiment_{exp_idx}_metadata.json"
                        with exp_path.open("w", encoding="utf-8") as f:
                            json.dump(results, f, indent=4)

                        metadata: dict[str, Any] = {
                            "index": exp_idx,
                            "qfa size": info["qfa_states"],
                            "qiskit circuit size": n_q,
                            "language": spec.language,
                            "copy_num": spec.copy_num,
                            "processor": proc_name,
                            "backend": backend.name,
                            "circuit_mode": spec.circuit_mode,
                            "name": spec.experiment_name,
                            "shots": SHOTS,
                            "simulator": True,
                            "mimic_rate_gate": eta_g,
                            "mimic_rate_readout": eta_r,
                            "use_density_mapping": use_d,
                            "use_entropy_mapping": use_d,
                            "use_mapping_noise_correction": False,
                            "mape_normalized": mape_norm,
                            "mean_invalid_shots": mean_inv,
                        }
                        with meta_path.open("w", encoding="utf-8") as f:
                            json.dump(metadata, f, indent=4)

                        summary_entries.append(metadata)
                        exp_idx += 1

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_entries, f, indent=4)

    print(f"Wrote {len(summary_entries)} experiment JSONs + summary to {output_dir}")
    return summary_entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Eagle r3 and Heron r2 noisy Qiskit Aer QFA benchmark suite."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "benchmark" / "results" / "eagle_heron",
        help="Destination directory for simulation JSON files and summary.json.",
    )
    args = parser.parse_args()
    run_all_simulations(args.output_dir)


if __name__ == "__main__":
    main()
