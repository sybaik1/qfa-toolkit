# QFA-Toolkit Benchmark Data and Reproducibility Scripts

This directory archives raw benchmark JSON datasets, reference plots, and
reproducible simulation scripts for *A Framework for Quantum Finite-State
Languages with Density Mapping*.

## Directory Layout

- `benchmark/results/eagle_heron/` — **Eagle r3 (`FakeBrisbane`) and Heron r2
  (`FakeFez`) Qiskit Aer noisy simulation dataset** (`360` runs indexed
  `0..359` plus `summary.json`).
  - Languages evaluated on unary inputs $a^2, \ldots, a^{40}$ ($39$
    strings, $10{,}000$ shots each):
    - `MOD_3`: 10-state (`copy_num=5`), 16-state (`copy_num=8`, zero unmapped
      basis states), and 18-state (`copy_num=9`) MO-QFAs.
    - `MOD_5`: 18-state (`copy_num=9`) and 26-state (`copy_num=0`) MO-QFAs.
    - `MOD_7`: 18-state (`copy_num=9`) and 32-state (`copy_num=0`) MO-QFAs.
    - `EQU_3`, `EQU_5`, `EQU_7`: 4-state co-end-decisive MM-QFAs executed with
      unconditioned static mid-circuit halting-flag projections (Section 5.2.2).
  - Noise grid: gate and readout mirroring factors ($\eta_g, \eta_r$) of `0.1%`,
    `0.5%`, and `1.0%` paired across naive binary mapping
    (`use_density_mapping=False`) and density mapping (`use_density_mapping=True`).
- `benchmark/results/experiment_*.json` & `results/` — Preserved May 2024
  simulation and IBM Quantum hardware measurement JSONs from the original
  `benchmark` branch.
- `benchmark/figures/reference/` — Archived Eagle and Heron reference plots from
  the manuscript.
- `benchmark/scripts/` — Experiment runner, noise model, and plotting/verification
  scripts (`run_eagle_heron_simulations.py`, `plot_eagle_heron.py`,
  `mimicnoisemodel.py`, `wrapper.py`, plus the original 2024 scripts).

## Reproducing the Eagle r3 & Heron r2 Simulations

From the `qfa-toolkit` repository root with the package environment activated:

```bash
python benchmark/scripts/run_eagle_heron_simulations.py
python benchmark/scripts/plot_eagle_heron.py --verify --out-dir /tmp/plots
```

## Implementation Notes

- **Offline Processor Calibrations:** `run_eagle_heron_simulations.py` uses
  Qiskit IBM Runtime's offline `FakeBrisbane` (Eagle r3, 127 qubits, native gates
  `ecr, id, rz, sx, x`) and `FakeFez` (Heron r2, 156 qubits, native gates
  `cz, id, rz, sx, x`). `MimicNoiseModel` filters out disabled/uncalibrated
  hardware sentinels (`error <= 0` or `error >= 0.5`) so average two-qubit and
  readout error rates reflect calibrated hardware links.
- **Uniform Unitary Synthesis in Qiskit 2.x:** In Qiskit 2.5+, the default
  `UnitaryGate` synthesizer applies an asymmetric `opt_a2=False` adjacent-LSB
  block-diagonal shortcut only for naive mapping. For MO-QFA blocks of $3+$
  qubits, `run_eagle_heron_simulations.py` applies uniform Quantum Shannon
  Decomposition (`qs_decomposition(..., opt_a1=True, opt_a2=True)`) and asserts
  exact `Operator` equivalence after native-gate transpilation.
- **Unconditioned Static Mid-Circuit `EQU_k` Circuits:** Matching Section 5.2.2
  of the manuscript, `EQU_k` benchmarks model static sequences of $\mathbf{U}_a$
  and $\mathbf{U}_{\text{end}}$ interleaved with unconditioned projective halting-flag
  measurements, reproducing baseline saturation around `~48–50%` MAPE unless
  dynamic classical feedforward (`if_test`) is used.
