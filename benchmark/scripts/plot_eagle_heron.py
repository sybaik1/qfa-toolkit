"""Summarize, verify, and plot the Eagle r3 and Heron r2 QFA benchmark suite."""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_qfa")

import json  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # type: ignore  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "benchmark" / "results" / "eagle_heron"

RATES: tuple[float, ...] = (0.001, 0.005, 0.01)
EXPECTED_KEYS: list[str] = ["1" * length for length in range(2, 41)]


def _load_experiment_series(
    data_dir: Path, entry: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load per-string (lengths, normalized observed, expected) arrays."""
    exp_path = data_dir / f"experiment_{entry['index']}.json"
    with exp_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    lengths = np.array([len(k) for k in EXPECTED_KEYS], dtype=int)
    ys = np.array(
        [
            data[k]["observed"][0] / (data[k]["observed"][0] + data[k]["observed"][1])
            for k in EXPECTED_KEYS
        ],
        dtype=float,
    )
    zs = np.array([data[k]["expected"][0] for k in EXPECTED_KEYS], dtype=float)
    return lengths, ys, zs


def _lookup(
    entries: list[dict[str, Any]],
    *,
    processor: str,
    language: str,
    copy_num: int | None,
    eta_g: float,
    eta_r: float,
    use_d: bool,
) -> dict[str, Any]:
    """Return the unique summary record matching the given parameters."""
    matches = [
        e for e in entries
        if e["processor"] == processor
        and e["language"] == language
        and e["copy_num"] == copy_num
        and np.isclose(e["mimic_rate_gate"], eta_g)
        and np.isclose(e["mimic_rate_readout"], eta_r)
        and e["use_density_mapping"] is use_d
    ]
    if len(matches) != 1:
        raise KeyError(
            f"Expected 1 match for {(processor, language, copy_num, eta_g, eta_r, use_d)}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _paired_diffs(
    entries: list[dict[str, Any]],
    specs: list[tuple[str, int | None]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return paired (naive_mapes, d_mapes, naive_minus_d) arrays across 9x2 regimes."""
    naive_vals: list[float] = []
    d_vals: list[float] = []
    for proc in ("eagle", "heron"):
        for lang, cnum in specs:
            for eg in RATES:
                for er in RATES:
                    e_n = _lookup(
                        entries,
                        processor=proc,
                        language=lang,
                        copy_num=cnum,
                        eta_g=eg,
                        eta_r=er,
                        use_d=False,
                    )
                    e_d = _lookup(
                        entries,
                        processor=proc,
                        language=lang,
                        copy_num=cnum,
                        eta_g=eg,
                        eta_r=er,
                        use_d=True,
                    )
                    naive_vals.append(float(e_n["mape_normalized"]))
                    d_vals.append(float(e_d["mape_normalized"]))
    naive_arr = np.asarray(naive_vals, dtype=float)
    d_arr = np.asarray(d_vals, dtype=float)
    return naive_arr, d_arr, naive_arr - d_arr


def _print_table_2(entries: list[dict[str, Any]]) -> None:
    """Print Table 2: MOD_3 with 10, 16, and 18 states at eta_g = eta_r in {0.5%, 1.0%}."""
    print("\n=== Table 2: MOD_3 MAPE (%) and Mean Invalid Shots ===")
    for rate in (0.01, 0.005):
        print(f"\n[eta_g = eta_r = {rate * 100:.1f}%]")
        header = (
            f"{'Simulator':10s} | "
            f"{'Naive 10':>14s} {'Naive 16':>9s} {'Naive 18':>14s} | "
            f"{'D-map 10':>14s} {'D-map 16':>9s} {'D-map 18':>14s}"
        )
        print(header)
        print("-" * len(header))
        for proc in ("eagle", "heron"):
            cells: list[str] = []
            for use_d in (False, True):
                for cnum in (5, 8, 9):
                    e = _lookup(
                        entries,
                        processor=proc,
                        language="mod_3",
                        copy_num=cnum,
                        eta_g=rate,
                        eta_r=rate,
                        use_d=use_d,
                    )
                    mape = float(e["mape_normalized"])
                    inv = round(float(e["mean_invalid_shots"]))
                    if cnum == 8:
                        cells.append(f"{mape:9.2f}")
                    else:
                        cells.append(f"{mape:6.2f} ({inv:4d})")
            print(
                f"{proc.capitalize():10s} | "
                f"{cells[0]:>14s} {cells[1]:>9s} {cells[2]:>14s} | "
                f"{cells[3]:>14s} {cells[4]:>9s} {cells[5]:>14s}"
            )


def _print_table_3(entries: list[dict[str, Any]], label: str, mod_specs: list[tuple[str, int | None]]) -> None:
    """Print Table 3: MAPE (%) across Eagle/Heron, 9 (eta_g, eta_r) pairs, naive vs D-mapping."""
    print(f"\n=== Table 3 ({label}): MOD_p & EQU_k MAPE (%) ===")
    equ_specs = [("singleton_3", None), ("singleton_5", None), ("singleton_7", None)]
    all_specs = mod_specs + equ_specs
    header = (
        "Proc   eta_g eta_r | "
        + " ".join(f"{'N-' + s[0].replace('singleton_', 'E').replace('mod_', 'M'):>6s}" for s in all_specs)
        + " | "
        + " ".join(f"{'D-' + s[0].replace('singleton_', 'E').replace('mod_', 'M'):>6s}" for s in all_specs)
    )
    print(header)
    print("-" * len(header))
    for proc in ("eagle", "heron"):
        for eg in RATES:
            for er in RATES:
                n_cells = [
                    f"{_lookup(entries, processor=proc, language=lang, copy_num=cn, eta_g=eg, eta_r=er, use_d=False)['mape_normalized']:6.2f}"
                    for lang, cn in all_specs
                ]
                d_cells = [
                    f"{_lookup(entries, processor=proc, language=lang, copy_num=cn, eta_g=eg, eta_r=er, use_d=True)['mape_normalized']:6.2f}"
                    for lang, cn in all_specs
                ]
                print(
                    f"{proc.capitalize():6s} {eg * 100:4.1f}% {er * 100:4.1f}% | "
                    + " ".join(n_cells)
                    + " | "
                    + " ".join(d_cells)
                )


def _print_stats(entries: list[dict[str, Any]]) -> None:
    """Print paired hypothesis-test summaries across MOD_p and EQU_k families."""
    print("\n=== Paired MAPE Reduction Summary (MAPE_naive - MAPE_D) ===")
    families = [
        ("MOD_p (18-state)", [("mod_3", 9), ("mod_5", 9), ("mod_7", 9)]),
        ("MOD_p (default size)", [("mod_3", 9), ("mod_5", 0), ("mod_7", 0)]),
        ("EQU_k (4-state)", [("singleton_3", None), ("singleton_5", None), ("singleton_7", None)]),
    ]
    for title, specs in families:
        _, _, diff = _paired_diffs(entries, specs)
        t_stat, p_t = stats.ttest_rel(diff, np.zeros_like(diff))
        _, p_w = stats.wilcoxon(diff)
        improved = int(np.sum(diff > 0))
        print(
            f"{title:22s}: n={len(diff):2d}, improved={improved:2d}/{len(diff):2d}, "
            f"mean={np.mean(diff):+6.2f} +/- {np.std(diff, ddof=1):5.2f} %p, "
            f"t={t_stat:+6.2f} (p={p_t:.3g}), Wilcoxon p={p_w:.3g}"
        )


def verify_dataset(data_dir: Path, entries: list[dict[str, Any]]) -> None:
    """Verify structural integrity and all benchmark claims against raw JSONs."""
    assert len(entries) == 360, f"Expected 360 summary entries, found {len(entries)}"

    for exp_idx in range(360):
        exp_path = data_dir / f"experiment_{exp_idx}.json"
        meta_path = data_dir / f"experiment_{exp_idx}_metadata.json"
        assert exp_path.is_file(), f"Missing {exp_path}"
        assert meta_path.is_file(), f"Missing {meta_path}"

        with exp_path.open("r", encoding="utf-8") as f:
            exp_data = json.load(f)
        with meta_path.open("r", encoding="utf-8") as f:
            meta_data = json.load(f)

        assert list(exp_data.keys()) == EXPECTED_KEYS, (
            f"Unexpected string keys in {exp_path}"
        )

        ys = np.array(
            [
                exp_data[k]["observed"][0]
                / (exp_data[k]["observed"][0] + exp_data[k]["observed"][1])
                for k in EXPECTED_KEYS
            ],
            dtype=float,
        )
        zs = np.array([exp_data[k]["expected"][0] for k in EXPECTED_KEYS], dtype=float)
        recomputed_mape = float(np.mean(np.abs((zs - ys) / zs)) * 100.0)
        recomputed_inv = float(
            np.mean(
                [
                    (1.0 - (exp_data[k]["observed"][0] + exp_data[k]["observed"][1]))
                    * meta_data["shots"]
                    for k in EXPECTED_KEYS
                ]
            )
        )
        assert abs(recomputed_mape - meta_data["mape_normalized"]) < 1e-6
        assert abs(recomputed_inv - meta_data["mean_invalid_shots"]) < 1e-4

    # 16-state MOD_3: zero invalid shots everywhere, lower MAPE under D-mapping on both procs.
    for proc in ("eagle", "heron"):
        diffs_16: list[float] = []
        for eg in RATES:
            for er in RATES:
                e_n = _lookup(
                    entries,
                    processor=proc,
                    language="mod_3",
                    copy_num=8,
                    eta_g=eg,
                    eta_r=er,
                    use_d=False,
                )
                e_d = _lookup(
                    entries,
                    processor=proc,
                    language="mod_3",
                    copy_num=8,
                    eta_g=eg,
                    eta_r=er,
                    use_d=True,
                )
                assert e_n["mean_invalid_shots"] == 0.0
                assert e_d["mean_invalid_shots"] == 0.0
                diffs_16.append(float(e_n["mape_normalized"]) - float(e_d["mape_normalized"]))
        assert float(np.mean(diffs_16)) > 0.0, (
            f"Expected D-mapping to reduce 16-state MOD_3 MAPE on {proc}, got {np.mean(diffs_16):.4f}"
        )

    # 18-state and default-size MOD_p: D-mapping reduces mean MAPE across 54 pairs.
    _, _, diff_18 = _paired_diffs(entries, [("mod_3", 9), ("mod_5", 9), ("mod_7", 9)])
    assert len(diff_18) == 54 and float(np.mean(diff_18)) > 0.0, (
        f"Expected positive mean MAPE reduction for 18-state MOD_p, got {np.mean(diff_18):.4f}"
    )

    _, _, diff_def = _paired_diffs(entries, [("mod_3", 9), ("mod_5", 0), ("mod_7", 0)])
    assert len(diff_def) == 54 and float(np.mean(diff_def)) > 0.0, (
        f"Expected positive mean MAPE reduction for default-size MOD_p, got {np.mean(diff_def):.4f}"
    )

    # EQU_k unconditioned baseline saturation near 50.07%, 49.46%, 48.12% and near-zero D diff.
    baselines = {"singleton_3": 50.07, "singleton_5": 49.46, "singleton_7": 48.12}
    for lang, expected_base in baselines.items():
        vals = [
            float(e["mape_normalized"])
            for e in entries
            if e["language"] == lang
        ]
        assert len(vals) == 36
        mean_val = float(np.mean(vals))
        assert abs(mean_val - expected_base) < 1.0, (
            f"{lang} mean MAPE {mean_val:.2f}% deviates >1.0%p from {expected_base:.2f}%"
        )
    _, _, diff_equ = _paired_diffs(
        entries, [("singleton_3", None), ("singleton_5", None), ("singleton_7", None)]
    )
    assert len(diff_equ) == 54 and abs(float(np.mean(diff_equ))) < 0.5, (
        f"Expected near-zero EQU_k paired diff, got {np.mean(diff_equ):.4f}"
    )
    print("\nAll verification assertions passed cleanly.")


def _plot_2x2_comparison(
    data_dir: Path,
    entries: list[dict[str, Any]],
    out_path: Path,
    panel_specs: list[tuple[str, str, int | None, float, str]],
    dashed_lines: tuple[float, ...],
) -> None:
    """Render a 2x2 naive-vs-D-mapping probability comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 6.8), sharex=True, sharey=True)
    for ax, (proc, lang, cnum, rate, title) in zip(axes.flat, panel_specs):
        e_n = _lookup(
            entries,
            processor=proc,
            language=lang,
            copy_num=cnum,
            eta_g=rate,
            eta_r=rate,
            use_d=False,
        )
        e_d = _lookup(
            entries,
            processor=proc,
            language=lang,
            copy_num=cnum,
            eta_g=rate,
            eta_r=rate,
            use_d=True,
        )
        xs, y_n, _ = _load_experiment_series(data_dir, e_n)
        _, y_d, _ = _load_experiment_series(data_dir, e_d)

        ax.plot(xs, y_n, label="w/o D-mapping", marker="o", markersize=4, linewidth=1.2)
        ax.plot(xs, y_d, label="with D-mapping", marker="^", markersize=4, linewidth=1.2)
        for level in dashed_lines:
            ax.axhline(level, color="gray", linestyle="--", linewidth=1.0)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0.1, 1.08)
        ax.set_xticks(range(2, 41, 2))
        ax.tick_params(labelsize=8)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=10)
    fig.supxlabel("Length of String", fontsize=11)
    fig.supylabel("Normalized Accepting Probability", fontsize=11)
    fig.tight_layout(rect=(0.02, 0.02, 0.99, 0.94))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_mapping_diff(
    data_dir: Path,
    entries: list[dict[str, Any]],
    out_path: Path,
    processor: str,
    title: str,
) -> None:
    """Render a per-string difference stem plot P_D(a^j) - P_naive(a^j) for MOD_5 (26 states)."""
    e_n = _lookup(
        entries,
        processor=processor,
        language="mod_5",
        copy_num=0,
        eta_g=0.005,
        eta_r=0.005,
        use_d=False,
    )
    e_d = _lookup(
        entries,
        processor=processor,
        language="mod_5",
        copy_num=0,
        eta_g=0.005,
        eta_r=0.005,
        use_d=True,
    )
    xs, y_n, _ = _load_experiment_series(data_dir, e_n)
    _, y_d, _ = _load_experiment_series(data_dir, e_d)
    diff = y_d - y_n

    fig, ax = plt.subplots(figsize=(8.0, 2.4))
    for mult in range(5, 41, 5):
        ax.axvspan(mult - 0.45, mult + 0.45, color="lightgray", alpha=0.85, zorder=0)
    markerline, stemlines, baseline = ax.stem(xs, diff, label="difference")
    plt.setp(markerline, markersize=4.5, color="tab:blue")
    plt.setp(stemlines, linewidth=1.0, color="tab:blue")
    plt.setp(baseline, color="tab:red", linewidth=1.2)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(range(2, 41, 2))
    ax.set_ylim(-0.06, 0.06)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def render_plots(data_dir: Path, entries: list[dict[str, Any]], out_dir: Path) -> None:
    """Generate the five reference comparison and difference PNG figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_2x2_comparison(
        data_dir,
        entries,
        out_dir / "MOD_5_eagle_heron_1_0.5.png",
        [
            ("eagle", "mod_5", 0, 0.01, "Eagle simulator (eta=1.0%)"),
            ("eagle", "mod_5", 0, 0.005, "Eagle simulator (eta=0.5%)"),
            ("heron", "mod_5", 0, 0.01, "Heron simulator (eta=1.0%)"),
            ("heron", "mod_5", 0, 0.005, "Heron simulator (eta=0.5%)"),
        ],
        dashed_lines=(1.0, 0.4),
    )
    _plot_mapping_diff(
        data_dir,
        entries,
        out_dir / "qfa_size_26_eagle_normalized_mapping_diff.png",
        processor="eagle",
        title="Eagle simulator",
    )
    _plot_mapping_diff(
        data_dir,
        entries,
        out_dir / "qfa_size_26_Heron_normalized_mapping_diff.png",
        processor="heron",
        title="Heron simulator",
    )
    _plot_2x2_comparison(
        data_dir,
        entries,
        out_dir / "qfa_size_10_eagle_heron_normalized_1_0.5.png",
        [
            ("eagle", "mod_3", 5, 0.01, "Eagle simulator (10-state, eta=1.0%)"),
            ("eagle", "mod_3", 5, 0.005, "Eagle simulator (10-state, eta=0.5%)"),
            ("heron", "mod_3", 5, 0.01, "Heron simulator (10-state, eta=1.0%)"),
            ("heron", "mod_3", 5, 0.005, "Heron simulator (10-state, eta=0.5%)"),
        ],
        dashed_lines=(1.0, 0.25),
    )
    _plot_2x2_comparison(
        data_dir,
        entries,
        out_dir / "qfasize_16_18_eagle_heron.png",
        [
            ("eagle", "mod_3", 8, 0.01, "Eagle simulator (16-state, eta=1.0%)"),
            ("eagle", "mod_3", 9, 0.01, "Eagle simulator (18-state, eta=1.0%)"),
            ("heron", "mod_3", 8, 0.01, "Heron simulator (16-state, eta=1.0%)"),
            ("heron", "mod_3", 9, 0.01, "Heron simulator (18-state, eta=1.0%)"),
        ],
        dashed_lines=(1.0, 0.25),
    )
    print(f"Rendered 5 benchmark plots into {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect, verify, and plot the Eagle r3 and Heron r2 benchmark results."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing summary.json and experiment_*.json files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional directory to write rendered PNG figures.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run consistency and statistical verification assertions.",
    )
    args = parser.parse_args()

    summary_path = args.data_dir / "summary.json"
    with summary_path.open("r", encoding="utf-8") as f:
        entries = json.load(f)

    _print_table_2(entries)
    _print_table_3(entries, "18-state MOD_p", [("mod_3", 9), ("mod_5", 9), ("mod_7", 9)])
    _print_table_3(entries, "default-size MOD_p", [("mod_3", 9), ("mod_5", 0), ("mod_7", 0)])
    _print_stats(entries)

    if args.verify:
        verify_dataset(args.data_dir, entries)
    if args.out_dir is not None:
        render_plots(args.data_dir, entries, args.out_dir)


if __name__ == "__main__":
    main()
