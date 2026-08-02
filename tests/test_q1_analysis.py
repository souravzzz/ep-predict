from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from ep_predict.analysis.q1 import _monotone_non_decreasing, analyze_q1


POLICIES = ("renormalize", "null")
POSITIONINGS = ("mass_omission", "random_within_route", "mass_adversarial")
MASS = (0.01, 0.05, 0.125, 0.25, 0.50)
TOPOLOGIES = ("spread", "layer_burst", "consecutive_block", "scattered")


def _base_config() -> dict:
    return {
        "output_dir": "/tmp/q1-test",
        "probe": {"mass_targets": list(MASS)},
        "decision_gate": {
            "headline_policy": "renormalize",
            "headline_positioning": "mass_omission",
            "headline_m": 0.125,
            "max_mean_forward_kl": 0.05,
            "min_top1_agreement": 0.99,
            "max_ppl_ratio": 1.05,
            "monotone_slack": 1e-6,
            "large_divergence_kl": 2.0,
            "large_divergence_max_fraction": 0.01,
        },
    }


def _write_aggregates(path: Path, *, headline_kl: float, monotone: bool) -> None:
    rows = []
    for policy in POLICIES:
        for positioning in POSITIONINGS:
            for m in MASS:
                kl = headline_kl * m / 0.125
                if not monotone and m == 0.25:
                    kl = 0.0  # make the curve non-monotone
                rows.append(
                    {
                        "policy": policy,
                        "positioning": positioning,
                        "target_m": m,
                        "topology": "spread",
                        "tokens": 4096,
                        "mean_forward_kl": kl,
                        "top1_agreement": 0.995 if kl < 0.05 else 0.95,
                        "perplexity_ratio": 1.02 if kl < 0.05 else 1.06,
                        "realized_missing_mass_mean": m,
                        "large_divergence_fraction": 0.0005,
                        "experts_erased_mean": max(1, round(8 * m / 0.125)),
                    }
                )
    # correlation cells at headline m (spread already exists above)
    for topology in ("layer_burst", "consecutive_block", "scattered"):
        rows.append(
            {
                "policy": "renormalize",
                "positioning": "mass_omission",
                "target_m": 0.125,
                "topology": topology,
                "tokens": 4096,
                "mean_forward_kl": headline_kl,
                "top1_agreement": 0.995,
                "perplexity_ratio": 1.02,
                "realized_missing_mass_mean": 0.125,
                "large_divergence_fraction": 0.0,
                "experts_erased_mean": 1.0,
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class MonotonicityTest(unittest.TestCase):
    def test_monotone_accepts_flat_and_increasing(self) -> None:
        self.assertTrue(_monotone_non_decreasing([0.0, 0.05, 0.10], 1e-6))
        self.assertTrue(_monotone_non_decreasing([0.05, 0.05, 0.05], 1e-6))

    def test_monotone_rejects_a_drop(self) -> None:
        self.assertFalse(_monotone_non_decreasing([0.10, 0.05, 0.20], 1e-6))


class AnalyzeQ1Test(unittest.TestCase):
    def _run(self, config: dict, headline_kl: float, monotone: bool) -> dict:
        with tempfile.TemporaryDirectory() as td:
            analysis = Path(td)
            config["output_dir"] = str(analysis)
            _write_aggregates(analysis / "quality_aggregates.csv", headline_kl=headline_kl, monotone=monotone)
            summary = analyze_q1(config)
            gate = summary["gate"]
            return {
                "decision": gate["decision"],
                "kl": gate["primary"]["mean_forward_kl"],
                "monotone": gate["monotonicity"]["monotone_non_decreasing"],
            }

    def test_gate_go_on_low_controlled_erasure_cost(self) -> None:
        out = self._run(_base_config(), headline_kl=0.02, monotone=True)
        self.assertEqual(out["decision"], "GO")
        self.assertTrue(out["monotone"])

    def test_gate_stop_on_high_divergence(self) -> None:
        out = self._run(_base_config(), headline_kl=0.30, monotone=True)
        self.assertEqual(out["decision"], "STOP")

    def test_gate_stop_on_non_monotone_curve(self) -> None:
        out = self._run(_base_config(), headline_kl=0.02, monotone=False)
        self.assertEqual(out["decision"], "STOP")
        self.assertFalse(out["monotone"])

    def test_artifacts_emitted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            analysis = Path(td)
            config = _base_config()
            config["output_dir"] = str(analysis)
            _write_aggregates(analysis / "quality_aggregates.csv", headline_kl=0.02, monotone=True)
            analyze_q1(config)
            for name in (
                "gate.json",
                "summary.json",
                "REPORT.md",
                "positioning_scan.csv",
                "correlation_scan.csv",
            ):
                self.assertTrue((analysis / name).is_file(), name)
            gate = json.loads((analysis / "gate.json").read_text())
            self.assertEqual(gate["decision"], "GO")


if __name__ == "__main__":
    unittest.main()


class TailAccumulatorTest(unittest.TestCase):
    def _make(self, **kw) -> "_TailCellAccumulator":
        from ep_predict.hardware.q1 import _TailCellAccumulator

        return _TailCellAccumulator(
            policy=kw.get("policy", "renormalize"),
            positioning=kw.get("positioning", "mass_omission"),
            incidence=kw.get("incidence", 0.009),
            run_length=kw.get("run_length", 1),
            experts_per_drop=kw.get("experts_per_drop", 1),
        )

    def test_conditional_separates_affected_from_untouched(self) -> None:
        import torch

        acc = self._make()
        acc.set_large_kl(2.0)
        # 8 positions: only positions 0,1 affected (index 7 = last).
        affected = torch.tensor([1, 1, 0, 0, 0, 0, 0, 1], dtype=torch.bool)
        metrics = {
            "kl": torch.tensor([0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5]),
            "top1_agree": torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
            "nll_c": torch.zeros(8),
            "nll_e": torch.ones(8),
        }
        acc.add(metrics, affected)
        row = acc.row()
        self.assertEqual(row["tokens_total"], 8)
        self.assertEqual(row["tokens_affected"], 3)
        self.assertAlmostEqual(row["realized_incidence"], 3 / 8)
        # conditional mean KL over [0.5,0.5,0.5] = 0.5
        self.assertAlmostEqual(row["affected_mean_forward_kl"], 0.5)
        self.assertAlmostEqual(row["affected_top1_agreement"], 2 / 3)

    def test_large_divergence_fraction_over_affected(self) -> None:
        import torch

        acc = self._make()
        acc.set_large_kl(0.3)
        affected = torch.tensor([1, 1, 1], dtype=torch.bool)
        metrics = {
            "kl": torch.tensor([0.1, 0.5, 0.6]),
            "top1_agree": torch.tensor([1.0, 1.0, 1.0]),
            "nll_c": torch.zeros(3),
            "nll_e": torch.zeros(3),
        }
        acc.add(metrics, affected)
        row = acc.row()
        self.assertAlmostEqual(row["affected_large_divergence_fraction"], 2 / 3)


TAIL_GATE = {
    "headline_policy": "renormalize",
    "headline_positioning": "mass_omission",
    "headline_incidence": 0.009,
    "headline_run_length": 1,
    "experts_per_drop": 1,
    "max_mean_forward_kl": 0.05,
    "min_top1_agreement": 0.99,
    "max_ppl_ratio": 1.05,
    "large_divergence_kl": 2.0,
    "large_divergence_max_fraction": 0.01,
    "monotone_slack": 1e-6,
}


def _write_tail_sweep(path: Path, *, headline_affected_kl: float, monotone: bool) -> None:
    from ep_predict.analysis.q1 import analyze_q1_tail  # noqa

    incidences = [0.002, 0.005, 0.009, 0.02, 0.05, 0.10, 0.20, 0.30]
    run_lengths = [1, 2, 4, 8]
    rows = []
    for policy in ("renormalize", "null"):
        for inc in incidences:
            if not monotone and inc == 0.10:
                akl = 1.0  # break overall monotonicity at this level
            else:
                akl = headline_affected_kl
            rows.append(
                {
                    "policy": policy,
                    "positioning": "mass_omission",
                    "incidence": inc,
                    "run_length": 1,
                    "experts_per_drop": 1,
                    "tokens_total": 4096,
                    "tokens_affected": int(4096 * inc),
                    "realized_incidence": inc,
                    "overall_mean_forward_kl": akl * inc / 0.009,
                    "overall_top1_agreement": 1.0 - 0.001 * inc,
                    "overall_perplexity_ratio": 1.01,
                    "affected_mean_forward_kl": akl,
                    "affected_p90_forward_kl": akl * 1.5,
                    "affected_top1_agreement": 0.998 if akl < 1.0 else 0.90,
                    "affected_perplexity_ratio": 1.01 if akl < 1.0 else 2.0,
                    "affected_large_divergence_fraction": 0.0002,
                    "affected_mean_missing_mass": 0.125,
                    "affected_experts_erased_mean": 1.0,
                }
            )
    for rl in run_lengths:
        if rl == 1:
            continue
        rows.append(
            {
                "policy": "renormalize",
                "positioning": "mass_omission",
                "incidence": 0.009,
                "run_length": rl,
                "experts_per_drop": 1,
                "tokens_total": 4096,
                "tokens_affected": int(4096 * 0.009),
                "realized_incidence": 0.009,
                "overall_mean_forward_kl": headline_affected_kl * 0.009 / 0.009,
                "overall_top1_agreement": 0.999,
                "overall_perplexity_ratio": 1.01,
                "affected_mean_forward_kl": headline_affected_kl * rl,
                "affected_p90_forward_kl": headline_affected_kl * rl * 1.5,
                "affected_top1_agreement": 0.998,
                "affected_perplexity_ratio": 1.01,
                "affected_large_divergence_fraction": 0.0002,
                "affected_mean_missing_mass": 0.125,
                "affected_experts_erased_mean": 1.0,
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class AnalyzeQ1TailTest(unittest.TestCase):
    def _base_tail_config(self) -> dict:
        return {
            "tail_output_dir": "/tmp/q1-tail-test",
            "tail_probe": {
                "incidence": [0.002, 0.005, 0.009, 0.02, 0.05, 0.10, 0.20, 0.30],
                "run_lengths": [1, 2, 4, 8],
            },
            "tail_gate": dict(TAIL_GATE),
        }

    def _run(self, config: dict, akl: float, monotone: bool) -> dict:
        from ep_predict.analysis.q1 import analyze_q1_tail

        with tempfile.TemporaryDirectory() as td:
            analysis = Path(td)
            config["tail_output_dir"] = str(analysis)
            _write_tail_sweep(analysis / "tail_sweep.csv",
                              headline_affected_kl=akl, monotone=monotone)
            summary = analyze_q1_tail(config)
            gate = summary["gate"]
            return {
                "decision": gate["decision"],
                "akl": gate["primary"]["mean_forward_kl"],
                "monotone": gate["monotonicity"]["overall_non_decreasing_in_incidence"],
            }

    def test_gate_go_on_low_tail_cost(self) -> None:
        out = self._run(self._base_tail_config(), akl=0.02, monotone=True)
        self.assertEqual(out["decision"], "GO")
        self.assertTrue(out["monotone"])

    def test_gate_stop_on_high_conditional_kld(self) -> None:
        out = self._run(self._base_tail_config(), akl=0.30, monotone=True)
        self.assertEqual(out["decision"], "STOP")

    def test_gate_stop_on_non_monotone_incidence(self) -> None:
        out = self._run(self._base_tail_config(), akl=0.02, monotone=False)
        self.assertEqual(out["decision"], "STOP")
        self.assertFalse(out["monotone"])

    def test_tail_artifacts_emitted(self) -> None:
        from ep_predict.analysis.q1 import analyze_q1_tail

        with tempfile.TemporaryDirectory() as td:
            analysis = Path(td)
            config = self._base_tail_config()
            config["tail_output_dir"] = str(analysis)
            _write_tail_sweep(analysis / "tail_sweep.csv",
                              headline_affected_kl=0.02, monotone=True)
            analyze_q1_tail(config)
            for name in ("tail_gate.json", "tail_summary.json", "TAIL_REPORT.md"):
                self.assertTrue((analysis / name).is_file(), name)
            gate = json.loads((analysis / "tail_gate.json").read_text())
            self.assertEqual(gate["decision"], "GO")


Q1B_GATE = {
    "headline_policy": "null",
    "headline_positioning": "mass_omission",
    "incidence": 0.009,
    "headline_run_length": 8,
    "experts_per_drop": 1,
    "max_superlinear_ratio": 3.0,
    "large_divergence_kl": 2.0,
    "large_divergence_max_fraction": 0.01,
    "monotone_slack": 1e-6,
}


def _write_q1b_depth(path: Path, *, kl_vals: list[float], large_frac: float = 0.0001) -> None:
    Ls = [1, 2, 4, 8]
    rows = []
    for L, kl in zip(Ls, kl_vals, strict=True):
        rows.append(
            {
                "policy": "null",
                "positioning": "mass_omission",
                "incidence": 0.009,
                "run_length": L,
                "experts_per_drop": 1,
                "tokens_total": 4096,
                "tokens_affected": int(4096 * 0.009),
                "realized_incidence": 0.009,
                "overall_mean_forward_kl": kl * 0.009,
                "overall_top1_agreement": 1.0,
                "overall_perplexity_ratio": 1.0,
                "affected_mean_forward_kl": kl,
                "affected_p90_forward_kl": kl * 1.5,
                "affected_top1_agreement": 0.998 if kl < 1.0 else 0.9,
                "affected_perplexity_ratio": 1.01,
                "affected_large_divergence_fraction": (
                    large_frac if L == 8 else 0.0001
                ),
                "affected_mean_missing_mass": 0.098,
                "affected_experts_erased_mean": 1.0,
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class AnalyzeQ1BTest(unittest.TestCase):
    def _config(self) -> dict:
        return {
            "q1b_output_dir": "/tmp/q1b-test",
            "q1b_probe": {"depth_run_lengths": [1, 2, 4, 8]},
            "q1b_gate": dict(Q1B_GATE),
        }

    def _run(self, config: dict, kl_vals: list[float], large_frac: float = 0.0001) -> dict:
        from ep_predict.analysis.q1 import analyze_q1b

        with tempfile.TemporaryDirectory() as td:
            analysis = Path(td)
            config["q1b_output_dir"] = str(analysis)
            _write_q1b_depth(analysis / "null_depth_scan.csv", kl_vals=kl_vals, large_frac=large_frac)
            summary = analyze_q1b(config)
            g = summary["gate"]
            return {
                "decision": g["decision"],
                "monotone": g["verdicts"]["monotone_in_l"],
                "superlinear": g["verdicts"]["superlinear_marginal_blowup"],
            }

    def test_gate_go_on_additive_depth(self) -> None:
        out = self._run(self._config(), kl_vals=[0.005, 0.010, 0.020, 0.040])
        self.assertEqual(out["decision"], "GO")
        self.assertTrue(out["monotone"])
        self.assertFalse(out["superlinear"])

    def test_gate_stop_on_non_monotone(self) -> None:
        out = self._run(self._config(), kl_vals=[0.005, 0.010, 0.005, 0.040])
        self.assertEqual(out["decision"], "STOP")
        self.assertFalse(out["monotone"])

    def test_gate_stop_on_superlinear_blowup(self) -> None:
        out = self._run(self._config(), kl_vals=[0.001, 0.002, 0.004, 0.080])
        self.assertEqual(out["decision"], "STOP")
        self.assertTrue(out["superlinear"])

    def test_gate_stop_on_large_divergence_at_worst_l(self) -> None:
        out = self._run(self._config(), kl_vals=[0.005, 0.010, 0.020, 0.040], large_frac=0.05)
        self.assertEqual(out["decision"], "STOP")

    def test_artifacts_emitted(self) -> None:
        from ep_predict.analysis.q1 import analyze_q1b

        with tempfile.TemporaryDirectory() as td:
            analysis = Path(td)
            config = self._config()
            config["q1b_output_dir"] = str(analysis)
            _write_q1b_depth(analysis / "null_depth_scan.csv", kl_vals=[0.005, 0.010, 0.020, 0.040])
            analyze_q1b(config)
            for name in ("null_gate.json", "null_summary.json", "NULL_REPORT.md"):
                self.assertTrue((analysis / name).is_file(), name)
            gate = json.loads((analysis / "null_gate.json").read_text())
            self.assertEqual(gate["decision"], "GO")
