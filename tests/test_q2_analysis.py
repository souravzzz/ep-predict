from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from ep_predict.analysis.q2 import (
    analyze_q2_cliff,
    analyze_q2_cross_domain,
    analyze_q2_decode,
    _monotone_non_decreasing,
)

CROSS_GATE = {
    "headline_run_length": 8,
    "max_superlinear_ratio": 3.0,
    "large_divergence_kl": 2.0,
    "large_divergence_max_fraction": 0.01,
    "monotone_slack": 1e-6,
}
DECODE_GATE = {
    "min_token_agreement": 0.80,
    "max_mean_step_kl": 0.05,
    "runaway_late_early_ratio": 3.0,
}
CLIFF_GATE = {
    "free_band_max_kl": 0.02,
    "ax4_nominal_incidence": 0.009,
    "ax4_nominal_run_length": 8,
    "ax4_nominal_experts": 1,
}


def _fieldnames(rows) -> list[str]:
    return list(rows[0])


def _write_csv(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)


def _base_depth_row(domain: str, L: int, kl: float, large_frac: float = 0.0001) -> dict:
    return {
        "domain": domain,
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
        "affected_large_divergence_fraction": large_frac if L == 8 else 0.0001,
        "affected_mean_missing_mass": 0.098,
        "affected_experts_erased_mean": 1.0,
    }


class Q2MonotonicityTest(unittest.TestCase):
    def test_monotone_helpers(self) -> None:
        self.assertTrue(_monotone_non_decreasing([0.0, 0.01, 0.02], 1e-6))
        self.assertFalse(_monotone_non_decreasing([0.02, 0.01, 0.03], 1e-6))


class AnalyzeQ2CrossDomainTest(unittest.TestCase):
    def _run(self, kl_by_domain: dict[str, list[float]], large_frac: float = 0.0001) -> dict:
        with tempfile.TemporaryDirectory() as td:
            analysis = Path(td)
            rows = []
            for domain, kls in kl_by_domain.items():
                for L, kl in zip([1, 2, 4, 8], kls, strict=True):
                    rows.append(_base_depth_row(domain, L, kl, large_frac))
            _write_csv(analysis / "depth_by_domain.csv", rows)
            cfg = {"incidence": 0.009, "experts_per_drop": 1}
            return analyze_q2_cross_domain(analysis, cfg, dict(CROSS_GATE))

    def test_go_when_all_domains_additive(self) -> None:
        g = self._run({"ref_wikitext2": [0.005, 0.010, 0.020, 0.040],
                       "math": [0.006, 0.011, 0.022, 0.043]})
        self.assertEqual(g["decision"], "GO")
        self.assertEqual(g["broken_domains"], [])

    def test_broken_domain_marks_candidate(self) -> None:
        g = self._run({"ref_wikitext2": [0.005, 0.010, 0.020, 0.040],
                       "math": [0.006, 0.011, 0.005, 0.043]})
        self.assertEqual(g["decision"], "CANDIDATE_ROBUSTNESS_TARGET")
        self.assertEqual(g["broken_domains"], ["math"])

    def test_artifacts_emitted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            analysis = Path(td)
            rows = [_base_depth_row("ref_wikitext2", L, 0.01 * L) for L in [1, 2, 4, 8]]
            _write_csv(analysis / "depth_by_domain.csv", rows)
            cfg = {"incidence": 0.009, "experts_per_drop": 1}
            analyze_q2_cross_domain(analysis, cfg, dict(CROSS_GATE))
            self.assertTrue((analysis / "gate.json").is_file())
            self.assertTrue((analysis / "CROSS_REPORT.md").is_file())


class AnalyzeQ2DecodeTest(unittest.TestCase):
    def _write(self, analysis: Path, *, mean_kl: float, agreement: float,
               late_ratio: float = 1.0) -> None:
        n = 64
        rows = []
        # Steps < agree_steps are marked as agreeing (clean vs erased same argmax).
        agree_steps = int(round(agreement * n))
        for rl in (1, 8):
            cum = 0.0
            for step in range(n):
                # early window more expensive -> high late_ratio = drops.
                if step < n // 4:
                    skl = mean_kl
                else:
                    skl = mean_kl * late_ratio
                cum = (cum * step + skl) / (step + 1)
                rows.append(
                    {
                        "run_length": rl,
                        "step": step,
                        "step_kl": f"{skl:.6f}",
                        "token_agree": 1 if step < agree_steps else 0,
                        "cumulative_kl": f"{cum:.6f}",
                    }
                )
        _write_csv(analysis / "continuation.csv", rows)
        (analysis / "decode_summary.json").write_text(
            json.dumps(
                {
                    "incidence": 0.009,
                    "max_new_tokens": n,
                    "prefix_tokens": 64,
                    "per_run_length": {
                        str(rl): {
                            "mean_step_kl": mean_kl,
                            "final_cumulative_kl": mean_kl,
                            "token_agreement": agreement,
                        }
                        for rl in (1, 8)
                    },
                }
            ),
            encoding="utf-8",
        )

    def _run(self, mean_kl: float, agreement: float, late_ratio: float = 1.0) -> dict:
        with tempfile.TemporaryDirectory() as td:
            analysis = Path(td)
            self._write(analysis, mean_kl=mean_kl, agreement=agreement, late_ratio=late_ratio)
            cfg = {"incidence": 0.009, "experts_per_drop": 1, "max_new_tokens": 64}
            return analyze_q2_decode(analysis, cfg, dict(DECODE_GATE))

    def test_go_on_coherent_continuation(self) -> None:
        g = self._run(mean_kl=0.01, agreement=0.95)
        self.assertEqual(g["decision"], "GO")
        self.assertFalse(g["runaway_detected"])

    def test_candidate_on_runaway_compounding(self) -> None:
        g = self._run(mean_kl=0.20, agreement=0.95, late_ratio=5.0)
        self.assertEqual(g["decision"], "CANDIDATE_ROBUSTNESS_TARGET")
        self.assertTrue(g["runaway_detected"])

    def test_candidate_on_low_agreement(self) -> None:
        g = self._run(mean_kl=0.01, agreement=0.50)
        self.assertEqual(g["decision"], "CANDIDATE_ROBUSTNESS_TARGET")

    def test_artifacts_emitted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            analysis = Path(td)
            self._write(analysis, mean_kl=0.01, agreement=0.95)
            cfg = {"incidence": 0.009, "experts_per_drop": 1, "max_new_tokens": 64}
            analyze_q2_decode(analysis, cfg, dict(DECODE_GATE))
            self.assertTrue((analysis / "gate.json").is_file())
            self.assertTrue((analysis / "DECODE_REPORT.md").is_file())


class AnalyzeQ2CliffTest(unittest.TestCase):
    def _row(self, axis, inc, rl, exp, kl) -> dict:
        return {
            "axis": axis,
            "policy": "null",
            "positioning": "mass_omission",
            "incidence": f"{inc}",
            "run_length": rl,
            "experts_per_drop": exp,
            "tokens_total": 4096,
            "tokens_affected": int(4096 * inc),
            "realized_incidence": inc,
            "overall_mean_forward_kl": f"{kl * inc:.5f}",
            "overall_top1_agreement": 1.0,
            "overall_perplexity_ratio": 1.0,
            "affected_mean_forward_kl": f"{kl:.5f}",
            "affected_p90_forward_kl": f"{kl * 1.5:.5f}",
            "affected_top1_agreement": 0.998,
            "affected_perplexity_ratio": 1.01,
            "affected_large_divergence_fraction": 0.0001,
            "affected_mean_missing_mass": 0.098,
            "affected_experts_erased_mean": 1.0,
        }

    def _run(self, kl_at_ax4: float, inc_high: float) -> dict:
        with tempfile.TemporaryDirectory() as td:
            analysis = Path(td)
            rows = []
            # incidence axis at L=8, 1 expert
            for inc in [0.009, 0.02, 0.05, 0.1, 0.3, 0.5]:
                kl = kl_at_ax4 if inc == 0.009 else (inc_high if inc >= 0.05 else kl_at_ax4 * 1.5)
                rows.append(self._row("incidence", inc, 8, 1, kl))
            # run-length axis at inc 0.009, 1 expert
            for rl in [8, 12, 16]:
                kl = kl_at_ax4 if rl == 8 else inc_high
                rows.append(self._row("run_length", 0.009, rl, 1, kl))
            # experts axis at inc 0.009, L=8
            for exp in [1, 2, 4]:
                kl = kl_at_ax4 if exp == 1 else inc_high
                rows.append(self._row("experts_per_layer", 0.009, 8, exp, kl))
            _write_csv(analysis / "cliff_surface.csv", rows)
            cfg = {"incidence": 0.009, "experts_per_drop": 1}
            return analyze_q2_cliff(analysis, cfg, dict(CLIFF_GATE))

    def test_with_margin_when_nominal_free_and_cliff_beyond(self) -> None:
        g = self._run(kl_at_ax4=0.010, inc_high=0.20)
        self.assertEqual(g["reading"], "WITH_MARGIN")
        self.assertTrue(g["ax4_nominal"]["free"])
        self.assertEqual(g["cliffs"]["incidence"]["crossing_at"], 0.05)

    def test_inside_when_nominal_not_free(self) -> None:
        g = self._run(kl_at_ax4=0.20, inc_high=0.40)
        self.assertEqual(g["reading"], "INSIDE")
        self.assertFalse(g["ax4_nominal"]["free"])

    def test_artifacts_emitted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            analysis = Path(td)
            rows = []
            for inc in [0.009, 0.02, 0.05, 0.1, 0.3, 0.5]:
                kl = 0.010 if inc == 0.009 else 0.20
                rows.append(self._row("incidence", inc, 8, 1, kl))
            for rl in [8, 12, 16]:
                rows.append(self._row("run_length", 0.009, rl, 1, 0.010 if rl == 8 else 0.20))
            for exp in [1, 2, 4]:
                rows.append(self._row("experts_per_layer", 0.009, 8, exp, 0.010 if exp == 1 else 0.20))
            _write_csv(analysis / "cliff_surface.csv", rows)
            cfg = {"incidence": 0.009, "experts_per_drop": 1}
            analyze_q2_cliff(analysis, cfg, dict(CLIFF_GATE))
            self.assertTrue((analysis / "gate.json").is_file())
            self.assertTrue((analysis / "CLIFF_REPORT.md").is_file())


if __name__ == "__main__":
    unittest.main()
