# -*- coding: utf-8 -*-
"""test_p4.py

Automated tests for P4 Evaluation & Integration using synthetic data.

IMPORTANT:  These tests use synthetic data only.
            Results must NOT be reported as actual model performance.
            Tests do NOT depend on any real dataset files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# sys.path setup
# ---------------------------------------------------------------------------
code_dir = str(Path(__file__).resolve().parents[2])
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from business_entity_resolution.src.p4_evaluation import (
    error_analysis,
    load_ground_truth,
    load_s1_entity_ids,
    load_scored_pairs,
    select_matches,
    tune_threshold,
    write_matching_results,
)
from business_entity_resolution.p3_matching.evaluate import (
    compute_entity_macro_f05,
)


# ===================================================================
# HELPERS
# ===================================================================

def _write_ground_truth(path: Path, rows: list[tuple[str, str]]) -> None:
    """Write a synthetic ground truth TSV.  rows = [(s1_id, ids_str), ...]"""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id, ids_str in rows:
            f.write(f"{s1_id}\t{ids_str}\n")


def _write_scored_pairs(path: Path, rows: list[tuple[str, str, float]]) -> None:
    """Write a synthetic scored pairs TSV."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_id\tmatch_probability\n")
        for s1, cand, prob in rows:
            f.write(f"{s1}\t{cand}\t{prob:.6f}\n")


def _write_source1(path: Path, ids: list[str]) -> None:
    """Write a synthetic source1 TSV with entity_id column."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        for eid in ids:
            f.write(f"{eid}\tTest Business\t123 Test St\tUS\n")


# ===================================================================
# TESTS
# ===================================================================


class TestLoadGroundTruth:
    """Tests for load_ground_truth."""

    def test_basic_parsing(self, tmp_path: Path) -> None:
        gt_file = tmp_path / "gt.tsv"
        _write_ground_truth(gt_file, [
            ("S1-1", "S2-1,S3-1"),
            ("S1-2", "S2-2"),
            ("S1-3", ""),           # singleton
            ("S1-4", "S2-4,S3-4,S3-5"),
            ("S1-5", ""),           # singleton
        ])
        gt = load_ground_truth(gt_file)

        assert len(gt) == 5
        assert gt["S1-1"] == {"S2-1", "S3-1"}
        assert gt["S1-2"] == {"S2-2"}
        assert gt["S1-3"] == set()
        assert gt["S1-4"] == {"S2-4", "S3-4", "S3-5"}
        assert gt["S1-5"] == set()

    def test_singleton_count(self, tmp_path: Path) -> None:
        gt_file = tmp_path / "gt.tsv"
        _write_ground_truth(gt_file, [
            ("S1-A", ""),
            ("S1-B", ""),
            ("S1-C", "S2-C"),
        ])
        gt = load_ground_truth(gt_file)
        singletons = sum(1 for v in gt.values() if not v)
        assert singletons == 2


class TestLoadScoredPairs:
    """Tests for load_scored_pairs."""

    def test_column_names_and_types(self, tmp_path: Path) -> None:
        sp_file = tmp_path / "scored.tsv"
        _write_scored_pairs(sp_file, [
            ("S1-1", "S2-1", 0.95),
            ("S1-1", "S3-1", 0.30),
            ("S1-2", "S2-2", 0.60),
        ])
        df = load_scored_pairs(sp_file)
        assert list(df.columns) == [
            "source1_entity_id", "candidate_entity_id", "match_probability",
        ]
        assert df["match_probability"].dtype == np.float64


class TestSelectMatches:
    """Tests for select_matches."""

    def test_threshold_partitioning(self) -> None:
        df = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-1", "S1-2", "S1-3"],
            "candidate_entity_id": ["S2-1", "S3-1", "S2-2", "S2-3"],
            "match_probability": [0.90, 0.40, 0.60, 0.10],
        })
        matches = select_matches(df, threshold=0.5)

        assert "S1-1" in matches
        assert matches["S1-1"] == {"S2-1"}       # 0.90 >= 0.5, 0.40 < 0.5
        assert matches["S1-2"] == {"S2-2"}       # 0.60 >= 0.5
        assert "S1-3" not in matches              # 0.10 < 0.5

    def test_all_below_threshold(self) -> None:
        df = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2"],
            "candidate_entity_id": ["S2-1", "S2-2"],
            "match_probability": [0.10, 0.20],
        })
        matches = select_matches(df, threshold=0.5)
        assert len(matches) == 0

    def test_multiple_matches_per_entity(self) -> None:
        df = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-1", "S1-1"],
            "candidate_entity_id": ["S2-1", "S2-2", "S3-1"],
            "match_probability": [0.80, 0.70, 0.90],
        })
        matches = select_matches(df, threshold=0.5)
        assert matches["S1-1"] == {"S2-1", "S2-2", "S3-1"}


class TestWriteMatchingResults:
    """Tests for write_matching_results."""

    def test_complete_output(self, tmp_path: Path) -> None:
        out_file = tmp_path / "matching_results.tsv"
        matches = {"S1-1": {"S3-1", "S2-1"}, "S1-2": set()}
        all_s1_ids = ["S1-1", "S1-2", "S1-3"]
        count = write_matching_results(matches, all_s1_ids, out_file)

        assert count == 3

        lines = out_file.read_text(encoding="utf-8").rstrip("\n").split("\n")
        assert len(lines) == 4  # header + 3 rows

        # Header
        assert lines[0] == "source1_entity_id\tmatched_entity_ids"

        # S1-1 — two matches, sorted
        assert lines[1] == "S1-1\tS2-1,S3-1"

        # S1-2 — explicit empty set
        assert lines[2] == "S1-2\t"

        # S1-3 — not in matches dict, should still appear as singleton
        assert lines[3] == "S1-3\t"

    def test_all_singletons(self, tmp_path: Path) -> None:
        out_file = tmp_path / "singletons.tsv"
        all_s1_ids = ["S1-A", "S1-B", "S1-C"]
        count = write_matching_results({}, all_s1_ids, out_file)

        assert count == 3
        lines = out_file.read_text(encoding="utf-8").rstrip("\n").split("\n")
        for line in lines[1:]:
            parts = line.split("\t")
            assert len(parts) == 2
            assert parts[1] == ""

    def test_tab_separated_not_csv(self, tmp_path: Path) -> None:
        out_file = tmp_path / "format.tsv"
        matches = {"S1-1": {"S2-1"}}
        write_matching_results(matches, ["S1-1"], out_file)

        content = out_file.read_text(encoding="utf-8")
        assert "\t" in content
        # The second column should not be quoted
        assert '"' not in content


class TestSingletonScoring:
    """Verify singleton scoring via the P3 evaluate function."""

    def test_correct_singleton_prediction(self) -> None:
        gt = {"S1-1": set()}
        pred = {}  # no predictions for S1-1 → empty
        _, _, f05 = compute_entity_macro_f05(gt, pred)
        assert f05 == 1.0

    def test_spurious_singleton_prediction(self) -> None:
        gt = {"S1-1": set()}
        pred = {"S1-1": {"S2-1"}}
        _, _, f05 = compute_entity_macro_f05(gt, pred)
        assert f05 == 0.0

    def test_mixed_singleton_and_matched(self) -> None:
        gt = {"S1-1": set(), "S1-2": {"S2-2"}}
        # Perfect on both
        pred = {"S1-2": {"S2-2"}}
        _, _, f05 = compute_entity_macro_f05(gt, pred)
        assert f05 == 1.0


class TestThresholdTuning:
    """Tests for tune_threshold."""

    def test_finds_correct_region(self, tmp_path: Path) -> None:
        # Construct data where the optimal threshold is around 0.60:
        #   S1-1 matches S2-1 (prob 0.95) — TP above any threshold
        #   S1-2 matches S2-2 (prob 0.65) — TP only above ≤ 0.65
        #   S1-2 non-match S3-2 (prob 0.55) — FP if threshold ≤ 0.55
        #   S1-3 singleton, no scored pairs
        scored_df = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2", "S1-2"],
            "candidate_entity_id": ["S2-1", "S2-2", "S3-2"],
            "match_probability": [0.95, 0.65, 0.55],
        })
        gt = {
            "S1-1": {"S2-1"},
            "S1-2": {"S2-2"},
            "S1-3": set(),
        }
        best_thr, results_df = tune_threshold(
            scored_df, gt,
            coarse_start=0.10, coarse_end=0.90, coarse_step=0.05,
            refine_window=0.10, refine_step=0.01,
        )
        # The optimal threshold should be in the range that captures S2-2
        # but rejects S3-2: between 0.56 and 0.65
        assert 0.50 <= best_thr <= 0.70
        assert "macro_f05" in results_df.columns


class TestErrorAnalysis:
    """Tests for error_analysis."""

    def test_tp_fp_fn_counts(self) -> None:
        scored_df = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
            "candidate_entity_id": ["S2-1", "S2-2", "S2-3"],
            "match_probability": [0.90, 0.90, 0.90],
        })
        gt = {
            "S1-1": {"S2-1"},     # S2-1 predicted → TP
            "S1-2": set(),         # S2-2 predicted on singleton → FP
            "S1-3": {"S2-4"},     # S2-4 not predicted, S2-3 predicted → FP + FN
        }
        results = error_analysis(scored_df, gt, threshold=0.5)

        assert results["metrics"]["TP"] == 1
        assert results["metrics"]["FP"] == 2   # S2-2 on singleton + S2-3 wrong
        assert results["metrics"]["FN"] == 1   # S2-4 missed

    def test_singleton_error_tracking(self) -> None:
        scored_df = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-1"],
            "candidate_entity_id": ["S2-1", "S2-2"],
            "match_probability": [0.80, 0.70],
        })
        gt = {"S1-1": set()}  # true singleton
        results = error_analysis(scored_df, gt, threshold=0.5)

        assert results["singleton_analysis"]["spurious_predictions_on_singletons"] == 2

    def test_candidate_recall_breakdown(self) -> None:
        scored_df = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-1"],
            "candidate_entity_id": ["S2-1", "S2-2"],
            "match_probability": [0.90, 0.30],  # S2-2 scored but below threshold
        })
        gt = {
            "S1-1": {"S2-1", "S2-2", "S2-3"},
            # S2-1 → TP (prob 0.90 ≥ 0.5)
            # S2-2 → FN scored below threshold (prob 0.30 < 0.5)
            # S2-3 → FN missing from candidates entirely
        }
        results = error_analysis(scored_df, gt, threshold=0.5)

        assert results["candidate_recall"]["below_threshold_FN"] == 1
        assert results["candidate_recall"]["missing_from_candidates_FN"] == 1


class TestUnseenCountryHandling:
    """Verify P4 processes France entities without filtering."""

    def test_france_entities_pass_through(self) -> None:
        scored_df = pd.DataFrame({
            "source1_entity_id": ["S1-FR-1", "S1-FR-2"],
            "candidate_entity_id": ["S2-FR-1", "S3-FR-2"],
            "match_probability": [0.95, 0.85],
        })
        matches = select_matches(scored_df, threshold=0.5)
        assert matches["S1-FR-1"] == {"S2-FR-1"}
        assert matches["S1-FR-2"] == {"S3-FR-2"}

    def test_france_entities_in_output(self, tmp_path: Path) -> None:
        out_file = tmp_path / "results.tsv"
        matches = {"S1-FR-1": {"S2-FR-1"}}
        all_s1_ids = ["S1-FR-1", "S1-FR-2"]
        write_matching_results(matches, all_s1_ids, out_file)

        lines = out_file.read_text(encoding="utf-8").rstrip("\n").split("\n")
        assert lines[1] == "S1-FR-1\tS2-FR-1"
        assert lines[2] == "S1-FR-2\t"


class TestCandidateSubsetConstraint:
    """Verify select_matches only returns IDs from the scored input."""

    def test_no_phantom_ids(self) -> None:
        scored_df = pd.DataFrame({
            "source1_entity_id": ["S1-1"],
            "candidate_entity_id": ["S2-1"],
            "match_probability": [0.90],
        })
        matches = select_matches(scored_df, threshold=0.5)
        assert matches["S1-1"] == {"S2-1"}
        # S3-1 was never scored, so it cannot appear
        assert "S3-1" not in matches.get("S1-1", set())


class TestFormatCompliance:
    """Verify matching_results.tsv format meets validator requirements."""

    def test_full_format_check(self, tmp_path: Path) -> None:
        out_file = tmp_path / "check.tsv"
        matches = {
            "S1-1": {"S3-1", "S2-1"},
            "S1-2": {"S2-2"},
        }
        all_s1_ids = ["S1-1", "S1-2", "S1-3"]
        write_matching_results(matches, all_s1_ids, out_file)

        content = out_file.read_text(encoding="utf-8")
        lines = content.rstrip("\n").split("\n")

        # Correct header
        header_parts = lines[0].split("\t")
        assert header_parts == ["source1_entity_id", "matched_entity_ids"]

        # Every S1 ID present exactly once
        s1_ids_in_file = [line.split("\t")[0] for line in lines[1:]]
        assert sorted(s1_ids_in_file) == sorted(all_s1_ids)
        assert len(s1_ids_in_file) == len(set(s1_ids_in_file))  # no duplicates

        # No S1- IDs in match lists
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) > 1 and parts[1]:
                for mid in parts[1].split(","):
                    assert not mid.startswith("S1-"), f"Self-match found: {mid}"
                    assert mid.startswith("S2-") or mid.startswith("S3-"), \
                        f"Invalid prefix: {mid}"

    def test_load_s1_entity_ids(self, tmp_path: Path) -> None:
        s1_file = tmp_path / "source1.tsv"
        _write_source1(s1_file, ["S1-C", "S1-A", "S1-B"])
        ids = load_s1_entity_ids(s1_file)
        # Should be sorted
        assert ids == ["S1-A", "S1-B", "S1-C"]
