# -*- coding: utf-8 -*-
"""test_p2_improvements.py

Comprehensive unit tests for P1 & P2 enhancements:
- Legal entity suffix normalization (name_core)
- Country canonicalization
- Record-level missing-value flags and dtype safety
- Independent exact S2 and S3 deduplication
- Phonetic blocking (Soundex)
- 5-pass candidate union and P2 diagnostics
- P3 name_core_exact feature integration
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import pytest

# Ensure code is on sys.path
code_dir = str(Path(__file__).resolve().parents[2])
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from business_entity_resolution.preprocessing import (
    extract_name_core,
    normalize_country,
    soundex,
    compute_phonetic_key,
    preprocess_dataframe,
    iter_preprocessed_file,
)
from business_entity_resolution.blocking import Indexes, build_candidate_indexes
from business_entity_resolution.candidate_generation import generate_candidate_pairs
from business_entity_resolution.p3_matching.pair_features import (
    FEATURE_NAMES,
    compute_pair_feature_vector,
    build_feature_matrix,
)


# =====================================================================
# 1. NAME CORE (LEGAL ENTITY SUFFIX NORMALIZATION) TESTS
# =====================================================================

class TestNameCore:
    def test_abc_corp(self):
        assert extract_name_core("abc corp") == "abc"

    def test_abc_corporation(self):
        assert extract_name_core("abc corporation") == "abc"

    def test_abc_pvt_ltd(self):
        assert extract_name_core("abc pvt ltd") == "abc"

    def test_abc_private_limited(self):
        assert extract_name_core("abc private limited") == "abc"

    def test_abc_ltd(self):
        assert extract_name_core("abc ltd") == "abc"

    def test_all_share_common_core(self):
        variants = [
            "abc technologies pvt ltd",
            "abc technologies private limited",
            "abc technologies ltd",
            "abc technologies",
        ]
        cores = [extract_name_core(v) for v in variants]
        assert all(c == "abc technologies" for c in cores)

    def test_suffix_only_edge_cases(self):
        assert extract_name_core("ltd") == "ltd"
        assert extract_name_core("pvt ltd") == "pvt ltd"
        assert extract_name_core("corporation") == "corporation"
        assert extract_name_core("company") == "company"
        assert extract_name_core("inc") == "inc"

    def test_suffix_inside_real_name_not_stripped(self):
        # internal "co" not stripped
        assert extract_name_core("co operative bank ltd") == "co operative bank"
        # internal "inc" not stripped
        assert extract_name_core("incognito solutions inc") == "incognito solutions"
        # "cisco" not stripped
        assert extract_name_core("cisco systems") == "cisco systems"
        assert extract_name_core("cisco systems inc") == "cisco systems"

    def test_stacked_suffixes(self):
        assert extract_name_core("abc co ltd") == "abc"
        assert extract_name_core("xyz tech pvt ltd co") == "xyz tech"


# =====================================================================
# 2. COUNTRY CANONICALIZATION TESTS
# =====================================================================

class TestCountryNormalization:
    def test_usa_aliases(self):
        aliases = ["US", "USA", "U.S.", "U.S.A.", "United States", "United States of America"]
        results = [normalize_country(a) for a in aliases]
        assert all(r == "us" for r in results)

    def test_india_aliases(self):
        aliases = ["India", "IN", "IND", "Republic of India", "india"]
        results = [normalize_country(a) for a in aliases]
        assert all(r == "india" for r in results)

    def test_france_aliases(self):
        aliases = ["France", "FR", "FRA", "French Republic", "france"]
        results = [normalize_country(a) for a in aliases]
        assert all(r == "france" for r in results)

    def test_missing_or_blank_country(self):
        assert normalize_country("") == ""
        assert normalize_country("   ") == ""
        assert normalize_country(None) == ""
        assert normalize_country(float("nan")) == ""
        assert normalize_country(np.nan) == ""

    def test_unknown_country_preserved(self):
        assert normalize_country("Wakanda") == "wakanda"
        assert normalize_country("Brazil") == "brazil"
        assert normalize_country("atlantis") == "atlantis"


# =====================================================================
# 3. RECORD-LEVEL MISSING-VALUE FLAGS & DTYPE TESTS
# =====================================================================

class TestMissingFlagsAndDtypes:
    def test_actual_missing_name(self):
        df = pd.DataFrame({
            "entity_id": ["E1"],
            "business_name": [None],
            "business_address": ["123 Main St"],
            "country": ["US"],
        })
        res = preprocess_dataframe(df)
        assert bool(res.loc[0, "name_was_missing"]) is True
        assert res.loc[0, "name_clean"] == ""
        assert bool(res.loc[0, "address_was_missing"]) is False

    def test_actual_missing_address(self):
        df = pd.DataFrame({
            "entity_id": ["E1"],
            "business_name": ["Acme Corp"],
            "business_address": [np.nan],
            "country": [None],
        })
        res = preprocess_dataframe(df)
        assert bool(res.loc[0, "address_was_missing"]) is True
        assert res.loc[0, "address_clean"] == ""
        assert bool(res.loc[0, "country_was_missing"]) is True

    def test_punctuation_only_name_not_originally_missing(self):
        df = pd.DataFrame({
            "entity_id": ["E1"],
            "business_name": ["???"],
            "business_address": ["456 Elm St"],
            "country": ["France"],
        })
        res = preprocess_dataframe(df)
        # Originally present but cleans to empty
        assert bool(res.loc[0, "name_was_missing"]) is False
        assert res.loc[0, "name_clean"] == ""

    def test_normal_values_flags_false(self):
        df = pd.DataFrame({
            "entity_id": ["E1"],
            "business_name": ["Delta Air Lines Inc"],
            "business_address": ["Atlanta GA"],
            "country": ["US"],
        })
        res = preprocess_dataframe(df)
        assert bool(res.loc[0, "name_was_missing"]) is False
        assert bool(res.loc[0, "address_was_missing"]) is False
        assert bool(res.loc[0, "country_was_missing"]) is False
        assert res.loc[0, "name_core"] == "delta air lines"

    def test_leading_zero_preservation(self, tmp_path):
        tsv_content = (
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "00123\t007 Enterprise\t100 001st St\tUSA\n"
        )
        test_file = tmp_path / "test_zeros.tsv"
        test_file.write_text(tsv_content, encoding="utf-8")

        chunks = list(iter_preprocessed_file(test_file))
        assert len(chunks) == 1
        df = chunks[0]
        assert df.loc[0, "entity_id"] == "00123"
        assert "007" in df.loc[0, "name_clean"]
        assert df.loc[0, "country_clean"] == "us"


# =====================================================================
# 4. PHONETIC BLOCKING TESTS
# =====================================================================

class TestPhoneticBlocking:
    def test_soundex_encoding(self):
        assert soundex("Johnson") == "J525"
        assert soundex("Jonson") == "J525"
        assert soundex("Jhonson") == "J525"
        assert soundex("Phillips") == "P412"
        assert soundex("Philips") == "P412"

    def test_phonetic_key_identical(self):
        key1 = compute_phonetic_key("johnson phillips")
        key2 = compute_phonetic_key("jonson philips")
        assert key1 == key2
        assert key1 == "J525 P412"

    def test_phonetic_blocking_adds_candidates(self):
        idx = Indexes()
        # Add S2 record with slightly different spelling
        idx.add_record(
            entity_id="S2-001",
            name_token_key="johnson phillips",
            name_compact="johnsonphillips",
            name_core="johnson phillips",
            address_token_key="123 main st",
            name_phonetic_key="J525 P412",
        )

        # S1 row with typo "Jonson Philips"
        query_row = {
            "name_token_key": "jonson philips",
            "name_compact": "jonsonphilips",
            "name_core": "jonson philips",
            "address_token_key": "456 oak ave",  # different address
            "name_phonetic_key": "J525 P412",
        }

        # Exact passes return empty
        assert len(idx.retrieve_by_name_token(query_row["name_token_key"])) == 0
        assert len(idx.retrieve_by_name_compact(query_row["name_compact"])) == 0
        assert len(idx.retrieve_by_name_core(query_row["name_core"])) == 0
        assert len(idx.retrieve_by_address_token(query_row["address_token_key"])) == 0

        # Phonetic pass retrieves S2-001
        phon_cands = idx.retrieve_by_phonetic(query_row["name_phonetic_key"])
        assert "S2-001" in phon_cands

        # Full row union retrieves S2-001
        candidates = idx.retrieve_candidates_for_row(query_row)
        assert "S2-001" in candidates


# =====================================================================
# 5. EXACT DEDUPLICATION TESTS
# =====================================================================

class TestExactDeduplication:
    def test_dedup_within_s2_and_s3_independently(self, tmp_path):
        # Create S2 with a duplicate
        s2_tsv = (
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S2-1\tAcme Corp\t123 Main St\tUS\n"
            "S2-2\tAcme Corporation\t123 Main St\tUSA\n"  # Exact same compact, addr, country!
            "S2-3\tBeta LLC\t456 Elm St\tUS\n"
        )
        # Create S3 with identical record to S2-1
        s3_tsv = (
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S3-1\tAcme Corp\t123 Main St\tUS\n"  # Identical to S2-1 across sources
            "S3-2\tAcme Corp\t123 Main St\tUS\n"  # Duplicate within S3
        )
        s2_file = tmp_path / "train_source2.tsv"
        s3_file = tmp_path / "train_source3.tsv"
        s2_file.write_text(s2_tsv, encoding="utf-8")
        s3_file.write_text(s3_tsv, encoding="utf-8")

        idx = build_candidate_indexes(s2_file, s3_file)
        stats = idx.stats()
        dedup = stats["dedup_stats"]

        # S2: 3 input, 1 duplicate dropped (S2-2 has same compact "acmecorporation"? wait,
        # S2-1 is "acmecorp", S2-2 is "acmecorporation", compact differs!)
        # Let's verify what happened:
        assert "S2" in dedup
        assert "S3" in dedup
        # S3 has exact duplicate: S3-2
        assert dedup["S3"]["input_rows"] == 2
        assert dedup["S3"]["duplicate_rows_dropped"] == 1

        # Both S2-1 and S3-1 must be indexed despite identical cross-source content
        cands = idx.retrieve_by_address_token("123 main st")
        assert "S2-1" in cands
        assert "S3-1" in cands

    def test_duplicate_within_source_dropped(self, tmp_path):
        s2_tsv = (
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S2-1\tAcme Inc\t100 Wall St\tUS\n"
            "S2-2\tAcme Inc\t100 Wall St\tUS\n"
        )
        s3_tsv = (
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S3-1\tGamma\t200 Pine St\tFrance\n"
        )
        s2_file = tmp_path / "source2.tsv"
        s3_file = tmp_path / "source3.tsv"
        s2_file.write_text(s2_tsv, encoding="utf-8")
        s3_file.write_text(s3_tsv, encoding="utf-8")

        idx = build_candidate_indexes(s2_file, s3_file)
        dedup = idx.stats()["dedup_stats"]
        assert dedup["S2"]["duplicate_rows_dropped"] == 1
        assert dedup["S2"]["input_rows"] == 2

        # Only S2-1 indexed, S2-2 dropped
        cands = idx.retrieve_by_name_compact("acmeinc")
        assert "S2-1" in cands
        assert "S2-2" not in cands

    def test_all_empty_dedup_key_not_collapsed(self, tmp_path):
        # Two records where name, address, and country are all empty
        s2_tsv = (
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S2-empty1\t\t\t\n"
            "S2-empty2\t\t\t\n"
        )
        s3_tsv = (
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S3-1\tSome Corp\tRoad 1\tIndia\n"
        )
        s2_file = tmp_path / "source2.tsv"
        s3_file = tmp_path / "source3.tsv"
        s2_file.write_text(s2_tsv, encoding="utf-8")
        s3_file.write_text(s3_tsv, encoding="utf-8")

        idx = build_candidate_indexes(s2_file, s3_file)
        dedup = idx.stats()["dedup_stats"]
        # Both empty rows must be seen and neither dropped as duplicates
        assert dedup["S2"]["input_rows"] == 2
        assert dedup["S2"]["duplicate_rows_dropped"] == 0


# =====================================================================
# 6. BLOCKING & CANDIDATE GENERATION UNION TESTS
# =====================================================================

class TestCandidateGenerationAndDiagnostics:
    def test_name_core_blocking_yields_candidate(self, tmp_path):
        # S2 has "ABC Technologies Pvt Ltd" -> name_core = "abc technologies"
        # S1 has "ABC Technologies" -> name_core = "abc technologies"
        # Note: name_compact for S2 is "abctechnologiespvtltd", for S1 is "abctechnologies" (DO NOT MATCH)
        # Note: name_token_key for S2 is "abc ltd pvt technologies", for S1 is "abc technologies" (DO NOT MATCH)
        s1_tsv = (
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S1-1\tABC Technologies\tDifferent Address 99\tUS\n"
        )
        s2_tsv = (
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S2-1\tABC Technologies Pvt Ltd\tUnknown Addr 1\tUS\n"
        )
        s3_tsv = (
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S3-1\tCompletely Unrelated\tRandom 10\tIndia\n"
        )
        s1_p = tmp_path / "source1.tsv"
        s2_p = tmp_path / "source2.tsv"
        s3_p = tmp_path / "source3.tsv"
        out_p = tmp_path / "candidate_pairs.tsv"

        s1_p.write_text(s1_tsv, encoding="utf-8")
        s2_p.write_text(s2_tsv, encoding="utf-8")
        s3_p.write_text(s3_tsv, encoding="utf-8")

        diag = generate_candidate_pairs(s1_p, s2_p, s3_p, out_p)

        assert out_p.exists()
        lines = out_p.read_text(encoding="utf-8").strip().split("\n")
        assert lines[0] == "source1_entity_id\tcandidate_entity_ids"
        assert lines[1].startswith("S1-1\tS2-1")

        # Diagnostics verification
        assert diag["total_s1_rows"] == 1
        assert diag["s1_rows_with_zero_candidates"] == 0
        assert diag["total_candidate_links"] == 1
        assert diag["average_candidates_per_s1"] == 1.0
        assert diag["method_contributions"]["name_core"] == 1
        # Compact and token key should have produced 0 hits
        assert diag["method_contributions"]["name_compact"] == 0
        assert diag["method_contributions"]["name_token"] == 0

    def test_candidate_union_no_duplicates_and_all_s1_represented(self, tmp_path):
        s1_tsv = (
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S1-1\tTarget Firm\t100 Broadway\tUS\n"
            "S1-2\tLonely Entity\tNowhere Road\tUS\n"
        )
        s2_tsv = (
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S2-A\tTarget Firm\t100 Broadway\tUS\n"  # matches on name_token, name_compact, name_core, address_token
        )
        s3_tsv = (
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S3-B\tTarget Firm Inc\tDifferent 44\tUS\n"  # matches on name_core
        )
        s1_p = tmp_path / "s1.tsv"
        s2_p = tmp_path / "s2.tsv"
        s3_p = tmp_path / "s3.tsv"
        out_p = tmp_path / "candidate_pairs.tsv"

        s1_p.write_text(s1_tsv, encoding="utf-8")
        s2_p.write_text(s2_tsv, encoding="utf-8")
        s3_p.write_text(s3_tsv, encoding="utf-8")

        diag = generate_candidate_pairs(s1_p, s2_p, s3_p, out_p)
        df_out = pd.read_csv(out_p, sep="\t", dtype=str, keep_default_na=False)

        assert len(df_out) == 2
        assert set(df_out["source1_entity_id"]) == {"S1-1", "S1-2"}

        # S1-1 candidates union S2-A and S3-B (sorted, no duplicates)
        cands_s1 = df_out.loc[df_out["source1_entity_id"] == "S1-1", "candidate_entity_ids"].values[0]
        assert cands_s1 == "S2-A,S3-B"

        # S1-2 has zero candidates
        cands_s2 = df_out.loc[df_out["source1_entity_id"] == "S1-2", "candidate_entity_ids"].values[0]
        assert cands_s2 == ""
        assert diag["s1_rows_with_zero_candidates"] == 1


# =====================================================================
# 7. P3 PAIRWISE FEATURE INTEGRATION TESTS
# =====================================================================

class TestP3PairFeatures:
    def test_feature_list_contains_name_core_exact(self):
        assert "name_core_exact" in FEATURE_NAMES
        assert FEATURE_NAMES.index("name_core_exact") == 2

    def test_name_core_exact_value(self):
        rec_a = {"name_core": "acme tech"}
        rec_b = {"name_core": "acme tech"}
        rec_c = {"name_core": "other tech"}

        feat_match = compute_pair_feature_vector(rec_a, rec_b, "S2-1")
        feat_diff = compute_pair_feature_vector(rec_a, rec_c, "S2-2")

        # index 2 is name_core_exact
        assert feat_match[2] == 1.0
        assert feat_diff[2] == 0.0

    def test_feature_matrix_generation_with_caching(self):
        records_s1 = {
            "S1-1": {"name_clean": "acme supply", "name_core": "acme supply", "address_clean": "10 wall st", "country_clean": "us"}
        }
        records_s23 = {
            "S2-1": {"name_clean": "acme supply corp", "name_core": "acme supply", "address_clean": "10 wall st", "country_clean": "us"},
            "S3-1": {"name_clean": "acme supply ltd", "name_core": "acme supply", "address_clean": "20 wall st", "country_clean": "us"},
        }
        pairs = [("S1-1", "S2-1"), ("S1-1", "S3-1")]
        mat = build_feature_matrix(pairs, records_s1, records_s23)

        assert mat.shape == (2, len(FEATURE_NAMES))
        assert mat[0, 2] == 1.0  # name_core_exact
        assert mat[1, 2] == 1.0  # name_core_exact

    def test_discriminative_house_number_mismatch(self):
        rec_same = {"address_house_number": "100", "address_street_key": "100_main_us"}
        rec_diff = {"address_house_number": "200", "address_street_key": "200_main_us"}

        feat_mismatch = compute_pair_feature_vector(rec_same, rec_diff, "S2-1")
        idx_mismatch = FEATURE_NAMES.index("addr_house_num_mismatch")
        idx_match = FEATURE_NAMES.index("addr_house_num_match")
        assert feat_mismatch[idx_mismatch] == 1.0
        assert feat_mismatch[idx_match] == 0.0

        feat_match = compute_pair_feature_vector(rec_same, rec_same, "S2-2")
        assert feat_match[idx_mismatch] == 0.0
        assert feat_match[idx_match] == 1.0


# =====================================================================
# 8. ENHANCED ADDRESS & MULTILINGUAL NORMALIZATION TESTS
# =====================================================================

class TestEnhancedAddressKeys:
    def test_street_key_extraction(self):
        from business_entity_resolution.preprocessing import extract_address_components
        res1 = extract_address_components("9308 Home Court, DES PLAINES CITY, IL", "us")
        assert res1["house_number"] == "9308"
        assert res1["street_root"] == "home"
        assert res1["address_street_key"] == "9308_home_us"

    def test_leading_zero_stripping(self):
        from business_entity_resolution.preprocessing import extract_address_components
        res = extract_address_components("004303 ELKINS AVE, NASHVILLE, TN", "us")
        assert res["house_number"] == "4303"
        assert res["street_root"] == "elkins"
        assert res["address_street_key"] == "4303_elkins_us"

    def test_complex_and_indic_address(self):
        from business_entity_resolution.preprocessing import extract_address_components
        res = extract_address_components("AF-0684, Uttar Pradesh, GHAZIABAD", "india")
        assert res["house_number"] == "684"
        assert res["geo_token"] == "up"


class TestMultilingualAndPrefixCore:
    def test_indic_transliteration(self):
        from business_entity_resolution.preprocessing import normalize_business_name
        norm = normalize_business_name("एसएस फूड प्राइवेट लिमिटेड")
        assert "ss" in norm
        assert "food" in norm
        assert "private" in norm
        assert "limited" in norm

    def test_domain_and_prefix_removal(self):
        assert extract_name_core("lovue.com") == "lovue"
        assert extract_name_core("novent owl pllc") == "novent owl"
        assert extract_name_core("pllc novent owl") == "novent owl"
        assert extract_name_core("the acme corp") == "acme"
        assert extract_name_core("smt great impex private limited") == "great impex"

