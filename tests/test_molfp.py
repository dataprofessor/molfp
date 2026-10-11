"""
Comprehensive tests for all molfp features.

Run with: python -m pytest tests/test_molfp.py -v
Or:       python tests/test_molfp.py
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import molfp


# ── Test molecules ────────────────────────────────────────────────────────────

ETHANOL = "CCO"
BENZENE = "c1ccccc1"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
SALICYLIC = "OC(=O)c1ccccc1O"
ANTHRACENE = "c1ccc2c(c1)cc1ccc3ccccc3c1c2"

SMILES_SET = [ETHANOL, BENZENE, ASPIRIN, CAFFEINE, SALICYLIC, ANTHRACENE]
ACTIVITIES = [4.5, 5.0, 7.2, 6.8, 3.1, 5.5]


# =============================================================================
# 1. CORE FINGERPRINTS
# =============================================================================

class TestCoreFingerprints:

    def test_available_returns_26(self):
        fps = molfp.available()
        assert len(fps) == 26

    @pytest.mark.parametrize("fp_name,expected_len", [
        ("maccs", 166),
        ("ecfp4", 1024),
        ("pubchem", 881),
        ("substructure", 307),
        ("substructure_count", 307),
        ("klekota_roth", 4860),
        ("klekota_roth_count", 4860),
        ("estate", 79),
        ("atompairs2d", 780),
        ("atompairs2d_count", 780),
        ("rdk", 1024),
        ("rdk_ext", 1024),
        ("rdk_graphonly", 1024),
    ])
    def test_fingerprint_length(self, fp_name, expected_len):
        fn, n_bits, desc = molfp.get(fp_name)
        fp = fn(ASPIRIN)
        assert len(fp) == expected_len
        assert n_bits == expected_len

    @pytest.mark.parametrize("fp_name", [
        "maccs", "ecfp4", "pubchem", "substructure", "estate",
        "atompairs2d", "rdk", "rdk_ext", "rdk_graphonly",
    ])
    def test_binary_fp_values_are_0_or_1(self, fp_name):
        fn = molfp.get(fp_name)[0]
        fp = fn(ASPIRIN)
        assert all(v in (0, 1) for v in fp), f"{fp_name} has non-binary values"

    def test_count_fp_can_exceed_1(self):
        fp = molfp.substructure_count(ANTHRACENE)
        # Anthracene has multiple aromatic carbons, some counts should exceed 1
        assert any(v > 1 for v in fp), "Count FP should have values > 1 for complex molecules"

    def test_smiles_and_mol_give_same_result(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles(ASPIRIN)
        fp_smi = molfp.maccs(ASPIRIN)
        fp_mol = molfp.maccs(mol)
        assert fp_smi == fp_mol

    def test_invalid_smiles_raises(self):
        with pytest.raises(ValueError, match="Invalid SMILES"):
            molfp.maccs("NOT_A_SMILES")

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            molfp.maccs(12345)

    def test_unknown_fp_raises(self):
        with pytest.raises(KeyError, match="Unknown fingerprint"):
            molfp.get("nonexistent_fp")

    def test_different_molecules_give_different_fps(self):
        fp1 = molfp.ecfp4(ETHANOL)
        fp2 = molfp.ecfp4(CAFFEINE)
        assert fp1 != fp2

    def test_same_molecule_gives_same_fp(self):
        fp1 = molfp.ecfp4(ASPIRIN)
        fp2 = molfp.ecfp4(ASPIRIN)
        assert fp1 == fp2

    # ── Direct calls for every FP function ──────────────────────────────────

    def test_direct_atompairs2d(self):
        assert len(molfp.atompairs2d(ASPIRIN)) == 780

    def test_direct_atompairs2d_count(self):
        assert len(molfp.atompairs2d_count(ASPIRIN)) == 780

    def test_direct_estate(self):
        assert len(molfp.estate(ASPIRIN)) == 79

    def test_direct_klekota_roth(self):
        assert len(molfp.klekota_roth(ASPIRIN)) == 4860

    def test_direct_klekota_roth_count(self):
        assert len(molfp.klekota_roth_count(ASPIRIN)) == 4860

    def test_direct_rdk(self):
        assert len(molfp.rdk(ASPIRIN)) == 1024

    def test_direct_rdk_ext(self):
        assert len(molfp.rdk_ext(ASPIRIN)) == 1024

    def test_direct_rdk_graphonly(self):
        assert len(molfp.rdk_graphonly(ASPIRIN)) == 1024

    def test_direct_atompairs2d_batch(self):
        assert molfp.atompairs2d_batch(SMILES_SET[:3]).shape == (3, 780)

    def test_direct_atompairs2d_count_batch(self):
        assert molfp.atompairs2d_count_batch(SMILES_SET[:3]).shape == (3, 780)

    def test_direct_estate_batch(self):
        assert molfp.estate_batch(SMILES_SET[:3]).shape == (3, 79)

    def test_direct_klekota_roth_batch(self):
        assert molfp.klekota_roth_batch(SMILES_SET[:3]).shape == (3, 4860)

    def test_direct_klekota_roth_count_batch(self):
        assert molfp.klekota_roth_count_batch(SMILES_SET[:3]).shape == (3, 4860)

    def test_direct_rdk_batch(self):
        assert molfp.rdk_batch(SMILES_SET[:3]).shape == (3, 1024)

    def test_direct_rdk_ext_batch(self):
        assert molfp.rdk_ext_batch(SMILES_SET[:3]).shape == (3, 1024)

    def test_direct_rdk_graphonly_batch(self):
        assert molfp.rdk_graphonly_batch(SMILES_SET[:3]).shape == (3, 1024)

    def test_direct_substructure_batch(self):
        assert molfp.substructure_batch(SMILES_SET[:3]).shape == (3, 307)

    def test_direct_substructure_count_batch(self):
        assert molfp.substructure_count_batch(SMILES_SET[:3]).shape == (3, 307)

    @pytest.mark.parametrize("fp_name", ["maccs", "ecfp4", "pubchem", "estate", "rdk"])
    def test_at_least_one_bit_on(self, fp_name):
        fn = molfp.get(fp_name)[0]
        fp = fn(ASPIRIN)
        assert sum(fp) > 0, f"{fp_name} has no bits ON for aspirin"


# =============================================================================
# 2. BATCH COMPUTATION
# =============================================================================

class TestBatch:

    def test_batch_shape(self):
        fps = molfp.maccs_batch(SMILES_SET)
        assert isinstance(fps, np.ndarray)
        assert fps.shape == (6, 166)
        assert fps.dtype == np.int32

    def test_batch_matches_individual(self):
        batch = molfp.ecfp4_batch(SMILES_SET[:3])
        for i, smi in enumerate(SMILES_SET[:3]):
            individual = molfp.ecfp4(smi)
            np.testing.assert_array_equal(batch[i], individual)

    def test_pubchem_batch_shape(self):
        fps = molfp.pubchem_batch(SMILES_SET[:3])
        assert fps.shape == (3, 881)


# =============================================================================
# 3. EXPLAIN
# =============================================================================

class TestExplain:

    def test_explain_ecfp4_returns_dict(self):
        result = molfp.explain(ASPIRIN, "ecfp4")
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_explain_ecfp4_bits_are_atom_environments(self):
        result = molfp.explain(ASPIRIN, "ecfp4")
        for bit, envs in result.items():
            assert isinstance(bit, int)
            for env in envs:
                center, radius = env
                assert isinstance(center, int)
                assert isinstance(radius, int)
                assert radius >= 0

    def test_explain_substructure_returns_atom_matches(self):
        result = molfp.explain(ASPIRIN, "substructure")
        assert isinstance(result, dict)
        assert len(result) > 0
        for bit, matches in result.items():
            for match in matches:
                assert all(isinstance(a, int) for a in match)

    def test_explain_pubchem_has_section_info(self):
        result = molfp.explain(ASPIRIN, "pubchem")
        assert len(result) > 0
        sections_seen = set()
        for bit, info in result.items():
            assert "section" in info
            sections_seen.add(info["section"])
        assert "element_count" in sections_seen
        assert "smarts" in sections_seen

    def test_explain_rdk_returns_bond_paths(self):
        result = molfp.explain(ASPIRIN, "rdk")
        assert len(result) > 0
        for bit, paths in result.items():
            for path in paths:
                assert isinstance(path, list)

    def test_explain_maccs(self):
        result = molfp.explain(ASPIRIN, "maccs")
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_explain_estate(self):
        result = molfp.explain(ASPIRIN, "estate")
        assert isinstance(result, dict)

    def test_explain_klekota_roth(self):
        result = molfp.explain(ASPIRIN, "klekota_roth")
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_explain_atompairs2d(self):
        result = molfp.explain(ASPIRIN, "atompairs2d")
        assert isinstance(result, dict)

    def test_explain_unknown_fp_raises(self):
        with pytest.raises(KeyError):
            molfp.explain(ASPIRIN, "nonexistent")

    def test_explain_on_bits_match_fingerprint(self):
        fp = molfp.ecfp4(ASPIRIN)
        explained = molfp.explain(ASPIRIN, "ecfp4")
        on_bits_fp = {i for i, v in enumerate(fp) if v == 1}
        on_bits_explained = set(explained.keys())
        assert on_bits_explained == on_bits_fp


# =============================================================================
# 4. ANNOTATE
# =============================================================================

class TestAnnotate:

    def test_annotate_substructure_has_name(self):
        ann = molfp.annotate("substructure", 0)
        assert "name" in ann
        assert "Primary_carbon" in ann["name"]
        assert "smarts" in ann

    def test_annotate_pubchem_element_count(self):
        ann = molfp.annotate("pubchem", 9)
        assert "element_count" in ann.get("category", "")
        assert "C >= 2" in ann["description"]

    def test_annotate_pubchem_ring(self):
        ann = molfp.annotate("pubchem", 178)
        assert "ring" in ann.get("category", "")

    def test_annotate_maccs(self):
        ann = molfp.annotate("maccs", 0)
        assert "name" in ann
        assert "MACCS" in ann["name"]

    def test_annotate_estate(self):
        ann = molfp.annotate("estate", 0)
        assert "name" in ann
        assert ann["category"] == "atom_type"

    def test_annotate_kr(self):
        ann = molfp.annotate("klekota_roth", 0)
        assert "smarts" in ann
        assert ann["category"] == "bioactivity_substructure"

    def test_annotate_hashed_returns_empty(self):
        anns = molfp.get_annotations("ecfp4")
        assert anns == {}

    def test_annotate_unknown_bit(self):
        ann = molfp.annotate("substructure", 9999)
        assert ann["category"] == "unknown"

    def test_get_annotations_returns_all(self):
        anns = molfp.get_annotations("substructure")
        assert len(anns) == 307

    def test_get_annotations_pubchem_covers_881(self):
        anns = molfp.get_annotations("pubchem")
        assert len(anns) == 881


# =============================================================================
# 5. COLLISIONS
# =============================================================================

class TestCollisions:

    def test_detect_collisions_returns_dict(self):
        coll = molfp.detect_collisions(ASPIRIN, "ecfp4", 1024)
        assert isinstance(coll, dict)

    def test_small_fp_has_more_collisions(self):
        coll_small = molfp.detect_collisions(ANTHRACENE, "ecfp4", 64)
        coll_large = molfp.detect_collisions(ANTHRACENE, "ecfp4", 4096)
        assert len(coll_small) >= len(coll_large)

    def test_collision_values_have_multiple_hashes(self):
        coll = molfp.detect_collisions(ANTHRACENE, "ecfp4", 64)
        for bit, hashes in coll.items():
            assert len(hashes) > 1

    def test_detect_collisions_atompairs(self):
        coll = molfp.detect_collisions(ANTHRACENE, "atompairs2d", 128)
        assert isinstance(coll, dict)

    def test_detect_collisions_rdk(self):
        coll = molfp.detect_collisions(ASPIRIN, "rdk")
        assert isinstance(coll, dict)

    def test_detect_collisions_rejects_smarts_fp(self):
        with pytest.raises(ValueError, match="hashed FPs"):
            molfp.detect_collisions(ASPIRIN, "substructure")

    def test_collision_rate_returns_stats(self):
        stats = molfp.collision_rate(SMILES_SET[:3], "ecfp4", 1024)
        assert "molecules" in stats
        assert "collision_rate" in stats
        assert stats["molecules"] == 3
        assert 0 <= stats["collision_rate"] <= 1

    def test_collision_pairs_finds_identical(self):
        # Two very similar molecules with small FP should collide
        smiles = ["C", "CC", "CCC", "CCCC", "CCCCC"]
        pairs = molfp.collision_pairs(smiles, "ecfp4", n_bits=16)
        # With only 16 bits, some alkanes may collide
        assert isinstance(pairs, list)


# =============================================================================
# 6. SIMILARITY
# =============================================================================

class TestSimilarity:

    def test_similarity_matrix_shape(self):
        mat = molfp.similarity_matrix(SMILES_SET[:4], "ecfp4")
        assert mat.shape == (4, 4)
        assert mat.dtype == np.float32

    def test_similarity_matrix_diagonal_is_1(self):
        mat = molfp.similarity_matrix(SMILES_SET[:4], "ecfp4")
        np.testing.assert_allclose(np.diag(mat), 1.0, atol=0.01)

    def test_similarity_matrix_is_symmetric(self):
        mat = molfp.similarity_matrix(SMILES_SET[:4], "ecfp4")
        np.testing.assert_array_almost_equal(mat, mat.T)

    def test_similarity_values_in_range(self):
        mat = molfp.similarity_matrix(SMILES_SET[:4], "maccs")
        assert np.all(mat >= 0)
        assert np.all(mat <= 1)

    def test_similarity_profile_keys(self):
        prof = molfp.similarity_profile(SMILES_SET, "ecfp4")
        for key in ["mean", "std", "min", "max", "median", "q25", "q75",
                     "histogram", "n_molecules", "n_pairs"]:
            assert key in prof, f"Missing key: {key}"

    def test_similarity_profile_values_sensible(self):
        prof = molfp.similarity_profile(SMILES_SET, "maccs")
        assert 0 <= prof["mean"] <= 1
        assert prof["min"] <= prof["mean"] <= prof["max"]
        assert prof["n_molecules"] == len(SMILES_SET)
        assert prof["n_pairs"] == len(SMILES_SET) * (len(SMILES_SET) - 1) // 2

    def test_compare_fps_returns_list(self):
        result = molfp.compare_fps(SMILES_SET[:3], ["maccs", "ecfp4"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert all("fp_type" in r for r in result)

    def test_dice_metric(self):
        mat = molfp.similarity_matrix(SMILES_SET[:3], "ecfp4", metric="dice")
        assert mat.shape == (3, 3)
        assert np.all(mat >= 0)


# =============================================================================
# 7. ACTIVITY CLIFFS
# =============================================================================

class TestCliffs:

    def test_activity_cliffs_returns_list(self):
        cliffs = molfp.activity_cliffs(
            SMILES_SET, ACTIVITIES, "maccs",
            sim_threshold=0.2, activity_threshold=1.0
        )
        assert isinstance(cliffs, list)

    def test_cliff_structure(self):
        cliffs = molfp.activity_cliffs(
            SMILES_SET, ACTIVITIES, "maccs",
            sim_threshold=0.1, activity_threshold=0.5
        )
        if cliffs:
            c = cliffs[0]
            assert "idx_a" in c
            assert "idx_b" in c
            assert "smiles_a" in c
            assert "similarity" in c
            assert "delta_activity" in c

    def test_cliffs_sorted_by_delta(self):
        cliffs = molfp.activity_cliffs(
            SMILES_SET, ACTIVITIES, "maccs",
            sim_threshold=0.1, activity_threshold=0.5
        )
        deltas = [c["delta_activity"] for c in cliffs]
        assert deltas == sorted(deltas, reverse=True)

    def test_cliffs_respect_thresholds(self):
        sim_t, act_t = 0.3, 2.0
        cliffs = molfp.activity_cliffs(
            SMILES_SET, ACTIVITIES, "maccs",
            sim_threshold=sim_t, activity_threshold=act_t
        )
        for c in cliffs:
            assert c["similarity"] >= sim_t
            assert c["delta_activity"] >= act_t

    def test_high_threshold_gives_fewer_cliffs(self):
        cliffs_low = molfp.activity_cliffs(
            SMILES_SET, ACTIVITIES, "maccs", sim_threshold=0.1, activity_threshold=0.5)
        cliffs_high = molfp.activity_cliffs(
            SMILES_SET, ACTIVITIES, "maccs", sim_threshold=0.5, activity_threshold=2.0)
        assert len(cliffs_low) >= len(cliffs_high)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            molfp.activity_cliffs(SMILES_SET, [1.0, 2.0], "maccs")

    def test_cliff_summary(self):
        result = molfp.cliff_summary(
            SMILES_SET, ACTIVITIES, ["maccs", "ecfp4"],
            sim_threshold=0.1, activity_threshold=0.5
        )
        assert isinstance(result, list)
        assert len(result) == 2
        assert all("n_cliffs" in r for r in result)


# =============================================================================
# 8. FUSION
# =============================================================================

class TestFusion:

    def test_fuse_returns_expected_keys(self):
        result = molfp.fuse(
            SMILES_SET, ACTIVITIES,
            fp_types=["maccs", "estate"], n_bits=50
        )
        assert "fused_fps" in result
        assert "selected_bits" in result
        assert "source_map" in result
        assert result["n_bits"] == 50

    def test_fuse_output_shape(self):
        result = molfp.fuse(
            SMILES_SET, ACTIVITIES,
            fp_types=["maccs", "estate"], n_bits=30
        )
        assert result["fused_fps"].shape == (len(SMILES_SET), 30)

    def test_fuse_selected_bits_have_scores(self):
        result = molfp.fuse(
            SMILES_SET, ACTIVITIES,
            fp_types=["maccs", "estate"], n_bits=20
        )
        for fp_type, bit_idx, score in result["selected_bits"]:
            assert isinstance(fp_type, str)
            assert isinstance(bit_idx, int)
            assert isinstance(score, float)

    def test_fuse_source_map_matches(self):
        result = molfp.fuse(
            SMILES_SET, ACTIVITIES,
            fp_types=["maccs", "estate"], n_bits=20
        )
        assert len(result["source_map"]) == 20
        for col_idx, (fp_type, bit_idx) in result["source_map"].items():
            assert fp_type in ("maccs", "estate")

    def test_fuse_variance_method(self):
        result = molfp.fuse(
            SMILES_SET, ACTIVITIES,
            fp_types=["maccs", "estate"], n_bits=20, method="variance"
        )
        assert result["fused_fps"].shape[1] == 20

    def test_complementarity_returns_matrix(self):
        result = molfp.complementarity(SMILES_SET, ["maccs", "ecfp4"])
        assert "correlation_matrix" in result
        mat = result["correlation_matrix"]
        assert len(mat) == 2
        assert len(mat[0]) == 2


# =============================================================================
# 9. ISOMERS
# =============================================================================

class TestIsomers:

    def test_find_identical_fps_returns_groups(self):
        groups = molfp.find_identical_fps(SMILES_SET, "ecfp4")
        assert isinstance(groups, list)
        # All molecules in our test set are distinct, so expect no groups
        # (or possibly some with very small FP sizes)

    def test_find_identical_fps_with_duplicates(self):
        smiles = [ETHANOL, BENZENE, ETHANOL, CAFFEINE, ETHANOL]
        groups = molfp.find_identical_fps(smiles, "maccs")
        assert len(groups) == 1
        assert len(groups[0]) == 3  # 3 copies of ethanol

    def test_discriminative_power_returns_all_fps(self):
        result = molfp.discriminative_power(SMILES_SET, ["maccs", "ecfp4", "pubchem"])
        assert len(result) == 3
        for r in result:
            assert "fp_type" in r
            assert "unique_fps" in r
            assert "discrimination_ratio" in r
            assert 0 <= r["discrimination_ratio"] <= 1

    def test_discriminative_power_all_unique(self):
        result = molfp.discriminative_power(SMILES_SET, ["ecfp4"])
        r = result[0]
        assert r["unique_fps"] == r["total_molecules"]
        assert r["discrimination_ratio"] == 1.0

    def test_discriminative_power_with_duplicates(self):
        smiles = [ETHANOL, BENZENE, ETHANOL]
        result = molfp.discriminative_power(smiles, ["maccs"])
        r = result[0]
        assert r["total_molecules"] == 3
        assert r["unique_fps"] == 2
        assert r["duplicates"] == 1

    def test_discriminative_power_sorted_desc(self):
        result = molfp.discriminative_power(SMILES_SET)
        ratios = [r["discrimination_ratio"] for r in result]
        assert ratios == sorted(ratios, reverse=True)

    def test_isomer_sensitivity(self):
        # Butanol isomers
        pairs = [
            ("CCCO", "CC(C)O"),     # 1-propanol vs 2-propanol
            ("CCCCO", "CC(C)CO"),   # 1-butanol vs 2-methyl-1-propanol
        ]
        result = molfp.isomer_sensitivity(pairs, ["maccs", "ecfp4"])
        assert "maccs" in result
        assert "ecfp4" in result
        for fp_type, stats in result.items():
            assert "sensitivity" in stats
            assert 0 <= stats["sensitivity"] <= 1

    def test_isomer_sensitivity_ecfp_distinguishes_isomers(self):
        pairs = [("CCCO", "CC(C)O")]
        result = molfp.isomer_sensitivity(pairs, ["ecfp4"])
        # ECFP4 should distinguish propanol isomers
        assert result["ecfp4"]["distinguished"] == 1


# =============================================================================
# 10. CROSS-MODULE INTEGRATION
# =============================================================================

class TestIntegration:

    def test_explain_bits_match_annotate(self):
        """Bits returned by explain() should be annotatable."""
        explained = molfp.explain(ASPIRIN, "substructure")
        for bit_idx in explained:
            ann = molfp.annotate("substructure", bit_idx)
            assert ann["category"] != "unknown"

    def test_collision_bits_are_in_fingerprint(self):
        """Collided bits should actually be ON in the fingerprint."""
        fp = molfp.ecfp4(ANTHRACENE)
        coll = molfp.detect_collisions(ANTHRACENE, "ecfp4", 1024)
        for bit in coll:
            assert fp[bit] == 1

    def test_fusion_uses_real_fp_bits(self):
        """Fused FP values should come from actual fingerprint computations."""
        result = molfp.fuse(
            SMILES_SET, ACTIVITIES,
            fp_types=["maccs", "estate"], n_bits=10
        )
        fused = result["fused_fps"]
        assert fused.shape == (len(SMILES_SET), 10)
        # All values should be non-negative integers
        assert np.all(fused >= 0)


# =============================================================================
# 11. SKLEARN TRANSFORMERS
# =============================================================================

class TestSklearnTransformers:
    """Test scikit-learn compatible transformer classes."""

    # ── Basic transformer API ────────────────────────────────────────────────

    @pytest.mark.parametrize("cls,expected_bits", [
        (molfp.MACCSTransformer, 166),
        (molfp.PubChemTransformer, 881),
        (molfp.SubstructureTransformer, 307),
        (molfp.SubstructureCountTransformer, 307),
        (molfp.KlekotaRothTransformer, 4860),
        (molfp.KlekotaRothCountTransformer, 4860),
        (molfp.EStateTransformer, 79),
    ])
    def test_fixed_size_transformer_shape(self, cls, expected_bits):
        t = cls()
        X = t.fit_transform(SMILES_SET[:3])
        assert X.shape == (3, expected_bits)
        assert X.dtype == np.int32

    @pytest.mark.parametrize("cls,default_bits", [
        (molfp.ECFPTransformer, 1024),
        (molfp.AtomPairs2DTransformer, 780),
        (molfp.AtomPairs2DCountTransformer, 780),
        (molfp.RDKTransformer, 1024),
        (molfp.RDKExtTransformer, 1024),
        (molfp.RDKGraphOnlyTransformer, 1024),
    ])
    def test_variable_size_transformer_default(self, cls, default_bits):
        t = cls()
        X = t.fit_transform(SMILES_SET[:3])
        assert X.shape == (3, default_bits)

    def test_ecfp_custom_n_bits(self):
        t = molfp.ECFPTransformer(n_bits=512)
        X = t.fit_transform(SMILES_SET[:3])
        assert X.shape == (3, 512)

    def test_atompairs_custom_n_bits(self):
        t = molfp.AtomPairs2DTransformer(n_bits=2048)
        X = t.fit_transform(SMILES_SET[:3])
        assert X.shape == (3, 2048)

    def test_rdk_custom_n_bits(self):
        t = molfp.RDKTransformer(n_bits=2048)
        X = t.fit_transform(SMILES_SET[:3])
        assert X.shape == (3, 2048)

    # ── fit/transform/fit_transform consistency ──────────────────────────────

    def test_fit_returns_self(self):
        t = molfp.MACCSTransformer()
        result = t.fit(SMILES_SET)
        assert result is t

    def test_transform_matches_fit_transform(self):
        t = molfp.ECFPTransformer(n_bits=512)
        X1 = t.fit_transform(SMILES_SET[:4])
        t2 = molfp.ECFPTransformer(n_bits=512)
        t2.fit(SMILES_SET[:4])
        X2 = t2.transform(SMILES_SET[:4])
        np.testing.assert_array_equal(X1, X2)

    # ── get_params / set_params ──────────────────────────────────────────────

    def test_get_params_fixed(self):
        t = molfp.MACCSTransformer(n_jobs=2, sparse=True)
        params = t.get_params()
        assert params["n_jobs"] == 2
        assert params["sparse"] is True

    def test_get_params_variable(self):
        t = molfp.ECFPTransformer(n_bits=2048, n_jobs=-1)
        params = t.get_params()
        assert params["n_bits"] == 2048
        assert params["n_jobs"] == -1

    def test_set_params(self):
        t = molfp.ECFPTransformer(n_bits=1024)
        t.set_params(n_bits=512)
        assert t.n_bits == 512
        X = t.transform(SMILES_SET[:2])
        assert X.shape == (2, 512)

    # ── get_feature_names_out ────────────────────────────────────────────────

    def test_feature_names_maccs(self):
        t = molfp.MACCSTransformer()
        names = t.get_feature_names_out()
        assert len(names) == 166
        assert names[0] == "MACCS_0"
        assert names[-1] == "MACCS_165"

    def test_feature_names_ecfp(self):
        t = molfp.ECFPTransformer(n_bits=512)
        names = t.get_feature_names_out()
        assert len(names) == 512

    def test_feature_names_pubchem(self):
        names = molfp.PubChemTransformer().get_feature_names_out()
        assert len(names) == 881

    # ── Sparse output ────────────────────────────────────────────────────────

    def test_sparse_output_type(self):
        from scipy.sparse import issparse
        t = molfp.MACCSTransformer(sparse=True)
        X = t.fit_transform(SMILES_SET[:3])
        assert issparse(X)
        assert X.shape == (3, 166)

    def test_sparse_matches_dense(self):
        X_dense = molfp.MACCSTransformer().fit_transform(SMILES_SET[:4])
        X_sparse = molfp.MACCSTransformer(sparse=True).fit_transform(SMILES_SET[:4])
        np.testing.assert_array_equal(X_dense, X_sparse.toarray())

    def test_sparse_kr_memory_savings(self):
        X_dense = molfp.KlekotaRothTransformer().fit_transform(SMILES_SET[:3])
        X_sparse = molfp.KlekotaRothTransformer(sparse=True).fit_transform(SMILES_SET[:3])
        sparse_bytes = X_sparse.data.nbytes + X_sparse.indices.nbytes + X_sparse.indptr.nbytes
        assert sparse_bytes < X_dense.nbytes

    # ── Parallelism ──────────────────────────────────────────────────────────

    def test_parallel_matches_sequential(self):
        X_seq = molfp.MACCSTransformer(n_jobs=1).fit_transform(SMILES_SET)
        X_par = molfp.MACCSTransformer(n_jobs=2).fit_transform(SMILES_SET)
        np.testing.assert_array_equal(X_seq, X_par)

    def test_parallel_all_cpus(self):
        X = molfp.EStateTransformer(n_jobs=-1).fit_transform(SMILES_SET[:4])
        assert X.shape == (4, 79)

    # ── sklearn Pipeline integration ─────────────────────────────────────────

    def test_pipeline_fit_predict(self):
        from sklearn.pipeline import make_pipeline
        from sklearn.ensemble import RandomForestClassifier

        smiles = SMILES_SET
        y = [0, 0, 1, 1, 0, 1]

        pipe = make_pipeline(
            molfp.ECFPTransformer(n_bits=256),
            RandomForestClassifier(n_estimators=10, random_state=42),
        )
        pipe.fit(smiles, y)
        preds = pipe.predict(smiles)
        assert len(preds) == len(smiles)
        assert all(p in (0, 1) for p in preds)

    def test_pipeline_cross_val_score(self):
        from sklearn.pipeline import make_pipeline
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        smiles = SMILES_SET
        y = [0, 0, 1, 1, 0, 1]

        pipe = make_pipeline(
            molfp.MACCSTransformer(),
            RandomForestClassifier(n_estimators=10, random_state=42),
        )
        scores = cross_val_score(pipe, smiles, y, cv=2)
        assert len(scores) == 2
        assert all(0 <= s <= 1 for s in scores)

    def test_pipeline_feature_union(self):
        from sklearn.pipeline import make_union

        union = make_union(
            molfp.ECFPTransformer(n_bits=256),
            molfp.MACCSTransformer(),
            molfp.EStateTransformer(),
        )
        X = union.fit_transform(SMILES_SET[:3])
        assert X.shape == (3, 256 + 166 + 79)

    def test_pipeline_gridsearch(self):
        from sklearn.pipeline import make_pipeline
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import GridSearchCV

        pipe = make_pipeline(
            molfp.ECFPTransformer(),
            RandomForestRegressor(random_state=42),
        )
        param_grid = {
            "ecfptransformer__n_bits": [256, 512],
            "randomforestregressor__n_estimators": [10, 50],
        }
        gs = GridSearchCV(pipe, param_grid, cv=2, scoring="r2")
        gs.fit(SMILES_SET, ACTIVITIES)
        assert gs.best_params_ is not None
        assert "ecfptransformer__n_bits" in gs.best_params_

    def test_pipeline_feature_selection(self):
        from sklearn.pipeline import make_pipeline
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import SelectKBest, f_classif

        smiles = SMILES_SET
        y = [0, 0, 1, 1, 0, 1]

        pipe = make_pipeline(
            molfp.MACCSTransformer(),
            SelectKBest(f_classif, k=10),
            RandomForestClassifier(n_estimators=10, random_state=42),
        )
        pipe.fit(smiles, y)
        preds = pipe.predict(smiles)
        assert len(preds) == len(smiles)

    # ── get_transformer convenience ──────────────────────────────────────────

    def test_get_transformer_by_name(self):
        t = molfp.get_transformer("ecfp4", n_bits=2048)
        assert isinstance(t, molfp.ECFPTransformer)
        assert t.n_bits == 2048

    def test_get_transformer_all_names(self):
        for name in molfp.TRANSFORMERS:
            t = molfp.get_transformer(name)
            assert t is not None

    def test_get_transformer_unknown_raises(self):
        with pytest.raises(KeyError):
            molfp.get_transformer("nonexistent")

    def test_get_transformer_with_kwargs(self):
        t = molfp.get_transformer("maccs", n_jobs=4, sparse=True)
        params = t.get_params()
        assert params["n_jobs"] == 4
        assert params["sparse"] is True

    # ── compute_batch universal API ──────────────────────────────────────────

    def test_compute_batch_dense(self):
        X = molfp.compute_batch(SMILES_SET[:3], "maccs")
        assert X.shape == (3, 166)
        assert X.dtype == np.int32

    def test_compute_batch_sparse(self):
        from scipy.sparse import issparse
        X = molfp.compute_batch(SMILES_SET[:3], "ecfp4", sparse=True)
        assert issparse(X)
        assert X.shape == (3, 1024)

    def test_compute_batch_parallel(self):
        X_seq = molfp.compute_batch(SMILES_SET, "estate", n_jobs=1)
        X_par = molfp.compute_batch(SMILES_SET, "estate", n_jobs=2)
        np.testing.assert_array_equal(X_seq, X_par)

    def test_compute_batch_all_fp_types(self):
        for fp_type in molfp.available():
            X = molfp.compute_batch(SMILES_SET[:2], fp_type)
            expected_bits = molfp.FINGERPRINTS[fp_type][1]
            assert X.shape == (2, expected_bits), f"{fp_type}: {X.shape} != (2, {expected_bits})"

    # ── TRANSFORMERS registry ────────────────────────────────────────────────

    def test_transformers_registry_has_all_22(self):
        assert len(molfp.TRANSFORMERS) == 22

    def test_transformers_registry_matches_base_fingerprints(self):
        # Transformers cover the 22 non-graph fingerprints
        graph_fps = {"graph_standard", "graph_distance", "graph_line", "graph_fragment"}
        fp_names = set(molfp.FINGERPRINTS.keys()) - graph_fps
        assert set(molfp.TRANSFORMERS.keys()) == fp_names


# =============================================================================
# 12. STANDARDIZE
# =============================================================================

class TestStandardize:

    def test_standardize_smiles_clean(self):
        result = molfp.standardize_smiles("CC(=O)Oc1ccccc1C(=O)O")
        assert result is not None
        assert "." not in result

    def test_standardize_desalts(self):
        result = molfp.standardize_smiles("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]")
        assert "[Na+]" not in result
        assert "." not in result

    def test_standardize_canonical_tautomer(self):
        keto = molfp.standardize_smiles("O=C1CC=CC=C1")
        enol = molfp.standardize_smiles("Oc1ccccc1")
        assert keto == enol

    def test_get_inchikey_same_molecule(self):
        k1 = molfp.get_inchikey("OC(=O)c1ccccc1O")
        k2 = molfp.get_inchikey("c1ccc(O)c(C(=O)O)c1")
        assert k1 == k2
        assert k1 is not None

    def test_get_inchikey_different_molecules(self):
        k1 = molfp.get_inchikey("CCO")
        k2 = molfp.get_inchikey("c1ccccc1")
        assert k1 != k2

    def test_deduplicate(self):
        smiles = ["CCO", "c1ccccc1", "CCO", "OCC"]
        result = molfp.deduplicate(smiles)
        assert result["n_unique"] < result["n_input"]
        assert result["n_duplicates"] > 0

    def test_preprocess(self):
        smiles = ["CCO", "[Na+].OC(=O)c1ccccc1O", "INVALID", "CCO"]
        result = molfp.preprocess(smiles)
        assert result["n_input"] == 4
        assert result["n_after_standardization"] == 3
        assert result["n_after_deduplication"] <= 3
        assert len(result["clean_mols"]) == result["n_after_deduplication"]

    def test_standardize_invalid_returns_none(self):
        assert molfp.standardize_smiles("NOT_VALID") is None
        assert molfp.get_inchikey("NOT_VALID") is None


# =============================================================================
# 13. BIOACTIVITY
# =============================================================================

class TestBioactivity:

    def test_classify_default_thresholds(self):
        labels = molfp.classify([6.5, 5.5, 4.5, 7.0], metric="pchembl")
        assert labels[0] == "active"      # 6.5 > 6.0
        assert labels[1] == "intermediate" # 5.0 <= 5.5 <= 6.0
        assert labels[2] == "inactive"    # 4.5 < 5.0
        assert labels[3] == "active"      # 7.0 > 6.0

    def test_classify_custom_thresholds(self):
        labels = molfp.classify([8.0, 5.0], thresholds={"active": 7, "inactive": 6})
        assert labels[0] == "active"
        assert labels[1] == "inactive"

    def test_classify_ic50(self):
        labels = molfp.classify([50, 5000, 50000], metric="ic50_nm")
        assert labels[0] == "active"
        assert labels[1] == "intermediate"
        assert labels[2] == "inactive"

    def test_class_similarity_keys(self):
        labels = molfp.classify(ACTIVITIES, metric="pchembl")
        result = molfp.class_similarity(SMILES_SET, labels, "maccs")
        assert "intra_class" in result
        assert "inter_class" in result
        assert "separation" in result
        assert "class_counts" in result

    def test_enriched_bits_returns_per_class(self):
        labels = molfp.classify(ACTIVITIES, metric="pchembl")
        result = molfp.enriched_bits(SMILES_SET, labels, "maccs", min_enrichment=1.0)
        assert isinstance(result, dict)
        for cls in set(labels):
            assert cls in result


# =============================================================================
# 14. INTERPRET (SHAP + chemical meaning)
# =============================================================================

# Use a larger set for interpret tests (SHAP needs enough samples)
INTERPRET_SMILES = [
    "CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "OC(=O)c1ccccc1O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "c1ccc2c(c1)cc1ccc3ccccc3c1c2",
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "c1ccc(-c2ccccc2)cc1",
    "CC(=O)O", "CCCCCCCC", "c1ccncc1", "C1CCCCC1", "CC(C)(C)c1ccc(O)cc1",
]
INTERPRET_ACTIVITIES = [4.5, 5.0, 7.2, 6.8, 3.1, 5.5, 6.9, 7.1, 4.0, 5.3,
                        4.2, 4.8, 6.1, 5.8, 6.5]


class TestInterpret:

    @pytest.fixture(scope="class")
    def interpret_result(self):
        return molfp.interpret(
            INTERPRET_SMILES, INTERPRET_ACTIVITIES,
            fp_type="maccs", metric="pchembl", top_n=10
        )

    def test_interpret_returns_expected_keys(self, interpret_result):
        result = interpret_result
        for key in ["shap_values", "feature_importance", "class_drivers",
                     "model", "labels", "label_set", "accuracy"]:
            assert key in result

    def test_accuracy_is_reasonable(self, interpret_result):
        assert 0 <= interpret_result["accuracy"] <= 1

    def test_labels_match_input_length(self, interpret_result):
        assert len(interpret_result["labels"]) == len(INTERPRET_SMILES)

    def test_label_set_has_classes(self, interpret_result):
        assert len(interpret_result["label_set"]) >= 2

    def test_feature_importance_is_sorted(self, interpret_result):
        fi = interpret_result["feature_importance"]
        shap_vals = [f["mean_abs_shap"] for f in fi]
        assert shap_vals == sorted(shap_vals, reverse=True)

    def test_feature_importance_has_chemical_meaning(self, interpret_result):
        fi = interpret_result["feature_importance"]
        assert all("chemical_meaning" in f for f in fi)

    def test_feature_importance_has_smarts(self, interpret_result):
        fi = interpret_result["feature_importance"]
        # At least some should have SMARTS (MACCS bits do)
        has_smarts = sum(1 for f in fi if f.get("smarts"))
        assert has_smarts > 0

    def test_class_drivers_has_all_classes(self, interpret_result):
        for cls in interpret_result["label_set"]:
            assert cls in interpret_result["class_drivers"]

    def test_class_drivers_have_value_effect(self, interpret_result):
        for cls, drivers in interpret_result["class_drivers"].items():
            for d in drivers:
                assert d["value_effect"] in ("present", "absent")
                assert "shap_when_on" in d
                assert "shap_when_off" in d
                assert "freq_in_class" in d
                assert "freq_overall" in d
                assert 0 <= d["freq_in_class"] <= 1
                assert 0 <= d["freq_overall"] <= 1

    def test_class_drivers_direction(self, interpret_result):
        for cls, drivers in interpret_result["class_drivers"].items():
            for d in drivers:
                assert d["direction"] in ("promotes", "inhibits")
                if d["direction"] == "promotes":
                    assert d["mean_shap"] > 0
                else:
                    assert d["mean_shap"] < 0

    def test_interpret_molecule(self, interpret_result):
        mol_r = molfp.interpret_molecule(INTERPRET_SMILES[0], interpret_result)
        assert "prediction" in mol_r
        assert "prediction_proba" in mol_r
        assert "top_bits" in mol_r
        assert mol_r["prediction"] in interpret_result["label_set"]
        assert sum(mol_r["prediction_proba"].values()) == pytest.approx(1.0, abs=0.01)

    def test_interpret_molecule_top_bits_have_atoms(self, interpret_result):
        mol_r = molfp.interpret_molecule(INTERPRET_SMILES[2], interpret_result)
        # At least some top bits should have atom mappings
        has_atoms = sum(1 for b in mol_r["top_bits"] if b["atoms"])
        assert has_atoms > 0

    def test_compare_molecules(self, interpret_result):
        comp = molfp.compare_molecules(
            INTERPRET_SMILES[0], INTERPRET_SMILES[2], interpret_result
        )
        assert "tanimoto" in comp
        assert 0 <= comp["tanimoto"] <= 1
        assert "shared_bits" in comp
        assert "unique_to_a" in comp
        assert "unique_to_b" in comp
        assert "important_shared" in comp
        assert "important_unique_a" in comp
        assert "important_unique_b" in comp

    def test_compare_molecules_same_molecule(self, interpret_result):
        comp = molfp.compare_molecules(
            INTERPRET_SMILES[0], INTERPRET_SMILES[0], interpret_result
        )
        assert comp["tanimoto"] == pytest.approx(1.0)
        assert comp["unique_to_a"] == 0
        assert comp["unique_to_b"] == 0


# =============================================================================
# 15. BIOACTIVITY (deeper coverage)
# =============================================================================

class TestBioactivityDeep:

    def test_classify_nan_handling(self):
        labels = molfp.classify([6.5, float("nan"), 4.5])
        assert labels[1] == "unassigned"

    def test_classify_boundary_values(self):
        # Default pchembl: active > 6, inactive < 5
        labels = molfp.classify([6.0, 5.0], metric="pchembl")
        assert labels[0] == "intermediate"  # 6.0 is not > 6.0
        assert labels[1] == "intermediate"  # 5.0 is not < 5.0

    def test_classify_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            molfp.classify([5.0], metric="unknown_metric")

    def test_class_similarity_separation_positive(self):
        # Active molecules should be more similar to each other than to inactives
        smiles = INTERPRET_SMILES
        labels = molfp.classify(INTERPRET_ACTIVITIES, metric="pchembl")
        result = molfp.class_similarity(smiles, labels, "maccs")
        # Separation can be positive or negative — just check it's computed
        assert all(isinstance(v, float) for v in result["separation"].values())

    def test_class_similarity_length_mismatch(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            molfp.class_similarity(["CCO", "c1ccccc1"], ["active"], "maccs")

    def test_enriched_bits_enrichment_ratio(self):
        labels = molfp.classify(INTERPRET_ACTIVITIES, metric="pchembl")
        result = molfp.enriched_bits(INTERPRET_SMILES, labels, "maccs", min_enrichment=1.0)
        for cls, bits in result.items():
            for bit, cf, of, ratio in bits:
                assert ratio >= 1.0
                assert cf > 0

    def test_depleted_bits(self):
        labels = molfp.classify(INTERPRET_ACTIVITIES, metric="pchembl")
        result = molfp.depleted_bits(INTERPRET_SMILES, labels, "maccs", max_depletion=0.5)
        assert isinstance(result, dict)
        for cls, bits in result.items():
            for bit, cf, of, ratio in bits:
                assert ratio <= 0.5

    def test_bioactivity_profile(self):
        result = molfp.bioactivity_profile(
            INTERPRET_SMILES, INTERPRET_ACTIVITIES,
            metric="pchembl", fp_types=["maccs", "estate"]
        )
        assert "labels" in result
        assert "class_counts" in result
        assert "fp_results" in result
        assert "best_separator" in result
        assert result["best_separator"] in ("maccs", "estate")


# =============================================================================
# 16. STANDARDIZE (deeper coverage)
# =============================================================================

class TestStandardizeDeep:

    def test_desalt_sodium_salt(self):
        from rdkit import Chem
        mol = molfp.desalt("[Na+].OC(=O)c1ccccc1O")
        assert mol is not None
        smi = Chem.MolToSmiles(mol)
        assert "Na" not in smi

    def test_neutralize_charged(self):
        from rdkit import Chem
        mol = molfp.neutralize("[O-]c1ccccc1")
        smi = Chem.MolToSmiles(mol)
        assert "[O-]" not in smi

    def test_canonicalize_tautomer(self):
        from rdkit import Chem
        mol = molfp.canonicalize_tautomer("O=C1CC=CC=C1")
        smi = Chem.MolToSmiles(mol)
        assert smi == Chem.MolToSmiles(molfp.canonicalize_tautomer("Oc1ccccc1"))

    def test_standardize_mol_returns_mol(self):
        from rdkit import Chem
        mol = molfp.standardize_mol("CCO")
        assert isinstance(mol, Chem.rdchem.Mol)

    def test_standardize_mol_invalid_returns_none(self):
        assert molfp.standardize_mol("INVALID") is None

    def test_standardize_batch_tracks_failures(self):
        results, failures = molfp.standardize_batch(
            ["CCO", "INVALID", "c1ccccc1", "ALSO_BAD"], return_failures=True
        )
        assert len(results) == 4
        assert results[0] is not None
        assert results[1] is None
        assert len(failures) == 2
        assert failures[0] == (1, "INVALID")

    def test_deduplicate_inchikey_method(self):
        result = molfp.deduplicate(
            ["OC(=O)c1ccccc1O", "c1ccc(O)c(C(=O)O)c1", "CCO"],
            method="inchikey"
        )
        assert result["n_unique"] == 2  # two salicylic acid variants collapse

    def test_deduplicate_canonical_method(self):
        result = molfp.deduplicate(["CCO", "OCC", "c1ccccc1"], method="canonical")
        assert result["n_unique"] == 2  # CCO and OCC are the same

    def test_preprocess_full_pipeline(self):
        smiles = [
            "CC(=O)Oc1ccccc1C(=O)O",
            "CC(=O)Oc1ccccc1C(=O)[O-].[Na+]",  # salt form of aspirin
            "INVALID",
            "CCO",
        ]
        result = molfp.preprocess(smiles)
        assert result["n_input"] == 4
        assert result["n_after_standardization"] == 3
        # Aspirin and its salt should deduplicate
        assert result["n_after_deduplication"] <= 3
        assert len(result["clean_mols"]) == result["n_after_deduplication"]


# =============================================================================
# 17. DISTANCE FUNCTIONS
# =============================================================================

class TestDistance:

    def test_tanimoto_identical(self):
        a = np.array([1, 0, 1, 1, 0])
        assert molfp.tanimoto(a, a) == pytest.approx(1.0)

    def test_tanimoto_disjoint(self):
        a = np.array([1, 1, 0, 0])
        b = np.array([0, 0, 1, 1])
        assert molfp.tanimoto(a, b) == pytest.approx(0.0)

    def test_tanimoto_partial(self):
        a = np.array([1, 1, 1, 0])
        b = np.array([1, 1, 0, 0])
        # intersection=2, union=3 → 2/3
        assert molfp.tanimoto(a, b) == pytest.approx(2 / 3)

    def test_tanimoto_zeros(self):
        a = np.array([0, 0, 0])
        b = np.array([0, 0, 0])
        assert molfp.tanimoto(a, b) == 0.0

    def test_dice_identical(self):
        a = np.array([1, 0, 1, 1])
        assert molfp.dice(a, a) == pytest.approx(1.0)

    def test_dice_partial(self):
        a = np.array([1, 1, 1, 0])
        b = np.array([1, 1, 0, 0])
        # 2*dot / (norm_a^2 + norm_b^2) = 2*2 / (3+2) = 0.8
        assert molfp.dice(a, b) == pytest.approx(0.8)

    def test_cosine_identical(self):
        a = np.array([1, 0, 1, 1])
        assert molfp.cosine(a, a) == pytest.approx(1.0)

    def test_cosine_orthogonal(self):
        a = np.array([1, 0, 0, 0])
        b = np.array([0, 1, 0, 0])
        assert molfp.cosine(a, b) == pytest.approx(0.0)

    def test_sokal_michener(self):
        a = np.array([1, 1, 0, 0])
        b = np.array([1, 0, 0, 1])
        # matches: positions 0 and 2 → 2/4
        assert molfp.sokal_michener(a, b) == pytest.approx(0.5)

    def test_russel_rao(self):
        a = np.array([1, 1, 0, 0])
        b = np.array([1, 0, 1, 0])
        # both ON: position 0 only → 1/4
        assert molfp.russel_rao(a, b) == pytest.approx(0.25)

    def test_distance_is_one_minus_similarity(self):
        a = np.array([1, 1, 1, 0])
        b = np.array([1, 1, 0, 0])
        assert molfp.distance(a, b, "tanimoto") == pytest.approx(1.0 - molfp.tanimoto(a, b))

    def test_pairwise_shape(self):
        X = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
        mat = molfp.pairwise(X, "tanimoto")
        assert mat.shape == (3, 3)
        assert mat.dtype == np.float32

    def test_pairwise_diagonal_is_one(self):
        X = np.array([[1, 0, 1], [0, 1, 1]])
        mat = molfp.pairwise(X, "tanimoto")
        np.testing.assert_allclose(np.diag(mat), 1.0, atol=1e-5)

    def test_pairwise_symmetric(self):
        X = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
        mat = molfp.pairwise(X, "tanimoto")
        np.testing.assert_array_almost_equal(mat, mat.T)

    def test_pairwise_tanimoto_matches_single(self):
        X = np.array([[1, 1, 1, 0], [1, 1, 0, 0]])
        mat = molfp.pairwise(X, "tanimoto")
        expected = molfp.tanimoto(X[0], X[1])
        assert mat[0, 1] == pytest.approx(expected, abs=1e-5)

    def test_pairwise_dice(self):
        X = np.array([[1, 0, 1], [0, 1, 1]])
        mat = molfp.pairwise(X, "dice")
        assert mat.shape == (2, 2)
        assert 0 <= mat[0, 1] <= 1

    def test_pairwise_cosine(self):
        X = np.array([[1, 0, 1], [0, 1, 1]])
        mat = molfp.pairwise(X, "cosine")
        assert mat.shape == (2, 2)

    def test_pairwise_sokal_michener(self):
        X = np.array([[1, 0, 1], [0, 1, 1]])
        mat = molfp.pairwise(X, "sokal_michener")
        assert mat.shape == (2, 2)

    def test_pairwise_distance(self):
        X = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
        dist = molfp.pairwise_distance(X, "tanimoto")
        sim = molfp.pairwise(X, "tanimoto")
        np.testing.assert_array_almost_equal(dist, 1.0 - sim)

    def test_available_metrics(self):
        metrics = molfp.available_metrics()
        assert "tanimoto" in metrics
        assert "dice" in metrics
        assert "cosine" in metrics
        assert len(metrics) >= 5

    def test_unknown_metric_raises(self):
        a = np.array([1, 0])
        with pytest.raises(ValueError, match="Unknown metric"):
            molfp.distance(a, a, "nonexistent")

    def test_pairwise_on_real_fps(self):
        fps = molfp.maccs_batch(SMILES_SET)
        mat = molfp.pairwise(fps, "tanimoto")
        assert mat.shape == (len(SMILES_SET), len(SMILES_SET))
        assert np.all(mat >= 0)
        assert np.all(mat <= 1)

    def test_count_fp_tanimoto(self):
        a = np.array([0, 3, 2, 0, 1])
        b = np.array([1, 2, 0, 0, 1])
        sim = molfp.tanimoto(a, b)
        assert 0 <= sim <= 1


# =============================================================================
# 18. 3D FINGERPRINTS
# =============================================================================

class Test3DFingerprints:

    @pytest.mark.parametrize("fp_name,expected_len", [
        ("atompairs3d", 780),
        ("atompairs3d_count", 780),
        ("usr", 12),
        ("usrcat", 60),
        ("autocorr3d", 80),
        ("rdf", 210),
        ("morse", 224),
        ("whim", 114),
        ("getaway", 273),
    ])
    def test_3d_fp_length(self, fp_name, expected_len):
        fn, n_bits, _ = molfp.get(fp_name)
        fp = fn(ASPIRIN)
        assert len(fp) == expected_len

    def test_usr_returns_floats(self):
        fp = molfp.usr(ASPIRIN)
        assert isinstance(fp, list)
        assert any(v != 0 and v != 1 for v in fp)

    def test_usrcat_returns_floats(self):
        fp = molfp.usrcat(ASPIRIN)
        assert any(v != 0 and v != 1 for v in fp)

    def test_rdf_returns_floats(self):
        fp = molfp.rdf(ASPIRIN)
        assert any(v != 0 and v != 1 for v in fp)

    def test_atompairs3d_binary(self):
        fp = molfp.atompairs3d(ASPIRIN)
        assert all(v in (0, 1) for v in fp)

    def test_atompairs3d_count_can_exceed_1(self):
        fp = molfp.atompairs3d_count(ANTHRACENE)
        assert any(v > 1 for v in fp)

    def test_3d_differs_from_2d(self):
        fp2d = molfp.atompairs2d(ASPIRIN)
        fp3d = molfp.atompairs3d(ASPIRIN)
        assert fp2d != fp3d

    @pytest.mark.parametrize("fp_name,expected_shape", [
        ("usr", (3, 12)),
        ("usrcat", (3, 60)),
        ("autocorr3d", (3, 80)),
        ("rdf", (3, 210)),
        ("morse", (3, 224)),
        ("whim", (3, 114)),
        ("getaway", (3, 273)),
        ("atompairs3d", (3, 780)),
        ("atompairs3d_count", (3, 780)),
    ])
    def test_3d_batch_shape(self, fp_name, expected_shape):
        fps = molfp.compute_batch(SMILES_SET[:3], fp_name)
        assert fps.shape == expected_shape

    def test_3d_batch_dtype_float_for_descriptors(self):
        fps = molfp.usr_batch(SMILES_SET[:3])
        assert fps.dtype == np.float32

    def test_3d_batch_dtype_int_for_atompairs(self):
        fps = molfp.atompairs3d_batch(SMILES_SET[:3])
        assert fps.dtype == np.int32

    def test_batch_no_nan(self):
        for fp_name in ["usr", "usrcat", "autocorr3d", "rdf", "morse", "whim", "getaway"]:
            fps = molfp.compute_batch(SMILES_SET[:3], fp_name)
            assert not np.any(np.isnan(fps)), f"{fp_name} contains NaN"

    def test_3d_deterministic(self):
        fp1 = molfp.usr(ASPIRIN)
        fp2 = molfp.usr(ASPIRIN)
        assert fp1 == fp2

    def test_3d_different_molecules(self):
        fp1 = molfp.usr(ETHANOL)
        fp2 = molfp.usr(ANTHRACENE)
        assert fp1 != fp2

    def test_3d_list_input(self):
        fps = molfp.usrcat(SMILES_SET[:3])
        assert isinstance(fps, np.ndarray)
        assert fps.shape == (3, 60)

    def test_3d_dataframe_input(self):
        import pandas as pd
        df = pd.DataFrame({"SMILES": SMILES_SET[:3]})
        fps = molfp.rdf(df)
        assert fps.shape == (3, 210)

    def test_3d_sparse(self):
        from scipy.sparse import issparse
        fps = molfp.morse(SMILES_SET[:3], sparse=True)
        assert issparse(fps)
        assert fps.shape == (3, 224)

    def test_3d_transformers_exist(self):
        for name in ["atompairs3d", "atompairs3d_count", "usr", "usrcat",
                      "autocorr3d", "rdf", "morse", "whim", "getaway"]:
            assert name in molfp.TRANSFORMERS

    def test_3d_transformer_works(self):
        t = molfp.get_transformer("usrcat")
        X = t.fit_transform(SMILES_SET[:3])
        assert X.shape == (3, 60)

    def test_conformer_from_smiles(self):
        fp = molfp.usr("C")  # methane — simplest molecule
        assert len(fp) == 12
        assert any(v != 0 for v in fp)


# =============================================================================
# 19. Visualization (viz module)
# =============================================================================

class TestVisualization:
    """Tests for draw() and view3d()."""

    def test_draw_single_returns_image(self):
        img = molfp.draw("CCO")
        from PIL import Image
        assert isinstance(img, Image.Image)

    def test_draw_single_custom_size(self):
        img = molfp.draw("CCO", size=(400, 400))
        assert img.size[0] == 400
        assert img.size[1] == 400

    def test_draw_batch_returns_image(self):
        img = molfp.draw(["CCO", "c1ccccc1", "CC(=O)O"])
        from PIL import Image
        assert isinstance(img, Image.Image)

    def test_draw_batch_with_legends(self):
        img = molfp.draw(
            ["CCO", "c1ccccc1"],
            legends=["ethanol", "benzene"]
        )
        from PIL import Image
        assert isinstance(img, Image.Image)

    def test_draw_with_highlight_bits(self):
        img = molfp.draw("c1ccccc1", highlight_bits=[0, 1], fp_type="maccs")
        from PIL import Image
        assert isinstance(img, Image.Image)

    def test_draw_rdkit_mol_input(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CCO")
        img = molfp.draw(mol)
        from PIL import Image
        assert isinstance(img, Image.Image)

    def test_view3d_returns_html_in_script_mode(self):
        try:
            import py3Dmol
        except ImportError:
            pytest.skip("py3Dmol not installed")
        result = molfp.view3d("CCO")
        assert isinstance(result, str)
        assert "3Dmol" in result or "viewer" in result.lower() or "<div" in result

    def test_view3d_different_styles(self):
        try:
            import py3Dmol
        except ImportError:
            pytest.skip("py3Dmol not installed")
        for style in ["stick", "sphere", "line", "ball_and_stick"]:
            result = molfp.view3d("CCO", style=style)
            assert isinstance(result, str)

    def test_view3d_missing_py3dmol_error(self):
        """view3d raises ImportError with install instructions if py3Dmol missing."""
        try:
            import py3Dmol
            # py3Dmol is available — just verify view3d works
            result = molfp.view3d("CCO")
            assert result is not None
        except ModuleNotFoundError:
            # py3Dmol not installed — verify our error message
            with pytest.raises(ImportError, match="pip install py3Dmol"):
                molfp.view3d("CCO")


# =============================================================================
# 20. Graph representations for deep learning
# =============================================================================

class TestGraph:
    """Tests for mol_graph and related functions."""

    def test_single_molecule_keys(self):
        g = molfp.mol_graph("CCO")
        assert "atom_features" in g
        assert "edge_index" in g
        assert "edge_features" in g
        assert "num_atoms" in g
        assert "num_edges" in g
        assert "smiles" in g

    def test_atom_features_shape(self):
        g = molfp.mol_graph("CCO")  # ethanol: 3 heavy atoms
        assert g["atom_features"].shape == (3, molfp.ATOM_FEATURE_DIM)
        assert g["atom_features"].dtype == np.float32

    def test_edge_index_shape(self):
        g = molfp.mol_graph("CCO")  # 2 bonds → 4 edges (undirected)
        assert g["edge_index"].shape[0] == 2
        assert g["edge_index"].shape[1] == 4  # 2 bonds * 2 directions
        assert g["edge_index"].dtype == np.int64

    def test_edge_features_shape(self):
        g = molfp.mol_graph("CCO")
        n_edges = g["edge_index"].shape[1]
        assert g["edge_features"].shape == (n_edges, molfp.EDGE_FEATURE_DIM)
        assert g["edge_features"].dtype == np.float32

    def test_undirected_symmetry(self):
        """Edge index should be symmetric (each bond in both directions)."""
        g = molfp.mol_graph("c1ccccc1")
        edge_set = set()
        for k in range(g["edge_index"].shape[1]):
            i, j = int(g["edge_index"][0, k]), int(g["edge_index"][1, k])
            edge_set.add((i, j))
        # Every (i,j) should have a matching (j,i)
        for i, j in list(edge_set):
            assert (j, i) in edge_set

    def test_benzene_atoms_and_edges(self):
        g = molfp.mol_graph("c1ccccc1")
        assert g["num_atoms"] == 6
        assert g["num_edges"] == 12  # 6 bonds * 2 directions

    def test_single_atom(self):
        """Single atom molecule should have no edges."""
        g = molfp.mol_graph("[Na+]")
        assert g["num_atoms"] == 1
        assert g["num_edges"] == 0
        assert g["edge_index"].shape == (2, 0)
        assert g["edge_features"].shape == (0, molfp.EDGE_FEATURE_DIM)

    def test_aromatic_features(self):
        """Benzene atoms should be marked aromatic."""
        g = molfp.mol_graph("c1ccccc1")
        # is_aromatic is the 20th feature (index 19)
        aromatic_col = 19
        assert all(g["atom_features"][:, aromatic_col] == 1)

    def test_batch_input(self):
        graphs = molfp.mol_graph(["CCO", "c1ccccc1", "CC(=O)O"])
        assert isinstance(graphs, list)
        assert len(graphs) == 3
        for g in graphs:
            assert "atom_features" in g

    def test_rdkit_mol_input(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CCO")
        g = molfp.mol_graph(mol)
        assert g["num_atoms"] == 3

    def test_atom_feature_names(self):
        names = molfp.atom_feature_names()
        assert len(names) == molfp.ATOM_FEATURE_DIM
        assert "atom_type_C" in names
        assert "is_aromatic" in names

    def test_edge_feature_names(self):
        names = molfp.edge_feature_names()
        assert len(names) == molfp.EDGE_FEATURE_DIM
        assert "bond_type_single" in names
        assert "is_conjugated" in names

    def test_atom_type_one_hot(self):
        """Carbon atom should have atom_type_C = 1."""
        g = molfp.mol_graph("C")  # methane
        names = molfp.atom_feature_names()
        c_idx = names.index("atom_type_C")
        assert g["atom_features"][0, c_idx] == 1.0

    def test_nitrogen_atom_type(self):
        g = molfp.mol_graph("N")  # ammonia
        names = molfp.atom_feature_names()
        n_idx = names.index("atom_type_N")
        assert g["atom_features"][0, n_idx] == 1.0

    def test_to_pyg_import_error(self):
        """to_pyg should raise ImportError with install instructions."""
        try:
            import torch_geometric
            g = molfp.mol_graph("CCO")
            data = molfp.to_pyg(g)
            assert data.x.shape[0] == 3
        except ModuleNotFoundError:
            g = molfp.mol_graph("CCO")
            with pytest.raises(ImportError, match="pip install"):
                molfp.to_pyg(g)

    def test_to_dgl_import_error(self):
        """to_dgl should raise ImportError with install instructions."""
        try:
            import dgl
            g = molfp.mol_graph("CCO")
            dgl_g = molfp.to_dgl(g)
            assert dgl_g.num_nodes() == 3
        except ModuleNotFoundError:
            g = molfp.mol_graph("CCO")
            with pytest.raises(ImportError, match="pip install"):
                molfp.to_dgl(g)

    def test_feature_dims_constants(self):
        assert molfp.ATOM_FEATURE_DIM == 20
        assert molfp.EDGE_FEATURE_DIM == 12

    # ── Enriched graph features ──────────────────────────────────────────

    def test_enriched_atom_features_shape(self):
        g = molfp.mol_graph("CCO", enriched=True)
        assert g["atom_features"].shape == (3, molfp.ENRICHED_ATOM_FEATURE_DIM)

    def test_enriched_has_more_features(self):
        g_base = molfp.mol_graph("c1ccccc1")
        g_enr = molfp.mol_graph("c1ccccc1", enriched=True)
        assert g_enr["atom_features"].shape[1] > g_base["atom_features"].shape[1]

    def test_enriched_base_features_preserved(self):
        """First 20 columns of enriched should match base features."""
        g_base = molfp.mol_graph("CCO")
        g_enr = molfp.mol_graph("CCO", enriched=True)
        np.testing.assert_array_equal(
            g_base["atom_features"],
            g_enr["atom_features"][:, :molfp.ATOM_FEATURE_DIM]
        )

    def test_enriched_morgan_bits_set(self):
        """Enriched features should have some Morgan bits active."""
        g = molfp.mol_graph("c1ccccc1", enriched=True)
        morgan_start = molfp.ATOM_FEATURE_DIM
        morgan_end = morgan_start + 3 * 128
        morgan_feats = g["atom_features"][:, morgan_start:morgan_end]
        assert morgan_feats.sum() > 0

    def test_graph_vectors_enriched(self):
        X = molfp.graph_vectors(["CCO", "c1ccccc1"], enriched=True)
        assert X.shape[0] == 2
        assert X.shape[1] > 298  # enriched should be much larger

    def test_graph_vectors_base_shape(self):
        X = molfp.graph_vectors(["CCO", "c1ccccc1"])
        assert X.shape == (2, 298)

    def test_graph_transformer_enriched(self):
        t = molfp.GraphTransformer(enriched=True)
        X = t.fit_transform(["CCO", "c1ccccc1", "CC(=O)O"])
        assert X.shape[0] == 3
        assert X.shape[1] > 298

    def test_graph_transformer_select_k_removed(self):
        """GraphTransformer no longer has select_k — use select_features() instead."""
        t = molfp.GraphTransformer(enriched=True)
        assert "select_k" not in t.get_params()

    def test_graph_transformer_get_params(self):
        t = molfp.GraphTransformer(enriched=True)
        params = t.get_params()
        assert params["enriched"] is True

    # ── Distance graph ───────────────────────────────────────────────────

    def test_distance_graph_complete(self):
        g = molfp.mol_graph("CCO", type="distance")
        n = g["num_atoms"]  # includes explicit Hs after conformer generation
        assert g["num_edges"] == n * (n - 1)  # complete graph
        assert g.get("type") == "distance"

    def test_distance_edge_features_dim(self):
        g = molfp.mol_graph("CCO", type="distance")
        assert g["edge_features"].shape[1] == 9

    def test_distance_positive(self):
        g = molfp.mol_graph("CCO", type="distance")
        assert all(g["edge_features"][:, 0] > 0)

    def test_distance_bonded_flag(self):
        g = molfp.mol_graph("CCO", type="distance")
        is_bonded = g["edge_features"][:, 2]
        # Should have some bonded and some non-bonded edges
        assert sum(is_bonded) > 0
        assert sum(is_bonded) < g["num_edges"]

    def test_distance_benzene(self):
        g = molfp.mol_graph("c1ccccc1", type="distance")
        n = g["num_atoms"]
        assert g["num_edges"] == n * (n - 1)

    def test_distance_batch(self):
        graphs = molfp.mol_graph(["CCO", "c1ccccc1"], type="distance")
        assert len(graphs) == 2

    def test_distance_graph_vectors(self):
        X = molfp.graph_vectors(["CCO", "c1ccccc1"], type="distance")
        assert X.shape[0] == 2

    # ── Line graph ───────────────────────────────────────────────────────

    def test_line_graph_nodes_are_bonds(self):
        g = molfp.mol_graph("CCO", type="line")
        assert g["num_atoms"] == 2
        assert g.get("type") == "line"

    def test_line_node_dim(self):
        g = molfp.mol_graph("CCO", type="line")
        assert g["atom_features"].shape[1] == 52  # 20+20+12

    def test_line_edge_dim(self):
        g = molfp.mol_graph("CCO", type="line")
        if g["num_edges"] > 0:
            assert g["edge_features"].shape[1] == 20

    def test_line_connectivity(self):
        g = molfp.mol_graph("CCO", type="line")
        assert g["num_edges"] == 2  # 2 adjacent bonds

    def test_line_benzene(self):
        g = molfp.mol_graph("c1ccccc1", type="line")
        assert g["num_atoms"] == 6

    def test_line_single_bond(self):
        g = molfp.mol_graph("CO", type="line")
        assert g["num_atoms"] == 1
        assert g["num_edges"] == 0

    def test_line_batch(self):
        graphs = molfp.mol_graph(["CCO", "c1ccccc1"], type="line")
        assert len(graphs) == 2

    def test_line_graph_vectors(self):
        X = molfp.graph_vectors(["CCO", "c1ccccc1"], type="line")
        assert X.shape[0] == 2

    # ── Fragment graph ───────────────────────────────────────────────────

    def test_fragment_aspirin(self):
        g = molfp.mol_graph("CC(=O)Oc1ccccc1C(=O)O", type="fragment")
        assert g["num_atoms"] > 0
        assert g.get("type") == "fragment"

    def test_fragment_node_dim(self):
        g = molfp.mol_graph("CC(=O)Oc1ccccc1C(=O)O", type="fragment")
        assert g["atom_features"].shape[1] == 41  # 20+1+20

    def test_fragment_edge_dim(self):
        g = molfp.mol_graph("CC(=O)Oc1ccccc1C(=O)O", type="fragment")
        if g["num_edges"] > 0:
            assert g["edge_features"].shape[1] == 2

    def test_fragment_alkane_fallback(self):
        g = molfp.mol_graph("CCCCCC", type="fragment")
        assert g["num_atoms"] >= 1

    def test_fragment_batch(self):
        graphs = molfp.mol_graph(["CC(=O)O", "c1ccccc1"], type="fragment")
        assert len(graphs) == 2

    def test_fragment_graph_vectors(self):
        X = molfp.graph_vectors(["CC(=O)O", "c1ccccc1"], type="fragment")
        assert X.shape[0] == 2

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown type"):
            molfp.mol_graph("CCO", type="invalid")


# =============================================================================
# 21. Standalone feature selection
# =============================================================================

class TestSelection:
    """Tests for select_features() and remove_low_variance()."""

    def _make_data(self):
        smiles = SMILES_SET
        X = np.array([molfp.FINGERPRINTS["klekota_roth"][0](s) for s in smiles], dtype=np.float32)
        y = np.array([0, 0, 1, 1, 0, 1])
        return X, y

    def test_remove_low_variance_keys(self):
        X, _ = self._make_data()
        result = molfp.remove_low_variance(X, threshold=0.0)
        assert "X" in result
        assert "mask" in result
        assert "n_removed" in result
        assert "n_remaining" in result

    def test_remove_low_variance_reduces_dims(self):
        X, _ = self._make_data()
        result = molfp.remove_low_variance(X, threshold=0.0)
        assert result["X"].shape[1] < X.shape[1]
        assert result["n_removed"] + result["n_remaining"] == X.shape[1]

    def test_remove_low_variance_zero_keeps_varying(self):
        X, _ = self._make_data()
        result = molfp.remove_low_variance(X, threshold=0.0)
        variances = np.var(result["X"], axis=0)
        assert all(v > 0 for v in variances)

    def test_remove_low_variance_higher_threshold_removes_more(self):
        X, _ = self._make_data()
        r1 = molfp.remove_low_variance(X, threshold=0.0)
        r2 = molfp.remove_low_variance(X, threshold=0.1)
        assert r2["n_remaining"] <= r1["n_remaining"]

    def test_select_features_keys(self):
        X, y = self._make_data()
        result = molfp.select_features(X, y, k=20)
        assert "X" in result
        assert "indices" in result
        assert "scores" in result
        assert "n_input" in result
        assert "n_selected" in result

    def test_select_features_shape(self):
        X, y = self._make_data()
        result = molfp.select_features(X, y, k=20)
        assert result["X"].shape == (len(y), 20)
        assert len(result["indices"]) == 20
        assert len(result["scores"]) == 20

    def test_select_features_indices_valid(self):
        X, y = self._make_data()
        result = molfp.select_features(X, y, k=10)
        assert all(0 <= idx < X.shape[1] for idx in result["indices"])

    def test_select_features_with_threshold(self):
        X, y = self._make_data()
        result = molfp.select_features(X, y, k=15, threshold=0.01)
        assert result["X"].shape == (len(y), 15)
        assert result["n_after_variance"] < result["n_input"]

    def test_select_features_k_larger_than_available(self):
        X, y = self._make_data()
        result = molfp.select_features(X, y, k=99999)
        assert result["X"].shape[1] <= X.shape[1]

    def test_select_features_on_ecfp4(self):
        smiles = SMILES_SET
        X = np.array([molfp.FINGERPRINTS["ecfp4"][0](s) for s in smiles], dtype=np.float32)
        y = np.array([0, 0, 1, 1, 0, 1])
        result = molfp.select_features(X, y, k=50)
        assert result["X"].shape == (6, 50)

    def test_select_features_on_graph_vectors(self):
        X = molfp.graph_vectors(SMILES_SET, enriched=True)
        y = np.array([0, 0, 1, 1, 0, 1])
        result = molfp.select_features(X, y, k=100, threshold=0.01)
        assert result["X"].shape == (6, 100)


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
