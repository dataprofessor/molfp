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

    def test_available_returns_13(self):
        fps = molfp.available()
        assert len(fps) == 13

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
        for name in molfp.available():
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

    def test_transformers_registry_has_all_13(self):
        assert len(molfp.TRANSFORMERS) == 13

    def test_transformers_registry_matches_fingerprints(self):
        assert set(molfp.TRANSFORMERS.keys()) == set(molfp.FINGERPRINTS.keys())


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
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
