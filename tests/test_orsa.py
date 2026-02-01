"""Tests pour l'algorithme ORSA."""

import numpy as np
import pytest
from orsa_homography.orsa import orsa_homography, OrsaResult


# Forme d'image par defaut correspondant au generate_synthetic_pair de 640x480
IMG_SHAPE = (480, 640)


class TestORSA:
    """Tests pour le pipeline ORSA complet."""

    def test_finds_meaningful_model(self, synthetic_correspondences):
        """ORSA doit trouver un modèle avec log10(NFA) < 0 sur des donnees propres."""
        src, dst, H_gt, mask = synthetic_correspondences
        result = orsa_homography(src, dst, IMG_SHAPE, IMG_SHAPE, max_iter=500, seed=42)
        assert result.H is not None
        assert result.log_nfa < 0

    def test_inlier_recovery(self, synthetic_correspondences):
        """ORSA doit retrouver la plupart des vrais inliers."""
        src, dst, H_gt, mask = synthetic_correspondences
        result = orsa_homography(src, dst, IMG_SHAPE, IMG_SHAPE, max_iter=500, seed=42)
        inliers = result.inlier_mask
        précision = (inliers & mask).sum() / (inliers.sum() + 1e-12)
        recall = (inliers & mask).sum() / (mask.sum() + 1e-12)
        assert précision > 0.7, f"Precision trop basse : {précision:.2f}"
        assert recall > 0.7, f"Rappel trop bas : {recall:.2f}"

    def test_homography_quality(self, synthetic_correspondences):
        """La H estimee doit produire une petite erreur de reprojection sur les inliers."""
        src, dst, H_gt, mask = synthetic_correspondences
        result = orsa_homography(src, dst, IMG_SHAPE, IMG_SHAPE, max_iter=500, seed=42)
        from orsa_homography.homography import symmetric_transfer_error
        errors, _ = symmetric_transfer_error(result.H, src[mask], dst[mask])
        assert np.median(errors) < 5.0, f"Erreur mediane des inliers trop elevee : {np.median(errors):.2f}"

    def test_nfa_positive_on_shuffled_data(self, synthetic_correspondences):
        """Melanger dst detruit le vrai modèle ; le NFA doit augmenter."""
        src, dst, _, _ = synthetic_correspondences
        rng = np.random.default_rng(99)
        dst_shuffled = dst[rng.permutation(len(dst))]
        result_shuffled = orsa_homography(src, dst_shuffled, IMG_SHAPE, IMG_SHAPE, max_iter=500, seed=42)
        result_real = orsa_homography(src, dst, IMG_SHAPE, IMG_SHAPE, max_iter=500, seed=42)
        # On utilise raw_log_nfa pour comparer même quand aucun ne franchit le seuil de détection
        assert result_shuffled.raw_log_nfa > result_real.raw_log_nfa, \
            "Les donnees melangees doivent avoir un NFA plus eleve (pire)"

    def test_deterministic_with_seed(self, synthetic_correspondences):
        """La même graine doit produire des résultats identiques."""
        src, dst, _, _ = synthetic_correspondences
        r1 = orsa_homography(src, dst, IMG_SHAPE, IMG_SHAPE, max_iter=100, seed=123)
        r2 = orsa_homography(src, dst, IMG_SHAPE, IMG_SHAPE, max_iter=100, seed=123)
        assert np.array_equal(r1.inlier_mask, r2.inlier_mask)
        assert r1.log_nfa == r2.log_nfa

    def test_returns_orsa_result(self, synthetic_correspondences):
        """orsa_homography doit retourner une dataclass OrsaResult."""
        src, dst, _, _ = synthetic_correspondences
        result = orsa_homography(src, dst, IMG_SHAPE, IMG_SHAPE, max_iter=100, seed=42)
        assert isinstance(result, OrsaResult)
        assert hasattr(result, 'n_inliers')
        assert hasattr(result, 'epsilon')
        assert hasattr(result, 'n_iterations')
