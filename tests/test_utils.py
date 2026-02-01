"""Tests pour les fonctions utilitaires."""

import numpy as np
import pytest
from orsa_homography.utils import generate_synthetic_pair


class TestGenerateSyntheticPair:
    """Tests pour la génération de donnees synthetiques."""

    def test_output_shapes(self):
        """Vérification que les sorties ont les bonnes formes."""
        src, dst, H, mask = generate_synthetic_pair(n_inliers=50, n_outliers=10, seed=0)
        assert src.shape == (60, 2)
        assert dst.shape == (60, 2)
        assert H.shape == (3, 3)
        assert mask.shape == (60,)

    def test_inlier_count(self):
        """Le masque doit contenir le bon nombre d'inliers."""
        src, dst, H, mask = generate_synthetic_pair(n_inliers=70, n_outliers=30, seed=0)
        assert mask.sum() == 70

    def test_deterministic_with_seed(self):
        """La même graine doit produire des sorties identiques."""
        a1 = generate_synthetic_pair(seed=42)
        a2 = generate_synthetic_pair(seed=42)
        for x, y in zip(a1, a2):
            assert np.array_equal(x, y)

    def test_inliers_follow_homography(self):
        """Les points dst inliers doivent être proches de H @ src (au bruit près)."""
        src, dst, H, mask = generate_synthetic_pair(
            n_inliers=50, n_outliers=0, noise_std=0.0, seed=0
        )
        src_h = np.column_stack([src, np.ones(50)]).T
        dst_h = H @ src_h
        dst_proj = (dst_h[:2] / dst_h[2:]).T
        assert np.allclose(dst, dst_proj, atol=1e-6)
