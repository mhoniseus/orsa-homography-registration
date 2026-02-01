"""Tests pour l'estimation d'homographie par DLT."""

import numpy as np
import pytest
from orsa_homography.homography import normalize_points, dlt_homography, symmetric_transfer_error


class TestNormalizePoints:
    """Tests pour la normalisation de Hartley."""

    def test_centroid_at_origin(self, rng):
        """Les points normalises doivent être centres a l'origine."""
        pts = rng.uniform(0, 100, size=(20, 2))
        pts_norm, T = normalize_points(pts)
        assert np.allclose(pts_norm.mean(axis=0), 0, atol=1e-10)

    def test_mean_distance_sqrt2(self, rng):
        """La distance moyenne a l'origine doit être sqrt(2) après normalisation."""
        pts = rng.uniform(0, 100, size=(20, 2))
        pts_norm, T = normalize_points(pts)
        mean_dist = np.mean(np.linalg.norm(pts_norm, axis=1))
        assert np.isclose(mean_dist, np.sqrt(2), atol=1e-10)

    def test_transformation_matrix_shape(self, rng):
        """La matrice de normalisation doit être 3x3."""
        pts = rng.uniform(0, 100, size=(10, 2))
        _, T = normalize_points(pts)
        assert T.shape == (3, 3)


class TestDLTHomography:
    """Tests pour la Transformation Lineaire Directe."""

    def test_identity_on_same_points(self, rng):
        """Le DLT avec src/dst identiques doit retourner l'identite."""
        pts = rng.uniform(0, 100, size=(10, 2))
        H = dlt_homography(pts, pts)
        assert H is not None
        assert np.allclose(H / H[2, 2], np.eye(3), atol=1e-6)

    def test_recovers_known_translation(self):
        """Le DLT doit retrouver une translation pure."""
        src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        dst = src + np.array([5.0, 3.0])
        H = dlt_homography(src, dst)
        assert H is not None
        assert np.isclose(H[0, 2] / H[2, 2], 5.0, atol=1e-6)
        assert np.isclose(H[1, 2] / H[2, 2], 3.0, atol=1e-6)

    def test_minimum_four_points(self):
        """Le DLT avec moins de 4 points doit retourner None."""
        src = np.array([[0, 0], [1, 0], [1, 1]], dtype=float)
        dst = src.copy()
        H = dlt_homography(src, dst)
        assert H is None

    def test_overdetermined_system(self, rng):
        """Le DLT avec plus de 4 points sans bruit doit retrouver H exactement."""
        # On utilise une homographie avec translation+rotation visibles pour qu'elle
        # reste bien conditionnee après la normalisation de Hartley
        H_gt = np.array([
            [0.9, -0.1, 30.0],
            [0.1,  0.95, 20.0],
            [0.0001, 0.0002, 1.0],
        ])
        src = rng.uniform(50, 600, size=(20, 2))
        src_h = np.column_stack([src, np.ones(20)]).T
        dst_h = H_gt @ src_h
        dst = (dst_h[:2] / dst_h[2:]).T

        H_est = dlt_homography(src, dst)
        assert H_est is not None, "Le DLT doit reussir sur des donnees sans bruit"
        H_est /= H_est[2, 2]
        H_gt_n = H_gt / H_gt[2, 2]
        assert np.allclose(H_est, H_gt_n, atol=1e-6)


class TestSymmetricTransferError:
    """Tests pour la metrique d'erreur de transfert symétrique."""

    def test_zero_error_for_exact_correspondences(self, rng, ground_truth_homography):
        """Les correspondances exactes doivent avoir une erreur quasi nulle."""
        H_gt = ground_truth_homography
        src = rng.uniform(0, 640, size=(10, 2))
        src_h = np.column_stack([src, np.ones(10)]).T
        dst_h = H_gt @ src_h
        dst = (dst_h[:2] / dst_h[2:]).T

        errors, sides = symmetric_transfer_error(H_gt, src, dst)
        assert np.allclose(errors, 0, atol=1e-6)

    def test_error_increases_with_noise(self, rng, ground_truth_homography):
        """Les erreurs doivent augmenter avec le bruit croissant."""
        H_gt = ground_truth_homography
        src = rng.uniform(0, 640, size=(30, 2))
        src_h = np.column_stack([src, np.ones(30)]).T
        dst_h = H_gt @ src_h
        dst_clean = (dst_h[:2] / dst_h[2:]).T

        err_low, _ = symmetric_transfer_error(H_gt, src, dst_clean + 0.1 * rng.standard_normal((30, 2)))
        err_high, _ = symmetric_transfer_error(H_gt, src, dst_clean + 10.0 * rng.standard_normal((30, 2)))
        assert err_high.mean() > err_low.mean()

    def test_output_shape(self, rng, ground_truth_homography):
        """La sortie doit avoir une erreur et un cote par correspondance."""
        H_gt = ground_truth_homography
        n = 15
        src = rng.uniform(0, 640, size=(n, 2))
        dst = rng.uniform(0, 640, size=(n, 2))
        errors, sides = symmetric_transfer_error(H_gt, src, dst)
        assert errors.shape == (n,)
        assert sides.shape == (n,)
