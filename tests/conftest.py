"""Fixtures partagées pour les tests d'homographie ORSA."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Générateur aléatoire deterministe."""
    return np.random.default_rng(3407)


@pytest.fixture
def ground_truth_homography(rng):
    """Homographie a perspective moderee avec verite terrain connue."""
    H = np.eye(3) + 0.001 * rng.standard_normal((3, 3))
    H[2, 2] = 1.0
    return H


@pytest.fixture
def synthetic_correspondences(rng, ground_truth_homography):
    """80 inliers + 20 aberrants avec verite terrain connue.

    Retourne (src, dst, H_gt, inlier_mask).
    """
    H_gt = ground_truth_homography
    n_in, n_out = 80, 20

    src_in = rng.uniform([0, 0], [640, 480], size=(n_in, 2))
    src_h = np.column_stack([src_in, np.ones(n_in)]).T
    dst_h = H_gt @ src_h
    dst_in = (dst_h[:2] / dst_h[2:]).T + 0.5 * rng.standard_normal((n_in, 2))

    src_out = rng.uniform([0, 0], [640, 480], size=(n_out, 2))
    dst_out = rng.uniform([0, 0], [640, 480], size=(n_out, 2))

    src = np.vstack([src_in, src_out])
    dst = np.vstack([dst_in, dst_out])
    mask = np.array([True] * n_in + [False] * n_out)

    perm = rng.permutation(n_in + n_out)
    return src[perm], dst[perm], H_gt, mask[perm]
