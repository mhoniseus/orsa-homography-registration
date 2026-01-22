"""
Fonctions utilitaires pour la génération de donnees synthetiques.

Auteur : Mouhssine Rifaki
"""

import numpy as np


def generate_synthetic_pair(n_inliers=80, n_outliers=20, noise_std=1.0, img_size=(640, 480), seed=None):
    """Génère une paire synthetique de correspondances de points avec une homographie verite terrain connue.

    Retourne
    --------
    src, dst : np.ndarray, forme (n, 2)
    H_gt : np.ndarray, forme (3, 3)
    inlier_mask : np.ndarray, forme (n,), bool
    """
    rng = np.random.default_rng(seed)
    w, h = img_size
    n = n_inliers + n_outliers

    # Homographie verite terrain aléatoire (perspective moderee)
    H_gt = np.eye(3) + 0.001 * rng.standard_normal((3, 3))
    H_gt[2, 2] = 1.0

    # Correspondances inliers
    src_in = rng.uniform([0, 0], [w, h], size=(n_inliers, 2))
    src_h = np.column_stack([src_in, np.ones(n_inliers)]).T
    dst_h = H_gt @ src_h
    dst_in = (dst_h[:2] / dst_h[2:]).T + noise_std * rng.standard_normal((n_inliers, 2))

    # Correspondances aberrantes (aléatoires)
    src_out = rng.uniform([0, 0], [w, h], size=(n_outliers, 2))
    dst_out = rng.uniform([0, 0], [w, h], size=(n_outliers, 2))

    src = np.vstack([src_in, src_out])
    dst = np.vstack([dst_in, dst_out])
    inlier_mask = np.array([True] * n_inliers + [False] * n_outliers)

    # Melange aléatoire
    perm = rng.permutation(n)
    return src[perm], dst[perm], H_gt, inlier_mask[perm]
