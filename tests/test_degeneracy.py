# tests/test_degeneracy.py
"""Tests pour le module de vérification de dégénérescence."""

import numpy as np
import pytest

from orsa_homography.degeneracy import (
    check_collinearity,
    check_conditioning,
    check_orientation_preserving,
    check_valid_warp,
)


class TestCheckCollinearity:
    def test_collinear_points(self):
        """4 points alignes doivent être signales comme degeneres."""
        pts = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=float)
        assert check_collinearity(pts)

    def test_three_collinear(self):
        """3 points sur 4 colineaires doivent aussi être signales."""
        pts = np.array([[0, 0], [1, 0], [2, 0], [1, 5]], dtype=float)
        assert check_collinearity(pts)

    def test_general_position(self):
        """4 points en position generale : aucun triplet colineaire."""
        pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
        assert not check_collinearity(pts)

    def test_near_collinear(self):
        """Points presque mais pas tout a fait colineaires."""
        pts = np.array([[0, 0], [100, 0], [200, 0.5], [50, 80]], dtype=float)
        assert not check_collinearity(pts)


class TestCheckConditioning:
    def test_identity(self):
        """La matrice identite est bien conditionnee."""
        assert check_conditioning(np.eye(3))

    def test_singular(self):
        """Une matrice singuliere doit echouer."""
        H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float)
        assert not check_conditioning(H)

    def test_near_singular(self):
        """Une matrice quasi singuliere doit echouer."""
        H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1e-12]], dtype=float)
        assert not check_conditioning(H)

    def test_good_homography(self):
        """Une homographie normalisee doit passer (le conditionnement est vérifie
        sur la H normalisee dans le pipeline DLT, ou cond ~= 1-10)."""
        # Une homographie normalisee bien conditionnee (telle que produite dans le DLT)
        H = np.array([
            [1.05, 0.08, 0.01],
            [-0.03, 0.98, 0.02],
            [0.01, 0.005, 1.0],
        ])
        assert check_conditioning(H)


class TestCheckOrientationPreserving:
    def test_identity(self):
        """L'identite préserve l'orientation."""
        pts1 = np.array([[100, 100], [200, 200], [300, 150]], dtype=float)
        pts2 = pts1.copy()
        assert check_orientation_preserving(np.eye(3), pts1, pts2)

    def test_flip(self):
        """Une homographie qui projette w' < 0 doit echouer."""
        H = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0.01, 0, -1],  # w' = 0.01*x - 1, négatif pour x < 100
        ], dtype=float)
        pts1 = np.array([[50, 50], [30, 100]], dtype=float)
        pts2 = pts1.copy()
        assert not check_orientation_preserving(H, pts1, pts2)


class TestCheckValidWarp:
    def test_identity(self):
        """La deformation identite doit être valide."""
        assert check_valid_warp(np.eye(3), (480, 640))

    def test_mild_perspective(self):
        """Une perspective moderee doit être valide."""
        H = np.array([
            [1.05, 0.08, 15],
            [-0.03, 0.98, 10],
            [0.0001, 0.00005, 1],
        ])
        assert check_valid_warp(H, (480, 640))

    def test_degenerate_warp(self):
        """Une deformation qui envoie les coins a l'infini doit echouer."""
        H = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0.01, 0.01, 0],  # w' = 0 pour de nombreux points
        ], dtype=float)
        assert not check_valid_warp(H, (480, 640))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
