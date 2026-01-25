"""
Vérification de dégénérescence et controles de validite pour l'estimation d'homographie.

Rejette les configurations dégénérées (points colineaires, matrices mal conditionnees,
deformations inversant l'orientation) avant qu'elles ne polluent le calcul du NFA.
"""

import numpy as np
from itertools import combinations


def check_collinearity(pts: np.ndarray, threshold: float = 1e-4) -> bool:
    """Vérifie si 3 des points donnes sont quasi colineaires.

    Paramètres
    ----------
    pts : (m, 2) tableau de points 2D (typiquement m=4 pour un echantillon minimal)
    threshold : aire minimale de triangle pour considerer la configuration comme non dégénérée

    Retourne
    --------
    True si la configuration est dégénérée (colineaire), False sinon.
    """
    # si 3 points sont colineaires le système DLT devient rang-deficient
    # et on obtient une homographie aberrante, donc on rejette ces échantillons tot
    n = len(pts)
    for i, j, k in combinations(range(n), 3):
        # le produit vectoriel donne le double de l'aire du triangle
        v1 = pts[j] - pts[i]
        v2 = pts[k] - pts[i]
        area = abs(v1[0] * v2[1] - v1[1] * v2[0])
        if area < threshold:
            return True  # degenere
    return False


def check_conditioning(H: np.ndarray, threshold: float = 0.1) -> bool:
    """Vérifie que H est bien conditionnee.

    Paramètres
    ----------
    H : (3, 3) matrice d'homographie
    threshold : nombre de condition inverse minimal acceptable.
        La référence IPOL rejette les homographies avec un nombre de condition
        superieur a ~10 (inv_cond < 0.1).

    Retourne
    --------
    True si H est bien conditionnee, False si dégénérée.
    """
    # on vérifie sur la H normalisee avant denormalisation comme recommande par IPOL
    # une H mal conditionnee signifie que la transformation est quasi singuliere et peu fiable
    sigma = np.linalg.svd(H, compute_uv=False)
    if sigma[0] < 1e-15:
        return False
    inv_cond = sigma[-1] / sigma[0]
    return bool(inv_cond >= threshold)


def check_orientation_preserving(
    H: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    indices: np.ndarray | None = None,
) -> bool:
    """Vérifie que H préserve l'orientation aux correspondances donnees.

    On a besoin que w' > 0 quand on projette un point à travers H, sinon cela signifie
    que le point est projete derriere la camera, ce qui est physiquement absurde.
    On ne vérifie que sur les points de l'echantillon et non sur tous les points car les
    correspondances invalides auront simplement une erreur infinie dans symmetric_transfer_error.
    """
    if indices is not None:
        pts = pts1[indices]
    else:
        pts = pts1

    # w' = H[2,0]*x + H[2,1]*y + H[2,2]
    w_prime = H[2, 0] * pts[:, 0] + H[2, 1] * pts[:, 1] + H[2, 2]
    return bool(np.all(w_prime > 0))


def check_valid_warp(H: np.ndarray, img_shape: tuple, max_area_ratio: float = 100.0) -> bool:
    """Vérifie que H ne deforme pas les coins de l'image vers des positions absurdes.

    On a ajoute cette vérification car certaines homographies passent tous les autres
    controles mais reduisent l'image a une fine bande ou la dilatent enormement.
    Cela attrape ces cas degeneres qui pollueraient le calcul du NFA.
    """
    h, w = img_shape[:2]
    corners = np.array([
        [0, 0, 1],
        [w, 0, 1],
        [w, h, 1],
        [0, h, 1],
    ], dtype=float).T  # (3, 4)

    mapped = H @ corners  # (3, 4)

    # Vérification que tous les w' > 0
    if np.any(mapped[2, :] <= 0):
        return False

    mapped_xy = mapped[:2, :] / mapped[2:, :]  # (2, 4)

    # Vérification que les coordonnees sont finies et pas trop grandes
    if not np.all(np.isfinite(mapped_xy)):
        return False
    if np.any(np.abs(mapped_xy) > 1e6):
        return False

    # on utilise la formule du lacet pour calculer le changement d'aire
    # si l'aire a grandi ou retreci de plus de 100x l'homographie est dégénérée
    x = mapped_xy[0, :]
    y = mapped_xy[1, :]
    area_mapped = 0.5 * abs(
        x[0] * y[1] - x[1] * y[0]
        + x[1] * y[2] - x[2] * y[1]
        + x[2] * y[3] - x[3] * y[2]
        + x[3] * y[0] - x[0] * y[3]
    )
    area_original = h * w
    if area_original < 1:
        return False
    ratio = area_mapped / area_original
    return bool(ratio < max_area_ratio and ratio > 1.0 / max_area_ratio)
