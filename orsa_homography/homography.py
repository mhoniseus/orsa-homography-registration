"""
Estimation d'homographie, calcul d'erreur et raffinement.

Implémente la Transformation Lineaire Directe (DLT) avec normalisation de Hartley,
l'erreur de transfert symétrique et le raffinement de Levenberg-Marquardt.
"""

import numpy as np
from scipy.optimize import least_squares

from .degeneracy import check_conditioning


def normalize_points(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalisation de Hartley : translation du centroide a l'origine, mise a l'echelle
    pour que la distance moyenne a l'origine soit egale a sqrt(2).

    On fait cela car sans normalisation le DLT est numeriquement instable
    -- le nombre de condition de la matrice A explose quand les coordonnees
    en pixels sont dans les centaines tandis que la coordonnee homogène est 1.
    """
    centroid = np.mean(pts, axis=0)
    pts_centered = pts - centroid
    mean_dist = np.mean(np.sqrt(np.sum(pts_centered ** 2, axis=1)))
    if mean_dist < 1e-10:
        mean_dist = 1e-10
    scale = np.sqrt(2.0) / mean_dist

    T = np.array([
        [scale, 0.0, -scale * centroid[0]],
        [0.0, scale, -scale * centroid[1]],
        [0.0, 0.0, 1.0],
    ])

    pts_h = np.column_stack([pts, np.ones(len(pts))])
    pts_norm_h = (T @ pts_h.T).T
    pts_norm = pts_norm_h[:, :2]

    return pts_norm, T


def fit_homography_dlt(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray | None:
    """Calcule l'homographie par DLT a partir de n >= 4 correspondances.

    Paramètres
    ----------
    pts1 : (n, 2) points source
    pts2 : (n, 2) points destination (pts2 ~ H @ pts1 en coordonnees homogènes)

    Retourne
    --------
    H : (3, 3) matrice d'homographie, ou None si l'estimation echoue.
    """
    n = len(pts1)
    if n < 4:
        return None

    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)

    # chaque correspondance donne 2 équations (une par coordonnee)
    # donc 4 points donnent 8 équations pour 8 inconnues (H a 9 entrees mais on fixe l'echelle)
    A = np.zeros((2 * n, 9))
    for i in range(n):
        x1, y1 = pts1_norm[i]
        x2, y2 = pts2_norm[i]
        A[2 * i] = [x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2]
        A[2 * i + 1] = [0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2]

    # la solution est le vecteur singulier droit correspondant a la plus petite
    # valeur singuliere, c'est-a-dire la derniere ligne de Vt
    try:
        _, S, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None

    h = Vt[-1, :]
    H_norm = h.reshape(3, 3)

    # on vérifie le conditionnement avant denormalisation car c'est ce que recommande IPOL
    if not check_conditioning(H_norm):
        return None

    # annulation de la normalisation : H = T2^-1 @ H_norm @ T1
    H = np.linalg.inv(T2) @ H_norm @ T1

    # Normalisation pour que H[2,2] = 1 (si possible)
    if abs(H[2, 2]) < 1e-15:
        return None
    H = H / H[2, 2]

    return H


# Conservation de l'ancien nom comme alias pour que le code existant (tests, notebooks) fonctionne
dlt_homography = fit_homography_dlt


def symmetric_transfer_error(
    H: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcule l'erreur de transfert symétrique pour chaque correspondance.

    On prend le maximum des erreurs directe et inverse car une bonne homographie
    doit bien fonctionner dans les deux sens -- si elle ne fonctionne que dans un
    sens elle ajuste probablement du bruit.

    On suit egalement quel cote domine car le NFA a besoin de savoir quelle
    image utiliser pour le calcul d'alpha (elles peuvent avoir des tailles différentes).

    Retourne
    --------
    errors : (n,) erreur de transfert symétrique au carre
    sides : (n,) int, 1 si l'image droite domine, 0 si la gauche
    """
    n = len(pts1)
    large_error = 1e18

    # Direct : projection de pts1 à travers H -> comparaison avec pts2
    pts1_h = np.column_stack([pts1, np.ones(n)])  # (n, 3)
    proj_fwd = (H @ pts1_h.T).T  # (n, 3)
    w_fwd = proj_fwd[:, 2]

    # Inverse : projection de pts2 à travers H^{-1} -> comparaison avec pts1
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full(n, large_error), np.zeros(n, dtype=int)

    pts2_h = np.column_stack([pts2, np.ones(n)])
    proj_bwd = (H_inv @ pts2_h.T).T
    w_bwd = proj_bwd[:, 2]

    # Erreur directe (image droite)
    err_right = np.full(n, large_error)
    valid_fwd = np.abs(w_fwd) > 1e-10
    if np.any(valid_fwd):
        proj_fwd_xy = proj_fwd[valid_fwd, :2] / proj_fwd[valid_fwd, 2:3]
        err_right[valid_fwd] = np.sum((pts2[valid_fwd] - proj_fwd_xy) ** 2, axis=1)

    # Erreur inverse (image gauche)
    err_left = np.full(n, large_error)
    valid_bwd = np.abs(w_bwd) > 1e-10
    if np.any(valid_bwd):
        proj_bwd_xy = proj_bwd[valid_bwd, :2] / proj_bwd[valid_bwd, 2:3]
        err_left[valid_bwd] = np.sum((pts1[valid_bwd] - proj_bwd_xy) ** 2, axis=1)

    # Symetrique : on prend le maximum
    errors = np.maximum(err_left, err_right)
    # side = 1 si l'erreur droite domine (directe), 0 si la gauche (inverse)
    sides = (err_right >= err_left).astype(int)

    return errors, sides


def compute_inliers(
    H: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Retourne un masque booleen des inliers (erreur <= epsilon^2).

    Paramètres
    ----------
    H : (3, 3) homographie
    pts1, pts2 : (n, 2) points
    epsilon : seuil de distance en pixels (non au carre)

    Retourne
    --------
    mask : (n,) booleen, True pour les inliers
    """
    errors, _ = symmetric_transfer_error(H, pts1, pts2)
    return errors <= epsilon ** 2


def refine_homography(
    H_init: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    inlier_mask: np.ndarray,
) -> np.ndarray | None:
    """Raffine H par Levenberg-Marquardt sur les correspondances inliers uniquement.

    Le DLT nous donne un bon point de depart mais LM peut obtenir une précision
    supplementaire en minimisant l'erreur de reprojection réelle de facon non lineaire.
    On fixe H[2,2]=1 donc on optimise 8 paramètres libres.
    """
    p1 = pts1[inlier_mask]
    p2 = pts2[inlier_mask]
    n_inliers = len(p1)

    if n_inliers < 4:
        return None

    # Normalisation de H pour que H[2,2] = 1
    H0 = H_init.copy()
    if abs(H0[2, 2]) < 1e-15:
        return None
    H0 = H0 / H0[2, 2]

    # Paramètres initiaux : 8 valeurs (sans H[2,2])
    h0 = H0.flatten()[:8]  # les 8 premiers elements

    p1_h = np.column_stack([p1, np.ones(n_inliers)])

    def residuals(h):
        H = np.array([
            [h[0], h[1], h[2]],
            [h[3], h[4], h[5]],
            [h[6], h[7], 1.0],
        ])
        # Projection directe
        proj = (H @ p1_h.T).T
        w = proj[:, 2]
        # Eviter la division par zero
        w = np.where(np.abs(w) < 1e-10, 1e-10, w)
        proj_xy = proj[:, :2] / w[:, np.newaxis]
        return (p2 - proj_xy).flatten()  # 2*n_inliers residus

    try:
        result = least_squares(residuals, h0, method='lm', max_nfev=200)
        h_opt = result.x
        H_refined = np.array([
            [h_opt[0], h_opt[1], h_opt[2]],
            [h_opt[3], h_opt[4], h_opt[5]],
            [h_opt[6], h_opt[7], 1.0],
        ])
        return H_refined
    except Exception:
        return None
