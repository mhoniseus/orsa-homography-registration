# experiments/synthetic.py
"""
Génération de donnees synthetiques pour les expériences ORSA.

Génère des correspondances de points avec une homographie de verite terrain connue,
des niveaux de bruit controles et des ratios inliers/aberrants configurables.
"""

import numpy as np


def generate_synthetic_matches(
    n_inliers: int,
    n_outliers: int,
    H_true: np.ndarray,
    noise_sigma: float = 1.0,
    img_shape: tuple = (480, 640),
    seed: int | None = None,
    margin: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Génère des correspondances de points synthetiques avec verite terrain connue.

    Paramètres
    ----------
    n_inliers : nombre de correspondances inliers
    n_outliers : nombre de correspondances aberrantes (aléatoires, indépendantes)
    H_true : (3, 3) homographie de verite terrain
    noise_sigma : écart-type du bruit gaussien ajoute aux points inliers projetes (pixels)
    img_shape : (hauteur, largeur) des deux images
    seed : graine aléatoire
    margin : marge par rapport aux bords de l'image pour la génération de points

    Retourne
    --------
    pts1 : (n, 2) points dans l'image 1
    pts2 : (n, 2) points dans l'image 2
    gt_mask : (n,) booleen, True = inlier
    """
    rng = np.random.default_rng(seed)
    h, w = img_shape

    pts1_list = []
    pts2_list = []
    gt_list = []

    # les inliers sont generes en projetant des points aléatoires à travers la vraie H
    # et en ajoutant du bruit gaussien pour simuler l'erreur de mesure réelle
    if n_inliers > 0:
        x1 = rng.uniform(margin, w - margin, n_inliers)
        y1 = rng.uniform(margin, h - margin, n_inliers)
        inlier_pts1 = np.column_stack([x1, y1])

        # Projection à travers H_true
        pts1_h = np.column_stack([inlier_pts1, np.ones(n_inliers)])
        proj = (H_true @ pts1_h.T).T
        proj_xy = proj[:, :2] / proj[:, 2:3]

        # Ajout de bruit
        noise = rng.normal(0, noise_sigma, (n_inliers, 2))
        inlier_pts2 = proj_xy + noise

        # Filtrage des points qui sortent des limites de l'image
        valid = (
            (inlier_pts2[:, 0] >= 0) & (inlier_pts2[:, 0] < w)
            & (inlier_pts2[:, 1] >= 0) & (inlier_pts2[:, 1] < h)
            & (proj[:, 2] > 0)  # preservation de l'orientation
        )
        pts1_list.append(inlier_pts1[valid])
        pts2_list.append(inlier_pts2[valid])
        gt_list.append(np.ones(np.sum(valid), dtype=bool))

    # les aberrants sont entierement aléatoires dans les deux images, sans aucune
    # relation géométrique -- c'est le modèle H0 du cadre a-contrario
    if n_outliers > 0:
        outlier_pts1 = np.column_stack([
            rng.uniform(margin, w - margin, n_outliers),
            rng.uniform(margin, h - margin, n_outliers),
        ])
        outlier_pts2 = np.column_stack([
            rng.uniform(margin, w - margin, n_outliers),
            rng.uniform(margin, h - margin, n_outliers),
        ])
        pts1_list.append(outlier_pts1)
        pts2_list.append(outlier_pts2)
        gt_list.append(np.zeros(n_outliers, dtype=bool))

    # on melange pour qu'ORSA ne puisse pas tricher en supposant que les inliers arrivent en premier
    pts1 = np.vstack(pts1_list) if pts1_list else np.zeros((0, 2))
    pts2 = np.vstack(pts2_list) if pts2_list else np.zeros((0, 2))
    gt_mask = np.concatenate(gt_list) if gt_list else np.zeros(0, dtype=bool)

    perm = rng.permutation(len(pts1))
    return pts1[perm], pts2[perm], gt_mask[perm]


def make_test_homographies(img_shape: tuple = (480, 640)) -> dict[str, np.ndarray]:
    """Retourne un dictionnaire d'homographies de test nommees.

    Toutes sont concues pour projeter des points dans les dimensions
    d'image donnees vers des positions raisonnables.
    """
    h, w = img_shape
    cx, cy = w / 2, h / 2

    homographies = {}

    # Identite
    homographies['identity'] = np.eye(3)

    # Translation pure (50px a droite, 30px vers le bas)
    homographies['translation'] = np.array([
        [1, 0, 50],
        [0, 1, 30],
        [0, 0, 1],
    ], dtype=float)

    # Rotation de 15 degres autour du centre de l'image
    theta = np.radians(15)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # Translation du centre vers l'origine, rotation, translation inverse
    T_to_origin = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=float)
    R = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=float)
    T_back = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=float)
    homographies['rotation_15deg'] = T_back @ R @ T_to_origin

    # Affine : mise a l'echelle + cisaillement
    homographies['affine'] = np.array([
        [1.1, 0.1, 20],
        [0.05, 0.95, -10],
        [0, 0, 1],
    ], dtype=float)

    # Perspective legere
    homographies['perspective_mild'] = np.array([
        [1.05, 0.08, 15],
        [-0.03, 0.98, 10],
        [0.0001, 0.00005, 1],
    ], dtype=float)

    # Perspective forte
    homographies['perspective_strong'] = np.array([
        [0.9, 0.2, 30],
        [-0.15, 1.1, -20],
        [0.0005, 0.0003, 1],
    ], dtype=float)

    return homographies


def evaluate_homography(
    H_estimated: np.ndarray,
    H_true: np.ndarray,
    img_shape: tuple = (480, 640),
) -> dict:
    """Evalue une homographie estimee par rapport a la verite terrain.

    On mesure l'erreur aux 4 coins car c'est la que les erreurs d'homographie
    sont les plus visibles et faciles a interpreter en pixels.
    On calcule egalement la distance de Frobenius sur les matrices normalisees
    pour obtenir une metrique indépendante de l'echelle.
    """
    h, w = img_shape
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
    corners_h = np.column_stack([corners, np.ones(4)])

    # Projection des coins à travers les deux homographies
    proj_true = (H_true @ corners_h.T).T
    proj_true_xy = proj_true[:, :2] / proj_true[:, 2:3]

    proj_est = (H_estimated @ corners_h.T).T
    proj_est_xy = proj_est[:, :2] / proj_est[:, 2:3]

    errors = np.sqrt(np.sum((proj_true_xy - proj_est_xy) ** 2, axis=1))

    # on normalise les deux matrices avant de comparer car H n'est definie
    # qu'a un facteur d'echelle près et on vérifie les deux signes car H et -H
    # representent la même transformation
    H_true_norm = H_true / np.linalg.norm(H_true)
    H_est_norm = H_estimated / np.linalg.norm(H_estimated)
    frob1 = np.linalg.norm(H_est_norm - H_true_norm)
    frob2 = np.linalg.norm(H_est_norm + H_true_norm)
    frob_error = min(frob1, frob2)

    return {
        'corner_error_mean': float(np.mean(errors)),
        'corner_error_max': float(np.max(errors)),
        'frobenius_error': float(frob_error),
    }
