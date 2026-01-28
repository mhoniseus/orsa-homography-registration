"""
Utilitaires de détection et de mise en correspondance de caracteristiques (encapsulation d'OpenCV).

Supporte les detecteurs SIFT et ORB avec le test du ratio de Lowe pour la
selection de correspondances putatives.

Auteur : Mouhssine Rifaki
"""

import cv2
import numpy as np


def detect_and_match(img1, img2, method="sift", ratio_threshold=0.75):
    """Detecte les points clés et calcule les correspondances putatives entre deux images.

    Paramètres
    ----------
    img1, img2 : np.ndarray
        Images d'entrée (niveaux de gris ou BGR).
    method : str
        Detecteur de caracteristiques : "sift" ou "orb".
    ratio_threshold : float
        Seuil du test du ratio de Lowe.

    Retourne
    --------
    src_pts : np.ndarray, forme (n, 2)
        Positions des points clés apparies dans img1.
    dst_pts : np.ndarray, forme (n, 2)
        Positions des points clés apparies dans img2.
    """
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1
    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2

    if method == "sift":
        detector = cv2.SIFT_create()
        norm_type = cv2.NORM_L2
    elif method == "orb":
        detector = cv2.ORB_create(nfeatures=5000)
        norm_type = cv2.NORM_HAMMING
    else:
        raise ValueError(f"Méthode inconnue : {method}")

    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        return np.empty((0, 2)), np.empty((0, 2))

    bf = cv2.BFMatcher(norm_type)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    # Test du ratio de Lowe
    good = []
    for m, n in raw_matches:
        if m.distance < ratio_threshold * n.distance:
            good.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

    return src_pts, dst_pts
