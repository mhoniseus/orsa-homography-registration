"""
Comparaison ORSA vs RANSAC sur différents taux d'aberrants et niveaux de bruit.

Enregistre les résultats CSV dans outputs/csv/ et les donnees brutes pour la génération de graphiques.

Auteur : Mouhssine Rifaki
"""

import time
import csv
import os
import numpy as np
from orsa_homography.homography import dlt_homography, symmetric_transfer_error
from orsa_homography.orsa import orsa_homography
from orsa_homography.utils import generate_synthetic_pair


# Dimensions d'image par defaut correspondant a generate_synthetic_pair : 640x480
IMG_SHAPE = (480, 640)


def ransac_homography(src, dst, threshold=5.0, n_iter=1000, seed=None):
    """RANSAC classique avec seuil fixe (référence pour comparaison)."""
    rng = np.random.default_rng(seed)
    n = len(src)
    best_inliers = np.zeros(n, dtype=bool)
    H_best = None

    for _ in range(n_iter):
        sample = rng.choice(n, size=4, replace=False)
        try:
            H = dlt_homography(src[sample], dst[sample])
        except (np.linalg.LinAlgError, TypeError):
            continue
        if H is None or not np.isfinite(H).all():
            continue

        errors, _ = symmetric_transfer_error(H, src, dst)
        inliers = errors < threshold
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            H_best = H.copy()

    if H_best is not None and best_inliers.sum() >= 4:
        try:
            H_best = dlt_homography(src[best_inliers], dst[best_inliers])
        except (np.linalg.LinAlgError, TypeError):
            pass

    return H_best, best_inliers


def homography_error(H_est, H_gt):
    """Distance de Frobenius entre homographies normalisees."""
    H_e = H_est / H_est[2, 2]
    H_g = H_gt / H_gt[2, 2]
    return np.linalg.norm(H_e - H_g, "fro")


def run_outlier_sweep(n_runs=5):
    """Balayage des taux d'aberrants de 10% a 70%."""
    outlier_fracs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    n_total = 100
    results = []

    for frac in outlier_fracs:
        n_out = int(n_total * frac)
        n_in = n_total - n_out

        orsa_prec, orsa_rec, orsa_err, orsa_time = [], [], [], []
        ransac_prec, ransac_rec, ransac_err, ransac_time = [], [], [], []

        for run in range(n_runs):
            src, dst, H_gt, mask = generate_synthetic_pair(
                n_inliers=n_in, n_outliers=n_out, noise_std=1.0, seed=run
            )

            t0 = time.time()
            result_o = orsa_homography(src, dst, IMG_SHAPE, IMG_SHAPE, max_iter=500, seed=run)
            orsa_time.append(time.time() - t0)

            t0 = time.time()
            H_r, inl_r = ransac_homography(src, dst, threshold=5.0, n_iter=500, seed=run)
            ransac_time.append(time.time() - t0)

            # Resultats ORSA
            H_o = result_o.H
            inl_o = result_o.inlier_mask

            for H, inl, prec_l, rec_l, err_l in [
                (H_o, inl_o, orsa_prec, orsa_rec, orsa_err),
                (H_r, inl_r, ransac_prec, ransac_rec, ransac_err),
            ]:
                if H is not None:
                    p = (inl & mask).sum() / (inl.sum() + 1e-12)
                    r = (inl & mask).sum() / (mask.sum() + 1e-12)
                    e = homography_error(H, H_gt)
                else:
                    p, r, e = 0.0, 0.0, float("inf")
                prec_l.append(p)
                rec_l.append(r)
                err_l.append(e)

        results.append({
            "outlier_frac": frac,
            "orsa_precision": np.mean(orsa_prec),
            "orsa_recall": np.mean(orsa_rec),
            "orsa_h_error": np.mean([e for e in orsa_err if np.isfinite(e)]),
            "orsa_time_s": np.mean(orsa_time),
            "ransac_precision": np.mean(ransac_prec),
            "ransac_recall": np.mean(ransac_rec),
            "ransac_h_error": np.mean([e for e in ransac_err if np.isfinite(e)]),
            "ransac_time_s": np.mean(ransac_time),
        })

    return results


def run_noise_sweep(n_runs=5):
    """Balayage des niveaux de bruit de 0.1 a 5.0 pixels."""
    noise_levels = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
    results = []

    for sigma in noise_levels:
        orsa_err, orsa_nfa = [], []

        for run in range(n_runs):
            src, dst, H_gt, mask = generate_synthetic_pair(
                n_inliers=80, n_outliers=20, noise_std=sigma, seed=run
            )
            result = orsa_homography(src, dst, IMG_SHAPE, IMG_SHAPE, max_iter=500, seed=run)
            if result.H is not None:
                orsa_err.append(homography_error(result.H, H_gt))
                orsa_nfa.append(result.log_nfa)

        results.append({
            "noise_std": sigma,
            "orsa_h_error": np.mean(orsa_err) if orsa_err else float("inf"),
            "orsa_nfa": np.mean(orsa_nfa) if orsa_nfa else 0.0,
        })

    return results


if __name__ == "__main__":
    os.makedirs("outputs/csv", exist_ok=True)

    print("Lancement du balayage des taux d'aberrants...")
    outlier_results = run_outlier_sweep()
    with open("outputs/csv/outlier_sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=outlier_results[0].keys())
        writer.writeheader()
        writer.writerows(outlier_results)
    print(f"  {len(outlier_results)} lignes enregistrees dans outputs/csv/outlier_sweep.csv")

    print("Lancement du balayage des niveaux de bruit...")
    noise_results = run_noise_sweep()
    with open("outputs/csv/noise_sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=noise_results[0].keys())
        writer.writeheader()
        writer.writerows(noise_results)
    print(f"  {len(noise_results)} lignes enregistrees dans outputs/csv/noise_sweep.csv")

    print("Termine.")
