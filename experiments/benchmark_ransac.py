"""
Comparaison ORSA vs RANSAC.

Lance des balayages de taux d'aberrants et de niveaux de bruit sur des donnees synthetiques,
produit des résultats CSV et trois graphiques de comparaison de qualite publication.

Utilisation :
    python -m experiments.benchmark_ransac [--output-dir DIR] [--n-runs N]

Executer depuis la racine du projet (MVA_detection_theory/).
"""

import argparse
import csv
import os
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from orsa_homography.homography import fit_homography_dlt, symmetric_transfer_error
from orsa_homography.orsa import orsa_homography
from experiments.synthetic import (
    evaluate_homography,
    generate_synthetic_matches,
    make_test_homographies,
)

# Paramètres de graphiques de qualite publication
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


# RANSAC de référence

def ransac_homography(pts1, pts2, threshold=5.0, n_iter=1000, seed=None):
    """RANSAC classique avec seuil fixe (référence pour comparaison).

    Paramètres
    ----------
    pts1, pts2 : (n, 2) correspondances de points
    threshold : seuil de distance pour les inliers en pixels
    n_iter : nombre maximal d'itérations
    seed : graine aléatoire

    Retourne
    --------
    H_best : (3, 3) homographie ou None
    best_inlier_mask : (n,) booleen
    """
    rng = np.random.default_rng(seed)
    n = len(pts1)
    best_inlier_mask = np.zeros(n, dtype=bool)
    H_best = None
    threshold_sq = threshold ** 2  # symmetric_transfer_error renvoie des distances au carre

    for _ in range(n_iter):
        sample = rng.choice(n, size=4, replace=False)
        H = fit_homography_dlt(pts1[sample], pts2[sample])
        if H is None:
            continue
        if not np.isfinite(H).all():
            continue

        errors, _ = symmetric_transfer_error(H, pts1, pts2)
        inliers = errors < threshold_sq
        if inliers.sum() > best_inlier_mask.sum():
            best_inlier_mask = inliers
            H_best = H.copy()

    # Re-estimation finale sur tous les inliers
    if H_best is not None and best_inlier_mask.sum() >= 4:
        H_refit = fit_homography_dlt(pts1[best_inlier_mask], pts2[best_inlier_mask])
        if H_refit is not None:
            H_best = H_refit

    return H_best, best_inlier_mask


# Balayages de comparaison

def run_outlier_sweep(n_runs=5, n_total=100, noise_sigma=1.0, max_iter=500,
                      img_shape=(480, 640)):
    """Balayage des taux d'aberrants de 10% a 70%, comparaison ORSA vs RANSAC."""
    outlier_fracs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    H_true = make_test_homographies(img_shape)['perspective_mild']
    results = []

    for frac in outlier_fracs:
        n_out = int(n_total * frac)
        n_in = n_total - n_out

        orsa_prec, orsa_rec, orsa_err, orsa_time = [], [], [], []
        ransac_prec, ransac_rec, ransac_err, ransac_time = [], [], [], []

        for run in range(n_runs):
            pts1, pts2, gt_mask = generate_synthetic_matches(
                n_inliers=n_in, n_outliers=n_out, H_true=H_true,
                noise_sigma=noise_sigma, img_shape=img_shape, seed=run,
            )

            # ORSA
            t0 = time.time()
            orsa_result = orsa_homography(
                pts1, pts2, img_shape, img_shape,
                max_iter=max_iter, seed=run,
            )
            orsa_time.append(time.time() - t0)

            # RANSAC
            t0 = time.time()
            H_r, inl_r = ransac_homography(
                pts1, pts2, threshold=5.0, n_iter=max_iter, seed=run,
            )
            ransac_time.append(time.time() - t0)

            # Metriques pour ORSA
            for H, inl, prec_l, rec_l, err_l in [
                (orsa_result.H, orsa_result.inlier_mask,
                 orsa_prec, orsa_rec, orsa_err),
                (H_r, inl_r, ransac_prec, ransac_rec, ransac_err),
            ]:
                if H is not None:
                    p = (inl & gt_mask).sum() / (inl.sum() + 1e-12)
                    r = (inl & gt_mask).sum() / (gt_mask.sum() + 1e-12)
                    e = evaluate_homography(H, H_true, img_shape)['frobenius_error']
                else:
                    p, r, e = 0.0, 0.0, float('inf')
                prec_l.append(p)
                rec_l.append(r)
                err_l.append(e)

        results.append({
            'outlier_frac': frac,
            'orsa_precision': np.mean(orsa_prec),
            'orsa_recall': np.mean(orsa_rec),
            'orsa_h_error': np.mean([e for e in orsa_err if np.isfinite(e)]) if any(np.isfinite(e) for e in orsa_err) else float('inf'),
            'orsa_time_s': np.mean(orsa_time),
            'ransac_precision': np.mean(ransac_prec),
            'ransac_recall': np.mean(ransac_rec),
            'ransac_h_error': np.mean([e for e in ransac_err if np.isfinite(e)]) if any(np.isfinite(e) for e in ransac_err) else float('inf'),
            'ransac_time_s': np.mean(ransac_time),
        })
        print(f"  outlier_frac={frac:.1f}: "
              f"ORSA prec={results[-1]['orsa_precision']:.3f} rec={results[-1]['orsa_recall']:.3f} | "
              f"RANSAC prec={results[-1]['ransac_precision']:.3f} rec={results[-1]['ransac_recall']:.3f}")

    return results


def run_noise_sweep(n_runs=5, n_inliers=80, n_outliers=20, max_iter=500,
                    img_shape=(480, 640)):
    """Balayage des niveaux de bruit de 0.1 a 5.0 pixels."""
    noise_levels = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
    H_true = make_test_homographies(img_shape)['perspective_mild']
    results = []

    for sigma in noise_levels:
        orsa_err, orsa_nfa = [], []
        ransac_err_list = []

        for run in range(n_runs):
            pts1, pts2, gt_mask = generate_synthetic_matches(
                n_inliers=n_inliers, n_outliers=n_outliers, H_true=H_true,
                noise_sigma=sigma, img_shape=img_shape, seed=run,
            )

            orsa_result = orsa_homography(
                pts1, pts2, img_shape, img_shape,
                max_iter=max_iter, seed=run,
            )
            if orsa_result.H is not None:
                orsa_err.append(evaluate_homography(orsa_result.H, H_true, img_shape)['frobenius_error'])
                orsa_nfa.append(orsa_result.log_nfa)

            H_r, _ = ransac_homography(pts1, pts2, threshold=5.0, n_iter=max_iter, seed=run)
            if H_r is not None:
                ransac_err_list.append(evaluate_homography(H_r, H_true, img_shape)['frobenius_error'])

        results.append({
            'noise_std': sigma,
            'orsa_h_error': np.mean(orsa_err) if orsa_err else float('inf'),
            'orsa_log_nfa': np.mean(orsa_nfa) if orsa_nfa else 0.0,
            'ransac_h_error': np.mean(ransac_err_list) if ransac_err_list else float('inf'),
        })
        print(f"  noise_std={sigma:.1f}: "
              f"ORSA H_err={results[-1]['orsa_h_error']:.4f} NFA={results[-1]['orsa_log_nfa']:.1f} | "
              f"RANSAC H_err={results[-1]['ransac_h_error']:.4f}")

    return results


# Sortie CSV

def save_csv(rows, path):
    """Ecrit une liste de dictionnaires dans un fichier CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Enregistre : {path}")


# Graphiques

def plot_outlier_precision_recall(rows, outdir):
    """Precision et rappel cote a cote en fonction du taux d'aberrants."""
    fracs = [r['outlier_frac'] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(fracs, [r['orsa_precision'] for r in rows],
             'o-', color='#3498db', label='ORSA', linewidth=2)
    ax1.plot(fracs, [r['ransac_precision'] for r in rows],
             's--', color='#e74c3c', label='RANSAC (ε=5px)', linewidth=2)
    ax1.set_xlabel("Fraction d'aberrants")
    ax1.set_ylabel('Precision')
    ax1.set_title("Precision des inliers vs taux d'aberrants")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    ax2.plot(fracs, [r['orsa_recall'] for r in rows],
             'o-', color='#3498db', label='ORSA', linewidth=2)
    ax2.plot(fracs, [r['ransac_recall'] for r in rows],
             's--', color='#e74c3c', label='RANSAC (ε=5px)', linewidth=2)
    ax2.set_xlabel("Fraction d'aberrants")
    ax2.set_ylabel('Rappel')
    ax2.set_title("Rappel des inliers vs taux d'aberrants")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(outdir, 'ransac_precision_recall.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Enregistre : {path}")


def plot_homography_error(rows, outdir):
    """Erreur de Frobenius de l'homographie vs taux d'aberrants (echelle log)."""
    fracs = [r['outlier_frac'] for r in rows]

    fig, ax = plt.subplots()
    ax.plot(fracs, [r['orsa_h_error'] for r in rows],
            'o-', color='#3498db', label='ORSA', linewidth=2)
    ax.plot(fracs, [r['ransac_h_error'] for r in rows],
            's--', color='#e74c3c', label='RANSAC (ε=5px)', linewidth=2)
    ax.set_xlabel("Fraction d'aberrants")
    ax.set_ylabel("Erreur d'homographie (Frobenius)")
    ax.set_title("Precision d'estimation vs taux d'aberrants")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    path = os.path.join(outdir, 'ransac_homography_error.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Enregistre : {path}")


def plot_noise_sensitivity(rows, outdir):
    """Erreur d'homographie et NFA vs niveau de bruit."""
    sigmas = [r['noise_std'] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(sigmas, [r['orsa_h_error'] for r in rows],
             'o-', color='#3498db', label='ORSA', linewidth=2)
    ax1.plot(sigmas, [r['ransac_h_error'] for r in rows],
             's--', color='#e74c3c', label='RANSAC (ε=5px)', linewidth=2)
    ax1.set_xlabel('Ecart-type du bruit (pixels)')
    ax1.set_ylabel("Erreur d'homographie (Frobenius)")
    ax1.set_title("Precision d'estimation vs niveau de bruit")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(sigmas, [r['orsa_log_nfa'] for r in rows],
             'o-', color='#9b59b6', linewidth=2)
    ax2.set_xlabel('Ecart-type du bruit (pixels)')
    ax2.set_ylabel('log$_{10}$(NFA)')
    ax2.set_title('Significativite ORSA vs niveau de bruit')
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Seuil de détection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, 'ransac_noise_sensitivity.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Enregistre : {path}")


# Principal

def main():
    parser = argparse.ArgumentParser(description='Comparaison ORSA vs RANSAC')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Repertoire de sortie pour les CSV et graphiques')
    parser.add_argument('--n-runs', type=int, default=5,
                        help="Nombre d'executions par configuration")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Lancement du balayage des taux d'aberrants...")
    outlier_results = run_outlier_sweep(n_runs=args.n_runs)
    save_csv(outlier_results, os.path.join(args.output_dir, 'ransac_outlier_sweep.csv'))

    print("\nLancement du balayage des niveaux de bruit...")
    noise_results = run_noise_sweep(n_runs=args.n_runs)
    save_csv(noise_results, os.path.join(args.output_dir, 'ransac_noise_sweep.csv'))

    print("\nGeneration des graphiques...")
    plot_outlier_precision_recall(outlier_results, args.output_dir)
    plot_homography_error(outlier_results, args.output_dir)
    plot_noise_sensitivity(noise_results, args.output_dir)

    print("\nTermine !")


if __name__ == '__main__':
    main()
