"""
Génération de graphiques de qualite publication a partir des résultats CSV.

Lit les fichiers depuis outputs/csv/ et ecrit les graphiques PNG dans outputs/plots/.

Auteur : Mouhssine Rifaki
"""

import os
import csv
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


def read_csv(path):
    """Lit un fichier CSV et le convertit en liste de dictionnaires avec conversion en flottants."""
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def plot_outlier_precision_recall(rows, outdir):
    """Precision et rappel vs fraction d'aberrants pour ORSA et RANSAC."""
    fracs = [r["outlier_frac"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(fracs, [r["orsa_precision"] for r in rows], "o-", color="#3498db", label="ORSA", linewidth=2)
    ax1.plot(fracs, [r["ransac_precision"] for r in rows], "s--", color="#e74c3c", label="RANSAC", linewidth=2)
    ax1.set_xlabel("Fraction d'aberrants")
    ax1.set_ylabel("Precision")
    ax1.set_title("Precision des inliers vs taux d'aberrants")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    ax2.plot(fracs, [r["orsa_recall"] for r in rows], "o-", color="#3498db", label="ORSA", linewidth=2)
    ax2.plot(fracs, [r["ransac_recall"] for r in rows], "s--", color="#e74c3c", label="RANSAC", linewidth=2)
    ax2.set_xlabel("Fraction d'aberrants")
    ax2.set_ylabel("Rappel")
    ax2.set_title("Rappel des inliers vs taux d'aberrants")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "precision_recall.png"), bbox_inches="tight")
    plt.close(fig)
    print("  precision_recall.png enregistre")


def plot_homography_error(rows, outdir):
    """Erreur de Frobenius de l'homographie vs fraction d'aberrants."""
    fracs = [r["outlier_frac"] for r in rows]

    fig, ax = plt.subplots()
    ax.plot(fracs, [r["orsa_h_error"] for r in rows], "o-", color="#3498db", label="ORSA", linewidth=2)
    ax.plot(fracs, [r["ransac_h_error"] for r in rows], "s--", color="#e74c3c", label="RANSAC", linewidth=2)
    ax.set_xlabel("Fraction d'aberrants")
    ax.set_ylabel("Erreur d'homographie (Frobenius)")
    ax.set_title("Precision d'estimation vs taux d'aberrants")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "homography_error.png"), bbox_inches="tight")
    plt.close(fig)
    print("  homography_error.png enregistre")


def plot_noise_sensitivity(rows, outdir):
    """Erreur d'homographie et NFA vs niveau de bruit."""
    sigmas = [r["noise_std"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(sigmas, [r["orsa_h_error"] for r in rows], "o-", color="#2ecc71", linewidth=2)
    ax1.set_xlabel("Ecart-type du bruit (pixels)")
    ax1.set_ylabel("Erreur d'homographie (Frobenius)")
    ax1.set_title("Precision ORSA vs niveau de bruit")
    ax1.grid(True, alpha=0.3)

    ax2.plot(sigmas, [r["orsa_nfa"] for r in rows], "o-", color="#9b59b6", linewidth=2)
    ax2.set_xlabel("Ecart-type du bruit (pixels)")
    ax2.set_ylabel("log10(NFA)")
    ax2.set_title("Significativite ORSA vs niveau de bruit")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5, label="Seuil de détection")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "noise_sensitivity.png"), bbox_inches="tight")
    plt.close(fig)
    print("  noise_sensitivity.png enregistre")


if __name__ == "__main__":
    outdir = "outputs/plots"
    os.makedirs(outdir, exist_ok=True)

    print("Génération des graphiques...")
    outlier_rows = read_csv("outputs/csv/outlier_sweep.csv")
    plot_outlier_precision_recall(outlier_rows, outdir)
    plot_homography_error(outlier_rows, outdir)

    noise_rows = read_csv("outputs/csv/noise_sweep.csv")
    plot_noise_sensitivity(noise_rows, outdir)

    print("Termine.")
