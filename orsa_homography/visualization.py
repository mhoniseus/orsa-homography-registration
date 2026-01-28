"""
Utilitaires de visualisation pour le recalage d'homographie par ORSA.

Fournit des fonctions pour dessiner les correspondances, afficher les images
deformees, tracer les courbes NFA et generer des resumes d'expériences.

Auteur : Mouhssine Rifaki
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2

matplotlib.use('Agg')  # backend non interactif pour la sauvegarde de figures


def draw_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    inlier_mask: np.ndarray | None = None,
    max_display: int = 200,
    title: str = "",
    figsize: tuple = (16, 8),
) -> plt.Figure:
    """Dessine les correspondances entre deux images cote a cote.

    Lignes vertes pour les inliers, rouges pour les aberrants. Si inlier_mask
    est None, toutes les correspondances sont dessinees en bleu.

    Paramètres
    ----------
    img1, img2 : images d'entrée
    pts1, pts2 : (n, 2) points apparies
    inlier_mask : (n,) masque booleen ; None signifie pas de classification
    max_display : nombre maximal de correspondances a dessiner (sous-echantillonnees si plus)
    title : titre de la figure
    figsize : taille de la figure

    Retourne
    -------
    fig : Figure matplotlib
    """
    # Conversion en RGB pour l'affichage
    if len(img1.shape) == 2:
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    else:
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    if len(img2.shape) == 2:
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    else:
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    h1, w1 = img1_rgb.shape[:2]
    h2, w2 = img2_rgb.shape[:2]
    h_max = max(h1, h2)

    # Image composite
    canvas = np.zeros((h_max, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1_rgb
    canvas[:h2, w1:w1 + w2] = img2_rgb

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(canvas)
    ax.set_axis_off()

    n = len(pts1)
    if n == 0:
        if title:
            ax.set_title(title)
        return fig

    # Sous-echantillonnage si trop de correspondances
    if n > max_display:
        indices = np.random.choice(n, max_display, replace=False)
    else:
        indices = np.arange(n)

    for idx in indices:
        x1, y1 = pts1[idx]
        x2, y2 = pts2[idx]
        x2_shifted = x2 + w1

        if inlier_mask is not None:
            if inlier_mask[idx]:
                color = (0, 0.8, 0)  # vert pour les inliers
                alpha = 0.7
                lw = 0.8
            else:
                color = (0.9, 0, 0)  # rouge pour les aberrants
                alpha = 0.3
                lw = 0.4
        else:
            color = (0.2, 0.5, 1.0)  # bleu
            alpha = 0.5
            lw = 0.6

        ax.plot([x1, x2_shifted], [y1, y2], '-', color=color, alpha=alpha, linewidth=lw)
        ax.plot(x1, y1, '.', color=color, markersize=3, alpha=alpha)
        ax.plot(x2_shifted, y2, '.', color=color, markersize=3, alpha=alpha)

    if title:
        ax.set_title(title, fontsize=14)

    fig.tight_layout()
    return fig


def warp_and_blend(
    img1: np.ndarray,
    img2: np.ndarray,
    H: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Deforme img1 dans le repere de img2 en utilisant H et effectue un melange alpha.

    Paramètres
    ----------
    img1 : image source
    img2 : image destination
    H : (3, 3) homographie projetant img1 -> img2
    alpha : poids du melange (0 = img2 seule, 1 = img1 deformee seule)

    Retourne
    -------
    blended : image RGB (tableau numpy)
    """
    h2, w2 = img2.shape[:2]
    warped = cv2.warpPerspective(img1, H, (w2, h2))

    # Conversion en flottant pour le melange
    warped_f = warped.astype(np.float64)
    img2_f = img2.astype(np.float64)

    # Melange uniquement la ou l'image deformee a du contenu
    mask = (warped.sum(axis=2) > 0) if len(warped.shape) == 3 else (warped > 0)
    blended = img2_f.copy()
    if len(blended.shape) == 3:
        blended[mask] = alpha * warped_f[mask] + (1 - alpha) * img2_f[mask]
    else:
        blended[mask] = alpha * warped_f[mask] + (1 - alpha) * img2_f[mask]

    return blended.astype(np.uint8)


def plot_nfa_curve(
    sorted_errors: np.ndarray,
    log_nfas: np.ndarray,
    best_k: int,
    title: str = "NFA en fonction du nombre d'inliers",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Trace log10(NFA) en fonction de k (nombre d'inliers).

    Paramètres
    ----------
    sorted_errors : (n,) residus tries par ordre croissant
    log_nfas : (n,) log10(NFA) pour chaque k (inf pour les k invalides)
    best_k : nombre optimal d'inliers
    title : titre de la figure
    figsize : taille de la figure

    Retourne
    -------
    fig : Figure matplotlib
    """
    fig, ax1 = plt.subplots(1, 1, figsize=figsize)

    n = len(sorted_errors)
    k_values = np.arange(1, n + 1)

    # Filtrage des valeurs NFA valides
    valid = np.isfinite(log_nfas)

    # Trace du log_NFA
    ax1.plot(k_values[valid], log_nfas[valid], 'b-', linewidth=1.5, label='log$_{10}$(NFA)')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Seuil NFA = 1')
    if best_k > 0:
        best_idx = best_k - 1
        if best_idx < n and valid[best_idx]:
            ax1.axvline(x=best_k, color='r', linestyle='--', alpha=0.7,
                        label=f'Meilleur k = {best_k}')
            ax1.plot(best_k, log_nfas[best_idx], 'ro', markersize=8,
                     label=f'min log$_{{10}}$(NFA) = {log_nfas[best_idx]:.1f}')

    ax1.set_xlabel("Nombre d'inliers (k)", fontsize=12)
    ax1.set_ylabel('log$_{10}$(NFA)', fontsize=12, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.legend(loc='upper right')

    # Axe secondaire : erreurs triees
    ax2 = ax1.twinx()
    ax2.plot(k_values, np.sqrt(np.maximum(sorted_errors, 0)), 'g-', alpha=0.4, linewidth=1,
             label='Erreur (px)')
    ax2.set_ylabel("Seuil d'erreur (px)", fontsize=12, color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    ax1.set_title(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_error_histogram(
    errors: np.ndarray,
    inlier_mask: np.ndarray,
    epsilon: float,
    title: str = "Distribution des erreurs de reprojection",
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """Trace l'histogramme des erreurs de reprojection, avec coloration inlier/aberrant.

    Paramètres
    ----------
    errors : (n,) erreurs au carre pour toutes les correspondances
    inlier_mask : (n,) booleen
    epsilon : seuil en pixels
    title : titre de la figure
    figsize : taille de la figure

    Retourne
    -------
    fig : Figure matplotlib
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    distances = np.sqrt(np.maximum(errors, 0))
    max_dist = min(np.percentile(distances, 95), epsilon * 5)

    inlier_dists = distances[inlier_mask]
    outlier_dists = distances[~inlier_mask]

    bins = np.linspace(0, max_dist, 50)
    ax.hist(inlier_dists[inlier_dists <= max_dist], bins=bins, alpha=0.7,
            color='green', label=f'Inliers ({len(inlier_dists)})')
    ax.hist(outlier_dists[outlier_dists <= max_dist], bins=bins, alpha=0.5,
            color='red', label=f'Aberrants ({len(outlier_dists)})')
    ax.axvline(x=epsilon, color='blue', linestyle='--', linewidth=2,
               label=f'Seuil = {epsilon:.1f} px')

    ax.set_xlabel('Erreur de reprojection (px)', fontsize=12)
    ax.set_ylabel('Effectif', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_experiment_summary(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    result,
    sorted_errors: np.ndarray | None = None,
    log_nfas: np.ndarray | None = None,
    title: str = "",
    figsize: tuple = (20, 12),
) -> plt.Figure:
    """Resume multi-panneaux d'une expérience ORSA.

    Panneau 1 : Correspondances avec coloration inlier/aberrant
    Panneau 2 : Melange deforme (si H trouve)
    Panneau 3 : Courbe NFA (si donnees fournies)
    Panneau 4 : Histogramme des erreurs (si H trouve)
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Panneau 1 : Correspondances
    _draw_matches_on_ax(axes[0, 0], img1, img2, pts1, pts2, result.inlier_mask)
    axes[0, 0].set_title(
        f'Correspondances : {result.n_inliers} inliers / {result.n_matches} total\n'
        f'log$_{{10}}$(NFA) = {result.log_nfa:.1f}, '
        f'eps = {result.epsilon:.1f} px',
        fontsize=11,
    )

    # Panneau 2 : Melange deforme
    if result.H is not None:
        blended = warp_and_blend(img1, img2, result.H)
        if len(blended.shape) == 3:
            blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        else:
            blended_rgb = blended
        axes[0, 1].imshow(blended_rgb)
        axes[0, 1].set_title('Melange deforme', fontsize=11)
    else:
        axes[0, 1].text(0.5, 0.5, 'Aucune homographie trouvee', ha='center', va='center',
                        fontsize=14, transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Melange deforme (N/A)', fontsize=11)
    axes[0, 1].set_axis_off()

    # Panneau 3 : Courbe NFA
    if sorted_errors is not None and log_nfas is not None:
        n = len(sorted_errors)
        k_values = np.arange(1, n + 1)
        valid = np.isfinite(log_nfas)
        axes[1, 0].plot(k_values[valid], log_nfas[valid], 'b-', linewidth=1.5)
        axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        if result.n_inliers > 0:
            axes[1, 0].axvline(x=result.n_inliers, color='r', linestyle='--', alpha=0.7)
        axes[1, 0].set_xlabel('k')
        axes[1, 0].set_ylabel('log$_{10}$(NFA)')
        axes[1, 0].set_title('Courbe NFA', fontsize=11)
    else:
        axes[1, 0].text(0.5, 0.5, 'Courbe NFA non disponible', ha='center', va='center',
                        fontsize=14, transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Courbe NFA (N/A)', fontsize=11)

    # Panneau 4 : Histogramme des erreurs
    if result.H is not None:
        from .homography import symmetric_transfer_error
        all_errors, _ = symmetric_transfer_error(result.H, pts1, pts2)
        distances = np.sqrt(np.maximum(all_errors, 0))
        max_dist = min(np.percentile(distances, 95), result.epsilon * 5) if result.epsilon > 0 else np.percentile(distances, 95)
        bins = np.linspace(0, max(max_dist, 0.1), 50)
        inlier_d = distances[result.inlier_mask]
        outlier_d = distances[~result.inlier_mask]
        axes[1, 1].hist(inlier_d[inlier_d <= max_dist], bins=bins, alpha=0.7,
                        color='green', label='Inliers')
        axes[1, 1].hist(outlier_d[outlier_d <= max_dist], bins=bins, alpha=0.5,
                        color='red', label='Aberrants')
        axes[1, 1].axvline(x=result.epsilon, color='blue', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Erreur (px)')
        axes[1, 1].set_ylabel('Effectif')
        axes[1, 1].set_title('Distribution des erreurs', fontsize=11)
        axes[1, 1].legend(fontsize=9)
    else:
        axes[1, 1].text(0.5, 0.5, 'Pas de modèle', ha='center', va='center',
                        fontsize=14, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Distribution des erreurs (N/A)', fontsize=11)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_experiment_summary_slide(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    result,
    sorted_errors: np.ndarray | None = None,
    log_nfas: np.ndarray | None = None,
    title: str = "",
    figsize: tuple = (16, 5),
) -> plt.Figure:
    """Resume a deux panneaux pour les diapositives : correspondances (gauche) et courbe NFA (droite).

    Panneau 1 : Correspondances avec coloration inlier/aberrant
    Panneau 2 : Courbe NFA (log10 NFA vs. k)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Panneau 1 : Correspondances
    _draw_matches_on_ax(axes[0], img1, img2, pts1, pts2, result.inlier_mask)
    axes[0].set_title(
        f'Correspondances : {result.n_inliers} inliers / {result.n_matches} total\n'
        f'log$_{{10}}$(NFA) = {result.log_nfa:.1f},  '
        f'$\\varepsilon$ = {result.epsilon:.1f} px',
        fontsize=11,
    )

    # Panneau 2 : Courbe NFA
    if sorted_errors is not None and log_nfas is not None:
        n = len(sorted_errors)
        k_values = np.arange(1, n + 1)
        valid = np.isfinite(log_nfas)
        axes[1].plot(k_values[valid], log_nfas[valid], 'b-', linewidth=1.5)
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.7, label='NFA = 1')
        if result.n_inliers > 0:
            best_idx = result.n_inliers - 1
            axes[1].axvline(x=result.n_inliers, color='r', linestyle='--', alpha=0.8,
                            label=f'$k^*$ = {result.n_inliers}')
            if best_idx < len(log_nfas) and np.isfinite(log_nfas[best_idx]):
                axes[1].plot(result.n_inliers, log_nfas[best_idx], 'ro', markersize=7)
        axes[1].set_xlabel("Nombre d'inliers $k$", fontsize=11)
        axes[1].set_ylabel('log$_{10}$(NFA)', fontsize=11)
        axes[1].set_title('Courbe NFA', fontsize=11)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'Courbe NFA non disponible', ha='center', va='center',
                     fontsize=14, transform=axes[1].transAxes)
        axes[1].set_title('Courbe NFA (N/A)', fontsize=11)

    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_registration(img1, img2, H, figsize=(14, 5)):
    """Visualise le résultat du recalage sous forme de figure a 3 panneaux."""
    blended = warp_and_blend(img1, img2, H)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, img, title in zip(axes, [img1, img2, blended], ["Image 1", "Image 2", "Recalage"]):
        if len(img.shape) == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.set_axis_off()
    plt.tight_layout()
    return fig


def _draw_matches_on_ax(ax, img1, img2, pts1, pts2, inlier_mask, max_display=200):
    """Fonction auxiliaire pour dessiner les correspondances sur un axe matplotlib donne."""
    if len(img1.shape) == 2:
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    else:
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    if len(img2.shape) == 2:
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    else:
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    h1, w1 = img1_rgb.shape[:2]
    h2, w2 = img2_rgb.shape[:2]
    h_max = max(h1, h2)
    canvas = np.zeros((h_max, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1_rgb
    canvas[:h2, w1:w1 + w2] = img2_rgb
    ax.imshow(canvas)
    ax.set_axis_off()

    n = len(pts1)
    if n == 0:
        return
    indices = np.random.choice(n, min(n, max_display), replace=False) if n > max_display else np.arange(n)
    for idx in indices:
        x1, y1 = pts1[idx]
        x2, y2 = pts2[idx]
        color = (0, 0.8, 0) if inlier_mask[idx] else (0.9, 0, 0)
        alpha = 0.7 if inlier_mask[idx] else 0.3
        lw = 0.8 if inlier_mask[idx] else 0.4
        ax.plot([x1, x2 + w1], [y1, y2], '-', color=color, alpha=alpha, linewidth=lw)
