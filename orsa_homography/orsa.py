"""
ORSA (Optimized Random SAmpling) pour l'estimation d'homographie.

Implémente la variante a-contrario de RANSAC de :
Moisan, Moulon, Monasse -- "Automatic Homographic Registration of a Pair
of Images, with A Contrario Elimination of Outliers", IPOL 2012.

La difference cle avec RANSAC : au lieu d'un seuil fixe pour les inliers,
ORSA sélectionné adaptativement le seuil qui minimise le NFA (Nombre de
Fausses Alarmes). Une détection est significative quand NFA < 1.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from .nfa import (
    compute_best_nfa,
    compute_nfa_for_all_k,
    precompute_log_combi_k,
    precompute_log_combi_n,
)
from .homography import (
    fit_homography_dlt,
    refine_homography,
    symmetric_transfer_error,
)
from .degeneracy import (
    check_collinearity,
    check_orientation_preserving,
    check_valid_warp,
)


@dataclass
class OrsaResult:
    """Resultat de l'estimation d'homographie par ORSA."""
    H: np.ndarray | None           # meilleure homographie 3x3 (None si pas de détection)
    inlier_mask: np.ndarray        # masque booleen (n,)
    nfa: float                     # valeur NFA (10^log_nfa)
    log_nfa: float                 # log10(NFA)
    n_inliers: int                 # nombre d'inliers
    epsilon: float                 # seuil d'erreur optimal (distance en pixels)
    n_iterations: int              # itérations effectuées
    n_models_tested: int           # modèles valides evalues (non rejetes par la dégénérescence)
    runtime: float                 # duree en secondes
    n_matches: int                 # total des correspondances en entrée
    reprojection_errors: np.ndarray | None = None  # erreurs sur les inliers
    log_nfa_history: list = field(default_factory=list)  # meilleur log_nfa au fil des itérations
    raw_log_nfa: float = 0.0  # log10(NFA) avant troncature (conserve même sans détection)


def orsa_homography(
    pts1: np.ndarray,
    pts2: np.ndarray,
    img1_shape: tuple,
    img2_shape: tuple,
    max_iter: int = 1000,
    confidence: float = 0.99,
    max_threshold: float | None = None,
    seed: int | None = None,
    verbose: bool = False,
) -> OrsaResult:
    """Boucle principale ORSA pour l'estimation d'homographie.

    Paramètres
    ----------
    pts1 : (n, 2) correspondances dans l'image 1
    pts2 : (n, 2) correspondances dans l'image 2
    img1_shape : (h1, w1) de l'image 1
    img2_shape : (h2, w2) de l'image 2
    max_iter : nombre maximal d'échantillons aléatoires
    confidence : probabilite de trouver le bon modèle (pour l'arret adaptatif)
    max_threshold : seuil maximal d'erreur au carre autorise. Si None,
        par defaut le carre de la diagonale de la plus grande image. Cela empeche
        de compter comme "inliers" les correspondances avec des erreurs plus grandes
        que l'image, ce qui violerait l'hypothese d'independance du NFA.
    seed : graine aléatoire pour la reproductibilité
    verbose : afficher la progression

    Retourne
    --------
    OrsaResult avec la meilleure homographie, les inliers, le NFA et les diagnostics.
    """
    t_start = time.time()
    n = len(pts1)
    sample_size = 4
    n_outcomes = 1  # un modèle par echantillon de 4 points

    # on borne l'erreur maximale au carre de la diagonale de l'image car toute correspondance
    # avec une erreur plus grande que la diagonale entiere ne peut pas être un inlier
    if max_threshold is None:
        h1, w1 = img1_shape[:2]
        h2, w2 = img2_shape[:2]
        diag = max(np.sqrt(w1**2 + h1**2), np.sqrt(w2**2 + h2**2))
        max_threshold = diag ** 2

    # Cas limite : pas assez de correspondances
    if n < sample_size + 1:
        return OrsaResult(
            H=None, inlier_mask=np.zeros(n, dtype=bool),
            nfa=1.0, log_nfa=0.0, n_inliers=0, epsilon=0.0,
            n_iterations=0, n_models_tested=0,
            runtime=time.time() - t_start, n_matches=n,
        )

    rng = np.random.default_rng(seed)

    # logalpha0 est log10(pi/(w*h)) pour chaque image
    # cela vient du modèle a-contrario : la probabilite qu'un point aléatoire
    # tombe a une distance epsilon d'un autre est pi*eps^2/(w*h)
    h1, w1 = img1_shape[:2]
    h2, w2 = img2_shape[:2]
    logalpha0 = np.array([
        np.log10(np.pi / (w1 * h1)),
        np.log10(np.pi / (w2 * h2)),
    ])

    # on precalcule ces tables une fois pour ne pas recalculer les coefficients
    # binomiaux a chaque itération
    log_combi_n = precompute_log_combi_n(n)
    log_combi_k = precompute_log_combi_k(sample_size, n)

    # on commence avec +inf pour que tout modèle trouve soit une amelioration
    # on suit le meilleur même si NFA > 1 car on en a besoin pour l'echantillonnage cible ensuite
    best_log_nfa = np.inf
    best_H = None
    best_inlier_mask = np.zeros(n, dtype=bool)
    best_epsilon = 0.0
    best_k = 0

    n_models_tested = 0
    n_iter = max_iter
    log_nfa_history = []

    # on reserve les derniers 10% des itérations pour l'echantillonnage cible
    # ou on ne tire que de l'ensemble d'inliers courant -- cela aide a
    # raffiner le modèle une fois qu'on a une idee approximative des inliers
    n_iter_reserve = max(1, max_iter // 10)
    n_iter_main = max_iter - n_iter_reserve

    actual_iter = 0
    for it in range(max_iter):
        actual_iter = it + 1

        # Arret adaptatif
        if it >= n_iter:
            break

        # Echantillonnage de 4 correspondances aléatoires
        if it < n_iter_main or best_H is None or best_k < sample_size:
            indices = rng.choice(n, size=sample_size, replace=False)
        else:
            # Echantillonnage cible depuis l'ensemble d'inliers courant
            inlier_indices = np.where(best_inlier_mask)[0]
            if len(inlier_indices) >= sample_size:
                indices = rng.choice(inlier_indices, size=sample_size, replace=False)
            else:
                indices = rng.choice(n, size=sample_size, replace=False)

        # on execute plusieurs controles de dégénérescence avant même de calculer les erreurs
        # cela economise du temps et evite de polluer le NFA avec des modèles aberrants
        if check_collinearity(pts1[indices]) or check_collinearity(pts2[indices]):
            continue

        H = fit_homography_dlt(pts1[indices], pts2[indices])
        if H is None:
            continue

        # on ne vérifie l'orientation que sur les 4 points de l'echantillon -- vérifier tous
        # les points serait du gaspillage car les correspondances ne preservant pas l'orientation
        # obtiennent simplement une erreur infinie dans l'erreur de transfert symétrique
        if not check_orientation_preserving(H, pts1, pts2, indices=indices):
            continue

        # ceci rejette les homographies qui reduisent ou dilatent l'image de facon aberrante
        if not check_valid_warp(H, img1_shape):
            continue

        n_models_tested += 1

        # Calcul de tous les residus
        errors, sides = symmetric_transfer_error(H, pts1, pts2)

        # Tri par erreur
        order = np.argsort(errors)
        sorted_errors = errors[order]
        sorted_sides = sides[order]

        # c'est ici que la magie a-contrario opere -- on laisse le NFA
        # trouver automatiquement le meilleur seuil au lieu d'en fixer un
        # on utilise mult_error=1.0 car nos erreurs sont des distances au carre
        # et alpha = pi*eps^2/(w*h) donc logalpha = logalpha0 + 1.0*log(erreur)
        log_nfa, k, err_at_k = compute_best_nfa(
            sorted_errors, sorted_sides, logalpha0,
            n, sample_size, n_outcomes,
            log_combi_n, log_combi_k,
            mult_error=1.0,
            max_threshold=max_threshold,
        )

        log_nfa_history.append(min(log_nfa, best_log_nfa if best_H is not None else log_nfa))

        # Mise a jour du meilleur modèle si amelioration
        if log_nfa < best_log_nfa:
            best_log_nfa = log_nfa
            best_H = H.copy()
            best_k = k
            best_epsilon = np.sqrt(max(err_at_k, 0.0))

            # Construction du masque d'inliers (les k premieres correspondances par erreur)
            inlier_indices_sorted = order[:k]
            best_inlier_mask = np.zeros(n, dtype=bool)
            best_inlier_mask[inlier_indices_sorted] = True

            # arret adaptatif standard RANSAC : une fois qu'on connait approximativement
            # le nombre d'inliers on peut estimer combien d'itérations sont necessaires
            # pour trouver un echantillon propre avec haute probabilite
            inlier_ratio = k / n
            if 0 < inlier_ratio < 1:
                p_all_inliers = inlier_ratio ** sample_size
                if p_all_inliers > 1e-15:
                    new_n_iter = int(np.ceil(
                        np.log(1 - confidence) / np.log(1 - p_all_inliers)
                    ))
                    n_iter = min(max_iter, max(it + 1, new_n_iter))

            if verbose:
                print(
                    f"  Iter {it} : log10(NFA) = {log_nfa:.2f}, "
                    f"inliers = {k}/{n}, eps = {best_epsilon:.2f} px"
                )

    # une fois l'echantillonnage aléatoire termine on re-estime H sur tous les inliers
    # cela ameliore souvent le NFA car la H initiale etait estimee sur seulement 4 points
    if best_H is not None and best_log_nfa < 0:
        best_H, best_inlier_mask, best_log_nfa, best_k, best_epsilon = _refine_until_convergence(
            best_H, best_inlier_mask, pts1, pts2,
            logalpha0, n, sample_size, n_outcomes,
            log_combi_n, log_combi_k,
            max_threshold, verbose,
        )

        # polissage final avec l'optimisation non lineaire de Levenberg-Marquardt
        H_refined = refine_homography(best_H, pts1, pts2, best_inlier_mask)
        if H_refined is not None:
            # on ne garde la H raffinee que si son NFA est au moins aussi bon
            errors_ref, sides_ref = symmetric_transfer_error(H_refined, pts1, pts2)
            order_ref = np.argsort(errors_ref)
            log_nfa_ref, k_ref, err_ref = compute_best_nfa(
                errors_ref[order_ref], sides_ref[order_ref], logalpha0,
                n, sample_size, n_outcomes,
                log_combi_n, log_combi_k,
                mult_error=1.0, max_threshold=max_threshold,
            )
            if log_nfa_ref <= best_log_nfa:
                best_H = H_refined
                best_log_nfa = log_nfa_ref
                best_k = k_ref
                best_epsilon = np.sqrt(max(err_ref, 0.0))
                inlier_indices_ref = order_ref[:k_ref]
                best_inlier_mask = np.zeros(n, dtype=bool)
                best_inlier_mask[inlier_indices_ref] = True

    # NFA < 1 signifie log_nfa < 0 ce qui signifie que la structure trouvee est
    # peu probable par hasard -- c'est tout l'interet de l'a-contrario
    is_meaningful = best_log_nfa < 0
    raw_log_nfa = best_log_nfa  # conserver avant troncature
    if not is_meaningful:
        best_H = None
        best_inlier_mask = np.zeros(n, dtype=bool)
        best_k = 0
        best_epsilon = 0.0
        best_log_nfa = 0.0  # NFA = 1

    # Calcul des erreurs de reprojection finales sur les inliers
    reproj_errors = None
    if best_H is not None and np.any(best_inlier_mask):
        all_errors, _ = symmetric_transfer_error(best_H, pts1, pts2)
        reproj_errors = np.sqrt(all_errors[best_inlier_mask])

    runtime = time.time() - t_start

    return OrsaResult(
        H=best_H,
        inlier_mask=best_inlier_mask,
        nfa=10 ** best_log_nfa if best_log_nfa < 300 else float('inf'),
        log_nfa=best_log_nfa,
        n_inliers=int(np.sum(best_inlier_mask)),
        epsilon=best_epsilon,
        n_iterations=actual_iter,
        n_models_tested=n_models_tested,
        runtime=runtime,
        n_matches=n,
        reprojection_errors=reproj_errors,
        log_nfa_history=log_nfa_history,
        raw_log_nfa=raw_log_nfa,
    )


def _refine_until_convergence(
    H, inlier_mask, pts1, pts2,
    logalpha0, n, sample_size, n_outcomes,
    log_combi_n, log_combi_k,
    max_threshold, verbose,
    max_refine_iter=10,
):
    """Re-estime iterativement H sur les inliers et recalcule le NFA jusqu'à convergence.

    Chaque re-estimation peut modifier l'ensemble d'inliers qui peut a son tour
    modifier la re-estimation et ainsi de suite.
    On s'arrete quand le NFA cesse de s'ameliorer.
    """
    best_log_nfa_ref = np.inf
    # Calcul du NFA initial
    errors, sides = symmetric_transfer_error(H, pts1, pts2)
    order = np.argsort(errors)
    current_log_nfa, current_k, current_err = compute_best_nfa(
        errors[order], sides[order], logalpha0,
        n, sample_size, n_outcomes,
        log_combi_n, log_combi_k,
        mult_error=1.0, max_threshold=max_threshold,
    )
    best_log_nfa_ref = current_log_nfa
    best_H = H.copy()
    best_mask = inlier_mask.copy()
    best_k = current_k
    best_eps = current_err

    for _ in range(max_refine_iter):
        # Re-estimation DLT sur les inliers courants
        inlier_pts1 = pts1[best_mask]
        inlier_pts2 = pts2[best_mask]

        if len(inlier_pts1) < 4:
            break

        H_refit = fit_homography_dlt(inlier_pts1, inlier_pts2)
        if H_refit is None:
            break

        # Recalcul des erreurs et du NFA
        errors_new, sides_new = symmetric_transfer_error(H_refit, pts1, pts2)
        order_new = np.argsort(errors_new)
        log_nfa_new, k_new, err_new = compute_best_nfa(
            errors_new[order_new], sides_new[order_new], logalpha0,
            n, sample_size, n_outcomes,
            log_combi_n, log_combi_k,
            mult_error=1.0, max_threshold=max_threshold,
        )

        if log_nfa_new < best_log_nfa_ref:
            best_log_nfa_ref = log_nfa_new
            best_H = H_refit.copy()
            best_k = k_new
            best_eps = err_new
            inlier_indices_new = order_new[:k_new]
            best_mask = np.zeros(n, dtype=bool)
            best_mask[inlier_indices_new] = True

            if verbose:
                print(
                    f"  Raffinement : log10(NFA) = {log_nfa_new:.2f}, "
                    f"inliers = {k_new}/{n}"
                )
        else:
            break  # Pas d'amelioration, arret

    return best_H, best_mask, best_log_nfa_ref, best_k, np.sqrt(max(best_eps, 0.0))
