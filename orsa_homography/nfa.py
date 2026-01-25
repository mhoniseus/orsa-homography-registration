"""
Calcul a-contrario du NFA (Nombre de Fausses Alarmes) pour ORSA.

Tous les calculs sont effectues en espace log10 pour eviter les depassements
de capacite avec les grands coefficients binomiaux et les petites probabilites
elevees a de grandes puissances.

Référence : Moisan, Moulon, Monasse -- "Automatic Homographic Registration
of a Pair of Images, with A Contrario Elimination of Outliers", IPOL 2012.
"""

import numpy as np


def log10_combi(n: int, k: int) -> float:
    """Calcule log10(C(n, k)) de maniere iterative pour la stabilité numérique.

    Utilise l'identite : C(n, k) = prod_{i=1}^{k} (n - i + 1) / i
    donc log10(C(n, k)) = sum_{i=1}^{k} [log10(n - i + 1) - log10(i)].
    """
    if k < 0 or k > n:
        return -np.inf
    if k == 0 or k == n:
        return 0.0
    # on utilise le plus petit entre k et n-k car C(n,k) = C(n,n-k) et cela economise des itérations
    if k > n - k:
        k = n - k
    # on somme les logarithmes au lieu de multiplier directement car les coefficients
    # binomiaux bruts deviennent astronomiquement grands et depasseraient la capacite
    result = 0.0
    for i in range(1, k + 1):
        result += np.log10(n - i + 1) - np.log10(i)
    return result


def precompute_log_combi_n(n: int) -> np.ndarray:
    """Tabule log10(C(n, k)) pour k = 0, 1, ..., n.

    Retourne un tableau de longueur n + 1 ou out[k] = log10(C(n, k)).
    """
    # on precalcule tout cela a l'avance pour que la boucle interne du NFA reste rapide
    # on utilise la recurrence C(n,k) = C(n,k-1) * (n-k+1)/k
    table = np.zeros(n + 1)
    for k in range(1, n + 1):
        table[k] = table[k - 1] + np.log10(n - k + 1) - np.log10(k)
    return table


def precompute_log_combi_k(k: int, n_max: int) -> np.ndarray:
    """Tabule log10(C(m, k)) pour m = 0, 1, ..., n_max.

    Retourne un tableau de longueur n_max + 1 ou out[m] = log10(C(m, k)).
    Les entrees pour m < k sont mises a 0 (log10(0) serait -inf, mais celles-ci
    ne sont jamais utilisees dans la boucle NFA).
    """
    table = np.zeros(n_max + 1)
    if k > n_max:
        return table
    # C(k, k) = 1
    table[k] = 0.0
    for m in range(k + 1, n_max + 1):
        # C(m, k) = C(m-1, k) * m / (m - k)
        table[m] = table[m - 1] + np.log10(m) - np.log10(m - k)
    return table


def compute_best_nfa(
    sorted_errors: np.ndarray,
    sorted_sides: np.ndarray,
    logalpha0: np.ndarray,
    n_data: int,
    sample_size: int,
    n_outcomes: int,
    log_combi_n: np.ndarray,
    log_combi_k: np.ndarray,
    mult_error: float = 1.0,
    max_threshold: float = np.inf,
) -> tuple[float, int, float]:
    """Calcul du NFA central avec epsilon adaptatif.

    Pour chaque nombre candidat d'inliers k (de sample_size+1 a n_data),
    on utilise le k-ieme residu trie comme seuil de précision et on calcule
    le NFA. On retourne le minimum.

    Paramètres
    ----------
    sorted_errors : (n_data,) residus tries par ordre croissant (distances au carre)
    sorted_sides : (n_data,) int, 0=image gauche domine, 1=droite
    logalpha0 : (2,) log10(pi / (w*h)) pour les images [gauche, droite]
    n_data : nombre total de correspondances
    sample_size : taille minimale de l'echantillon (4 pour l'homographie)
    n_outcomes : nombre de modèles par echantillon (1 pour l'homographie)
    log_combi_n : log10(C(n_data, k)) precalcule pour k=0..n
    log_combi_k : log10(C(m, sample_size)) precalcule pour m=0..n_data
    mult_error : multiplicateur pour log10(erreur) ; 0.5 pour les distances au carre
    max_threshold : seuil d'erreur maximal autorise

    Retourne
    --------
    best_log_nfa : log10(NFA) du meilleur (minimum) NFA trouve
    best_k : nombre d'inliers a l'optimum
    best_error : seuil d'erreur au carre a l'optimum
    """
    eps_machine = np.finfo(float).eps

    # c'est la correction du nombre de tests de l'eq 3 de l'article
    n_minus_p = n_data - sample_size
    if n_minus_p <= 0:
        return 0.0, 0, 0.0
    loge0 = np.log10(n_outcomes * n_minus_p)

    best_log_nfa = np.inf
    best_k = 0
    best_error = 0.0

    # on essaie chaque nombre possible d'inliers k et on choisit celui qui donne
    # le plus petit NFA -- c'est l'idee cle de l'a-contrario, on laisse les
    # donnees choisir le seuil au lieu de le fixer nous-mêmes
    for i in range(sample_size, n_data):
        error_i = sorted_errors[i]

        # on saute les erreurs au-dela du carre de la diagonale de l'image car ces
        # correspondances sont physiquement impossibles et casseraient le modèle
        if error_i > max_threshold:
            break

        # alpha est la probabilite qu'une correspondance aléatoire ait une erreur <= error_i
        # on le fait par cote car les deux images peuvent avoir des tailles différentes
        side_i = int(sorted_sides[i])
        logalpha = logalpha0[side_i] + mult_error * np.log10(error_i + eps_machine)

        # alpha ne peut pas depasser 1 donc on le tronque
        if logalpha > 0.0:
            logalpha = 0.0

        k = i + 1
        j = k - sample_size

        # c'est l'eq 3 de l'article IPOL
        # NFA = (n-4) * C(n,k) * C(k,4) * alpha^(k-4)
        # on utilise C(n,k)*C(k,4) et NON C(n-4,k-4)*C(k,4) qui serait faux
        log_nfa = (
            loge0
            + logalpha * j
            + log_combi_n[k]       # log10(C(n, k))
            + log_combi_k[k]       # log10(C(k, p))
        )

        if log_nfa < best_log_nfa:
            best_log_nfa = log_nfa
            best_k = i + 1
            best_error = error_i

    if best_log_nfa == np.inf:
        return 0.0, 0, 0.0

    return best_log_nfa, best_k, best_error


def compute_nfa_for_all_k(
    sorted_errors: np.ndarray,
    sorted_sides: np.ndarray,
    logalpha0: np.ndarray,
    n_data: int,
    sample_size: int,
    n_outcomes: int,
    log_combi_n: np.ndarray,
    log_combi_k: np.ndarray,
    mult_error: float = 1.0,
) -> np.ndarray:
    """Calcule log10(NFA) pour chaque k de sample_size+1 a n_data.

    Meme logique que compute_best_nfa mais on garde toutes les valeurs au lieu
    du minimum seul, afin de pouvoir tracer la courbe NFA complete et voir
    ou se situe le minimum.
    """
    eps_machine = np.finfo(float).eps
    n_minus_p = n_data - sample_size
    if n_minus_p <= 0:
        return np.full(n_data, np.inf)

    loge0 = np.log10(n_outcomes * n_minus_p)
    log_nfas = np.full(n_data, np.inf)

    for i in range(sample_size, n_data):
        error_i = sorted_errors[i]
        side_i = int(sorted_sides[i])
        logalpha = logalpha0[side_i] + mult_error * np.log10(error_i + eps_machine)
        if logalpha > 0.0:
            logalpha = 0.0
        k = i + 1
        j = k - sample_size
        log_nfas[i] = (
            loge0
            + logalpha * j
            + log_combi_n[k]       # log10(C(n, k))
            + log_combi_k[k]       # log10(C(k, p))
        )

    return log_nfas
