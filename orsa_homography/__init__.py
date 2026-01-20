"""Recalage homographique a-contrario avec ORSA."""

from .homography import dlt_homography, fit_homography_dlt, normalize_points, symmetric_transfer_error
from .orsa import orsa_homography, OrsaResult
from .matching import detect_and_match
from .visualization import draw_matches
from .utils import generate_synthetic_pair
from .visualization import warp_and_blend, plot_registration
