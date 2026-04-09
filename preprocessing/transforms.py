"""
preprocessing/transforms.py
────────────────────────────
Toutes les transformations d'images du pipeline de prétraitement :
    - Redimensionnement     (section 7)
    - Normalisation         (section 8)
    - Filtrage              (section 9)
    - Augmentation          (section 10)

Ces fonctions opèrent sur des np.ndarray et sont indépendantes
de toute logique de visualisation ou de chargement de fichier.
"""

import random
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from scipy.ndimage import gaussian_filter

from config.settings import SEED, IMG_SIZE_CNN


# ─────────────────────────────────────────────────────────────────────────────
# A — REDIMENSIONNEMENT
# ─────────────────────────────────────────────────────────────────────────────

def resize_image(img: np.ndarray, target_size: tuple,
                 interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    """
    Redimensionne une image à target_size (H, W).

    Méthode d'interpolation choisie : INTER_LINEAR (bilinéaire)
        → meilleur compromis vitesse / qualité visuelle pour réduire la résolution.
        → INTER_AREA serait optimal pour un fort downscaling (256→64),
          mais INTER_LINEAR est suffisant pour 256→128.

    Paramètres :
        img         (np.ndarray) : image RGB uint8
        target_size (tuple)      : (H, W) cible
        interpolation            : méthode OpenCV

    Retourne :
        np.ndarray uint8 redimensionné
    """
    # cv2.resize attend (width, height), donc on inverse (H, W) → (W, H)
    return cv2.resize(img, (target_size[1], target_size[0]), interpolation=interpolation)


# ─────────────────────────────────────────────────────────────────────────────
# B — NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def normalize_minmax(img: np.ndarray) -> np.ndarray:
    """
    Normalisation min-max : pixel / 255.0
    Valeurs uint8 [0,255] → float32 [0.0, 1.0]
    Utilisée pour les CNN entraînés from scratch.
    """
    return img.astype(np.float32) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# C — FILTRAGE
# ─────────────────────────────────────────────────────────────────────────────

def apply_gaussian_filter(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Filtre gaussien canal par canal.
    Lisse le bruit léger sans trop dégrader les contours.
    sigma=1.0 : lissage minimal adapté à PlantVillage (images propres).
    """
    out = np.zeros_like(img, dtype=np.float32)
    for ch in range(img.shape[2]):
        out[:, :, ch] = gaussian_filter(img[:, :, ch].astype(np.float32), sigma=sigma)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_median_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Filtre médian — élimine le bruit impulsionnel (sel & poivre).
    ksize=3 : noyau 3×3, préserve les détails fins des lésions.
    """
    return cv2.medianBlur(img, ksize)


def apply_bilateral_filter(img: np.ndarray,
                            d: int = 9, sc: float = 75, ss: float = 75) -> np.ndarray:
    """
    Filtre bilatéral — lisse ET préserve les contours des zones malades.
    Paramètres :
        d  : diamètre du voisinage (9 = bon équilibre vitesse/qualité)
        sc : sigma_color — sensibilité aux différences de couleur
        ss : sigma_space — portée spatiale
    """
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    res_bgr = cv2.bilateralFilter(img_bgr, d, sc, ss)
    return cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)


def apply_sharpening(img: np.ndarray) -> np.ndarray:
    """
    Renforcement des contours par filtre Laplacien.
    Améliore la netteté des bords des lésions (utile pour CNN).
    Noyau utilisé :
        [[ 0, -1,  0],
         [-1,  5, -1],
         [ 0, -1,  0]]
    """
    kernel    = np.array([[0, -1,  0],
                          [-1, 5, -1],
                          [0, -1,  0]], dtype=np.float32)
    sharpened = cv2.filter2D(img, -1, kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def compute_psnr(original: np.ndarray, filtered: np.ndarray) -> float:
    """
    Calcule le PSNR (Peak Signal-to-Noise Ratio) en dB.
    Mesure la qualité de la reconstruction par rapport à l'original.
    Plus le PSNR est élevé, plus le filtre préserve l'image originale.
    """
    mse = np.mean((original.astype(float) - filtered.astype(float)) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# D — AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def augment_image(img: np.ndarray, seed_offset: int = 0) -> dict:
    """
    Applique 8 transformations d'augmentation sur une image réelle.

    Chaque transformation simule une variation naturelle du terrain :
        flip_h      → feuille orientée à gauche ou à droite
        flip_v      → image prise à l'envers
        rotation    → inclinaison de la feuille ou de l'appareil
        zoom_in     → caméra plus proche de la feuille
        zoom_out    → caméra plus loin
        brightness  → ensoleillement / ombre
        contrast    → différences de qualité d'appareil
        noise       → bruit de capteur (smartphone bas de gamme)

    Paramètres :
        img         (np.ndarray) : image uint8 RGB 128×128
        seed_offset (int)        : pour varier les paramètres aléatoires

    Retourne :
        dict {nom: image_transformée}
    """
    random.seed(SEED + seed_offset)
    np.random.seed(SEED + seed_offset)

    H, W   = img.shape[:2]
    result = {}

    # ── 1. Flip horizontal (symétrie gauche-droite) ───────────────────────────
    result["Flip Horizontal"] = cv2.flip(img, 1)

    # ── 2. Flip vertical (symétrie haut-bas) ──────────────────────────────────
    result["Flip Vertical"] = cv2.flip(img, 0)

    # ── 3. Rotation aléatoire ±30° ────────────────────────────────────────────
    angle  = random.uniform(-30, 30)
    center = (W // 2, H // 2)
    M_rot  = cv2.getRotationMatrix2D(center, angle, 1.0)
    # BORDER_REFLECT_101 : remplissage miroir (pas de bords noirs)
    result[f"Rotation ({angle:+.0f}°)"] = cv2.warpAffine(
        img, M_rot, (W, H), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    # ── 4. Zoom In (recadrage central + redimensionnement) ────────────────────
    zoom   = 0.85   # on garde 85% du centre → effet zoom +15%
    cy, cx = H // 2, W // 2
    h_crop = int(H * zoom / 2)
    w_crop = int(W * zoom / 2)
    cropped = img[cy - h_crop: cy + h_crop, cx - w_crop: cx + w_crop]
    result["Zoom In (+15%)"] = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LINEAR)

    # ── 5. Zoom Out (padding + redimensionnement) ─────────────────────────────
    pad        = int(min(H, W) * 0.12)
    img_padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    result["Zoom Out (-12%)"] = cv2.resize(img_padded, (W, H), interpolation=cv2.INTER_LINEAR)

    # ── 6. Variation de luminosité ────────────────────────────────────────────
    factor = random.uniform(0.55, 1.50)   # 0.55 = très sombre, 1.5 = très clair
    pil    = Image.fromarray(img)
    result[f"Luminosité (×{factor:.2f})"] = np.array(
        ImageEnhance.Brightness(pil).enhance(factor)
    )

    # ── 7. Variation de contraste ─────────────────────────────────────────────
    factor2 = random.uniform(0.60, 1.50)
    result[f"Contraste (×{factor2:.2f})"] = np.array(
        ImageEnhance.Contrast(pil).enhance(factor2)
    )

    # ── 8. Bruit gaussien (σ=8) ────────────────────────────────────────────────
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    result["Bruit Gaussien (σ=8)"] = np.clip(
        img.astype(np.int16) + noise, 0, 255
    ).astype(np.uint8)

    return result
