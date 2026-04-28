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
# C — SHARPENING (renforcement de netteté)
# ─────────────────────────────────────────────────────────────────────────────
#
# Le dataset PlantVillage est acquis en conditions contrôlées (0 image corrompue),
# donc aucun débruitage n'est nécessaire — un gaussien lisserait justement les
# contours des lésions/nécroses qu'on cherche à détecter.
# À la place, on applique un sharpening léger pour compenser le flou introduit
# par le redimensionnement bilinéaire 256×256 → 128×128.

SHARPEN_KERNEL = np.array([[0, -1,  0],
                           [-1, 5, -1],
                           [0, -1,  0]], dtype=np.float32)


def apply_sharpening(img: np.ndarray) -> np.ndarray:
    """
    Sharpening léger via convolution 2D.
    Compense le flou introduit par le downscale 256→128 et accentue les
    contours des taches et nécroses.

    Accepte :
        - uint8  [0, 255]  → retourne uint8  [0, 255]
        - float  [0, 1]    → retourne float32 [0, 1]
        - float  [0, 255]  → retourne float32 [0, 255]
    """
    sharpened = cv2.filter2D(img, -1, SHARPEN_KERNEL)
    if img.dtype == np.uint8:
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    high = 1.0 if img.max() <= 1.0 else 255.0
    return np.clip(sharpened, 0, high).astype(np.float32)


def compute_psnr(original: np.ndarray, filtered: np.ndarray) -> float:
    """
    PSNR (Peak Signal-to-Noise Ratio) en dB. Mesure la déviation par rapport
    à l'image source — utile pour quantifier l'impact du sharpening.
    """
    mse = np.mean((original.astype(float) - filtered.astype(float)) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# D — AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def augment_image(img: np.ndarray, seed_offset: int = 0) -> dict:
    """
    Applique les 5 transformations d'augmentation retenues pour PlantVillage.

    Transformations conservées (cohérentes avec les générateurs Keras) :
        flip_h      → feuille orientée à gauche ou à droite
        rotation    → inclinaison de la feuille ou de l'appareil (±30°)
        zoom        → caméra plus proche / plus loin (±20%)
        brightness  → ensoleillement / ombre ([0.65, 1.35])
        contrast    → différences de qualité d'appareil ([0.60, 1.50])

    Transformations supprimées :
        flip_v   → non pertinent (les feuilles sont toujours photographiées
                   dans une orientation cohérente, pas à l'envers)
        noise    → non pertinent (PlantVillage est en conditions contrôlées,
                   ajouter du bruit risque de masquer les lésions à détecter)

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

    # ── 2. Rotation aléatoire ±30° ────────────────────────────────────────────
    angle  = random.uniform(-30, 30)
    center = (W // 2, H // 2)
    M_rot  = cv2.getRotationMatrix2D(center, angle, 1.0)
    result[f"Rotation ({angle:+.0f}°)"] = cv2.warpAffine(
        img, M_rot, (W, H), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    # ── 3. Zoom In (recadrage central, ~+20%) ────────────────────────────────
    zoom   = 0.80                          # 80% du centre → effet zoom +20%
    cy, cx = H // 2, W // 2
    h_crop = int(H * zoom / 2)
    w_crop = int(W * zoom / 2)
    cropped = img[cy - h_crop: cy + h_crop, cx - w_crop: cx + w_crop]
    result["Zoom In (+20%)"] = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LINEAR)

    # ── 4. Zoom Out (padding miroir, ~-20%) ──────────────────────────────────
    pad        = int(min(H, W) * 0.20)
    img_padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    result["Zoom Out (-20%)"] = cv2.resize(img_padded, (W, H), interpolation=cv2.INTER_LINEAR)

    # ── 5. Variation de luminosité [0.65, 1.35] ──────────────────────────────
    factor = random.uniform(0.65, 1.35)
    pil    = Image.fromarray(img)
    result[f"Luminosité (×{factor:.2f})"] = np.array(
        ImageEnhance.Brightness(pil).enhance(factor)
    )

    # ── 6. Variation de contraste [0.60, 1.50] ───────────────────────────────
    factor2 = random.uniform(0.60, 1.50)
    result[f"Contraste (×{factor2:.2f})"] = np.array(
        ImageEnhance.Contrast(pil).enhance(factor2)
    )

    return result
