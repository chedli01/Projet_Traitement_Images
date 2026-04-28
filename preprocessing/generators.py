"""
preprocessing/generators.py
────────────────────────────
Construction des générateurs Keras (ImageDataGenerator) pour l'entraînement.
L'augmentation est appliquée en temps réel (on-the-fly) sur le train set.
Correspond à la section 12 du pipeline original.
Prêt pour l'entraînement au CR3.
"""

from pathlib import Path
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config.settings import IMG_SIZE_CNN, BATCH_SIZE, SEED


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING_FUNCTION — Sharpening léger (étape ③ du pipeline)
# ─────────────────────────────────────────────────────────────────────────────
#
# Ordre Keras : load → resize → augment → preprocessing_function → rescale.
# Donc à ce stade l'image est en float32 dans [0, 255] (rescale n'a pas encore
# été appliqué). On fait le sharpening ici pour qu'il s'applique aux deux
# splits (train ET val) — c'est une transformation déterministe qui fait
# partie du pipeline, pas une augmentation.

_SHARPEN_KERNEL = np.array([[0, -1,  0],
                            [-1, 5, -1],
                            [0, -1,  0]], dtype=np.float32)


def _sharpen(img: np.ndarray) -> np.ndarray:
    """Sharpening léger : convolution + clip dans [0, 255]."""
    sharpened = cv2.filter2D(img.astype(np.float32), -1, _SHARPEN_KERNEL)
    return np.clip(sharpened, 0.0, 255.0)


def _train_preprocess(img: np.ndarray) -> np.ndarray:
    """Train : variation de contraste aléatoire [0.60, 1.50] + sharpening."""
    factor   = np.random.uniform(0.60, 1.50)
    mean     = img.mean()
    contrast = np.clip((img - mean) * factor + mean, 0.0, 255.0)
    return _sharpen(contrast)


def _val_preprocess(img: np.ndarray) -> np.ndarray:
    """Val : sharpening seul (aucune augmentation aléatoire)."""
    return _sharpen(img)


# ─────────────────────────────────────────────────────────────────────────────
# GÉNÉRATEURS KERAS
# ─────────────────────────────────────────────────────────────────────────────

def build_keras_generators(color_dir: Path):
    """
    Construit les générateurs Keras pour l'entraînement futur (CR3).

    Pipeline appliqué par les générateurs (par ordre d'exécution) :
        ① Resize bilinéaire 256×256 → 128×128         (target_size)
        ② Augmentation aléatoire (train uniquement)   (rotation, flip-h, zoom,
                                                       luminosité, contraste)
        ③ Sharpening léger                            (preprocessing_function)
        ④ Normalisation /255 → float32 [0, 1]         (rescale)

    Split utilisé : 85% train / 15% val (via validation_split Keras).

    Paramètres :
        color_dir (Path) : dossier 'color' du dataset PlantVillage

    Retourne :
        train_gen    : DirectoryIterator avec augmentation
        val_gen      : DirectoryIterator sans augmentation
        class_names  : liste des 38 classes dans l'ordre des indices
    """
    print("\n" + "─"*60)
    print("  GÉNÉRATEURS KERAS — Prêts pour l'entraînement (CR3)")
    print("─"*60)

    # ── Générateur entraînement (avec augmentation) ───────────────────────────
    # Augmentation pertinente pour PlantVillage uniquement :
    #   - flip_h     : symétrie gauche/droite OK (feuilles non orientées)
    #   - rotation   : ±30° (inclinaison appareil)
    #   - zoom       : ±20% (distance caméra)
    #   - brightness : [0.65, 1.35] (ensoleillement/ombre)
    #   - contrast   : [0.60, 1.50] géré dans preprocessing_function ci-dessous
    # Supprimés : vertical_flip (feuilles jamais à l'envers) et bruit gaussien
    # (dataset en conditions contrôlées, le bruit masquerait les lésions).
    train_datagen = ImageDataGenerator(
        preprocessing_function=_train_preprocess,
        rescale=1.0 / 255,
        rotation_range=30,
        zoom_range=0.20,
        horizontal_flip=True,
        brightness_range=[0.65, 1.35],
        fill_mode="reflect",
        validation_split=0.15,
    )

    # ── Générateur validation/test (SANS augmentation, AVEC sharpening) ──────
    val_datagen = ImageDataGenerator(
        preprocessing_function=_val_preprocess,
        rescale=1.0 / 255,
        validation_split=0.15,
    )

    # ── Flux d'images depuis le dossier ───────────────────────────────────────
    train_gen = train_datagen.flow_from_directory(
        directory=str(color_dir),
        target_size=IMG_SIZE_CNN,      # 128×128
        batch_size=BATCH_SIZE,
        class_mode="categorical",      # one-hot encoding, 38 classes
        subset="training",             # 85% des données
        shuffle=True,
        seed=SEED,
    )

    val_gen = val_datagen.flow_from_directory(
        directory=str(color_dir),
        target_size=IMG_SIZE_CNN,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",           # 15% des données
        shuffle=False,
        seed=SEED,
    )

    class_names = list(train_gen.class_indices.keys())

    print(f"  ✅ Train      : {train_gen.samples:>6} images")
    print(f"  ✅ Validation : {val_gen.samples:>6} images")
    print(f"  ✅ Classes    : {len(class_names)}")
    print(f"  ✅ Batch size : {BATCH_SIZE}")
    print(f"  ✅ Taille     : {IMG_SIZE_CNN}")

    # Exemple de forme d'un batch (pour vérification)
    x_batch, y_batch = next(train_gen)
    print(f"\n  Forme d'un batch train :")
    print(f"    x (images) : {x_batch.shape}   min={x_batch.min():.3f}, max={x_batch.max():.3f}")
    print(f"    y (labels) : {y_batch.shape}   (one-hot sur {len(class_names)} classes)")

    return train_gen, val_gen, class_names
