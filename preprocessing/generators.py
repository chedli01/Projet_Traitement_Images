"""
preprocessing/generators.py
────────────────────────────
Construction des générateurs Keras (ImageDataGenerator) pour l'entraînement.
L'augmentation est appliquée en temps réel (on-the-fly) sur le train set.
Correspond à la section 12 du pipeline original.
Prêt pour l'entraînement au CR3.
"""

from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config.settings import IMG_SIZE_CNN, BATCH_SIZE, SEED


# ─────────────────────────────────────────────────────────────────────────────
# GÉNÉRATEURS KERAS
# ─────────────────────────────────────────────────────────────────────────────

def build_keras_generators(color_dir: Path):
    """
    Construit les générateurs Keras pour l'entraînement futur (CR3).
    L'augmentation est appliquée en temps réel (on-the-fly) sur le train set.

    Split utilisé : 85% train / 15% val (via validation_split Keras)

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
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,             # normalisation min-max → [0, 1]
        rotation_range=30,             # rotation ±30°
        width_shift_range=0.10,        # translation horizontale ±10%
        height_shift_range=0.10,       # translation verticale ±10%
        zoom_range=0.20,               # zoom ±20%
        horizontal_flip=True,          # flip horizontal
        vertical_flip=True,            # flip vertical
        brightness_range=[0.65, 1.35], # luminosité ±35%
        fill_mode="nearest",           # remplissage pixel voisin le plus proche
        validation_split=0.15,         # 15% réservé à la validation
    )

    # ── Générateur validation/test (SANS augmentation) ────────────────────────
    val_datagen = ImageDataGenerator(
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
