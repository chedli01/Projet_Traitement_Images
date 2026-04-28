"""
cr4_train.py
────────────
CR4 — Entraînement du CNN sur PlantVillage.

Pipeline complet :
    1. Vérifications (dataset, GPU)
    2. Indexation et split stratifié 70/15/15
    3. Générateurs Keras (sharpening + augmentation train)
    4. Calcul des class_weight pour déséquilibre 36.2×
    5. Construction du CNN (4 blocs convolutifs)
    6. Entraînement avec callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
    7. Sauvegarde de l'historique JSON et des courbes (loss, accuracy, lr, gap)

Exécution :
    python cr4_train.py
"""

import os
import json
import random
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Suppressions verbeuses TF ─────────────────────────────────────────────────
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from config.settings import IMG_SIZE_CNN, BATCH_SIZE, SEED, BASE_DIR, C
from preprocessing.generators import _train_preprocess, _val_preprocess
from model.architecture import build_cnn


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

OUTPUTS_DIR = Path("./outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH      = OUTPUTS_DIR / "best_model.keras"
HISTORY_JSON    = OUTPUTS_DIR / "history.json"
SPLITS_CSV      = OUTPUTS_DIR / "splits.csv"
CLASSES_JSON    = OUTPUTS_DIR / "class_indices.json"

EPOCHS_MAX      = 100
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15   # ⇒ TEST = 0.15
COLOR_DIRNAME   = "color"


# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCTIBILITÉ
# ─────────────────────────────────────────────────────────────────────────────

def set_global_seeds(seed: int = SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# 1. VÉRIFICATIONS
# ─────────────────────────────────────────────────────────────────────────────

def check_environment() -> Path:
    """Vérifie présence du dataset et liste les GPUs disponibles."""
    print("\n" + "═"*70)
    print("  CR4 — ENTRAÎNEMENT CNN PlantVillage")
    print("═"*70)

    # ── Dataset ──────────────────────────────────────────────────────────────
    color_dir = Path(BASE_DIR) / COLOR_DIRNAME
    if not color_dir.exists():
        # Tolérance : le ZIP a parfois été extrait avec un dossier intermédiaire
        candidates = list(Path(BASE_DIR).rglob(COLOR_DIRNAME))
        if candidates:
            color_dir = candidates[0]
        else:
            raise FileNotFoundError(
                f"Dataset introuvable. Attendu : {color_dir}\n"
                f"Lancez d'abord `python main.py` (CR2) pour extraire le ZIP."
            )
    print(f"  ✅ Dataset trouvé : {color_dir}")

    # ── GPU ──────────────────────────────────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        print(f"  ✅ GPU(s) disponibles : {len(gpus)}  ({gpus[0].name})")
    else:
        print("  ⚠️  Aucun GPU détecté — entraînement sur CPU (très lent).")

    print(f"  ✅ TensorFlow         : {tf.__version__}")
    return color_dir


# ─────────────────────────────────────────────────────────────────────────────
# 2. INDEXATION + SPLIT STRATIFIÉ 70/15/15
# ─────────────────────────────────────────────────────────────────────────────

def build_splits(color_dir: Path) -> tuple:
    """
    Indexe toutes les images, applique un split stratifié 70/15/15 et
    sauvegarde les splits sur disque (CSV) pour réutilisation par le CR5.
    """
    print("\n" + "─"*70)
    print("  Étape 1 — Indexation et split 70/15/15 (stratifié)")
    print("─"*70)

    records = []
    for class_dir in sorted(color_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                records.append({"filepath": str(img_path),
                                "class":    class_dir.name})

    df = pd.DataFrame(records)
    n_total = len(df)
    n_classes = df["class"].nunique()
    print(f"  Images totales : {n_total:,}")
    print(f"  Classes        : {n_classes}")

    # 70% train / 30% temp
    train_df, temp_df = train_test_split(
        df, train_size=TRAIN_RATIO,
        stratify=df["class"], random_state=SEED, shuffle=True,
    )
    # 30% temp → 50/50 → 15% val / 15% test
    val_df, test_df = train_test_split(
        temp_df, train_size=0.50,
        stratify=temp_df["class"], random_state=SEED, shuffle=True,
    )
    train_df["split"] = "train"
    val_df["split"]   = "val"
    test_df["split"]  = "test"
    splits = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    splits.to_csv(SPLITS_CSV, index=False)

    print(f"  ✅ Train : {len(train_df):>6} ({len(train_df)/n_total:.1%})")
    print(f"  ✅ Val   : {len(val_df):>6} ({len(val_df)/n_total:.1%})")
    print(f"  ✅ Test  : {len(test_df):>6} ({len(test_df)/n_total:.1%})")
    print(f"  → Splits sauvegardés : {SPLITS_CSV}")

    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# 3. GÉNÉRATEURS KERAS
# ─────────────────────────────────────────────────────────────────────────────

def build_generators(train_df, val_df, test_df):
    """
    Crée les 3 générateurs (train/val/test).
    - train : sharpening + augmentation (rotation, zoom, flip-h, brightness, contrast)
    - val/test : sharpening seul (aucune augmentation aléatoire)
    Tous appliquent /255.
    """
    print("\n" + "─"*70)
    print("  Étape 2 — Construction des générateurs Keras")
    print("─"*70)

    # ── Train (avec augmentation) ────────────────────────────────────────────
    train_datagen = ImageDataGenerator(
        preprocessing_function=_train_preprocess,   # sharpening + contraste
        rescale=1.0 / 255,
        rotation_range=30,
        zoom_range=0.20,
        horizontal_flip=True,
        brightness_range=[0.65, 1.35],
        fill_mode="reflect",
    )
    val_datagen = ImageDataGenerator(
        preprocessing_function=_val_preprocess,     # sharpening seul
        rescale=1.0 / 255,
    )

    common = dict(
        x_col="filepath", y_col="class",
        target_size=IMG_SIZE_CNN,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )

    train_gen = train_datagen.flow_from_dataframe(
        train_df, **common, shuffle=True, seed=SEED,
    )
    val_gen = val_datagen.flow_from_dataframe(
        val_df, **common, shuffle=False,
    )
    test_gen = val_datagen.flow_from_dataframe(
        test_df, **common, shuffle=False,
    )

    # Sauvegarde du mapping classe → index pour le CR5
    with open(CLASSES_JSON, "w", encoding="utf-8") as f:
        json.dump(train_gen.class_indices, f, indent=2, ensure_ascii=False)

    print(f"  ✅ Train : {train_gen.samples} images, {len(train_gen)} batches")
    print(f"  ✅ Val   : {val_gen.samples} images, {len(val_gen)} batches")
    print(f"  ✅ Test  : {test_gen.samples} images, {len(test_gen)} batches")
    print(f"  ✅ Batch shape : (32, 128, 128, 3) — labels one-hot 38 classes")

    return train_gen, val_gen, test_gen


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLASS WEIGHTS — gestion du déséquilibre
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(train_df, class_indices: dict) -> dict:
    """
    Class weights inversement proportionnels à la fréquence de chaque classe.
    Compense le déséquilibre 36.2× du dataset.
    """
    print("\n" + "─"*70)
    print("  Étape 3 — Calcul des class_weight")
    print("─"*70)

    y_int = train_df["class"].map(class_indices).values
    classes = np.arange(len(class_indices))
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_int,
    )
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}

    counts = Counter(train_df["class"])
    max_c, min_c = max(counts.values()), min(counts.values())
    print(f"  Classe la + représentée : {max_c} images")
    print(f"  Classe la - représentée : {min_c} images")
    print(f"  Ratio déséquilibre      : {max_c / min_c:.1f}×")
    print(f"  Poids min / max         : {min(weights):.3f} / {max(weights):.3f}")

    return class_weight_dict


# ─────────────────────────────────────────────────────────────────────────────
# 5. CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def build_callbacks() -> list:
    return [
        EarlyStopping(
            monitor="val_loss", patience=10,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5,
            min_lr=1e-6, verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 6. VISUALISATIONS POST-ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(history_dict: dict):
    """
    Génère 4 figures :
        - Loss train vs val
        - Accuracy train vs val
        - Learning rate
        - Gap train-val (overfitting)
    Ainsi qu'une figure résumé combinée.
    """
    print("\n" + "─"*70)
    print("  Étape 5 — Visualisation de l'entraînement")
    print("─"*70)

    epochs    = np.arange(1, len(history_dict["loss"]) + 1)
    loss_tr   = np.array(history_dict["loss"])
    loss_val  = np.array(history_dict["val_loss"])
    acc_tr    = np.array(history_dict["accuracy"])
    acc_val   = np.array(history_dict["val_accuracy"])
    lr_hist   = np.array(history_dict.get("lr", history_dict.get("learning_rate", [])))
    gap       = acc_tr - acc_val

    # ── Loss ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, loss_tr,  color=C["red"],    label="Train",      lw=2)
    ax.plot(epochs, loss_val, color=C["accent"], label="Validation", lw=2)
    ax.set_xlabel("Époque"); ax.set_ylabel("Loss (categorical_crossentropy)")
    ax.set_title("CR4 — Évolution de la Loss")
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(OUTPUTS_DIR / "cr4_loss_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── Accuracy ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, acc_tr,  color=C["red"],    label="Train",      lw=2)
    ax.plot(epochs, acc_val, color=C["accent"], label="Validation", lw=2)
    ax.set_xlabel("Époque"); ax.set_ylabel("Accuracy")
    ax.set_title("CR4 — Évolution de l'Accuracy")
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(OUTPUTS_DIR / "cr4_accuracy_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── Learning rate ────────────────────────────────────────────────────────
    if len(lr_hist) > 0:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(epochs, lr_hist, color=C["blue"], lw=2, marker="o", markersize=4)
        ax.set_xlabel("Époque"); ax.set_ylabel("Learning rate")
        ax.set_yscale("log")
        ax.set_title("CR4 — Évolution du Learning Rate (ReduceLROnPlateau)")
        ax.grid(alpha=0.3, which="both")
        fig.savefig(OUTPUTS_DIR / "cr4_learning_rate.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ── Gap train-val (overfitting) ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, gap, color=C["orange"], lw=2)
    ax.axhline(0, color=C["grey"], ls="--", lw=1)
    ax.fill_between(epochs, gap, 0, where=(gap > 0),
                    color=C["red"], alpha=0.15, label="Overfitting (train > val)")
    ax.fill_between(epochs, gap, 0, where=(gap <= 0),
                    color=C["accent"], alpha=0.15, label="Underfitting")
    ax.set_xlabel("Époque"); ax.set_ylabel("Accuracy(train) − Accuracy(val)")
    ax.set_title("CR4 — Gap Train/Val (détection d'overfitting)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(OUTPUTS_DIR / "cr4_train_val_gap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── Figure résumé ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(epochs, loss_tr,  color=C["red"], label="Train")
    axes[0, 0].plot(epochs, loss_val, color=C["accent"], label="Val")
    axes[0, 0].set_title("Loss"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)
    axes[0, 1].plot(epochs, acc_tr,  color=C["red"], label="Train")
    axes[0, 1].plot(epochs, acc_val, color=C["accent"], label="Val")
    axes[0, 1].set_title("Accuracy"); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)
    if len(lr_hist) > 0:
        axes[1, 0].plot(epochs, lr_hist, color=C["blue"])
        axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Learning rate"); axes[1, 0].grid(alpha=0.3, which="both")
    axes[1, 1].plot(epochs, gap, color=C["orange"])
    axes[1, 1].axhline(0, color=C["grey"], ls="--")
    axes[1, 1].set_title("Gap train − val"); axes[1, 1].grid(alpha=0.3)
    for ax in axes.flat:
        ax.set_xlabel("Époque")
    fig.suptitle("CR4 — Synthèse de l'entraînement", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "cr4_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  ✅ 5 figures sauvegardées dans : {OUTPUTS_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    set_global_seeds(SEED)
    color_dir = check_environment()

    train_df, val_df, test_df = build_splits(color_dir)
    train_gen, val_gen, _     = build_generators(train_df, val_df, test_df)
    class_weights             = compute_class_weights(train_df, train_gen.class_indices)

    # ── Construction du modèle ───────────────────────────────────────────────
    print("\n" + "─"*70)
    print("  Étape 4 — Construction du CNN (4 blocs convolutifs)")
    print("─"*70)
    model = build_cnn(
        input_shape=(*IMG_SIZE_CNN, 3),
        num_classes=len(train_gen.class_indices),
    )
    model.summary(line_length=80)

    # ── Entraînement ─────────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print(f"  Lancement de l'entraînement (max {EPOCHS_MAX} époques)")
    print("─"*70)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_MAX,
        callbacks=build_callbacks(),
        class_weight=class_weights,
        verbose=1,
    )

    # ── Sauvegarde de l'historique ───────────────────────────────────────────
    history_dict = {k: [float(v) for v in vs] for k, vs in history.history.items()}
    with open(HISTORY_JSON, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)
    print(f"\n  ✅ Historique sauvegardé : {HISTORY_JSON}")
    print(f"  ✅ Meilleur modèle       : {MODEL_PATH}")

    # ── Visualisations ───────────────────────────────────────────────────────
    plot_training_history(history_dict)

    print("\n" + "═"*70)
    print("  ✅ CR4 terminé. Lancez `python cr5_evaluate.py` pour l'évaluation.")
    print("═"*70)


if __name__ == "__main__":
    main()
