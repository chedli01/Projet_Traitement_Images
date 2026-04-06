"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       PROJET TRAITEMENT D'IMAGES — Détection des Maladies des Plantes       ║
║                   CR2 — Pipeline de Prétraitement des Images                 ║
║                       Année Universitaire 2025–2026                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Membres du groupe :
    - Yasmine Zerai
    - Chedli Masmoudi
    - Koussay Jebir
    - Ahmed Gargouri

Description :
    Ce script implémente l'intégralité du pipeline de prétraitement des images
    du dataset PlantVillage pour la détection de maladies des plantes.

    Les étapes couvertes sont :
        1. Téléchargement et chargement du dataset
        2. Nettoyage et vérification de l'intégrité
        3. Analyse exploratoire (distribution des classes)
        4. Redimensionnement des images
        5. Normalisation des valeurs de pixels
        6. Filtrage et réduction du bruit
        7. Augmentation de données
        8. Génération des illustrations avant/après (pour le CR2)
        9. Construction des générateurs Keras pour l'entraînement

Dépendances :
    pip install tensorflow opencv-python matplotlib numpy pillow seaborn scipy

Dataset :
    PlantVillage — 54 306 images, 38 classes, 256×256 px (RGB)
    Source : https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
             ou tensorflow_datasets (plant_village)
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import os
import random
import warnings
import zipfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from PIL import Image, ImageEnhance
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONFIGURATION GLOBALE
# ─────────────────────────────────────────────────────────────────────────────

# Chemins principaux — adaptez selon votre environnement
BASE_DIR      = Path("./plantvillage")         # racine du dataset après extraction
OUTPUT_DIR    = Path("./output_cr2")           # dossier de sortie pour les figures
PROCESSED_DIR = Path("./plantvillage_processed")  # images après prétraitement

# Hyperparamètres de prétraitement
IMG_SIZE_CNN  = (128, 128)   # résolution pour un CNN entraîné from scratch
IMG_SIZE_TL   = (224, 224)   # résolution pour Transfer Learning (ResNet-50, EfficientNet)
BATCH_SIZE    = 32           # nombre d'images par mini-batch lors de l'entraînement
SEED          = 42           # graine aléatoire pour la reproductibilité

# Couleurs pour les visualisations matplotlib
COLORS = {
    "green":  "#2E8B57",
    "accent": "#10B981",
    "red":    "#EF4444",
    "blue":   "#3B82F6",
    "orange": "#F59E0B",
    "grey":   "#6B7280",
}

# Créer les dossiers de sortie s'ils n'existent pas
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("✅ Configuration initialisée.")
print(f"   → Répertoire dataset   : {BASE_DIR}")
print(f"   → Répertoire sorties   : {OUTPUT_DIR}")
print(f"   → Taille image CNN     : {IMG_SIZE_CNN}")
print(f"   → Taille image TL      : {IMG_SIZE_TL}")
print(f"   → Batch size           : {BATCH_SIZE}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CHARGEMENT DU DATASET
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset_via_tensorflow():
    """
    Charge le dataset PlantVillage directement via TensorFlow Datasets.
    Alternative si vous n'avez pas téléchargé le dataset manuellement.
    
    Retourne :
        ds_train  : tf.data.Dataset pour l'entraînement
        ds_test   : tf.data.Dataset pour le test
        info      : informations sur le dataset (classes, splits, etc.)
    """
    import tensorflow_datasets as tfds

    # Téléchargement automatique depuis le catalogue TensorFlow Datasets
    # 'color' = version couleur RGB (vs 'gray' ou 'segmented')
    ds_train, ds_test, info = tfds.load(
        name="plant_village",
        split=["train[:80%]", "train[80%:]"],  # 80% train / 20% test
        with_info=True,
        as_supervised=True   # retourne des tuples (image, label)
    )

    print(f"✅ Dataset chargé via TensorFlow Datasets.")
    print(f"   → Nombre de classes : {info.features['label'].num_classes}")
    print(f"   → Noms des classes  : {info.features['label'].names[:5]} ...")
    return ds_train, ds_test, info


def load_dataset_from_disk(dataset_path: Path):
    """
    Charge le dataset PlantVillage depuis un répertoire local.
    Structure attendue :
        dataset_path/
            Apple___Apple_scab/
                image1.jpg
                image2.jpg
            Apple___Black_rot/
                ...
            ...

    Paramètres :
        dataset_path (Path) : chemin vers le dossier racine du dataset

    Retourne :
        image_paths (list)  : liste des chemins vers chaque image
        labels      (list)  : liste des labels correspondants (noms de classes)
        class_names (list)  : liste ordonnée des noms de classes
    """
    image_paths = []
    labels      = []

    # Lister toutes les sous-classes (sous-dossiers)
    class_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]

    print(f"✅ Dataset détecté : {len(class_names)} classes trouvées.")

    for class_dir in class_dirs:
        # Extensions acceptées
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            for img_path in class_dir.glob(ext):
                image_paths.append(img_path)
                labels.append(class_dir.name)

    print(f"   → Total images chargées : {len(image_paths)}")
    return image_paths, labels, class_names


# ─────────────────────────────────────────────────────────────────────────────
# 3.  NETTOYAGE ET VÉRIFICATION DE L'INTÉGRITÉ
# ─────────────────────────────────────────────────────────────────────────────

def verify_dataset_integrity(image_paths: list, labels: list):
    """
    Vérifie l'intégrité de chaque image du dataset :
        - fichier lisible et non corrompu
        - dimensions conformes (au moins 32×32 px)
        - nombre de canaux correct (RGB = 3 canaux)

    Paramètres :
        image_paths (list) : liste des chemins vers les images
        labels      (list) : liste des labels associés

    Retourne :
        valid_paths  (list) : chemins des images valides
        valid_labels (list) : labels correspondants
        report       (dict) : rapport de vérification
    """
    valid_paths  = []
    valid_labels = []
    corrupt      = []   # images illisibles ou corrompues
    too_small    = []   # images trop petites (< 32×32)
    wrong_channels = [] # images en niveaux de gris ou RGBA

    print("\n🔍 Vérification de l'intégrité du dataset...")

    for path, label in zip(image_paths, labels):
        try:
            # Ouvrir l'image avec Pillow pour détecter les corruptions
            with Image.open(path) as img:
                img.verify()   # détecte les fichiers tronqués ou invalides

            # Recharger après verify() (verify() invalide l'objet Image)
            img = Image.open(path).convert("RGB")
            w, h = img.size

            # Vérification des dimensions minimales
            if w < 32 or h < 32:
                too_small.append(str(path))
                continue

            # Vérification du nombre de canaux
            img_array = np.array(img)
            if img_array.ndim != 3 or img_array.shape[2] != 3:
                wrong_channels.append(str(path))
                continue

            valid_paths.append(path)
            valid_labels.append(label)

        except Exception as e:
            # L'image est corrompue ou illisible
            corrupt.append((str(path), str(e)))

    # Construire le rapport de vérification
    report = {
        "total_initial"     : len(image_paths),
        "valides"           : len(valid_paths),
        "corrompues"        : len(corrupt),
        "trop_petites"      : len(too_small),
        "canaux_incorrects" : len(wrong_channels),
        "details_corrupt"   : corrupt,
    }

    print(f"   ✅ Images valides        : {report['valides']}")
    print(f"   ❌ Images corrompues     : {report['corrompues']}")
    print(f"   ⚠️  Images trop petites  : {report['trop_petites']}")
    print(f"   ⚠️  Canaux incorrects    : {report['canaux_incorrects']}")

    return valid_paths, valid_labels, report


def analyze_class_distribution(labels: list, class_names: list):
    """
    Analyse et visualise la distribution des classes du dataset.
    Identifie les classes sur-représentées et sous-représentées.

    Paramètres :
        labels      (list) : liste de tous les labels
        class_names (list) : liste ordonnée des noms de classes

    Retourne :
        class_counts (dict) : dictionnaire {nom_classe: nombre_images}
    """
    # Comptage des images par classe
    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1

    # Trier du plus fréquent au moins fréquent
    class_counts = dict(sorted(class_counts.items(), key=lambda x: x[1], reverse=True))

    total = sum(class_counts.values())
    print(f"\n📊 Distribution des classes ({len(class_counts)} classes, {total} images) :")
    print(f"   → Classe max : {list(class_counts.keys())[0]}  ({list(class_counts.values())[0]} images)")
    print(f"   → Classe min : {list(class_counts.keys())[-1]} ({list(class_counts.values())[-1]} images)")
    print(f"   → Ratio max/min : {list(class_counts.values())[0] / list(class_counts.values())[-1]:.1f}x")

    # ── Visualisation ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle("Distribution des Classes — PlantVillage", fontsize=14, fontweight="bold")

    # Graphique en barres (toutes les classes)
    ax = axes[0]
    names  = [k.replace("___", "\n").replace("_", " ") for k in class_counts.keys()]
    counts = list(class_counts.values())
    bars = ax.barh(names, counts, color=COLORS["accent"], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Nombre d'images", fontsize=11)
    ax.set_title("Nombre d'images par classe", fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", labelsize=7)
    ax.set_facecolor("#F8F9FA")
    # Annoter les 3 plus grandes et les 3 plus petites classes
    for i, (bar, count) in enumerate(zip(bars, counts)):
        if i < 3 or i >= len(counts) - 3:
            ax.text(count + 50, bar.get_y() + bar.get_height()/2,
                    f" {count}", va="center", fontsize=8, color=COLORS["green"])

    # Camembert pour les 10 plus grandes classes
    ax2 = axes[1]
    top10_names  = list(class_counts.keys())[:10]
    top10_counts = list(class_counts.values())[:10]
    others_count = sum(list(class_counts.values())[10:])
    pie_labels   = [n.split("___")[-1].replace("_", " ") for n in top10_names] + ["Autres"]
    pie_values   = top10_counts + [others_count]
    wedges, texts, autotexts = ax2.pie(
        pie_values, labels=pie_labels, autopct="%1.1f%%",
        startangle=140, pctdistance=0.8,
        colors=plt.cm.Set3.colors[:len(pie_values)]
    )
    for text in texts:
        text.set_fontsize(8)
    ax2.set_title("Répartition des 10 premières classes", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_distribution_classes.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   → Figure sauvegardée : {OUTPUT_DIR}/01_distribution_classes.png")

    return class_counts


# ─────────────────────────────────────────────────────────────────────────────
# 4.  REDIMENSIONNEMENT DES IMAGES
# ─────────────────────────────────────────────────────────────────────────────

def resize_image(image: np.ndarray, target_size: tuple, interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    """
    Redimensionne une image à la taille cible.

    Paramètres :
        image       (np.ndarray) : image source (H×W×C), valeurs uint8 [0,255]
        target_size (tuple)      : (hauteur, largeur) cible, ex: (128, 128)
        interpolation            : méthode d'interpolation OpenCV
                                   - cv2.INTER_LINEAR  : bilinéaire (défaut, bon compromis)
                                   - cv2.INTER_CUBIC   : bicubique (plus lent, meilleure qualité)
                                   - cv2.INTER_NEAREST : plus rapide, pixelisation
                                   - cv2.INTER_AREA    : recommandé pour réduire la taille

    Retourne :
        np.ndarray : image redimensionnée (target_size[0] × target_size[1] × C)

    Note :
        OpenCV attend (largeur, hauteur) dans cv2.resize, donc on inverse target_size.
    """
    # cv2.resize attend (width, height) = (target_size[1], target_size[0])
    resized = cv2.resize(image, (target_size[1], target_size[0]), interpolation=interpolation)
    return resized


def demonstrate_resize(sample_image_path: Path):
    """
    Génère une figure illustrant l'effet du redimensionnement
    à différentes résolutions (256, 128, 64 px).

    Paramètres :
        sample_image_path (Path) : chemin vers une image d'exemple
    """
    # Charger l'image en RGB (OpenCV charge en BGR par défaut, on convertit)
    img_bgr = cv2.imread(str(sample_image_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    sizes  = [256, 128, 64]
    labels = [f"Original\n{s}×{s} px" if i == 0 else f"Redimensionné\n{s}×{s} px"
              for i, s in enumerate(sizes)]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle("Illustration du Redimensionnement des Images", fontsize=13, fontweight="bold")

    for ax, size, lbl in zip(axes, sizes, labels):
        img_resized = resize_image(img_rgb, (size, size))
        ax.imshow(img_resized)
        ax.set_title(lbl, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel(f"Taille fichier ≈ {size*size*3 // 1024} Ko", fontsize=9, color=COLORS["grey"])
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_edgecolor("#CCCCCC")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_redimensionnement.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   → Figure sauvegardée : 02_redimensionnement.png")


def batch_resize_dataset(image_paths: list, labels: list,
                         output_root: Path, target_size: tuple = (128, 128)):
    """
    Redimensionne toutes les images du dataset et les sauvegarde dans output_root.
    Conserve la structure de dossiers par classe.

    Paramètres :
        image_paths (list)  : chemins des images sources
        labels      (list)  : labels correspondants
        output_root (Path)  : dossier de destination
        target_size (tuple) : (H, W) cible

    Retourne :
        int : nombre d'images traitées avec succès
    """
    success = 0
    errors  = 0

    for img_path, label in zip(image_paths, labels):
        # Créer le sous-dossier de la classe si nécessaire
        class_dir = output_root / label
        class_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Lire l'image
            img = cv2.imread(str(img_path))
            if img is None:
                errors += 1
                continue

            # Convertir BGR → RGB (OpenCV travaille en BGR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Redimensionner
            img_resized = resize_image(img_rgb, target_size)

            # Sauvegarder (reconvertir RGB → BGR pour cv2.imwrite)
            out_path = class_dir / img_path.name
            cv2.imwrite(str(out_path), cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
            success += 1

        except Exception as e:
            errors += 1
            print(f"   ⚠️ Erreur sur {img_path.name} : {e}")

    print(f"✅ Redimensionnement terminé : {success} images traitées, {errors} erreurs.")
    return success


# ─────────────────────────────────────────────────────────────────────────────
# 5.  NORMALISATION DES VALEURS DE PIXELS
# ─────────────────────────────────────────────────────────────────────────────

def normalize_minmax(image: np.ndarray) -> np.ndarray:
    """
    Normalisation min-max : divise chaque pixel par 255.
    Les valeurs passent de [0, 255] (uint8) à [0.0, 1.0] (float32).

    Paramètres :
        image (np.ndarray) : image uint8, shape (H, W, C) ou (H, W)

    Retourne :
        np.ndarray float32 avec valeurs dans [0.0, 1.0]

    Utilisation :
        Idéale pour un CNN entraîné from scratch.
    """
    return image.astype(np.float32) / 255.0


def normalize_imagenet(image: np.ndarray) -> np.ndarray:
    """
    Standardisation avec les statistiques ImageNet.
    Utilisée pour le Transfer Learning (ResNet, EfficientNet, VGG...).

    Formule : pixel_norm = (pixel / 255 - mean) / std
        mean = [0.485, 0.456, 0.406]  ← moyenne par canal R, G, B sur ImageNet
        std  = [0.229, 0.224, 0.225]  ← écart-type par canal R, G, B sur ImageNet

    Paramètres :
        image (np.ndarray) : image uint8 RGB, shape (H, W, 3)

    Retourne :
        np.ndarray float32 standardisé (valeurs centrées autour de 0)
    """
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img_float = image.astype(np.float32) / 255.0   # d'abord normaliser [0,1]
    img_norm  = (img_float - IMAGENET_MEAN) / IMAGENET_STD  # puis standardiser
    return img_norm


def demonstrate_normalization(sample_image: np.ndarray):
    """
    Génère une figure illustrant la normalisation :
        - image brute (valeurs [0, 255])
        - image normalisée (valeurs [0.0, 1.0])
        + histogrammes des valeurs de pixels pour chaque canal RGB

    Paramètres :
        sample_image (np.ndarray) : image uint8 RGB (H×W×3)
    """
    img_norm = normalize_minmax(sample_image)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle("Normalisation des Valeurs de Pixels — Avant / Après",
                 fontsize=13, fontweight="bold")

    channel_colors  = [COLORS["red"], COLORS["green"], COLORS["blue"]]
    channel_labels  = ["Rouge (R)", "Vert (G)", "Bleu (B)"]

    # ── Ligne 0 : affichage de l'image ───────────────────────────────────────
    axes[0, 0].imshow(sample_image)
    axes[0, 0].set_title("Image Brute\nValeurs : [0, 255]", fontsize=11, fontweight="bold")
    axes[0, 0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    axes[0, 1].imshow(img_norm)
    axes[0, 1].set_title("Image Normalisée\nValeurs : [0.0, 1.0]", fontsize=11, fontweight="bold")
    axes[0, 1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # ── Ligne 1 : histogrammes des canaux RGB ─────────────────────────────────
    for ch, (color, lbl) in enumerate(zip(channel_colors, channel_labels)):
        axes[1, 0].hist(sample_image[:, :, ch].ravel(), bins=64,
                        color=color, alpha=0.6, label=lbl)

    axes[1, 0].set_xlabel("Intensité (0 → 255)", fontsize=10)
    axes[1, 0].set_ylabel("Fréquence", fontsize=10)
    axes[1, 0].set_title("Histogramme Avant Normalisation", fontsize=11, fontweight="bold")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].set_facecolor("#F8F9FA")

    for ch, (color, lbl) in enumerate(zip(channel_colors, channel_labels)):
        axes[1, 1].hist(img_norm[:, :, ch].ravel(), bins=64,
                        color=color, alpha=0.6, label=lbl)

    axes[1, 1].set_xlabel("Intensité (0.0 → 1.0)", fontsize=10)
    axes[1, 1].set_ylabel("Fréquence", fontsize=10)
    axes[1, 1].set_title("Histogramme Après Normalisation", fontsize=11, fontweight="bold")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].set_facecolor("#F8F9FA")

    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_edgecolor("#CCCCCC")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_normalisation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   → Figure sauvegardée : 03_normalisation.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FILTRAGE ET RÉDUCTION DU BRUIT
# ─────────────────────────────────────────────────────────────────────────────

def apply_gaussian_filter(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Applique un filtre gaussien à l'image pour réduire le bruit léger.
    Utilise une convolution par noyau gaussien sur chaque canal séparément.

    Paramètres :
        image (np.ndarray) : image uint8 RGB (H×W×3)
        sigma (float)      : écart-type du noyau gaussien
                             - sigma faible (0.5–1.0) : lissage léger
                             - sigma élevé (2.0–3.0)  : fort lissage, perd les détails

    Retourne :
        np.ndarray uint8 : image filtrée
    """
    # Appliquer le filtre canal par canal pour préserver les couleurs
    filtered = np.zeros_like(image, dtype=np.float32)
    for ch in range(image.shape[2]):
        filtered[:, :, ch] = gaussian_filter(image[:, :, ch].astype(np.float32), sigma=sigma)
    return np.clip(filtered, 0, 255).astype(np.uint8)


def apply_median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Applique un filtre médian — efficace contre le bruit "sel et poivre".
    Remplace chaque pixel par la valeur médiane de son voisinage.

    Paramètres :
        image       (np.ndarray) : image uint8 RGB
        kernel_size (int)        : taille du voisinage (doit être impair : 3, 5, 7)

    Retourne :
        np.ndarray uint8 : image filtrée
    """
    # cv2.medianBlur fonctionne directement sur les 3 canaux (BGR ou RGB)
    return cv2.medianBlur(image, kernel_size)


def apply_bilateral_filter(image: np.ndarray,
                            d: int = 9,
                            sigma_color: float = 75,
                            sigma_space: float = 75) -> np.ndarray:
    """
    Applique un filtre bilatéral — lisse le bruit tout en PRÉSERVANT les contours.
    Très utile pour conserver les bords des zones de lésion sur les feuilles.

    Paramètres :
        image       (np.ndarray) : image uint8 RGB
        d           (int)        : diamètre du voisinage (9 = voisinage 9×9)
        sigma_color (float)      : contrôle la sensibilité aux différences de couleur
                                   (valeur haute = lisse à travers les bords de couleur)
        sigma_space (float)      : contrôle la portée spatiale du filtre

    Retourne :
        np.ndarray uint8 : image filtrée
    """
    # cv2.bilateralFilter travaille en BGR, on convertit
    img_bgr      = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    filtered_bgr = cv2.bilateralFilter(img_bgr, d, sigma_color, sigma_space)
    return cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)


def demonstrate_filtering(sample_image: np.ndarray):
    """
    Génère une figure comparant les différents filtres sur une image bruitée.

    Paramètres :
        sample_image (np.ndarray) : image uint8 RGB propre
    """
    # Simuler du bruit gaussien pour la démonstration
    np.random.seed(SEED)
    noise       = np.random.normal(0, 25, sample_image.shape).astype(np.int16)
    img_noisy   = np.clip(sample_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Appliquer les différents filtres
    img_gauss    = apply_gaussian_filter(img_noisy, sigma=1.5)
    img_median   = apply_median_filter(img_noisy, kernel_size=3)
    img_bilateral= apply_bilateral_filter(img_noisy)

    images = [sample_image, img_noisy, img_gauss, img_median, img_bilateral]
    titles = [
        "Original\n(sans bruit)",
        "Image bruitée\n(σ_bruit = 25)",
        "Filtre Gaussien\n(σ = 1.5)",
        "Filtre Médian\n(noyau 3×3)",
        "Filtre Bilatéral\n(d=9, σ=75)",
    ]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle("Comparaison des Filtres de Réduction du Bruit", fontsize=13, fontweight="bold")

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_edgecolor("#CCCCCC")

    # Calcul du PSNR (Peak Signal-to-Noise Ratio) pour chaque image filtrée
    def psnr(original, filtered):
        """Mesure la qualité de la reconstruction (dB) — plus haut = mieux."""
        mse = np.mean((original.astype(float) - filtered.astype(float)) ** 2)
        if mse == 0:
            return float("inf")
        return 20 * np.log10(255.0 / np.sqrt(mse))

    psnr_values = ["-", "-",
                   f"PSNR: {psnr(sample_image, img_gauss):.1f} dB",
                   f"PSNR: {psnr(sample_image, img_median):.1f} dB",
                   f"PSNR: {psnr(sample_image, img_bilateral):.1f} dB"]

    for ax, psnr_val in zip(axes, psnr_values):
        ax.set_xlabel(psnr_val, fontsize=9, color=COLORS["green"])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_filtrage.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   → Figure sauvegardée : 04_filtrage.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  AUGMENTATION DE DONNÉES
# ─────────────────────────────────────────────────────────────────────────────

def augment_image_manual(image: np.ndarray, seed: int = None) -> dict:
    """
    Applique différentes transformations d'augmentation à une image.
    Chaque transformation est appliquée indépendamment sur l'image originale.

    Paramètres :
        image (np.ndarray) : image uint8 RGB (H×W×3)
        seed  (int)        : graine pour la reproductibilité

    Retourne :
        dict : {nom_transformation: image_transformée}
    """
    if seed is not None:
        np.random.seed(seed)

    H, W = image.shape[:2]
    augmented = {}

    # ── 1. Flip horizontal ────────────────────────────────────────────────────
    # Simule une feuille vue en miroir (naturellement possible)
    augmented["Flip Horizontal"] = cv2.flip(image, 1)   # flipCode=1 → horizontal

    # ── 2. Flip vertical ──────────────────────────────────────────────────────
    # Simule une image prise à l'envers ou sous un autre angle
    augmented["Flip Vertical"] = cv2.flip(image, 0)     # flipCode=0 → vertical

    # ── 3. Rotation ───────────────────────────────────────────────────────────
    # Simule l'orientation variable de la feuille lors de la prise de photo
    angle  = random.uniform(-30, 30)
    center = (W // 2, H // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    augmented[f"Rotation ({angle:.0f}°)"] = cv2.warpAffine(
        image, M, (W, H), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101   # évite les bords noirs
    )

    # ── 4. Zoom (crop + resize) ────────────────────────────────────────────────
    # Simule une distance variable entre la caméra et la feuille
    zoom_factor = random.uniform(0.8, 1.2)
    if zoom_factor < 1.0:
        # Zoom out : ajouter du padding
        pad = int((1.0 - zoom_factor) * min(H, W) / 2)
        img_padded = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                                        cv2.BORDER_REFLECT_101)
        augmented["Zoom Out"] = cv2.resize(img_padded, (W, H))
    else:
        # Zoom in : recadrer le centre
        crop = int((zoom_factor - 1.0) * min(H, W) / 2)
        img_cropped = image[crop:H-crop, crop:W-crop]
        augmented["Zoom In"] = cv2.resize(img_cropped, (W, H))

    # ── 5. Variation de luminosité ────────────────────────────────────────────
    # Simule des conditions d'éclairage différentes (nuageux, ensoleillé...)
    pil_img   = Image.fromarray(image)
    enhancer  = ImageEnhance.Brightness(pil_img)
    factor    = random.uniform(0.6, 1.4)   # 0.6 = plus sombre, 1.4 = plus clair
    augmented["Luminosité"] = np.array(enhancer.enhance(factor))

    # ── 6. Variation de contraste ─────────────────────────────────────────────
    # Simule différentes qualités d'appareil photo
    enhancer2 = ImageEnhance.Contrast(pil_img)
    factor2   = random.uniform(0.7, 1.4)
    augmented["Contraste"] = np.array(enhancer2.enhance(factor2))

    # ── 7. Translation (décalage spatial) ─────────────────────────────────────
    # Simule un cadrage imparfait lors de la prise de photo
    tx = random.randint(-W // 10, W // 10)   # décalage horizontal ≤ 10% de la largeur
    ty = random.randint(-H // 10, H // 10)   # décalage vertical   ≤ 10% de la hauteur
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    augmented["Translation"] = cv2.warpAffine(
        image, M_trans, (W, H), borderMode=cv2.BORDER_REFLECT_101
    )

    # ── 8. Bruit gaussien léger ────────────────────────────────────────────────
    # Simule le bruit de capteur des smartphones dans des conditions difficiles
    noise  = np.random.normal(0, 8, image.shape).astype(np.int16)
    augmented["Bruit Gaussien"] = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return augmented


def demonstrate_augmentation(sample_image: np.ndarray):
    """
    Génère une figure illustrant toutes les transformations d'augmentation.

    Paramètres :
        sample_image (np.ndarray) : image uint8 RGB originale
    """
    augmented = augment_image_manual(sample_image, seed=SEED)

    all_images = [("Original", sample_image)] + list(augmented.items())
    n_cols = 5
    n_rows = (len(all_images) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle("Techniques d'Augmentation de Données — PlantVillage",
                 fontsize=13, fontweight="bold")

    axes_flat = axes.flat if n_rows > 1 else axes
    for ax, (title, img) in zip(axes_flat, all_images):
        ax.imshow(img)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_edgecolor("#CCCCCC")

    # Masquer les axes vides restants
    for ax in list(axes_flat)[len(all_images):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_augmentation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   → Figure sauvegardée : 05_augmentation.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  GÉNÉRATEURS KERAS POUR L'ENTRAÎNEMENT (on-the-fly)
# ─────────────────────────────────────────────────────────────────────────────

def build_keras_generators(dataset_path: Path,
                            target_size: tuple = IMG_SIZE_CNN,
                            batch_size: int = BATCH_SIZE):
    """
    Construit les générateurs Keras pour l'entraînement, la validation et le test.
    L'augmentation est appliquée en temps réel (on-the-fly), uniquement sur le train.

    Structure du dossier dataset_path attendue :
        dataset_path/
            Apple___Apple_scab/  ← un dossier par classe
            Apple___Black_rot/
            ...

    Paramètres :
        dataset_path (Path)  : chemin vers le dataset redimensionné
        target_size  (tuple) : (H, W) de sortie des images
        batch_size   (int)   : taille des mini-batchs

    Retourne :
        train_gen (DirectoryIterator) : générateur d'entraînement avec augmentation
        val_gen   (DirectoryIterator) : générateur de validation sans augmentation
        test_gen  (DirectoryIterator) : générateur de test sans augmentation
        class_names (list)            : noms des classes dans l'ordre des indices
    """
    # ── Générateur d'entraînement — AVEC augmentation ─────────────────────────
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,              # normalisation min-max → [0, 1]
        rotation_range=30,              # rotation aléatoire ±30°
        width_shift_range=0.10,         # translation horizontale ±10%
        height_shift_range=0.10,        # translation verticale ±10%
        zoom_range=0.20,                # zoom aléatoire ±20%
        horizontal_flip=True,           # flip horizontal (50% des cas)
        vertical_flip=True,             # flip vertical (50% des cas)
        brightness_range=[0.70, 1.30],  # variation de luminosité ±30%
        fill_mode="nearest",            # remplissage des pixels créés par rotation/translation
        validation_split=0.15,          # 15% des données pour la validation
    )

    # ── Générateur de validation/test — SANS augmentation ─────────────────────
    # La normalisation seule est appliquée, pas de transformations aléatoires
    val_test_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.15,
    )

    # ── Générateur entraînement ────────────────────────────────────────────────
    train_gen = train_datagen.flow_from_directory(
        directory=str(dataset_path),
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",    # one-hot encoding des labels
        subset="training",           # utilise 85% des données (1 - 0.15)
        shuffle=True,                # mélange les images à chaque époque
        seed=SEED,
    )

    # ── Générateur validation ──────────────────────────────────────────────────
    val_gen = val_test_datagen.flow_from_directory(
        directory=str(dataset_path),
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",         # utilise les 15% restants
        shuffle=False,               # pas de mélange pour la validation
        seed=SEED,
    )

    class_names = list(train_gen.class_indices.keys())

    print(f"\n✅ Générateurs Keras construits :")
    print(f"   → Train      : {train_gen.samples} images, {len(class_names)} classes")
    print(f"   → Validation : {val_gen.samples} images")
    print(f"   → Batch size : {batch_size}")
    print(f"   → Taille image : {target_size}")

    return train_gen, val_gen, class_names


def build_keras_generators_transfer_learning(dataset_path: Path,
                                              batch_size: int = BATCH_SIZE):
    """
    Générateurs optimisés pour le Transfer Learning avec les statistiques ImageNet.
    Utilise tf.keras.applications.resnet50.preprocess_input.

    Paramètres :
        dataset_path (Path) : chemin vers le dataset
        batch_size   (int)  : taille des mini-batchs

    Retourne :
        train_gen, val_gen, class_names
    """
    # Utiliser la fonction de prétraitement spécifique à ResNet-50
    # (standardisation ImageNet : mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

    train_datagen_tl = ImageDataGenerator(
        preprocessing_function=resnet_preprocess,  # standardisation ImageNet
        rotation_range=25,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=False,        # pour ResNet, on évite le flip vertical
        fill_mode="nearest",
        validation_split=0.15,
    )

    val_datagen_tl = ImageDataGenerator(
        preprocessing_function=resnet_preprocess,
        validation_split=0.15,
    )

    train_gen = train_datagen_tl.flow_from_directory(
        directory=str(dataset_path),
        target_size=IMG_SIZE_TL,    # 224×224 requis par ResNet-50
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=SEED,
    )

    val_gen = val_datagen_tl.flow_from_directory(
        directory=str(dataset_path),
        target_size=IMG_SIZE_TL,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=SEED,
    )

    class_names = list(train_gen.class_indices.keys())
    print(f"\n✅ Générateurs Transfer Learning construits (224×224, standardisation ImageNet).")
    return train_gen, val_gen, class_names


# ─────────────────────────────────────────────────────────────────────────────
# 9.  PIPELINE COMPLET — FONCTION PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(dataset_path: Path = BASE_DIR, use_sample: bool = True):
    """
    Exécute l'intégralité du pipeline de prétraitement du CR2.

    Si use_sample=True, utilise des images synthétiques pour la démonstration
    (utile si le dataset PlantVillage n'est pas encore téléchargé).

    Paramètres :
        dataset_path (Path) : chemin vers le dataset PlantVillage
        use_sample   (bool) : True = utiliser des images de démonstration synthétiques
    """
    print("=" * 70)
    print("   PIPELINE DE PRÉTRAITEMENT CR2 — DÉTECTION MALADIES DES PLANTES   ")
    print("=" * 70)

    # ── Étape 0 : Créer une image de démonstration synthétique ────────────────
    # (remplacer par load_dataset_from_disk() avec le vrai dataset)
    sample_image = _create_demo_leaf_image()

    print("\n📂 ÉTAPE 1 — Vérification du dataset")
    print("-" * 40)
    if dataset_path.exists() and not use_sample:
        image_paths, labels, class_names = load_dataset_from_disk(dataset_path)
        valid_paths, valid_labels, report = verify_dataset_integrity(image_paths, labels)
        class_counts = analyze_class_distribution(valid_labels, class_names)
    else:
        print("   ℹ️  Dataset non trouvé — utilisation du mode démonstration.")
        print("   ℹ️  Téléchargez PlantVillage depuis Kaggle ou TensorFlow Datasets.")

    print("\n🔄 ÉTAPE 2 — Redimensionnement")
    print("-" * 40)
    demonstrate_resize_synthetic()   # figure de démonstration sans dataset

    print("\n📐 ÉTAPE 3 — Normalisation")
    print("-" * 40)
    demonstrate_normalization(sample_image)

    print("\n🔍 ÉTAPE 4 — Filtrage")
    print("-" * 40)
    demonstrate_filtering(sample_image)

    print("\n🔀 ÉTAPE 5 — Augmentation de données")
    print("-" * 40)
    demonstrate_augmentation(sample_image)

    print("\n" + "=" * 70)
    print(f"✅ Pipeline terminé. Figures sauvegardées dans : {OUTPUT_DIR}/")
    print("=" * 70)


def _create_demo_leaf_image(size: int = 256) -> np.ndarray:
    """
    Crée une image synthétique de feuille pour les démonstrations.
    Simule une feuille verte avec des taches de maladie brune/orange.

    Retourne :
        np.ndarray uint8 RGB (size×size×3)
    """
    np.random.seed(SEED)
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Fond vert (chlorophylle)
    img[:, :] = [55, 140, 45]

    # Nervures (plus sombres)
    for i in range(1, 6):
        y = int(i * size / 6)
        img[max(0, y-2):y+2, :] = [25, 90, 25]

    # Nervure centrale diagonale
    for x in range(size):
        y = int(size * 0.15 + x * 0.7)
        if 0 <= y < size:
            img[y:min(size, y+3), x] = [25, 90, 25]

    # Zones de maladie (taches brunes — lésions fongiques)
    centers = [(int(size * 0.20), int(size * 0.30), 18),
               (int(size * 0.65), int(size * 0.60), 24),
               (int(size * 0.45), int(size * 0.75), 12)]

    for (cx, cy, radius) in centers:
        yy, xx = np.ogrid[:size, :size]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        # Centre de la lésion (brun foncé)
        inner = (yy - cy) ** 2 + (xx - cx) ** 2 <= (radius // 2) ** 2
        img[mask]  = [180, 110, 20]   # bord orange-brun
        img[inner] = [120, 60, 10]    # centre brun foncé

    # Légère variabilité de texture
    noise = np.random.normal(0, 6, img.shape).astype(np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def demonstrate_resize_synthetic():
    """
    Version de la démonstration de redimensionnement sans dataset réel.
    Utilise l'image synthétique pour illustrer les différentes résolutions.
    """
    img = _create_demo_leaf_image(256)
    sizes  = [256, 128, 64]
    labels_sz = ["Original (256×256)", "Redimensionné (128×128)", "Redimensionné (64×64)"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle("Illustration du Redimensionnement des Images (Synthétique)",
                 fontsize=13, fontweight="bold")

    for ax, size, lbl in zip(axes, sizes, labels_sz):
        img_resized = resize_image(img, (size, size))
        ax.imshow(img_resized)
        ax.set_title(lbl, fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel(f"≈ {size*size*3 // 1024} Ko (non compressé)", fontsize=9, color=COLORS["grey"])
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_edgecolor("#CCCCCC")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_redimensionnement.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   → Figure sauvegardée : 02_redimensionnement.png")


# ─────────────────────────────────────────────────────────────────────────────
# 10. FONCTIONS UTILITAIRES COMPLÉMENTAIRES
# ─────────────────────────────────────────────────────────────────────────────

def compute_dataset_statistics(image_paths: list, sample_size: int = 1000):
    """
    Calcule les statistiques globales du dataset (moyenne et écart-type par canal).
    Ces valeurs sont nécessaires pour la standardisation personnalisée.

    Paramètres :
        image_paths (list) : liste des chemins d'images
        sample_size (int)  : nombre d'images à échantillonner (pour la rapidité)

    Retourne :
        dict : {"mean": [R, G, B], "std": [R, G, B]}
    """
    # Échantillonner un sous-ensemble pour gagner du temps
    sample_paths = random.sample(image_paths, min(sample_size, len(image_paths)))

    means = []
    stds  = []

    for path in sample_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        means.append(img_rgb.mean(axis=(0, 1)))   # moyenne par canal sur (H, W)
        stds.append(img_rgb.std(axis=(0, 1)))

    global_mean = np.mean(means, axis=0)
    global_std  = np.mean(stds,  axis=0)

    print(f"\n📊 Statistiques du dataset PlantVillage (sur {len(means)} images) :")
    print(f"   → Moyenne  R={global_mean[0]:.4f}, G={global_mean[1]:.4f}, B={global_mean[2]:.4f}")
    print(f"   → Écart-type R={global_std[0]:.4f}, G={global_std[1]:.4f}, B={global_std[2]:.4f}")

    return {"mean": global_mean.tolist(), "std": global_std.tolist()}


def visualize_sample_images(image_paths: list, labels: list,
                             n_samples: int = 16, n_cols: int = 4):
    """
    Affiche un échantillon aléatoire d'images du dataset avec leurs labels.

    Paramètres :
        image_paths (list) : liste des chemins d'images
        labels      (list) : labels correspondants
        n_samples   (int)  : nombre d'images à afficher
        n_cols      (int)  : nombre de colonnes dans la grille
    """
    n_rows   = (n_samples + n_cols - 1) // n_cols
    indices  = random.sample(range(len(image_paths)), n_samples)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(f"Échantillon d'images PlantVillage ({n_samples} images)",
                 fontsize=13, fontweight="bold")

    for ax, idx in zip(axes.flat, indices):
        img = Image.open(image_paths[idx]).convert("RGB")
        ax.imshow(img)
        # Raccourcir le label pour l'affichage
        short_label = labels[idx].replace("___", "\n").replace("_", " ")
        ax.set_title(short_label, fontsize=7, fontweight="bold", pad=4)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_edgecolor("#CCCCCC")

    for ax in list(axes.flat)[n_samples:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "00_echantillon_dataset.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   → Figure sauvegardée : 00_echantillon_dataset.png")


def compute_class_weights(labels: list):
    """
    Calcule les poids de classes pour compenser le déséquilibre du dataset.
    Les classes minoritaires reçoivent un poids plus élevé lors de l'entraînement.

    Formule : w_i = n_total / (n_classes × n_i)

    Paramètres :
        labels (list) : liste des labels de toutes les images

    Retourne :
        dict : {index_classe: poids} — à passer à model.fit(class_weight=...)
    """
    from collections import Counter
    counter    = Counter(labels)
    n_total    = len(labels)
    n_classes  = len(counter)
    class_names_sorted = sorted(counter.keys())

    class_weights = {}
    for i, cls in enumerate(class_names_sorted):
        # Plus n_i est petit, plus le poids est élevé
        class_weights[i] = n_total / (n_classes * counter[cls])

    print(f"\n⚖️ Poids des classes (pour équilibrer le dataset) :")
    for i, cls in enumerate(class_names_sorted[:5]):
        print(f"   → {cls}: {class_weights[i]:.3f}")
    print("   → ...")

    return class_weights


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Exécution du pipeline complet.
    
    Pour utiliser avec le vrai dataset PlantVillage :
        1. Télécharger depuis Kaggle :
           kaggle datasets download -d abdallahalidev/plantvillage-dataset
        2. Extraire dans le dossier ./plantvillage/
        3. Changer use_sample=False dans l'appel ci-dessous
    
    Pour tester sans le dataset :
        python CR2_Pretraitement_Pipeline.py
        → Génère toutes les figures avec des images synthétiques
    """
    run_full_pipeline(
        dataset_path=BASE_DIR,
        use_sample=False    # ← passer à False si le dataset est disponible
    )