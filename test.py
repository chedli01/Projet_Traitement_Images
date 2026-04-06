"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       PROJET TRAITEMENT D'IMAGES — Détection des Maladies des Plantes        ║
║           CR2 — Pipeline COMPLET de Prétraitement des Images                 ║
║                 Dataset RÉEL : PlantVillage (Kaggle)                         ║
║                       Année Universitaire 2025–2026                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Membres du groupe :
    - Yasmine Zerai
    - Chedli Masmoudi
    - Koussay Jebir
    - Ahmed Gargouri

Ce script fonctionne avec le dataset PlantVillage téléchargé MANUELLEMENT.
Placez le fichier ZIP dans le dossier du script ou spécifiez son chemin.

Prérequis :
    pip install tensorflow opencv-python matplotlib numpy pillow scipy scikit-learn seaborn

Exécution :
    python CR2_Pipeline_Complet.py

Sorties générées :
    output_cr2/
    ├── 00_echantillon_brut.png          ← images brutes du dataset
    ├── 01_distribution_classes.png      ← analyse de la distribution
    ├── 02_avant_apres_resize.png        ← avant/après redimensionnement
    ├── 03_avant_apres_normalisation.png ← avant/après normalisation + histogrammes
    ├── 04_avant_apres_filtrage.png      ← comparaison des 4 filtres
    ├── 05_avant_apres_augmentation.png  ← les 8 transformations appliquées
    └── 06_pipeline_final.png            ← image brute → résultat final
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import random
import shutil
import zipfile
import warnings
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")   # backend sans affichage (compatible Colab/serveur)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image, ImageEnhance
from scipy.ndimage import gaussian_filter

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # Supprimer les logs TensorFlow verbeux

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONFIGURATION GLOBALE
# ─────────────────────────────────────────────────────────────────────────────

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR        = Path("./plantvillage_dataset")   # dossier après extraction du zip
OUTPUT_DIR      = Path("./output_cr2")             # figures générées
PROCESSED_DIR   = Path("./plantvillage_processed") # images prétraitées
ZIP_NAME        = "plantvillage-dataset.zip"       # nom attendu du fichier zip

# ── Hyperparamètres ───────────────────────────────────────────────────────────
IMG_SIZE_CNN = (128, 128)   # résolution CNN from scratch
IMG_SIZE_TL  = (224, 224)   # résolution Transfer Learning (ResNet-50, EfficientNet)
BATCH_SIZE   = 32
SEED         = 42
N_SAMPLES_VIZ = 6           # nombre d'images dans les grilles de comparaison

# ── Palette de couleurs ────────────────────────────────────────────────────────
C = {
    "green":  "#2E8B57",
    "accent": "#10B981",
    "red":    "#EF4444",
    "blue":   "#3B82F6",
    "orange": "#F59E0B",
    "grey":   "#6B7280",
    "dark":   "#1F2937",
    "light":  "#F8F9FA",
}

# Créer les dossiers de sortie
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)

print("╔" + "═"*68 + "╗")
print("║   CR2 — Pipeline Prétraitement PlantVillage                        ║")
print("╚" + "═"*68 + "╝")
print(f"  → Sorties    : {OUTPUT_DIR.resolve()}")
print(f"  → Taille CNN : {IMG_SIZE_CNN}   Taille TL : {IMG_SIZE_TL}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  EXTRACTION DU DATASET (MANUEL)
# ─────────────────────────────────────────────────────────────────────────────

def extract_plantvillage():
    """
    Extrait le dataset PlantVillage depuis un fichier ZIP téléchargé manuellement.
    
    Cherche le fichier zip dans l'ordre:
    1. Le chemin spécifié par l'utilisateur (argument ligne de commande)
    2. Le fichier ZIP_NAME dans le dossier courant
    3. Tout fichier .zip dans le dossier courant
    
    Le dataset contient ~54 306 images JPEG réparties en 38 dossiers (classes).
    """
    print("\n" + "─"*60)
    print("  ÉTAPE 0 — Extraction du dataset PlantVillage")
    print("─"*60)

    # Vérifier si le dataset est déjà extrait
    color_dir = BASE_DIR / "plantvillage dataset" / "color"
    if color_dir.exists() and any(color_dir.iterdir()):
        n_classes = sum(1 for d in color_dir.iterdir() if d.is_dir())
        print(f"  ✅ Dataset déjà disponible ({n_classes} classes trouvées).")
        return color_dir

    # ── Trouver le fichier zip ─────────────────────────────────────────────
    zip_path = None
    
    # Option 1: Chemin passé en argument
    if len(sys.argv) > 1:
        zip_path = Path(sys.argv[1])
        if not zip_path.exists():
            print(f"  ⚠️  Fichier spécifié introuvable: {zip_path}")
            zip_path = None
    
    # Option 2: Fichier avec le nom par défaut dans le dossier courant
    if zip_path is None:
        default_zip = Path(ZIP_NAME)
        if default_zip.exists():
            zip_path = default_zip
            print(f"  📦 Fichier trouvé: {zip_path}")
    
    # Option 3: Chercher tout fichier .zip récent
    if zip_path is None:
        zips = sorted(Path(".").glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        if zips:
            zip_path = zips[0]
            print(f"  📦 Fichier zip détecté: {zip_path}")
    
    # Aucun fichier trouvé
    if zip_path is None:
        print("  ❌ ERREUR : Aucun fichier ZIP trouvé!")
        print(f"""
  → Solutions:
     1. Placez '{ZIP_NAME}' dans ce dossier: {Path(".").resolve()}
     2. Ou spécifiez le chemin: python {Path(__file__).name} chemin/vers/fichier.zip
     3. Téléchargez depuis: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
        """)
        sys.exit(1)

    # ── Extraire le zip ─────────────────────────────────────────────────────
    print(f"  📦 Extraction de {zip_path.name}...")
    print(f"     Taille: {zip_path.stat().st_size / (1024**3):.2f} Go")
    print("     Cela peut prendre quelques minutes...")
    
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Afficher la progression
            total = len(zf.namelist())
            for i, member in enumerate(zf.namelist(), 1):
                zf.extract(member, BASE_DIR)
                if i % 1000 == 0 or i == total:
                    print(f"     Extrait {i}/{total} fichiers...", end="\r")
        print(f"\n  ✅ Extraction terminée ({total} fichiers)")
    except zipfile.BadZipFile:
        print(f"  ❌ Erreur: {zip_path} est corrompu ou n'est pas un fichier zip valide")
        sys.exit(1)
    except Exception as e:
        print(f"  ❌ Erreur lors de l'extraction: {e}")
        sys.exit(1)

    # ── Localiser le dossier 'color' ───────────────────────────────────────
    color_dir = BASE_DIR / "plantvillage dataset" / "color"
    if not color_dir.exists():
        # Chercher récursivement si la structure est différente
        found = list(BASE_DIR.rglob("color"))
        if found:
            color_dir = found[0]
        else:
            print("  ❌ Dossier 'color' introuvable après extraction.")
            print(f"     Structure extraite : {list(BASE_DIR.iterdir())}")
            sys.exit(1)

    n_classes = sum(1 for d in color_dir.iterdir() if d.is_dir())
    n_images = sum(1 for _ in color_dir.rglob("*.jpg")) + sum(1 for _ in color_dir.rglob("*.JPG"))
    print(f"  ✅ Dataset prêt — {n_classes} classes, ~{n_images} images dans {color_dir}")
    
    # Option: Supprimer le zip après extraction pour économiser l'espace
    # zip_path.unlink()
    # print(f"  🗑️  Fichier zip supprimé pour libérer de l'espace")
    
    return color_dir


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CHARGEMENT ET INDEXATION DU DATASET
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset_index(color_dir: Path):
    """
    Parcourt le dataset et construit les listes (chemin, label, classe).

    Structure PlantVillage :
        color/
            Apple___Apple_scab/image1.JPG
            Apple___Black_rot/image2.JPG
            Tomato___healthy/image3.JPG
            ...

    Paramètres :
        color_dir (Path) : dossier 'color' du dataset

    Retourne :
        image_paths  (list[Path]) : chemin vers chaque image
        labels       (list[str])  : nom de classe de chaque image
        class_names  (list[str])  : liste triée des 38 classes
        class_to_idx (dict)       : {nom_classe: index_entier}
    """
    image_paths = []
    labels      = []

    class_dirs  = sorted([d for d in color_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    # Extensions acceptées (PlantVillage utilise JPG en majorité)
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

    for class_dir in class_dirs:
        for f in class_dir.iterdir():
            if f.suffix in EXTENSIONS:
                image_paths.append(f)
                labels.append(class_dir.name)

    print(f"\n  ✅ Index construit : {len(image_paths)} images, {len(class_names)} classes")
    return image_paths, labels, class_names, class_to_idx


# ─────────────────────────────────────────────────────────────────────────────
# 4.  NETTOYAGE ET VÉRIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def verify_and_clean(image_paths: list, labels: list):
    """
    Vérifie l'intégrité de chaque image :
        - Fichier lisible (non corrompu)
        - Dimensions ≥ 32×32 px
        - 3 canaux RGB
    Retourne les listes nettoyées et un rapport.
    """
    print("\n" + "─"*60)
    print("  ÉTAPE 1 — Vérification de l'intégrité du dataset")
    print("─"*60)

    valid_paths, valid_labels = [], []
    n_corrupt, n_small, n_wrong_ch = 0, 0, 0

    for path, label in zip(image_paths, labels):
        try:
            # PIL.verify() détecte les fichiers tronqués sans charger toute l'image
            with Image.open(path) as img:
                img.verify()

            # Recharger (verify() invalide l'objet)
            img = Image.open(path).convert("RGB")
            w, h = img.size

            if w < 32 or h < 32:
                n_small += 1
                continue

            arr = np.array(img)
            if arr.ndim != 3 or arr.shape[2] != 3:
                n_wrong_ch += 1
                continue

            valid_paths.append(path)
            valid_labels.append(label)

        except Exception:
            n_corrupt += 1

    print(f"  Total initial          : {len(image_paths)}")
    print(f"  ✅ Images valides      : {len(valid_paths)}")
    print(f"  ❌ Corrompues          : {n_corrupt}")
    print(f"  ⚠️  Trop petites       : {n_small}")
    print(f"  ⚠️  Canaux incorrects  : {n_wrong_ch}")

    return valid_paths, valid_labels


# ─────────────────────────────────────────────────────────────────────────────
# 5.  ANALYSE DE LA DISTRIBUTION DES CLASSES
# ─────────────────────────────────────────────────────────────────────────────

def analyze_and_plot_distribution(labels: list):
    """
    Compte les images par classe et génère une figure de distribution.
    Identifie les classes sur/sous-représentées.
    """
    print("\n" + "─"*60)
    print("  ÉTAPE 2 — Analyse de la distribution des classes")
    print("─"*60)

    counter     = Counter(labels)
    class_names = sorted(counter.keys(), key=lambda k: counter[k], reverse=True)
    counts      = [counter[k] for k in class_names]

    print(f"  Classe la plus représentée : {class_names[0]}  ({counts[0]} images)")
    print(f"  Classe la moins représentée: {class_names[-1]} ({counts[-1]} images)")
    print(f"  Ratio déséquilibre         : {counts[0]/counts[-1]:.1f}x")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10), facecolor=C["light"])
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # Barres horizontales
    ax1 = fig.add_subplot(gs[0])
    short_names = [n.replace("___", "\n").replace("_", " ") for n in class_names]
    colors_bar  = [C["red"] if c < 2000 else C["orange"] if c < 3500 else C["accent"]
                   for c in counts]
    bars = ax1.barh(short_names, counts, color=colors_bar, edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Nombre d'images", fontsize=12, color=C["dark"])
    ax1.set_title("Distribution des 38 classes PlantVillage", fontsize=12,
                  fontweight="bold", color=C["dark"], pad=10)
    ax1.tick_params(axis="y", labelsize=7)
    ax1.set_facecolor(C["light"])
    ax1.invert_yaxis()  # plus représenté en haut
    # Annoter les valeurs
    for bar, count in zip(bars, counts):
        ax1.text(count + 80, bar.get_y() + bar.get_height()/2,
                 f"{count}", va="center", ha="left", fontsize=7, color=C["dark"])

    # Légende couleurs
    legend_elements = [
        mpatches.Patch(color=C["accent"], label="Bien représenté (≥3500)"),
        mpatches.Patch(color=C["orange"], label="Moyen (2000–3500)"),
        mpatches.Patch(color=C["red"],    label="Sous-représenté (<2000)"),
    ]
    ax1.legend(handles=legend_elements, fontsize=9, loc="lower right")

    # Camembert des 10 premières + Autres
    ax2 = fig.add_subplot(gs[1])
    top_n = 10
    pie_labels = [n.split("___")[-1].replace("_", " ") for n in class_names[:top_n]]
    pie_vals   = counts[:top_n] + [sum(counts[top_n:])]
    pie_labels.append("Autres (28 classes)")
    colors_pie = list(plt.cm.tab20.colors[:top_n]) + [(0.85, 0.85, 0.85)]
    wedges, texts, autotexts = ax2.pie(
        pie_vals, labels=pie_labels, autopct="%1.1f%%",
        startangle=140, pctdistance=0.82, colors=colors_pie,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    for t in texts:     t.set_fontsize(8)
    for at in autotexts: at.set_fontsize(7.5)
    ax2.set_title("Répartition — Top 10 classes + Autres", fontsize=12,
                  fontweight="bold", color=C["dark"], pad=10)

    fig.suptitle("CR2 — Distribution des Classes du Dataset PlantVillage (54 306 images)",
                 fontsize=14, fontweight="bold", color=C["dark"], y=1.01)

    out = OUTPUT_DIR / "01_distribution_classes.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C["light"])
    plt.close()
    print(f"  → Sauvegardé : {out}")

    return counter


# ─────────────────────────────────────────────────────────────────────────────
# 6.  SÉLECTION D'UN ÉCHANTILLON REPRÉSENTATIF (une image par espèce)
# ─────────────────────────────────────────────────────────────────────────────

def select_sample_images(image_paths: list, labels: list,
                          class_names: list, n: int = 6) -> list:
    """
    Sélectionne n images réelles depuis des classes variées pour les illustrations.
    Choisit des classes différentes (maladies et saine) pour la diversité.

    Paramètres :
        image_paths (list)  : tous les chemins
        labels      (list)  : tous les labels
        class_names (list)  : liste des classes
        n           (int)   : nombre d'images à sélectionner

    Retourne :
        list of (Path, str) : [(chemin, label), ...]
    """
    # Grouper les images par classe
    by_class = {}
    for path, label in zip(image_paths, labels):
        by_class.setdefault(label, []).append(path)

    # Choisir des classes représentatives et variées
    priority_keywords = [
        "Tomato", "Apple", "Corn", "Grape", "Potato",
        "Pepper", "Cherry", "Peach", "Strawberry"
    ]

    selected = []
    used_plants = set()

    # D'abord essayer les classes prioritaires
    for keyword in priority_keywords:
        if len(selected) >= n:
            break
        plant = keyword.lower()
        if plant in used_plants:
            continue
        for cls in by_class:
            if keyword in cls and by_class[cls]:
                path = random.choice(by_class[cls])
                selected.append((path, cls))
                used_plants.add(plant)
                break

    # Compléter si nécessaire
    remaining = [c for c in by_class if not any(k.lower() in c.lower() for k in used_plants)]
    random.shuffle(remaining)
    for cls in remaining:
        if len(selected) >= n:
            break
        if by_class[cls]:
            selected.append((random.choice(by_class[cls]), cls))

    return selected[:n]


def show_raw_samples(samples: list):
    """
    Affiche les images brutes du dataset (avant tout traitement) pour référence.
    C'est la figure de base servant de référence pour toutes les comparaisons.

    Paramètres :
        samples (list) : list of (Path, label_str)
    """
    n = len(samples)
    fig, axes = plt.subplots(2, n, figsize=(n * 3.2, 7),
                             facecolor=C["light"],
                             gridspec_kw={"height_ratios": [4, 1]})
    fig.suptitle("Images BRUTES du Dataset PlantVillage (avant tout prétraitement)",
                 fontsize=13, fontweight="bold", color=C["dark"])

    for col, (path, label) in enumerate(samples):
        img = np.array(Image.open(path).convert("RGB"))

        axes[0, col].imshow(img)
        axes[0, col].set_title(
            label.replace("___", "\n").replace("_", " "),
            fontsize=8, fontweight="bold", pad=5, color=C["dark"]
        )
        axes[0, col].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in axes[0, col].spines.values():
            spine.set_edgecolor("#CCCCCC")

        # Infos techniques sous chaque image
        h, w, c = img.shape
        info_txt = f"{w}×{h} px\nRGB — valeurs [0,255]"
        axes[1, col].text(0.5, 0.5, info_txt,
                          ha="center", va="center", fontsize=8,
                          color=C["grey"], transform=axes[1, col].transAxes)
        axes[1, col].axis("off")

    plt.tight_layout()
    out = OUTPUT_DIR / "00_echantillon_brut.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C["light"])
    plt.close()
    print(f"  → Sauvegardé : {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  ÉTAPE A : REDIMENSIONNEMENT
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


def step_resize_and_plot(samples: list):
    """
    Applique le redimensionnement sur les images réelles et génère
    une figure avant / après avec 3 résolutions.

    Figure :
        Ligne 0 : images brutes 256×256
        Ligne 1 : images 128×128 (cible CNN from scratch)
        Ligne 2 : images 224×224 (cible Transfer Learning)
    """
    print("\n" + "─"*60)
    print("  ÉTAPE A — Redimensionnement")
    print("─"*60)

    n      = len(samples)
    sizes  = [256, 128, 224]
    titles = ["AVANT — Brut (256×256)", "APRÈS — CNN (128×128)", "APRÈS — Transfer Learning (224×224)"]
    colors_title = [C["red"], C["accent"], C["blue"]]

    fig, axes = plt.subplots(3, n, figsize=(n * 3.2, 10.5), facecolor=C["light"])
    fig.suptitle("CR2 — Étape A : Redimensionnement des Images",
                 fontsize=13, fontweight="bold", color=C["dark"])

    for row, (size, title, col_t) in enumerate(zip(sizes, titles, colors_title)):
        # Titre de la ligne
        axes[row, 0].set_ylabel(title, fontsize=10, fontweight="bold",
                                color=col_t, rotation=90, labelpad=8)

        for col, (path, label) in enumerate(samples):
            img_orig = np.array(Image.open(path).convert("RGB"))
            img_res  = resize_image(img_orig, (size, size))

            axes[row, col].imshow(img_res)
            if row == 0:
                axes[row, col].set_title(
                    label.split("___")[-1].replace("_", " "),
                    fontsize=8, fontweight="bold", pad=5
                )
            axes[row, col].tick_params(left=False, bottom=False,
                                       labelleft=False, labelbottom=False)
            axes[row, col].set_xlabel(f"{size}×{size} px",
                                      fontsize=8, color=C["grey"])
            for spine in axes[row, col].spines.values():
                spine.set_edgecolor("#CCCCCC")

    plt.tight_layout()
    out = OUTPUT_DIR / "02_avant_apres_resize.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C["light"])
    plt.close()
    print(f"  ✅ Redimensionnement appliqué → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  ÉTAPE B : NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def normalize_minmax(img: np.ndarray) -> np.ndarray:
    """
    Normalisation min-max : pixel / 255.0
    Valeurs uint8 [0,255] → float32 [0.0, 1.0]
    Utilisée pour les CNN entraînés from scratch.
    """
    return img.astype(np.float32) / 255.0


def step_normalize_and_plot(samples: list):
    """
    Applique la normalisation sur des images réelles redimensionnées,
    et génère une figure complète :
        - Colonne gauche  : image brute + histogramme RGB
        - Colonne droite  : image normalisée + histogramme RGB
    Une ligne par image d'exemple.
    """
    print("\n" + "─"*60)
    print("  ÉTAPE B — Normalisation")
    print("─"*60)

    # On travaille sur 3 images pour ne pas surcharger la figure
    sub_samples = samples[:3]
    n = len(sub_samples)

    fig = plt.figure(figsize=(16, n * 5), facecolor=C["light"])
    fig.suptitle("CR2 — Étape B : Normalisation des Valeurs de Pixels",
                 fontsize=13, fontweight="bold", color=C["dark"])

    outer_gs = gridspec.GridSpec(n, 1, figure=fig, hspace=0.55)

    ch_colors  = [C["red"], C["accent"], C["blue"]]
    ch_labels  = ["R", "G", "B"]

    for row, (path, label) in enumerate(sub_samples):
        # Charger et redimensionner
        img_orig = np.array(Image.open(path).convert("RGB"))
        img_128  = resize_image(img_orig, IMG_SIZE_CNN)   # uint8
        img_norm = normalize_minmax(img_128)              # float32

        # Sous-grille : [image_brute | histo_brut | image_norm | histo_norm]
        inner_gs = outer_gs[row].subgridspec(1, 4, wspace=0.35)

        # ── Image brute ──────────────────────────────────────────────────────
        ax_img_raw = fig.add_subplot(inner_gs[0])
        ax_img_raw.imshow(img_128)
        ax_img_raw.set_title(
            f"AVANT — {label.split('___')[-1].replace('_',' ')}\n(uint8 [0, 255])",
            fontsize=9, fontweight="bold", color=C["red"]
        )
        ax_img_raw.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # ── Histogramme brut ─────────────────────────────────────────────────
        ax_hist_raw = fig.add_subplot(inner_gs[1])
        for ch, (clr, lbl) in enumerate(zip(ch_colors, ch_labels)):
            ax_hist_raw.hist(img_128[:, :, ch].ravel(), bins=64,
                             color=clr, alpha=0.55, label=lbl, density=True)
        ax_hist_raw.set_title("Histogramme RGB (brut)", fontsize=9)
        ax_hist_raw.set_xlabel("Intensité [0, 255]", fontsize=8)
        ax_hist_raw.set_ylabel("Densité", fontsize=8)
        ax_hist_raw.legend(fontsize=8)
        ax_hist_raw.set_facecolor(C["light"])

        # ── Image normalisée ─────────────────────────────────────────────────
        ax_img_norm = fig.add_subplot(inner_gs[2])
        ax_img_norm.imshow(img_norm)
        ax_img_norm.set_title(
            f"APRÈS — {label.split('___')[-1].replace('_',' ')}\n(float32 [0.0, 1.0])",
            fontsize=9, fontweight="bold", color=C["accent"]
        )
        ax_img_norm.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # ── Histogramme normalisé ────────────────────────────────────────────
        ax_hist_norm = fig.add_subplot(inner_gs[3])
        for ch, (clr, lbl) in enumerate(zip(ch_colors, ch_labels)):
            ax_hist_norm.hist(img_norm[:, :, ch].ravel(), bins=64,
                              color=clr, alpha=0.55, label=lbl, density=True)
        ax_hist_norm.set_title("Histogramme RGB (normalisé)", fontsize=9)
        ax_hist_norm.set_xlabel("Intensité [0.0, 1.0]", fontsize=8)
        ax_hist_norm.set_ylabel("Densité", fontsize=8)
        ax_hist_norm.legend(fontsize=8)
        ax_hist_norm.set_facecolor(C["light"])

    out = OUTPUT_DIR / "03_avant_apres_normalisation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C["light"])
    plt.close()
    print(f"  ✅ Normalisation appliquée → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  ÉTAPE C : FILTRAGE ET RÉDUCTION DU BRUIT
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
    kernel = np.array([[0, -1,  0],
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


def step_filter_and_plot(samples: list):
    """
    Applique les 4 filtres sur des images réelles et génère une figure
    comparative montrant avant / après chaque filtre.

    Figure :
        Lignes = images d'exemple
        Colonnes = [Brute | Gaussien | Médian | Bilatéral | Sharpening]
    """
    print("\n" + "─"*60)
    print("  ÉTAPE C — Filtrage et réduction du bruit")
    print("─"*60)

    sub_samples = samples[:4]
    n = len(sub_samples)

    filter_names = [
        "AVANT\n(brut, 128×128)",
        "Filtre Gaussien\n(σ=1.0)",
        "Filtre Médian\n(noyau 3×3)",
        "Filtre Bilatéral\n(d=9, σ=75)",
        "Sharpening\n(Laplacien)",
    ]
    col_colors = [C["red"], C["accent"], C["blue"], C["orange"], C["grey"]]

    fig, axes = plt.subplots(n, 5, figsize=(18, n * 3.8), facecolor=C["light"])
    fig.suptitle("CR2 — Étape C : Filtrage des Images (Comparaison des 4 filtres)",
                 fontsize=13, fontweight="bold", color=C["dark"])

    for row, (path, label) in enumerate(sub_samples):
        img_orig = np.array(Image.open(path).convert("RGB"))
        img_128  = resize_image(img_orig, IMG_SIZE_CNN)   # uint8 128×128

        # Calculer les versions filtrées
        filtered_imgs = [
            img_128,
            apply_gaussian_filter(img_128, sigma=1.0),
            apply_median_filter(img_128,   ksize=3),
            apply_bilateral_filter(img_128),
            apply_sharpening(img_128),
        ]

        for col, (filt_img, fname, fcol) in enumerate(
            zip(filtered_imgs, filter_names, col_colors)
        ):
            axes[row, col].imshow(filt_img)

            # En-têtes de colonnes (seulement première ligne)
            if row == 0:
                axes[row, col].set_title(fname, fontsize=9, fontweight="bold",
                                         color=fcol, pad=6)

            # Label de la ligne (seulement première colonne)
            if col == 0:
                axes[row, col].set_ylabel(
                    label.split("___")[-1].replace("_", " "),
                    fontsize=8, fontweight="bold", labelpad=5
                )

            # PSNR (sauf pour l'original)
            if col > 0:
                psnr_val = compute_psnr(img_128, filt_img)
                axes[row, col].set_xlabel(f"PSNR = {psnr_val:.1f} dB",
                                          fontsize=8, color=C["dark"])

            axes[row, col].tick_params(left=False, bottom=False,
                                       labelleft=False, labelbottom=False)
            for spine in axes[row, col].spines.values():
                spine.set_edgecolor("#CCCCCC")

    plt.tight_layout()
    out = OUTPUT_DIR / "04_avant_apres_filtrage.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C["light"])
    plt.close()
    print(f"  ✅ Filtrage appliqué → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. ÉTAPE D : AUGMENTATION DE DONNÉES
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

    H, W = img.shape[:2]
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
    zoom = 0.85   # on garde 85% du centre → effet zoom +15%
    cy, cx = H // 2, W // 2
    h_crop = int(H * zoom / 2)
    w_crop = int(W * zoom / 2)
    cropped = img[cy - h_crop: cy + h_crop, cx - w_crop: cx + w_crop]
    result["Zoom In (+15%)"] = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LINEAR)

    # ── 5. Zoom Out (padding + redimensionnement) ─────────────────────────────
    pad  = int(min(H, W) * 0.12)
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


def step_augmentation_and_plot(samples: list):
    """
    Applique les 8 transformations sur 3 images réelles différentes.

    Figure :
        Chaque ligne = une image source
        Colonnes = [Original | transfo_1 | ... | transfo_8]
    """
    print("\n" + "─"*60)
    print("  ÉTAPE D — Augmentation de données")
    print("─"*60)

    sub_samples = samples[:3]
    n_rows = len(sub_samples)

    # Calculer d'abord toutes les augmentations pour connaître le nombre de colonnes
    all_augmented = []
    for i, (path, label) in enumerate(sub_samples):
        img_orig = np.array(Image.open(path).convert("RGB"))
        img_128  = resize_image(img_orig, IMG_SIZE_CNN)
        augs     = augment_image(img_128, seed_offset=i * 10)
        all_augmented.append((img_128, label, augs))

    n_cols = 1 + len(all_augmented[0][2])   # 1 original + 8 transformations

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.4, n_rows * 3.2),
        facecolor=C["light"]
    )
    fig.suptitle("CR2 — Étape D : Augmentation de Données (8 transformations sur images réelles)",
                 fontsize=13, fontweight="bold", color=C["dark"])

    col_names = ["ORIGINAL"] + list(all_augmented[0][2].keys())

    for row, (img_128, label, augs) in enumerate(all_augmented):
        all_imgs = [img_128] + list(augs.values())

        for col, (disp_img, col_name) in enumerate(zip(all_imgs, col_names)):
            axes[row, col].imshow(disp_img)

            # En-têtes de colonnes (première ligne seulement)
            if row == 0:
                color = C["red"] if col == 0 else C["accent"]
                axes[row, col].set_title(col_name, fontsize=8, fontweight="bold",
                                         color=color, pad=5)

            # Label de l'image (première colonne seulement)
            if col == 0:
                axes[row, col].set_ylabel(
                    label.split("___")[-1].replace("_", " "),
                    fontsize=8, fontweight="bold", labelpad=5
                )

            axes[row, col].tick_params(left=False, bottom=False,
                                       labelleft=False, labelbottom=False)
            for spine in axes[row, col].spines.values():
                spine.set_edgecolor("#CCCCCC")

    plt.tight_layout()
    out = OUTPUT_DIR / "05_avant_apres_augmentation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C["light"])
    plt.close()
    print(f"  ✅ Augmentation appliquée → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. FIGURE FINALE : PIPELINE COMPLET (brut → étape par étape → final)
# ─────────────────────────────────────────────────────────────────────────────

def plot_full_pipeline(samples: list):
    """
    Génère la figure synthétique finale montrant UNE image traversant
    toutes les étapes du pipeline de A à Z :

    Image brute (256×256)
        ↓  Redimensionnement
    128×128 (uint8)
        ↓  Normalisation
    128×128 (float32, [0,1])
        ↓  Filtre Gaussien
    128×128 (filtré)
        ↓  Augmentation (exemple : rotation)
    128×128 (augmenté)
    """
    print("\n" + "─"*60)
    print("  FIGURE FINALE — Pipeline complet illustré")
    print("─"*60)

    # Choisir une image avec une maladie visible pour mieux illustrer les étapes
    disease_samples = [(p, l) for p, l in samples if "healthy" not in l.lower()]
    chosen_path, chosen_label = disease_samples[0] if disease_samples else samples[0]

    # ── Calculer chaque étape ─────────────────────────────────────────────────
    img_brut   = np.array(Image.open(chosen_path).convert("RGB"))         # brut 256×256
    img_256    = resize_image(img_brut, (256, 256))                       # déjà 256 normalement
    img_128    = resize_image(img_brut, IMG_SIZE_CNN)                     # ① redimension
    img_norm   = normalize_minmax(img_128)                                 # ② normalisation
    img_gauss  = apply_gaussian_filter(img_128, sigma=1.0)                # ③ filtre gaussien
    img_aug    = augment_image(img_128, seed_offset=99)                   # ④ augmentation
    img_final  = list(img_aug.values())[2]                                 #    prendre rotation

    steps = [
        (img_256,   "ORIGINAL (BRUT)\n256×256 px — uint8 [0,255]",     C["red"]),
        (img_128,   "① REDIMENSIONNEMENT\n128×128 px — uint8 [0,255]", C["orange"]),
        (img_norm,  "② NORMALISATION\n128×128 px — float32 [0,1]",     C["blue"]),
        (img_gauss, "③ FILTRE GAUSSIEN\n128×128 px — σ=1.0",           C["grey"]),
        (img_final, "④ AUGMENTATION\n128×128 px — Rotation",           C["accent"]),
    ]
    n = len(steps)

    fig = plt.figure(figsize=(n * 3.4, 7.5), facecolor=C["light"])
    gs  = gridspec.GridSpec(2, n, figure=fig,
                            height_ratios=[5, 0.6], hspace=0.08, wspace=0.08)

    fig.suptitle(
        f"CR2 — Pipeline Complet : {chosen_label.replace('___', ' — ').replace('_', ' ')}",
        fontsize=13, fontweight="bold", color=C["dark"]
    )

    for col, (img, title, color) in enumerate(steps):
        # Image
        ax_img = fig.add_subplot(gs[0, col])
        # Normaliser pour l'affichage si float
        disp = np.clip(img, 0, 1) if img.dtype == np.float32 else img
        ax_img.imshow(disp)
        ax_img.set_title(title, fontsize=9, fontweight="bold",
                         color=color, pad=7, multialignment="center")
        ax_img.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax_img.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

        # Flèche entre les étapes
        ax_arrow = fig.add_subplot(gs[1, col])
        ax_arrow.axis("off")
        if col < n - 1:
            ax_arrow.annotate(
                "", xy=(1.15, 0.5), xytext=(0.9, 0.5),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=C["dark"],
                                lw=2.0, mutation_scale=18)
            )

    plt.tight_layout()
    out = OUTPUT_DIR / "06_pipeline_final_complet.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C["light"])
    plt.close()
    print(f"  ✅ Pipeline final → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 12. CONSTRUCTION DES GÉNÉRATEURS KERAS POUR L'ENTRAÎNEMENT (CR3)
# ─────────────────────────────────────────────────────────────────────────────

def build_keras_generators(color_dir: Path):
    """
    Construit les générateurs Keras pour l'entraînement futur (CR3).
    L'augmentation est appliquée en temps réel (on-the-fly) sur le train set.

    Split utilisé : 70% train / 15% val / 15% test (via deux passes)

    Paramètres :
        color_dir (Path) : dossier 'color' du dataset PlantVillage

    Retourne :
        train_gen, val_gen, class_names
    """
    print("\n" + "─"*60)
    print("  GÉNÉRATEURS KERAS — Prêts pour l'entraînement (CR3)")
    print("─"*60)

    # ── Générateur entraînement (avec augmentation) ───────────────────────────
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,            # normalisation min-max → [0, 1]
        rotation_range=30,            # rotation ±30°
        width_shift_range=0.10,       # translation horizontale ±10%
        height_shift_range=0.10,      # translation verticale ±10%
        zoom_range=0.20,              # zoom ±20%
        horizontal_flip=True,         # flip horizontal
        vertical_flip=True,           # flip vertical
        brightness_range=[0.65, 1.35],# luminosité ±35%
        fill_mode="nearest",          # remplissage pixel voisin le plus proche
        validation_split=0.15,        # 15% réservé à la validation
    )

    # ── Générateur validation/test (SANS augmentation) ─────────────────────────
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.15,
    )

    # ── Flux d'images depuis le dossier ──────────────────────────────────────
    train_gen = train_datagen.flow_from_directory(
        directory=str(color_dir),
        target_size=IMG_SIZE_CNN,     # 128×128
        batch_size=BATCH_SIZE,
        class_mode="categorical",     # one-hot encoding, 38 classes
        subset="training",            # 85% des données
        shuffle=True,
        seed=SEED,
    )

    val_gen = val_datagen.flow_from_directory(
        directory=str(color_dir),
        target_size=IMG_SIZE_CNN,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",          # 15% des données
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


# ─────────────────────────────────────────────────────────────────────────────
# 13. RÉSUMÉ FINAL — STATISTIQUES DU DATASET APRÈS PRÉTRAITEMENT
# ─────────────────────────────────────────────────────────────────────────────

def print_final_summary(image_paths: list, labels: list, class_names: list):
    """
    Affiche un tableau récapitulatif de toutes les étapes du prétraitement.
    """
    print("\n" + "═"*60)
    print("  RÉSUMÉ DU PIPELINE DE PRÉTRAITEMENT — CR2")
    print("═"*60)

    counter = Counter(labels)
    counts  = sorted(counter.values())

    rows = [
        ("Dataset",            f"PlantVillage (Manuel)"),
        ("Total images",       f"{len(image_paths):,}"),
        ("Nombre de classes",  f"{len(class_names)}"),
        ("Résolution initiale","256×256 px (RGB)"),
        ("─── ÉTAPES ───",     ""),
        ("A — Redim. CNN",     f"128×128 px (INTER_LINEAR)"),
        ("A — Redim. TL",      f"224×224 px (INTER_LINEAR)"),
        ("B — Normalisation",  "uint8 [0,255] → float32 [0.0, 1.0]"),
        ("C — Filtre retenu",  "Gaussien σ=1.0 (+ Bilatéral optionnel)"),
        ("D — Augmentation",   "8 transformations (rotation, flip, zoom...)"),
        ("Split train/val",    "85% / 15% (via Keras validation_split)"),
        ("Batch size",         f"{BATCH_SIZE}"),
        ("─── SORTIE ───",     ""),
        ("Figures générées",   str(OUTPUT_DIR.resolve())),
    ]

    col_w = 28
    for key, val in rows:
        if key.startswith("─"):
            print(f"  {'─'*56}")
        else:
            print(f"  {key:<{col_w}} {val}")

    print("═"*60)
    print("  → Figures sauvegardées dans :", OUTPUT_DIR.resolve())
    print("  → Prochain CR3 : Conception du modèle CNN / Transfer Learning")
    print("═"*60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── 1. Extraction du dataset (manuel) ────────────────────────────────────
    color_dir = extract_plantvillage()

    # ── 2. Indexation du dataset ──────────────────────────────────────────────
    image_paths, labels, class_names, class_to_idx = load_dataset_index(color_dir)

    # ── 3. Nettoyage / vérification ──────────────────────────────────────────
    image_paths, labels = verify_and_clean(image_paths, labels)

    # ── 4. Distribution des classes ───────────────────────────────────────────
    counter = analyze_and_plot_distribution(labels)

    # ── 5. Sélectionner un échantillon représentatif (6 images réelles) ──────
    print("\n" + "─"*60)
    print("  Sélection des images d'illustration (réelles du dataset)...")
    print("─"*60)
    samples = select_sample_images(image_paths, labels, class_names, n=N_SAMPLES_VIZ)
    for p, l in samples:
        print(f"    {l:50s}  → {p.name}")

    # ── 6. Afficher les images brutes (référence) ─────────────────────────────
    show_raw_samples(samples)

    # ── 7. Étape A : Redimensionnement ────────────────────────────────────────
    step_resize_and_plot(samples)

    # ── 8. Étape B : Normalisation ────────────────────────────────────────────
    step_normalize_and_plot(samples)

    # ── 9. Étape C : Filtrage ────────────────────────────────────────────────
    step_filter_and_plot(samples)

    # ── 10. Étape D : Augmentation ────────────────────────────────────────────
    step_augmentation_and_plot(samples)

    # ── 11. Figure pipeline complet ───────────────────────────────────────────
    plot_full_pipeline(samples)

    # ── 12. Générateurs Keras (prêts pour le CR3) ────────────────────────────
    train_gen, val_gen, class_names_out = build_keras_generators(color_dir)

    # ── 13. Résumé final ──────────────────────────────────────────────────────
    print_final_summary(image_paths, labels, class_names)

    print("\n  ✅ CR2 terminé ! Toutes les figures sont dans :", OUTPUT_DIR.resolve())