"""
config/settings.py
──────────────────
Configuration globale du projet CR2.
Tous les hyperparamètres, chemins et constantes sont centralisés ici.
Importez ce module dans tous les autres fichiers.
"""

from pathlib import Path
import random
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CHEMINS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR        = Path("./plantvillage_dataset")   # dossier après extraction du zip
OUTPUT_DIR      = Path("./output_cr2")             # figures générées
PROCESSED_DIR   = Path("./plantvillage_processed") # images prétraitées
ZIP_NAME        = " "       # nom attendu du fichier zip

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMÈTRES
# ─────────────────────────────────────────────────────────────────────────────

IMG_SIZE_CNN  = (128, 128)   # résolution CNN from scratch
IMG_SIZE_TL   = (224, 224)   # résolution Transfer Learning (ResNet-50, EfficientNet)
BATCH_SIZE    = 32
SEED          = 42
N_SAMPLES_VIZ = 6            # nombre d'images dans les grilles de comparaison

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE DE COULEURS (matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def init():
    """
    Crée les dossiers de sortie et fixe les graines aléatoires.
    À appeler une seule fois au démarrage du programme (dans main.py).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)

    print("╔" + "═"*68 + "╗")
    print("║   CR2 — Pipeline Prétraitement PlantVillage                        ║")
    print("╚" + "═"*68 + "╝")
    print(f"  → Sorties    : {OUTPUT_DIR.resolve()}")
    print(f"  → Taille CNN : {IMG_SIZE_CNN}   Taille TL : {IMG_SIZE_TL}")
