"""
data/cleaner.py
───────────────
Nettoyage et vérification de l'intégrité du dataset PlantVillage.
Correspond à la section 4 du pipeline original.
"""

import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# VÉRIFICATION ET NETTOYAGE
# ─────────────────────────────────────────────────────────────────────────────

def verify_and_clean(image_paths: list, labels: list):
    """
    Vérifie l'intégrité de chaque image :
        - Fichier lisible (non corrompu)
        - Dimensions ≥ 32×32 px
        - 3 canaux RGB
    Retourne les listes nettoyées et un rapport.

    Paramètres :
        image_paths (list) : liste des chemins d'images
        labels      (list) : liste des labels associés

    Retourne :
        valid_paths  (list) : chemins des images valides
        valid_labels (list) : labels correspondants
    """
    print("\n" + "─"*60)
    print("  ÉTAPE 1 — Vérification de l'intégrité du dataset")
    print("─"*60)

    valid_paths, valid_labels   = [], []
    n_corrupt, n_small, n_wrong_ch = 0, 0, 0

    for path, label in zip(image_paths, labels):
        try:
            # PIL.verify() détecte les fichiers tronqués sans charger toute l'image
            with Image.open(path) as img:
                img.verify()

            # Recharger (verify() invalide l'objet)
            img  = Image.open(path).convert("RGB")
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
