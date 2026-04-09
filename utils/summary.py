"""
utils/summary.py
────────────────
Affichage du tableau récapitulatif final du pipeline CR2.
Correspond à la section 13 du pipeline original.
"""

from collections import Counter
from config.settings import OUTPUT_DIR, IMG_SIZE_CNN, IMG_SIZE_TL, BATCH_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# RÉSUMÉ FINAL
# ─────────────────────────────────────────────────────────────────────────────

def print_final_summary(image_paths: list, labels: list, class_names: list):
    """
    Affiche un tableau récapitulatif de toutes les étapes du prétraitement.

    Paramètres :
        image_paths (list) : liste des chemins valides après nettoyage
        labels      (list) : labels correspondants
        class_names (list) : liste des 38 classes
    """
    print("\n" + "═"*60)
    print("  RÉSUMÉ DU PIPELINE DE PRÉTRAITEMENT — CR2")
    print("═"*60)

    rows = [
        ("Dataset",            "PlantVillage (Manuel)"),
        ("Total images",       f"{len(image_paths):,}"),
        ("Nombre de classes",  f"{len(class_names)}"),
        ("Résolution initiale","256×256 px (RGB)"),
        ("─── ÉTAPES ───",     ""),
        ("A — Redim. CNN",     f"{IMG_SIZE_CNN[0]}×{IMG_SIZE_CNN[1]} px (INTER_LINEAR)"),
        ("A — Redim. TL",      f"{IMG_SIZE_TL[0]}×{IMG_SIZE_TL[1]} px (INTER_LINEAR)"),
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
