"""
utils/sampling.py
─────────────────
Sélection d'un échantillon représentatif d'images du dataset
pour alimenter les fonctions de visualisation.
Correspond à la section 6 du pipeline original.
"""

import random
from pathlib import Path

from config.settings import SEED


# ─────────────────────────────────────────────────────────────────────────────
# SÉLECTION D'ÉCHANTILLON
# ─────────────────────────────────────────────────────────────────────────────

def select_sample_images(image_paths: list, labels: list,
                          class_names: list, n: int = 6) -> list:
    """
    Sélectionne n images réelles depuis des classes variées pour les illustrations.
    Choisit des classes différentes (maladies et saines) pour la diversité.

    Paramètres :
        image_paths (list) : tous les chemins du dataset
        labels      (list) : tous les labels associés
        class_names (list) : liste des classes disponibles
        n           (int)  : nombre d'images à sélectionner

    Retourne :
        list of (Path, str) : [(chemin, label), ...]
    """
    random.seed(SEED)

    # Grouper les images par classe
    by_class = {}
    for path, label in zip(image_paths, labels):
        by_class.setdefault(label, []).append(path)

    # Classes représentatives et variées à prioriser
    priority_keywords = [
        "Tomato", "Apple", "Corn", "Grape", "Potato",
        "Pepper", "Cherry", "Peach", "Strawberry"
    ]

    selected    = []
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
    remaining = [c for c in by_class
                 if not any(k.lower() in c.lower() for k in used_plants)]
    random.shuffle(remaining)
    for cls in remaining:
        if len(selected) >= n:
            break
        if by_class[cls]:
            selected.append((random.choice(by_class[cls]), cls))

    return selected[:n]
