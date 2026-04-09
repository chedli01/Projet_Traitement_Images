"""
data/loader.py
──────────────
Gestion du dataset PlantVillage :
    - Extraction depuis le fichier ZIP (téléchargé manuellement)
    - Indexation des images et labels
Correspond aux sections 2 et 3 du pipeline original.
"""

import sys
import zipfile
from pathlib import Path

from config.settings import BASE_DIR, ZIP_NAME


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION DU DATASET (MANUEL)
# ─────────────────────────────────────────────────────────────────────────────

def extract_plantvillage() -> Path:
    """
    Extrait le dataset PlantVillage depuis un fichier ZIP téléchargé manuellement.

    Cherche le fichier zip dans l'ordre :
        1. Le chemin spécifié par l'utilisateur (argument ligne de commande)
        2. Le fichier ZIP_NAME dans le dossier courant
        3. Tout fichier .zip dans le dossier courant

    Le dataset contient ~54 306 images JPEG réparties en 38 dossiers (classes).

    Retourne :
        Path : chemin vers le dossier 'color' extrait
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

    # ── Trouver le fichier zip ────────────────────────────────────────────────
    zip_path = None

    # Option 1 : Chemin passé en argument
    if len(sys.argv) > 1:
        zip_path = Path(sys.argv[1])
        if not zip_path.exists():
            print(f"  ⚠️  Fichier spécifié introuvable: {zip_path}")
            zip_path = None

    # Option 2 : Fichier avec le nom par défaut dans le dossier courant
    if zip_path is None:
        default_zip = Path(ZIP_NAME)
        if default_zip.exists():
            zip_path = default_zip
            print(f"  📦 Fichier trouvé: {zip_path}")

    # Option 3 : Chercher tout fichier .zip récent
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
     2. Ou spécifiez le chemin: python main.py chemin/vers/fichier.zip
     3. Téléchargez depuis: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
        """)
        sys.exit(1)

    # ── Extraire le zip ───────────────────────────────────────────────────────
    print(f"  📦 Extraction de {zip_path.name}...")
    print(f"     Taille: {zip_path.stat().st_size / (1024**3):.2f} Go")
    print("     Cela peut prendre quelques minutes...")

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
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

    # ── Localiser le dossier 'color' ──────────────────────────────────────────
    color_dir = BASE_DIR / "plantvillage dataset" / "color"
    if not color_dir.exists():
        found = list(BASE_DIR.rglob("color"))
        if found:
            color_dir = found[0]
        else:
            print("  ❌ Dossier 'color' introuvable après extraction.")
            print(f"     Structure extraite : {list(BASE_DIR.iterdir())}")
            sys.exit(1)

    n_classes = sum(1 for d in color_dir.iterdir() if d.is_dir())
    n_images  = sum(1 for _ in color_dir.rglob("*.jpg")) + \
                sum(1 for _ in color_dir.rglob("*.JPG"))
    print(f"  ✅ Dataset prêt — {n_classes} classes, ~{n_images} images dans {color_dir}")

    return color_dir


# ─────────────────────────────────────────────────────────────────────────────
# INDEXATION DU DATASET
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
    image_paths  = []
    labels       = []

    class_dirs   = sorted([d for d in color_dir.iterdir() if d.is_dir()])
    class_names  = [d.name for d in class_dirs]
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
