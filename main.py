"""
main.py
───────
Point d'entrée principal du pipeline CR2.
Orchestre tous les modules dans l'ordre du pipeline original.

Exécution :
    python main.py                          # ZIP dans le dossier courant
    python main.py chemin/vers/fichier.zip  # ZIP ailleurs

Modules appelés dans l'ordre :
    config.settings         → initialisation (chemins, seeds, dossiers)
    data.loader             → extraction ZIP + indexation
    data.cleaner            → vérification intégrité
    utils.sampling          → sélection d'échantillon représentatif
    visualization.plots     → toutes les figures avant/après
    preprocessing.generators→ générateurs Keras (prêts pour CR3)
    utils.summary           → tableau récapitulatif
"""

import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── Config ────────────────────────────────────────────────────────────────────
import config.settings as settings
settings.init()

# ── Data ──────────────────────────────────────────────────────────────────────
from data.loader  import extract_plantvillage, load_dataset_index
from data.cleaner import verify_and_clean

# ── Utils ─────────────────────────────────────────────────────────────────────
from utils.sampling import select_sample_images
from utils.summary  import print_final_summary

# ── Visualization ─────────────────────────────────────────────────────────────
from visualization.plots import (
    plot_distribution,
    show_raw_samples,
    plot_resize,
    plot_normalization,
    plot_filtering,
    plot_augmentation,
    plot_full_pipeline,
)

# ── Preprocessing ─────────────────────────────────────────────────────────────
from preprocessing.generators import build_keras_generators


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Extraction du dataset (ZIP manuel)
    color_dir = extract_plantvillage()

    # 2. Indexation
    image_paths, labels, class_names, class_to_idx = load_dataset_index(color_dir)

    # 3. Nettoyage / vérification intégrité
    image_paths, labels = verify_and_clean(image_paths, labels)

    # 4. Distribution des classes → figure 01
    counter = plot_distribution(labels)

    # 5. Sélection d'un échantillon représentatif (6 images réelles)
    print("\n" + "─"*60)
    print("  Sélection des images d'illustration (réelles du dataset)...")
    print("─"*60)
    samples = select_sample_images(image_paths, labels, class_names,
                                   n=settings.N_SAMPLES_VIZ)
    for p, l in samples:
        print(f"    {l:50s}  → {p.name}")

    # 6. Images brutes → figure 00
    show_raw_samples(samples)

    # 7. Étape A — Redimensionnement → figure 02
    plot_resize(samples)

    # 8. Étape B — Normalisation → figure 03
    plot_normalization(samples)

    # 9. Étape C — Filtrage → figure 04
    plot_filtering(samples)

    # 10. Étape D — Augmentation → figure 05
    plot_augmentation(samples)

    # 11. Pipeline complet → figure 06
    plot_full_pipeline(samples)

    # 12. Générateurs Keras (prêts pour CR3)
    train_gen, val_gen, class_names_out = build_keras_generators(color_dir)

    # 13. Résumé final
    print_final_summary(image_paths, labels, class_names)

    print("\n  ✅ CR2 terminé ! Toutes les figures sont dans :",
          settings.OUTPUT_DIR.resolve())
