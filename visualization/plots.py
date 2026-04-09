"""
visualization/plots.py
───────────────────────
Toutes les fonctions de génération de figures du CR2.
Chaque fonction produit un fichier PNG dans OUTPUT_DIR.

Fonctions exportées :
    plot_distribution()       → 01_distribution_classes.png
    show_raw_samples()        → 00_echantillon_brut.png
    plot_resize()             → 02_avant_apres_resize.png
    plot_normalization()      → 03_avant_apres_normalisation.png
    plot_filtering()          → 04_avant_apres_filtrage.png
    plot_augmentation()       → 05_avant_apres_augmentation.png
    plot_full_pipeline()      → 06_pipeline_final_complet.png
"""

import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from collections import Counter
from pathlib import Path
from PIL import Image

from config.settings import C, OUTPUT_DIR, IMG_SIZE_CNN, SEED
from preprocessing.transforms import (
    resize_image, normalize_minmax,
    apply_gaussian_filter, apply_median_filter,
    apply_bilateral_filter, apply_sharpening,
    compute_psnr, augment_image,
)


# ─────────────────────────────────────────────────────────────────────────────
# DISTRIBUTION DES CLASSES  (section 5)
# ─────────────────────────────────────────────────────────────────────────────

def plot_distribution(labels: list) -> Counter:
    """
    Compte les images par classe et génère une figure de distribution.
    Identifie les classes sur/sous-représentées.

    Paramètres :
        labels (list) : liste de tous les labels du dataset

    Retourne :
        Counter : {nom_classe: nombre_images}
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

    fig = plt.figure(figsize=(18, 10), facecolor=C["light"])
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # ── Barres horizontales ───────────────────────────────────────────────────
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
    ax1.invert_yaxis()
    for bar, count in zip(bars, counts):
        ax1.text(count + 80, bar.get_y() + bar.get_height()/2,
                 f"{count}", va="center", ha="left", fontsize=7, color=C["dark"])
    legend_elements = [
        mpatches.Patch(color=C["accent"], label="Bien représenté (≥3500)"),
        mpatches.Patch(color=C["orange"], label="Moyen (2000–3500)"),
        mpatches.Patch(color=C["red"],    label="Sous-représenté (<2000)"),
    ]
    ax1.legend(handles=legend_elements, fontsize=9, loc="lower right")

    # ── Camembert top 10 ─────────────────────────────────────────────────────
    ax2    = fig.add_subplot(gs[1])
    top_n  = 10
    pie_labels = [n.split("___")[-1].replace("_", " ") for n in class_names[:top_n]]
    pie_vals   = counts[:top_n] + [sum(counts[top_n:])]
    pie_labels.append("Autres (28 classes)")
    colors_pie = list(plt.cm.tab20.colors[:top_n]) + [(0.85, 0.85, 0.85)]
    wedges, texts, autotexts = ax2.pie(
        pie_vals, labels=pie_labels, autopct="%1.1f%%",
        startangle=140, pctdistance=0.82, colors=colors_pie,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    for t in texts:      t.set_fontsize(8)
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
# ÉCHANTILLON BRUT  (section 6)
# ─────────────────────────────────────────────────────────────────────────────

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

        h, w, c = img.shape
        axes[1, col].text(0.5, 0.5, f"{w}×{h} px\nRGB — valeurs [0,255]",
                          ha="center", va="center", fontsize=8,
                          color=C["grey"], transform=axes[1, col].transAxes)
        axes[1, col].axis("off")

    plt.tight_layout()
    out = OUTPUT_DIR / "00_echantillon_brut.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C["light"])
    plt.close()
    print(f"  → Sauvegardé : {out}")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE A — REDIMENSIONNEMENT  (section 7)
# ─────────────────────────────────────────────────────────────────────────────

def plot_resize(samples: list):
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

    n            = len(samples)
    sizes        = [256, 128, 224]
    titles       = ["AVANT — Brut (256×256)",
                    "APRÈS — CNN (128×128)",
                    "APRÈS — Transfer Learning (224×224)"]
    colors_title = [C["red"], C["accent"], C["blue"]]

    fig, axes = plt.subplots(3, n, figsize=(n * 3.2, 10.5), facecolor=C["light"])
    fig.suptitle("CR2 — Étape A : Redimensionnement des Images",
                 fontsize=13, fontweight="bold", color=C["dark"])

    for row, (size, title, col_t) in enumerate(zip(sizes, titles, colors_title)):
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
            axes[row, col].set_xlabel(f"{size}×{size} px", fontsize=8, color=C["grey"])
            for spine in axes[row, col].spines.values():
                spine.set_edgecolor("#CCCCCC")

    plt.tight_layout()
    out = OUTPUT_DIR / "02_avant_apres_resize.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C["light"])
    plt.close()
    print(f"  ✅ Redimensionnement appliqué → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE B — NORMALISATION  (section 8)
# ─────────────────────────────────────────────────────────────────────────────

def plot_normalization(samples: list):
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

    sub_samples = samples[:3]
    n           = len(sub_samples)
    ch_colors   = [C["red"], C["accent"], C["blue"]]
    ch_labels   = ["R", "G", "B"]

    fig      = plt.figure(figsize=(16, n * 5), facecolor=C["light"])
    fig.suptitle("CR2 — Étape B : Normalisation des Valeurs de Pixels",
                 fontsize=13, fontweight="bold", color=C["dark"])
    outer_gs = gridspec.GridSpec(n, 1, figure=fig, hspace=0.55)

    for row, (path, label) in enumerate(sub_samples):
        img_orig = np.array(Image.open(path).convert("RGB"))
        img_128  = resize_image(img_orig, IMG_SIZE_CNN)
        img_norm = normalize_minmax(img_128)

        inner_gs = outer_gs[row].subgridspec(1, 4, wspace=0.35)

        # Image brute
        ax_img_raw = fig.add_subplot(inner_gs[0])
        ax_img_raw.imshow(img_128)
        ax_img_raw.set_title(
            f"AVANT — {label.split('___')[-1].replace('_',' ')}\n(uint8 [0, 255])",
            fontsize=9, fontweight="bold", color=C["red"]
        )
        ax_img_raw.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Histogramme brut
        ax_hist_raw = fig.add_subplot(inner_gs[1])
        for ch, (clr, lbl) in enumerate(zip(ch_colors, ch_labels)):
            ax_hist_raw.hist(img_128[:, :, ch].ravel(), bins=64,
                             color=clr, alpha=0.55, label=lbl, density=True)
        ax_hist_raw.set_title("Histogramme RGB (brut)", fontsize=9)
        ax_hist_raw.set_xlabel("Intensité [0, 255]", fontsize=8)
        ax_hist_raw.set_ylabel("Densité", fontsize=8)
        ax_hist_raw.legend(fontsize=8)
        ax_hist_raw.set_facecolor(C["light"])

        # Image normalisée
        ax_img_norm = fig.add_subplot(inner_gs[2])
        ax_img_norm.imshow(img_norm)
        ax_img_norm.set_title(
            f"APRÈS — {label.split('___')[-1].replace('_',' ')}\n(float32 [0.0, 1.0])",
            fontsize=9, fontweight="bold", color=C["accent"]
        )
        ax_img_norm.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Histogramme normalisé
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
# ÉTAPE C — FILTRAGE  (section 9)
# ─────────────────────────────────────────────────────────────────────────────

def plot_filtering(samples: list):
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

    sub_samples  = samples[:4]
    n            = len(sub_samples)
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
        img_128  = resize_image(img_orig, IMG_SIZE_CNN)

        filtered_imgs = [
            img_128,
            apply_gaussian_filter(img_128, sigma=1.0),
            apply_median_filter(img_128, ksize=3),
            apply_bilateral_filter(img_128),
            apply_sharpening(img_128),
        ]

        for col, (filt_img, fname, fcol) in enumerate(
            zip(filtered_imgs, filter_names, col_colors)
        ):
            axes[row, col].imshow(filt_img)
            if row == 0:
                axes[row, col].set_title(fname, fontsize=9, fontweight="bold",
                                         color=fcol, pad=6)
            if col == 0:
                axes[row, col].set_ylabel(
                    label.split("___")[-1].replace("_", " "),
                    fontsize=8, fontweight="bold", labelpad=5
                )
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
# ÉTAPE D — AUGMENTATION  (section 10)
# ─────────────────────────────────────────────────────────────────────────────

def plot_augmentation(samples: list):
    """
    Applique les 8 transformations sur 3 images réelles différentes.

    Figure :
        Chaque ligne = une image source
        Colonnes = [Original | transfo_1 | ... | transfo_8]
    """
    print("\n" + "─"*60)
    print("  ÉTAPE D — Augmentation de données")
    print("─"*60)

    sub_samples   = samples[:3]
    n_rows        = len(sub_samples)
    all_augmented = []

    for i, (path, label) in enumerate(sub_samples):
        img_orig = np.array(Image.open(path).convert("RGB"))
        img_128  = resize_image(img_orig, IMG_SIZE_CNN)
        augs     = augment_image(img_128, seed_offset=i * 10)
        all_augmented.append((img_128, label, augs))

    n_cols    = 1 + len(all_augmented[0][2])   # 1 original + 8 transformations
    col_names = ["ORIGINAL"] + list(all_augmented[0][2].keys())

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.4, n_rows * 3.2),
        facecolor=C["light"]
    )
    fig.suptitle("CR2 — Étape D : Augmentation de Données (8 transformations sur images réelles)",
                 fontsize=13, fontweight="bold", color=C["dark"])

    for row, (img_128, label, augs) in enumerate(all_augmented):
        all_imgs = [img_128] + list(augs.values())
        for col, (disp_img, col_name) in enumerate(zip(all_imgs, col_names)):
            axes[row, col].imshow(disp_img)
            if row == 0:
                color = C["red"] if col == 0 else C["accent"]
                axes[row, col].set_title(col_name, fontsize=8, fontweight="bold",
                                         color=color, pad=5)
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
# FIGURE FINALE — PIPELINE COMPLET  (section 11)
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

    # Choisir une image avec une maladie visible
    disease_samples    = [(p, l) for p, l in samples if "healthy" not in l.lower()]
    chosen_path, chosen_label = disease_samples[0] if disease_samples else samples[0]

    img_brut  = np.array(Image.open(chosen_path).convert("RGB"))
    img_256   = resize_image(img_brut, (256, 256))
    img_128   = resize_image(img_brut, IMG_SIZE_CNN)
    img_norm  = normalize_minmax(img_128)
    img_gauss = apply_gaussian_filter(img_128, sigma=1.0)
    img_aug   = augment_image(img_128, seed_offset=99)
    img_final = list(img_aug.values())[2]    # prendre la rotation

    steps = [
        (img_256,   "ORIGINAL (BRUT)\n256×256 px — uint8 [0,255]",     C["red"]),
        (img_128,   "① REDIMENSIONNEMENT\n128×128 px — uint8 [0,255]", C["orange"]),
        (img_norm,  "② NORMALISATION\n128×128 px — float32 [0,1]",     C["blue"]),
        (img_gauss, "③ FILTRE GAUSSIEN\n128×128 px — σ=1.0",           C["grey"]),
        (img_final, "④ AUGMENTATION\n128×128 px — Rotation",           C["accent"]),
    ]
    n   = len(steps)
    fig = plt.figure(figsize=(n * 3.4, 7.5), facecolor=C["light"])
    gs  = gridspec.GridSpec(2, n, figure=fig,
                            height_ratios=[5, 0.6], hspace=0.08, wspace=0.08)
    fig.suptitle(
        f"CR2 — Pipeline Complet : {chosen_label.replace('___', ' — ').replace('_', ' ')}",
        fontsize=13, fontweight="bold", color=C["dark"]
    )

    for col, (img, title, color) in enumerate(steps):
        ax_img = fig.add_subplot(gs[0, col])
        disp   = np.clip(img, 0, 1) if img.dtype == np.float32 else img
        ax_img.imshow(disp)
        ax_img.set_title(title, fontsize=9, fontweight="bold",
                         color=color, pad=7, multialignment="center")
        ax_img.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax_img.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

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
