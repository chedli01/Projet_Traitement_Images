"""
cr5_evaluate.py
───────────────
CR5 — Évaluation du CNN entraîné sur le jeu de TEST de PlantVillage.

Pipeline complet :
    1. Chargement du meilleur modèle et des splits sauvegardés par CR4
    2. Évaluation : accuracy + loss sur le test set
    3. Matrice de confusion 38×38 (heatmap seaborn)
    4. Classification report (precision / recall / f1) → CSV
    5. Top 5 meilleures et pires classes (par f1-score)
    6. Grille d'images mal classifiées (3×4)
    7. Courbes ROC multi-classe (one-vs-rest) avec AUC par classe
    8. Courbes Precision-Recall par classe
    9. Analyse du déséquilibre : scatter (samples vs f1-score)
   10. Résumé console des métriques globales

Exécution :
    python cr5_evaluate.py
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score, f1_score,
)

from config.settings import IMG_SIZE_CNN, BATCH_SIZE, SEED, C
from preprocessing.generators import _val_preprocess


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

OUTPUTS_DIR     = Path("./outputs")
MODEL_PATH      = OUTPUTS_DIR / "best_model.keras"
SPLITS_CSV      = OUTPUTS_DIR / "splits.csv"
CLASSES_JSON    = OUTPUTS_DIR / "class_indices.json"
REPORT_CSV      = OUTPUTS_DIR / "cr5_classification_report.csv"
SUMMARY_JSON    = OUTPUTS_DIR / "cr5_summary.json"


# ─────────────────────────────────────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def load_artifacts():
    """Vérifie et charge le modèle, les splits et le mapping de classes."""
    print("\n" + "═"*70)
    print("  CR5 — ÉVALUATION CNN PlantVillage")
    print("═"*70)

    for p in (MODEL_PATH, SPLITS_CSV, CLASSES_JSON):
        if not p.exists():
            raise FileNotFoundError(
                f"Fichier manquant : {p}. Lancez d'abord `python cr4_train.py`."
            )

    print(f"  → Chargement du modèle : {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    splits_df = pd.read_csv(SPLITS_CSV)
    test_df = splits_df[splits_df["split"] == "test"].reset_index(drop=True)

    with open(CLASSES_JSON, "r", encoding="utf-8") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    class_names  = [idx_to_class[i] for i in range(len(idx_to_class))]

    print(f"  ✅ {len(test_df)} images de test")
    print(f"  ✅ {len(class_names)} classes")
    return model, test_df, class_names, class_indices


# ─────────────────────────────────────────────────────────────────────────────
# 2. GÉNÉRATEUR TEST + PRÉDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_test(model, test_df, class_indices):
    """Construit le générateur test (sharpening + /255), prédit, et calcule
    accuracy + loss globales."""
    print("\n" + "─"*70)
    print("  Étape 1 — Évaluation sur le jeu de TEST")
    print("─"*70)

    test_datagen = ImageDataGenerator(
        preprocessing_function=_val_preprocess,
        rescale=1.0 / 255,
    )
    test_gen = test_datagen.flow_from_dataframe(
        test_df, x_col="filepath", y_col="class",
        target_size=IMG_SIZE_CNN, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=False,
        classes=list(class_indices.keys()),  # ordre cohérent avec CR4
    )

    # Métriques globales (loss + accuracy)
    eval_loss, eval_acc = model.evaluate(test_gen, verbose=1)
    print(f"  ✅ Test accuracy : {eval_acc:.4f}")
    print(f"  ✅ Test loss     : {eval_loss:.4f}")

    # Prédictions probabilistes
    test_gen.reset()
    y_pred_proba = model.predict(test_gen, verbose=1)
    y_pred       = np.argmax(y_pred_proba, axis=1)
    y_true       = test_gen.classes
    filepaths    = np.array(test_gen.filepaths)

    return {
        "loss": float(eval_loss),
        "accuracy": float(eval_acc),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_pred_proba,
        "filepaths": filepaths,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. MATRICE DE CONFUSION
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion(y_true, y_pred, class_names):
    print("\n" + "─"*70)
    print("  Étape 2 — Matrice de confusion 38×38")
    print("─"*70)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_norm = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    short_names = [n.replace("___", "\n").replace("_", " ") for n in class_names]

    fig, ax = plt.subplots(figsize=(20, 17))
    sns.heatmap(
        cm_norm, annot=False, cmap="viridis",
        xticklabels=short_names, yticklabels=short_names,
        ax=ax, cbar_kws={"label": "Proportion (par ligne)"},
    )
    ax.set_xlabel("Prédiction"); ax.set_ylabel("Vérité terrain")
    ax.set_title("CR5 — Matrice de confusion (normalisée par classe vraie)",
                 fontsize=14, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "cr5_confusion_matrix.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Sauvegardé : cr5_confusion_matrix.png")
    return cm


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLASSIFICATION REPORT + TOP/FLOP 5
# ─────────────────────────────────────────────────────────────────────────────

def make_classification_report(y_true, y_pred, class_names) -> pd.DataFrame:
    print("\n" + "─"*70)
    print("  Étape 3 — Classification report (precision / recall / f1)")
    print("─"*70)

    report = classification_report(
        y_true, y_pred,
        labels=range(len(class_names)), target_names=class_names,
        output_dict=True, zero_division=0,
    )
    df = pd.DataFrame(report).T
    df.to_csv(REPORT_CSV, index=True)
    print(f"  ✅ CSV sauvegardé : {REPORT_CSV}")

    per_class = df.loc[class_names].copy()
    per_class = per_class.sort_values("f1-score", ascending=False)

    print("\n  🏆 Top 5 classes les MIEUX classifiées :")
    for name, row in per_class.head(5).iterrows():
        print(f"    {name:50s}  f1={row['f1-score']:.3f}  "
              f"(prec={row['precision']:.2f}, rec={row['recall']:.2f}, n={int(row['support'])})")

    print("\n  ⚠️  Top 5 classes les PIRES classifiées :")
    for name, row in per_class.tail(5).iterrows():
        print(f"    {name:50s}  f1={row['f1-score']:.3f}  "
              f"(prec={row['precision']:.2f}, rec={row['recall']:.2f}, n={int(row['support'])})")

    return per_class


# ─────────────────────────────────────────────────────────────────────────────
# 5. EXEMPLES MAL CLASSIFIÉS
# ─────────────────────────────────────────────────────────────────────────────

def plot_misclassified(y_true, y_pred, filepaths, class_names, n_show: int = 12):
    print("\n" + "─"*70)
    print(f"  Étape 4 — Grille de {n_show} images mal classifiées")
    print("─"*70)

    mis_idx = np.where(y_true != y_pred)[0]
    if len(mis_idx) == 0:
        print("  ✅ Aucune erreur (test parfait).")
        return

    rng = np.random.default_rng(SEED)
    selected = rng.choice(mis_idx, size=min(n_show, len(mis_idx)), replace=False)

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for ax, idx in zip(axes.flat, selected):
        img = np.array(Image.open(filepaths[idx]).convert("RGB"))
        ax.imshow(img)
        true_n = class_names[y_true[idx]].split("___")[-1].replace("_", " ")
        pred_n = class_names[y_pred[idx]].split("___")[-1].replace("_", " ")
        ax.set_title(f"Vrai : {true_n}\nPrédit : {pred_n}",
                     fontsize=8, color=C["red"])
        ax.axis("off")
    # Cases vides si moins de n_show erreurs
    for ax in axes.flat[len(selected):]:
        ax.axis("off")
    fig.suptitle("CR5 — Exemples d'images mal classifiées",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "cr5_misclassified.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Sauvegardé : cr5_misclassified.png  ({len(mis_idx)} erreurs au total)")


# ─────────────────────────────────────────────────────────────────────────────
# 6. ROC + PR multi-classe
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_pr(y_true, y_proba, class_names):
    print("\n" + "─"*70)
    print("  Étape 5 — Courbes ROC et Precision-Recall (one-vs-rest)")
    print("─"*70)

    n_classes  = len(class_names)
    y_true_oh  = tf.keras.utils.to_categorical(y_true, n_classes)

    # ── ROC : courbes par classe + macro / micro average ─────────────────────
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_oh[:, i], y_proba[:, i])
        roc_auc[i]        = auc(fpr[i], tpr[i])

    # Micro average (toutes classes confondues)
    fpr_micro, tpr_micro, _ = roc_curve(y_true_oh.ravel(), y_proba.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.tab20
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], lw=0.8, alpha=0.5, color=cmap(i % 20))
    ax.plot(fpr_micro, tpr_micro, lw=2.5, color="black",
            label=f"Micro-average (AUC = {auc_micro:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=C["grey"], lw=1)
    ax.set_xlabel("Taux de faux positifs (FPR)")
    ax.set_ylabel("Taux de vrais positifs (TPR)")
    ax.set_title(f"CR5 — Courbes ROC One-vs-Rest ({n_classes} classes)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "cr5_roc_curves.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ ROC sauvegardé   : AUC micro = {auc_micro:.4f}")

    # ── Precision-Recall ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    ap_per_class = []
    for i in range(n_classes):
        prec, rec, _ = precision_recall_curve(y_true_oh[:, i], y_proba[:, i])
        ap = average_precision_score(y_true_oh[:, i], y_proba[:, i])
        ap_per_class.append(ap)
        ax.plot(rec, prec, lw=0.8, alpha=0.5, color=cmap(i % 20))

    prec_m, rec_m, _ = precision_recall_curve(y_true_oh.ravel(), y_proba.ravel())
    ap_micro = average_precision_score(y_true_oh, y_proba, average="micro")
    ax.plot(rec_m, prec_m, lw=2.5, color="black",
            label=f"Micro-average (AP = {ap_micro:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"CR5 — Courbes Precision-Recall ({n_classes} classes)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower left"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "cr5_pr_curves.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ PR sauvegardé    : AP micro  = {ap_micro:.4f}")

    return {
        "auc_micro": float(auc_micro),
        "auc_per_class": {class_names[i]: float(roc_auc[i]) for i in range(n_classes)},
        "ap_micro": float(ap_micro),
        "ap_per_class": {class_names[i]: float(ap_per_class[i]) for i in range(n_classes)},
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. ANALYSE DU DÉSÉQUILIBRE
# ─────────────────────────────────────────────────────────────────────────────

def plot_imbalance_vs_f1(per_class: pd.DataFrame, train_df: pd.DataFrame):
    """Scatter : nombre d'images d'entraînement (x) vs f1-score test (y)."""
    print("\n" + "─"*70)
    print("  Étape 6 — Analyse déséquilibre (samples vs f1-score)")
    print("─"*70)

    counts = train_df["class"].value_counts()
    samples = per_class.index.map(counts).astype(int)
    f1      = per_class["f1-score"].values

    median_count = np.median(samples)
    majority = f1[samples >= median_count]
    minority = f1[samples <  median_count]
    print(f"  f1-score moyen — classes majoritaires : {majority.mean():.3f}")
    print(f"  f1-score moyen — classes minoritaires : {minority.mean():.3f}")
    print(f"  Écart                                  : {majority.mean()-minority.mean():+.3f}")

    fig, ax = plt.subplots(figsize=(11, 7))
    sc = ax.scatter(samples, f1, c=f1, cmap="viridis",
                    s=80, edgecolor="white", linewidth=0.8)
    plt.colorbar(sc, ax=ax, label="f1-score")

    # Annoter les 3 plus mauvaises classes
    worst = per_class.nsmallest(3, "f1-score")
    for name, row in worst.iterrows():
        ax.annotate(
            name.split("___")[-1].replace("_", " "),
            xy=(counts[name], row["f1-score"]),
            xytext=(8, 0), textcoords="offset points",
            fontsize=8, color=C["red"],
        )
    ax.axvline(median_count, color=C["grey"], ls="--", lw=1,
               label=f"Médiane = {median_count:.0f}")
    ax.set_xlabel("Nombre d'images d'entraînement par classe")
    ax.set_ylabel("f1-score (test)")
    ax.set_title("CR5 — Impact du déséquilibre sur les performances",
                 fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "cr5_imbalance_vs_f1.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Sauvegardé : cr5_imbalance_vs_f1.png")

    return {
        "majority_mean_f1": float(majority.mean()),
        "minority_mean_f1": float(minority.mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. RÉSUMÉ FINAL
# ─────────────────────────────────────────────────────────────────────────────

def print_final_summary(eval_metrics, roc_pr, imbalance, per_class):
    print("\n" + "═"*70)
    print("  ✅ CR5 — RÉSUMÉ FINAL")
    print("═"*70)
    print(f"  Test accuracy            : {eval_metrics['accuracy']:.4f}")
    print(f"  Test loss                : {eval_metrics['loss']:.4f}")
    print(f"  Macro f1-score           : {per_class['f1-score'].mean():.4f}")
    print(f"  Médiane f1-score         : {per_class['f1-score'].median():.4f}")
    print(f"  AUC micro-average        : {roc_pr['auc_micro']:.4f}")
    print(f"  AP  micro-average        : {roc_pr['ap_micro']:.4f}")
    print(f"  f1 majoritaires (≥ méd)  : {imbalance['majority_mean_f1']:.4f}")
    print(f"  f1 minoritaires (< méd)  : {imbalance['minority_mean_f1']:.4f}")
    print("═"*70)

    summary = {
        "test_accuracy": eval_metrics["accuracy"],
        "test_loss": eval_metrics["loss"],
        "macro_f1": float(per_class["f1-score"].mean()),
        "auc_micro": roc_pr["auc_micro"],
        "ap_micro": roc_pr["ap_micro"],
        "majority_mean_f1": imbalance["majority_mean_f1"],
        "minority_mean_f1": imbalance["minority_mean_f1"],
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  → Résumé JSON : {SUMMARY_JSON}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(SEED); tf.random.set_seed(SEED)

    model, test_df, class_names, class_indices = load_artifacts()

    eval_metrics = predict_test(model, test_df, class_indices)
    plot_confusion(eval_metrics["y_true"], eval_metrics["y_pred"], class_names)
    per_class = make_classification_report(
        eval_metrics["y_true"], eval_metrics["y_pred"], class_names,
    )
    plot_misclassified(
        eval_metrics["y_true"], eval_metrics["y_pred"],
        eval_metrics["filepaths"], class_names,
    )
    roc_pr = plot_roc_pr(
        eval_metrics["y_true"], eval_metrics["y_proba"], class_names,
    )

    splits_df = pd.read_csv(SPLITS_CSV)
    train_df  = splits_df[splits_df["split"] == "train"]
    imbalance = plot_imbalance_vs_f1(per_class, train_df)

    print_final_summary(eval_metrics, roc_pr, imbalance, per_class)


if __name__ == "__main__":
    main()
