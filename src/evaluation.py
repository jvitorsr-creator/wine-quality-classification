"""Metricas, ajuste de limiar e graficos de avaliacao."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_curve)


def best_threshold(y_true, proba):
    """Limiar que maximiza o F1 da classe positiva (usar probabilidades out-of-fold do treino)."""
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p * r / (p + r + 1e-9)
    return t[np.argmax(f1[:-1])]


def metrics_row(name, y_true, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return {"modelo": name, "limiar": round(float(thr), 2),
            "precisao": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred),
            "f1": f1_score(y_true, pred),
            "roc_auc": roc_auc_score(y_true, proba),
            "pr_auc": average_precision_score(y_true, proba)}


def plot_confusions(results, path):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    for ax, (name, (y, proba, thr)) in zip(np.atleast_1d(axes), results.items()):
        ConfusionMatrixDisplay(confusion_matrix(y, proba >= thr),
                               display_labels=["Baixa/Media", "Alta"]).plot(ax=ax, colorbar=False, cmap="Purples")
        ax.set_title(f"{name} (limiar {thr:.2f})")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_pr(results, path):
    plt.figure(figsize=(6, 4.5))
    for name, (y, proba, _) in results.items():
        p, r, _ = precision_recall_curve(y, proba)
        plt.plot(r, p, label=f"{name} (PR-AUC={average_precision_score(y, proba):.2f})")
    plt.axhline(np.mean(list(results.values())[0][0]), ls="--", c="gray", label="Aleatorio")
    plt.xlabel("Recall"); plt.ylabel("Precisao"); plt.legend(); plt.title("Curva Precisao-Recall")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
