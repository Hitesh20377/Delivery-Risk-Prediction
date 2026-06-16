import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc
)

def plot_confusion_matrix(y_test, y_pred):

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(
        "reports/figures/confusion_matrix.png"
    )

    plt.close()


def plot_roc_curve(y_test, y_prob):

    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.figure(figsize=(6,5))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.3f}"
    )

    plt.plot(
        [0,1],
        [0,1],
        linestyle="--"
    )

    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    plt.savefig(
        "reports/figures/roc_curve.png"
    )

    plt.close()