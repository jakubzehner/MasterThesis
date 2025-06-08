import matplotlib as mpl
from matplotlib import pyplot as plt

plt.style.use("seaborn-v0_8-muted")

mpl.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "text.usetex": False,
    }
)


def save_loss_plot(path, train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(val_losses) + 1)

    plt.plot(epochs, val_losses, label="Validation Loss", linewidth=2)
    plt.plot(epochs, train_losses, label="Training Loss", linewidth=2)

    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{path}.pdf", format="pdf")
    plt.savefig(f"{path}.png", format="png")

    plt.close()
