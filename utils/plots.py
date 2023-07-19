import seaborn as sns
import matplotlib.pyplot as plt

from utils.dataset import label_names

# -----------------------------------------------------------------------------

# Plot confusion matrix
def plot_confusion_matrix(cm, title, file):

    # Plot confusion matrix
    ax = sns.heatmap(100*cm, vmin=0, vmax=100, annot=True, fmt='.1f')
    ax.set_aspect("equal")

    # Set title and labes
    ax.set_title(title)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

    # Set Ticklabels
    ax.xaxis.set_ticklabels(label_names)
    ax.yaxis.set_ticklabels(label_names)

    plt.tight_layout()

    # Save image
    plt.savefig(file, dpi=180)
    plt.clf()

# -----------------------------------------------------------------------------

# Plot confusion matrix with mean and standard deviation
def plot_confusion_matrix_mean_std(cmm, cms, title, file):
    
    # Plot confusion matrix mean
    ax = sns.heatmap(100*cmm, vmin=0, vmax=100)
    ax.set_aspect("equal")

    # Write mean and standard deviation
    for i in range(cmm.shape[0]):
        for j in range(cms.shape[1]):

            # Write
            plt.text(0.5 + j,
                     0.5 + i,
                     f'{100*cmm[i, j]:.1f}$\pm${100*cms[i, j]:.1f}',
                     horizontalalignment="center",
                     verticalalignment="center",
                     fontsize=7,
                     color="black" if cmm[i, j] > 0.5 else "white")

    # Set title and labes
    ax.set_title(title)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

    # Set Ticklabels
    ax.xaxis.set_ticklabels(label_names)
    ax.yaxis.set_ticklabels(label_names)

    plt.tight_layout()

    # Save image
    plt.savefig(file, dpi=180)
    plt.clf()

# -----------------------------------------------------------------------------
