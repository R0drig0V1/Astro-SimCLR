import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils_dataset import feature_names
from utils_dataset import label_names


# -----------------------------------------------------------------------------

def stamps_plot(data, type_set, index):
    
    # Loads source's data
    label = data['Train']['labels'][index]
    features = data['Train']['features'][index]
    img = data['Train']['images'][index]
    
    # Prints image's label
    print("Label: {0}".format(label_names[label]))
    
    # Dataframe with features
    d = {'Feature': feature_names, 'Value': features[3:]}
    df = pd.DataFrame(data=d)
    
    # Plot of science, reference and difference images
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,15))
    
    ax1.set_title("Science")
    ax1.imshow(img[:,:,0])
    ax1.axis('off')
    
    ax2.set_title("Reference")
    ax2.imshow(img[:,:,1])
    ax2.axis('off')
    
    ax3.set_title("Difference")
    ax3.imshow(img[:,:,2])
    ax3.axis('off')
    
    # Displays images and features
    plt.show()
    display(df)

# -----------------------------------------------------------------------------

def plot_confusion_matrix(confusion_matrix, title, label_names, file):

    ax = sns.heatmap(confusion_matrix, vmin=0, vmax=1, annot=True, fmt='.2f');

    ax.set_title(title)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

    # set ticklabels
    ax.xaxis.set_ticklabels(label_names)
    ax.yaxis.set_ticklabels(label_names)

    plt.savefig(file, dpi=180)

# -----------------------------------------------------------------------------
