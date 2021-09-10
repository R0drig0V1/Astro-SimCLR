import torch
import numpy as np

# -----------------------------------------------------------------------------

# Dataset is classified, returns true and predicted labels
def results(model, loader, device=torch.device("cuda")):

    y_pred = np.array([])
    y_true = np.array([])

    # No gradients needed
    with torch.no_grad():

        for data in loader:

            images, features, labels = data

            # Cuda tensor is converted to cpu tensor
            images = images.to(device)
            features = features.to(device)

            outputs = model(images, features)

            prediction = torch.argmax(outputs, dim=1)
            true = torch.argmax(labels, dim=1).numpy()

            if(str(device)=="cuda"):
                prediction = prediction.cpu()

            prediction = prediction.numpy()

            # collect the correct predictions for each class
            y_pred = np.append(y_pred, prediction)
            y_true = np.append(y_true, true)

    return y_true, y_pred 

# -----------------------------------------------------------------------------
