import sys

from sklearn.metrics import accuracy_score

sys.path.append('utils')
from utils_metrics import results

# -----------------------------------------------------------------------------

def trainer_v1(model, epochs, train_dataloader, validation_dataloader, optimizer, criterion, device):

    model.train()

    for ep in range(epochs):

        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):


            # get the inputs; data is a list of [inputs, labels]
            images, features, labels = data

            images = images.to(device)
            features = features.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            for param in model.parameters():
                param.grad = None

            # forward + backward + optimize
            outputs = model(images, features)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # metrics
            running_loss += loss.item()


            if i % 100 == 99:    # print every 100 mini-batches

                model.eval()

                y_true, y_pred = results(model, validation_dataloader, device)
                acc = accuracy_score(y_true, y_pred)

                model.train()

                print('[%d, %5d] loss: %.5f acc: %.5f' %
                      (ep + 1, i + 1, running_loss / 100, acc))
                running_loss = 0.0

    print('Finished Training')

# -----------------------------------------------------------------------------