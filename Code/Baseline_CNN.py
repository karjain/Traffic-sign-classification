import torch
import torch.nn as nn
from data_loader import Dataset
from torch.nn import functional as f
from tqdm import tqdm
from utils import SaveBestModel, download_data
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import argparse
import os


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(3 * 3 * 64, 256)
        self.fc2 = nn.Linear(256, 43)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(f.max_pool2d(self.conv2(x), 2))
        x = f.dropout(x, p=0.5, training=self.training)
        x = f.relu(f.max_pool2d(self.conv3(x), 2))
        x = f.dropout(x, p=0.5, training=self.training)
        x = x.view(-1, 3 * 3 * 64)
        x = f.relu(self.fc1(x))
        x = f.dropout(x, training=self.training)
        x = self.fc2(x)
        return f.log_softmax(x, dim=1)


# Training process begins
def train(epoch):
    train_loss = 0
    total = 0
    train_acc = 0
    all_predictions = list()
    all_labels = list()
    metric_dict = dict()

    # Iterating over the training dataset in batches
    model.train()
    for images, labels in (pbar := tqdm(mnist.train_loader)):
        pbar.set_description(f"Training Epoch [{epoch+1}/{num_epochs}]: ")
        # Extracting images and target labels for the batch being iterated
        images = images.to(device)
        labels = labels.to(device)

        # Calculating the model output and the cross entropy loss
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Updating weights according to calculated loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # Calculated prediction labels from models
        _, y_pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        # Comparing predicted and true labels
        train_acc += torch.sum(torch.Tensor(y_pred == labels)).item()
        all_predictions.extend(list(y_pred.cpu().numpy()))
        all_labels.extend(list(labels.cpu().numpy()))
        pbar.set_postfix({
            "train_loss": train_loss / total,
            "train_acc": train_acc / total
        })

    # Printing loss for each epoch
    train_loss_list.append(train_loss / len(mnist.train_loader))
    metric_dict['train_loss'] = train_loss / len(mnist.train_loader)
    metric_dict['train_accuracy'] = accuracy_score(all_labels, all_predictions)
    metric_dict['train_f1_score_macro'] = f1_score(all_labels, all_predictions, average='macro')
    return metric_dict


def test(epoch):
    test_loss = 0
    test_acc = 0
    all_predictions = list()
    all_labels = list()
    metric_dict = dict()
    model.eval()
    total = 0
    with torch.no_grad():
        # Iterating over the training dataset in batches
        for i, (images, labels) in enumerate(pbar := tqdm(mnist.test_loader)):
            pbar.set_description(f"Testing Epoch [{epoch+1}/{num_epochs}]: ")
            images = images.to(device)
            y_true = labels.to(device)

            # Calculating outputs for the batch being iterated
            outputs = model(images)
            loss = criterion(outputs, y_true)
            test_loss += loss.item()
            # Calculated prediction labels from models`
            _, y_pred = torch.max(outputs.data, 1)
            total += y_true.size(0)
            # Comparing predicted and true labels
            test_acc += torch.sum(torch.Tensor(y_pred == y_true)).item()
            all_predictions.extend(list(y_pred.cpu().numpy()))
            all_labels.extend(list(labels.cpu().numpy()))
            pbar.set_postfix({
                "test_loss": test_loss / total,
                "test_acc": test_acc / total
            })
    test_loss_list.append(test_loss / len(mnist.test_loader))
    metric_dict[f'val_loss'] = test_loss / len(mnist.test_loader)
    metric_dict[f'val_accuracy'] = accuracy_score(all_labels, all_predictions)
    metric_dict[f'val_f1_macro_score'] = f1_score(all_labels, all_predictions, average='macro')
    return metric_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-ImageDownload', action='store_true')
    args = parser.parse_args()

    if args.ImageDownload:
        DOWNLOAD_IMG_DATA = True
    else:
        DOWNLOAD_IMG_DATA = False

    code_dir = os.getcwd()
    data_dir = os.path.join(os.path.split(code_dir)[0], 'Data')
    if DOWNLOAD_IMG_DATA:
        download_data(data_dir)
    # Selecting the appropriate training device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CNN().to(device)

    # Defining the model hyper parameters
    num_epochs = 10
    learning_rate = 0.001
    weight_decay = 0.01
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    code_dir = os.getcwd()
    data_dir = os.path.join(os.path.split(code_dir)[0], 'Data')

    BATCH_SIZE = 64
    mnist = Dataset(BATCH_SIZE)
    train_loss_list = list()
    test_loss_list = list()
    epoch_metrics = list()
    save_best_model = SaveBestModel(model_name='cnn-model.pt')
    for e in range(num_epochs):
        train_metrics = train(e)
        val_metrics = test(e)
        save_best_model(test_loss_list[-1], e + 1, model, optimizer)
        epoch_metrics.append({"epoch": e+1, **train_metrics, **val_metrics})
    metric_df = pd.DataFrame(epoch_metrics)
    # Creating metric outputs
    metric_df.to_csv(os.path.join(mnist.img_dir, 'CNN-Metrics.csv'), index=False)
