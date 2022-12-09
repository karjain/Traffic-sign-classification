import torch
import torch.nn as nn
from torchvision import transforms
from data_loader import Dataset
from torch.nn import functional as F
import numpy as np
import os
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
from lime.wrappers.scikit_image import SegmentationAlgorithm
import matplotlib.pyplot as plt

BATCH_SIZE = 10
mnist = Dataset(BATCH_SIZE)

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(3 * 3 * 64, 256)
        self.fc2 = nn.Linear(256, 43)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1, 3 * 3 * 64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



# Selecting the appropriate training device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNN().to(device)

# Defining the model hyper parameters
num_epochs = 1
learning_rate = 0.001
weight_decay = 0.01
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training process begins
def train(model, optimizer):
    train_loss_list = []
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}:', end=' ')
        train_loss = 0

        # Iterating over the training dataset in batches
        model.train()
        for i, (images, labels) in enumerate(mnist.train_loader):
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

        # Printing loss for each epoch
        train_loss_list.append(train_loss / len(mnist.train_loader))
        print(f"Training loss = {train_loss_list[-1]}")

def test(model):
    test_acc = 0
    model.eval()
    total = 0
    with torch.no_grad():
        # Iterating over the training dataset in batches
        for i, (images, labels) in enumerate(mnist.test_loader):
            images = images.to(device)
            y_true = labels.to(device)

            # Calculating outputs for the batch being iterated
            outputs = model(images)

            # Calculated prediction labels from models
            _, y_pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            # Comparing predicted and true labels
            test_acc += (y_pred == y_true).sum().item()

        print(f"Test set accuracy = {100 * (test_acc / total)} %")


# explaining with lime 
def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def lime(img):
    def get_pil_transform():
        transf = transforms.Compose([
            transforms.Resize((28, 28)),
            # transforms.CenterCrop(224)
        ])

        return transf

    def get_preprocess_transform():
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transf = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        return transf

    pill_transf = get_pil_transform()
    preprocess_transform = get_preprocess_transform()

    def batch_predict(images):
        model = CNN()
        # checkpoint = torch.load('saved_model.pth')
        #model.load_state_dict(torch.load('capsnet-model.pt'))
        # model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        batch = batch.to(device)

        output = model(batch)
        # probs = F.softmax(logits, dim=1)
        return output.data.cpu().numpy()

    test_pred = batch_predict([pill_transf(img)])

    explainer = lime_image.LimeImageExplainer()

    segmenter = SegmentationAlgorithm('quickshift')
    explanation = explainer.explain_instance(np.array(pill_transf(img)),
                                             batch_predict,  # classification function
                                             top_labels=5,
                                             # hide_color=0,
                                             segmentation_fn=segmenter,
                                             num_samples=1000,
                                             )  # number of images that will be sent to classification function

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    ax1.imshow(img)
    ax1.set_title('Image')

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=50,
                                                hide_rest=True)
    img_boundry2 = mark_boundaries(temp / 255.0, mask)
    ax2.imshow(img_boundry2)
    ax2.set_title('Image with Mask')
    plt.show()


train(model, optimizer)
test(model)

img = get_image(r'/home/ubuntu//Final_Project/Data/Test/07200.png')
lime(img)
