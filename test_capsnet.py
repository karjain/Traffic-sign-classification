import numpy as np
import torch
# import torch.nn as nn
# import torch.nn.functional as F
from torch.autograd import Variable
# from torchvision import datasets, transforms
from capsnet import CapsNet
from data_loader import Dataset
from tqdm import tqdm


torch.cuda.empty_cache()

USE_CUDA = True if torch.cuda.is_available() else False
BATCH_SIZE = 16
N_EPOCHS = 2
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
NUM_CLASSES = 43
'''
Config class to determine the parameters for capsule net
'''


class Config:
    def __init__(self):

        # CNN (cnn)
        self.cnn_in_channels = 3
        self.cnn_out_channels = 256
        self.cnn_kernel_size = 9

        # Primary Capsule (pc)
        self.pc_num_capsules = 8
        self.pc_in_channels = 256
        self.pc_out_channels = 32
        self.pc_kernel_size = 9
        self.pc_num_routes = 32 * 6 * 6

        # Digit Capsule (dc)
        self.dc_num_capsules = NUM_CLASSES
        self.dc_num_routes = 32 * 6 * 6
        self.dc_in_channels = 8
        self.dc_out_channels = 16

        # Decoder
        self.input_width = 28
        self.input_height = 28
        self.num_classes = NUM_CLASSES


def train(model, optim, train_loader, epoch):
    capsnet = model
    capsnet.train()
    n_batch = len(list(enumerate(train_loader)))
    total_loss = 0
    for batch_id, (data, target) in enumerate(tqdm(train_loader)):

        target = torch.sparse.torch.eye(NUM_CLASSES).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)

        if USE_CUDA:
            data, target = data.cuda(), target.cuda()

        optim.zero_grad()
        output, reconstructions, masked = capsnet(data)
        loss = capsnet.loss(data, output, target, reconstructions)
        loss.backward()
        optim.step()
        correct = sum(np.argmax(masked.data.cpu().numpy(), 1) == np.argmax(target.data.cpu().numpy(), 1))
        train_loss = loss.item()
        total_loss += train_loss
        if batch_id % 100 == 0:
            tqdm.write("Epoch: [{}/{}], Batch: [{}/{}], train accuracy: {:.6f}, loss: {:.6f}".format(
                epoch,
                N_EPOCHS,
                batch_id + 1,
                n_batch,
                correct / float(BATCH_SIZE),
                train_loss / float(BATCH_SIZE)
                ))
    tqdm.write('Epoch: [{}/{}], train loss: {:.6f}'.format(epoch, N_EPOCHS, total_loss / len(train_loader.dataset)))


def test(capsnet, test_loader, epoch):
    capsule_net.eval()
    test_loss = 0
    correct = 0
    for batch_id, (data, target) in enumerate(test_loader):

        target = torch.sparse.torch.eye(NUM_CLASSES).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)

        if USE_CUDA:
            data, target = data.cuda(), target.cuda()

        output, reconstructions, masked = capsnet(data)
        loss = capsnet.loss(data, output, target, reconstructions)

        test_loss += loss.item()
        correct += sum(np.argmax(masked.data.cpu().numpy(), 1) ==
                       np.argmax(target.data.cpu().numpy(), 1))

    tqdm.write(
        "Epoch: [{}/{}], test accuracy: {:.6f}, loss: {:.6f}".format(epoch, N_EPOCHS, correct/len(test_loader.dataset),
                                                                     test_loss / len(test_loader)))


if __name__ == '__main__':
    torch.manual_seed(1)

    config = Config()
    mnist = Dataset(BATCH_SIZE, download=False)

    capsule_net = CapsNet(config)
    capsule_net = torch.nn.DataParallel(capsule_net)
    if USE_CUDA:
        capsule_net = capsule_net.cuda()
    capsule_net = capsule_net.module

    optimizer = torch.optim.Adam(capsule_net.parameters(), lr=LEARNING_RATE)

    for e in range(1, N_EPOCHS + 1):
        train(capsule_net, optimizer, mnist.train_loader, e)
        test(capsule_net, mnist.test_loader, e)


# batch = next(iter(mnist.test_loader))
# images, _ = batch
# background = images[:100].cuda()
# test_images = images[100:105].cuda()
# explainer = shap.DeepExplainer(capsule_net, mnist)
