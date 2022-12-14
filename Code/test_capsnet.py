import numpy as np
import torch
# import torch.nn as nn
# import torch.nn.functional as F
from torch.autograd import Variable
# from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score
from capsnet import CapsNet
from data_loader import Dataset
from tqdm import tqdm
import argparse
import os
import pandas as pd
from utils import SaveBestModel, download_model, download_data, plot_metrics

torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)

USE_CUDA = True if torch.cuda.is_available() else False
BATCH_SIZE = 64
N_EPOCHS = 10
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
NUM_CLASSES = 43
'''
Config class to determine the parameters for capsule net
'''


class Config:
    def __init__(self):

        # CNN (cnn)
        self.cnn_in_channels = 3
        self.cnn_out_channels = 384
        self.cnn_kernel_size = 9

        # Primary Capsule (pc)
        self.pc_num_capsules = 8
        self.pc_in_channels = 384
        self.pc_out_channels = 32
        self.pc_kernel_size = 10  # 9
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


def train(train_loader, epoch):
    capsule_net.train()
    n_batch = len(list(enumerate(train_loader)))
    total_loss = 0
    all_predictions = list()
    all_labels = list()
    metric_dict = dict()
    for batch_id, (data, target) in enumerate(tqdm(train_loader)):

        target = torch.sparse.torch.eye(NUM_CLASSES).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)

        if USE_CUDA:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output, reconstructions, masked = capsule_net(data)
        loss = capsule_net.loss(data, output, target, reconstructions)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(capsule_net.parameters(), max_norm=1)
        optimizer.step()
        labels = np.argmax(target.data.cpu().numpy(), 1)
        predictions = np.argmax(masked.data.cpu().numpy(), 1)
        correct = sum(predictions == labels)
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
        all_predictions.extend(list(predictions))
        all_labels.extend(list(labels))
    tqdm.write('Epoch: [{}/{}], train loss: {:.6f}'.format(epoch, N_EPOCHS, total_loss / len(train_loader.dataset)))
    metric_dict['train_loss'] = total_loss / len(train_loader.dataset)
    metric_dict['train_accuracy'] = accuracy_score(all_labels, all_predictions)
    metric_dict['train_f1_score_macro'] = f1_score(all_labels, all_predictions, average='macro')
    return metric_dict


def test(test_loader, epoch, suffix):
    capsule_net.eval()
    test_loss = 0
    all_predictions = list()
    all_labels = list()
    metric_dict = dict()
    num_correct = 0
    epoch_print = f"{suffix}" if suffix == "test" else f"Epoch: [{epoch}/{N_EPOCHS}], {suffix}"
    for batch_id, (data, target) in enumerate(test_loader):

        target = torch.sparse.torch.eye(NUM_CLASSES).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)

        if USE_CUDA:
            data, target = data.cuda(), target.cuda()

        output, reconstructions, masked = capsule_net(data)
        loss = capsule_net.loss(data, output, target, reconstructions)

        test_loss += loss.item()
        labels = np.argmax(target.data.cpu().numpy(), 1)
        predictions = np.argmax(masked.data.cpu().numpy(), 1)
        correct = sum(predictions == labels)
        num_correct += correct
        all_predictions.extend(list(predictions))
        all_labels.extend(list(labels))

    tqdm.write(f"{epoch_print} accuracy: {num_correct/len(all_labels):.6f}, "
               f"{suffix} loss: {test_loss / len(test_loader.dataset):.6f}")
    metric_dict[f'{suffix}_loss'] = test_loss / len(test_loader.dataset)
    metric_dict[f'{suffix}_accuracy'] = accuracy_score(all_labels, all_predictions)
    metric_dict[f'{suffix}_f1_macro_score'] = f1_score(all_labels, all_predictions, average='macro')
    return metric_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-Train', action='store_true')
    parser.add_argument('-ImageDownload', action='store_true')
    args = parser.parse_args()

    option = args.Train
    print(f'option={option}')
    if args.Train:
        TRAIN_MODEL = True
        print('Training')
    else:
        print('only predict')
        TRAIN_MODEL = False

    if args.ImageDownload:
        DOWNLOAD_IMG_DATA = True
    else:
        DOWNLOAD_IMG_DATA = False

    print(f'TRAIN_MODEL={TRAIN_MODEL}')
    code_dir = os.getcwd()
    data_dir = os.path.join(os.path.split(code_dir)[0], 'Data')

    torch.manual_seed(1)

    config = Config()
    mnist = Dataset(BATCH_SIZE, download=DOWNLOAD_IMG_DATA)

    capsule_net = CapsNet(config)
    capsule_net = torch.nn.DataParallel(capsule_net)
    if USE_CUDA:
        capsule_net = capsule_net.cuda()
    capsule_net = capsule_net.module

    optimizer = torch.optim.Adam(capsule_net.parameters(), lr=LEARNING_RATE)
    model_dir = os.path.join(os.path.split(mnist.img_dir)[0], 'Model')

    if TRAIN_MODEL:
        save_best_model = SaveBestModel(model_name='capsnet-model.pt')
        epoch_metrics = list()
        print("Begin training")
        for e in range(1, N_EPOCHS + 1):
            train_metrics = train(mnist.train_loader, e)
            val_metrics = test(mnist.test_loader, e, suffix='val')
            save_best_model(val_metrics['val_loss'], e, capsule_net, optimizer)
            epoch_metrics.append({"epoch": e, **train_metrics, **val_metrics})
        metric_df = pd.DataFrame(epoch_metrics)
        # Creating metric outputs
        metric_df.to_csv(os.path.join(mnist.img_dir, 'Capsnet-Metrics.csv'), index=False)
        plot_metrics(mnist.img_dir, 'Capsnet-Metrics.csv')

    else:
        download_model(model_dir)

    print("Testing model")
    capsule_net.load_state_dict(torch.load(os.path.join(model_dir, 'capsnet-model.pt')))
    test_metrics = test(mnist.test_loader, 1, suffix='test')
