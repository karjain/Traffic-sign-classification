import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
import pandas as pd
import os
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi
import numpy as np
import torch
from torch.autograd import Variable
from capsnet import CapsNet
from tqdm import tqdm
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from lime import lime_image
import os, random
from skimage.segmentation import mark_boundaries
import utils

torch.cuda.empty_cache()

USE_CUDA = True if torch.cuda.is_available() else False
BATCH_SIZE = 16
N_EPOCHS = 2
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
NUM_CLASSES = 43
torch.manual_seed(1)

utils.download_model('.') 

torch.cuda.empty_cache()
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


def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def lime(img):
    def get_pil_transform():
        '''
        Preprocess image
        '''
        transf = transforms.Compose([
            transforms.Resize((28, 28)),
        ])

        return transf

    def get_preprocess_transform():
        '''
            Preprocess image
        '''
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transf = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        return transf

    pill_transf = get_pil_transform()
    preprocess_transform = get_preprocess_transform()

    def batch_predict(images):
        model = CapsNet(config)
        model.load_state_dict(torch.load('capsnet-model.pt'))
        model.eval()

        batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        batch = batch.to(device)

        output,_ ,masked = model(batch)
        # probs = F.softmax(logits, dim=1)
        return masked.data.cpu().numpy()

    test_pred = batch_predict([pill_transf(img)])

    # Lime image Explaaner
    explainer = lime_image.LimeImageExplainer()
    from lime.wrappers.scikit_image import SegmentationAlgorithm
    # segmenter = SegmentationAlgorithm('slic', n_segments=4, compactness=100, sigma=1)
    segmenter = SegmentationAlgorithm('quickshift')
    explanation = explainer.explain_instance(np.array(pill_transf(img)),
                                             batch_predict,  # classification function
                                             top_labels=5,
                                             # hide_color=0,
                                             segmentation_fn=segmenter,
                                             num_samples=1000,
                                             )  # number of images that will be sent to classification function


    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4))

    ax1.imshow(img)
    ax1.set_title('Image')
    
    ax2.imshow(explanation.segments)
    ax2.set_title('Segments')
    
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=50, hide_rest=True)
    img_boundry2 = mark_boundaries(temp / 255.0, mask)
    ax3.imshow(img_boundry2)
    ax3.set_title('Image with Mask')
    plt.show()


if __name__ == '__main__':
    torch.manual_seed(1)
    config = Config()

    code_dir = os.getcwd()
    img_dir = os.path.join(os.path.split(code_dir)[0], 'Data')


    for i in range(6):
        img_path = random.choice(os.listdir(os.path.join(img_dir,'Test')))
        print(img_path)
        img = get_image(os.path.join(img_dir,'Test', img_path))
        lime(img)








