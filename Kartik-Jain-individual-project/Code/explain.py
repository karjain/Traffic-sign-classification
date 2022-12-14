import numpy as np
import pandas as pd
import torch
from capsnet import CapsNet
from Baseline_CNN import CNN
# from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from lime import lime_image
import os
import random
from skimage.segmentation import mark_boundaries
from utils import download_data, download_model
from test_capsnet import Config
from lime.wrappers.scikit_image import SegmentationAlgorithm
from torch.autograd import Variable

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
N_EPOCHS = 2
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
NUM_CLASSES = 43
NUM_IMGS_2_VIZ = 8
torch.manual_seed(1)
torch.cuda.empty_cache()
code_dir = os.getcwd()
model_dir = os.path.join(os.path.split(code_dir)[0], 'Model')
data_dir = os.path.join(os.path.split(code_dir)[0], 'Data')



class Lime:
    def __init__(self):
        self.pil_transform = transforms.Compose([
            transforms.Resize((28, 28)),
        ])
        self.preprocess_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.capsnet_model, self.cnn_model = Lime.build_state()
        self.explainer = lime_image.LimeImageExplainer()
        self.segmenter = SegmentationAlgorithm('quickshift')
        self.cnn_pred_class = 99
        self.ann = pd.read_csv('annotations.csv')
        self.cnn_pred_label = ''
        self.cap_pred_class = 99
        self.cap_pred_label = ''

    @staticmethod
    def build_state():
        test_img_dir = os.path.join(data_dir, 'Test')
        if len(os.listdir(test_img_dir)) < NUM_IMGS_2_VIZ:
            download_data(data_dir)
        if not os.path.exists(os.path.join(model_dir, 'capsnet-model.pt')) or \
                not os.path.exists(os.path.join(model_dir, 'cnn-model.pt')):
            download_model(model_dir)
        capsnet_config = Config()
        capsnet_model = CapsNet(capsnet_config)
        capsnet_model.load_state_dict(torch.load(os.path.join(model_dir, 'capsnet-model.pt')))
        capsnet_model.to(device)
        cnn_model = CNN()
        cnn_model.load_state_dict(torch.load(os.path.join(model_dir, 'cnn-model.pt')))
        cnn_model.to(device)
        return capsnet_model, cnn_model

    def batch_predict_capsnet(self, images):
        self.capsnet_model.eval()
        batch = torch.stack(tuple(self.preprocess_transform(j) for j in images), dim=0)
        batch = batch.to(device)
        output, _, masked = self.capsnet_model(batch)
        return masked.data.cpu().numpy()

    # def pred(self, img):
    #     self.capsnet_model.eval()
    #     img_arr = np.array(self.pil_transform(img))
    #     batch = torch.stack(tuple(self.preprocess_transform(img_arr)), dim=0)
    #     batch = torch.unsqueeze(batch, dim=0)
    #     batch = batch.to(device)
    #     output, _, masked = self.capsnet_model(batch)
    #     _, y_pred = torch.max(output.data, 1)
    #     print(f'pred = {masked.data.cpu().numpy()}')
    #     print(f'predicted class= {np.argmax(masked.data.cpu().numpy(), 1)}')


    def batch_predict_cnn(self, images):
        self.cnn_model.eval()
        batch = torch.stack(tuple(self.preprocess_transform(j) for j in images), dim=0)
        batch = batch.to(device)
        output = self.cnn_model(batch)
        _, y_pred = torch.max(output.data, 1)
        if self.cnn_pred_class == 99:
            temp = y_pred[0].item()
            self.cnn_pred_class  = y_pred[0].item()
            print(y_pred[0])
            print(type(self.ann[self.ann['Id'] == y_pred[0]]['Name'].values))
            print(self.ann[self.ann['Id'] == self.cnn_pred_class]['Name'].values)
            self.cnn_pred_label = self.ann[self.ann['Id'] == y_pred[0].item()]['Name'].values[0]
            # df[df['Id'] == cls]['Name'].values[0]
        return output.data.cpu().numpy()



    def plot(self, image):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax2.imshow(self.explanation_capsnet.segments)
        ax2.set_title('Segments')

        temp, mask = self.explanation_capsnet.get_image_and_mask(
            self.explanation_capsnet.top_labels[0],
            positive_only=False,
            num_features=50,
            hide_rest=True
        )
        img_boundary2 = mark_boundaries(temp / 255.0, mask)
        ax3.imshow(img_boundary2)
        ax3.set_title(f'Capsnet with mask')

        temp, mask = self.explanation_cnn.get_image_and_mask(
            self.explanation_cnn.top_labels[0],
            positive_only=False,
            num_features=50,
            hide_rest=True
        )
        img_boundary2 = mark_boundaries(temp / 255.0, mask)
        ax4.imshow(img_boundary2)
        ax4.set_title(f'CNN with mask')
        plt.tight_layout()
        fig.suptitle(f'Predicted class={self.cnn_pred_label}, {self.cnn_pred_class}' )
        plt.subplots_adjust(top=0.9)
        plt.show()

    @staticmethod
    def get_image(path):
        with open(os.path.abspath(path), 'rb') as f:
            with Image.open(f) as image:
                return image.convert('RGB')

    def __call__(self, image):
        """
        Lime image explainer
        :param image:
        :return:
        """
        image = Lime.get_image(image)
        # self.pred(image)
        self.explanation_capsnet = self.explainer.explain_instance(
            np.array(self.pil_transform(image)),
            self.batch_predict_capsnet,  # classification function
            top_labels=5,
            segmentation_fn=self.segmenter,
            num_samples=1000,
        )  # number of images that will be sent to classification function

        # self.get_cnn_pred(image)
        self.explanation_cnn = self.explainer.explain_instance(
            np.array(self.pil_transform(image)),
            self.batch_predict_cnn,
            top_labels=5,
            segmentation_fn=self.segmenter,
            num_samples=1000,
        )
        self.plot(image)
        del self.explanation_capsnet
        del self.explanation_cnn

# avg dim 0
# squeeze on dim 0
# argmax

if __name__ == '__main__':
    lime = Lime()
    names = pd.read_csv('annotations.csv')
    for i in range(NUM_IMGS_2_VIZ):
        print(data_dir)
        img_path = random.choice(os.listdir(os.path.join(data_dir, 'Test')))
        img = os.path.join(data_dir, 'Test', img_path)
        lime(img)
