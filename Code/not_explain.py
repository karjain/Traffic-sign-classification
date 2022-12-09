
from data_loader import Dataset
import os
from lime.wrappers.scikit_image import SegmentationAlgorithm
import numpy as np
import torch
from capsnet import CapsNet
import matplotlib.pyplot as plt
from lime import lime_image
from test_capsnet import Config
from skimage.segmentation import mark_boundaries


torch.cuda.empty_cache()

USE_CUDA = True if torch.cuda.is_available() else False
BATCH_SIZE = 1
N_EPOCHS = 2
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
NUM_CLASSES = 43
torch.manual_seed(1)


torch.cuda.empty_cache()


def batch_predict_caps(images):
    config = Config()
    model = CapsNet(config)
    code_dir = os.getcwd()
    model_dir = os.path.join(os.path.split(code_dir)[0], 'Data')
    model.load_state_dict(torch.load(os.path.join(model_dir,'capsnet-model.pt')))
    model.eval()
    batch = torch.stack(tuple(i for i in images), dim=0)
    device = torch.device("cuda" if USE_CUDA else "cpu")
    model.to(device)
    batch = batch.to(device)
    output,_ ,masked = model(batch)
    return masked.data.cpu().numpy()



if __name__ == '__main__':
    torch.manual_seed(1)
    mnist = Dataset(BATCH_SIZE, download=False)
    explainer = lime_image.LimeImageExplainer()
    segmenter = SegmentationAlgorithm('quickshift')
    for batch_id, (data, target) in enumerate(mnist.test_loader):
        img = data.squeeze(0).permute(1,2,0).numpy()
        explanation = explainer.explain_instance(img, batch_predict_caps, top_labels=5,
                                                 segmentation_fn=segmenter,num_samples=1000)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.imshow(img)
        ax1.set_title('Image')
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=50,
                                                    hide_rest=True)
        img_boundry2 = mark_boundaries(temp / 255.0, mask)
        ax2.imshow(img_boundry2)
        ax2.set_title('Image with Mask')
        plt.show()







