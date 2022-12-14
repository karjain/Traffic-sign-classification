
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
import pandas as pd
import os
import matplotlib.pyplot as plt
# from utils import download_data


code_dir = os.getcwd()
img_dir = os.path.join(os.path.split(code_dir)[0], 'Data')
annotations_file = os.path.join(img_dir, "Train.csv")
df = pd.read_csv(annotations_file)[["Path", "ClassId"]]
# for path in os.path.join(img_dir, img_labels):
#     print(path)
gdf = df.groupby(['ClassId'])['ClassId'].count()
plt.figure(figsize=(15,10))
plt.title('Class Imbalance', fontsize = 20)
plt.ylabel('Count', fontsize = 15)
plt.xlabel('Class', fontsize = 15)
gdf.plot.bar(rot=0)
plt.show()
print(gdf)

