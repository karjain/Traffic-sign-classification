import torch
import os
import matplotlib.pyplot as plt
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi
import gdown

plt.style.use('ggplot')


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least, then save the
    model state.
    """

    def __init__(
            self, model_name, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        self.model_name = model_name
        code_dir = os.getcwd()
        self.model_dir = os.path.join(os.path.split(code_dir)[0], 'Model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

    def __call__(
            self, current_valid_loss,
            epoch, model, optimizer
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"Best validation loss: {self.best_valid_loss}")
            print(f"Saving best model for epoch: {epoch}\n")
            torch.save(
                model.state_dict(),
                os.path.join(self.model_dir, self.model_name)
            )


def download_data(img_dir):
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    else:
        for root, dirs, files in os.walk(img_dir):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
    os.chdir(img_dir)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        'meowmeowmeowmeowmeow/gtsrb-german-traffic-sign',
        path=img_dir)
    return_code = os.system("unzip gtsrb-german-traffic-sign.zip")
    if return_code != 0:
        print("Could not unzip file")
        exit(1)
    print("Download complete")


def download_model(model_dir):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    cur_dir = os.getcwd()
    os.chdir(model_dir)
    capsnet_model_id = "1LNMRafVrmCEOOMjbKJbrOvHLzgy1rCxu"
    capsnet_model_name = 'capsnet-model.pt'
    output_name = gdown.download(id=capsnet_model_id, quiet=False)
    if capsnet_model_name != output_name:
        print("Error in downloading capsnet model")
        exit(2)
    print("Capsnet model download complete")
    os.chdir(cur_dir)

