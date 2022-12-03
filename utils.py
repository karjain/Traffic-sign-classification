import torch
import os
import matplotlib.pyplot as plt

plt.style.use('ggplot')


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(
            self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        code_dir = os.getcwd()
        self.model_dir = os.path.join(os.path.split(code_dir)[0], 'Data')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

    def __call__(
            self, current_valid_loss,
            epoch, model, optimizer
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch + 1}\n")
            # torch.save({
            #     'epoch': epoch + 1,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict()
            # }, os.path.join(self.model_dir, 'capsnet-model.pt'))
            torch.save(
                model.state_dict(),
                os.path.join(self.model_dir, 'capsnet-model.pt')
            )
