import torch
import os

class EarlyStopping:
    def __init__(self, patience, delta):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model, file_path, model_name):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, file_path, model_name)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, file_path, model_name)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_loss, model, model_path, model_name):
        os.makedirs(model_path, exist_ok=True)
        torch.save(model.state_dict(), f"{model_path}{model_name}.pth")
        self.val_loss_min = val_loss