import torch
import os

class SLNN_EarlyStopping:
    def __init__(self, patience, delta, snr_dB, hidden_layer):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.snr_dB = snr_dB
        self.hidden_layer = hidden_layer

    def __call__(self, val_loss, model, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_loss, model, model_path):
        os.makedirs(model_path, exist_ok=True)
        torch.save(model.state_dict(), f"{model_path}SLNN_model_hiddenlayer{self.hidden_layer}_BER{self.snr_dB}.pth")
        self.val_loss_min = val_loss

class MLNN_EarlyStopping:
    def __init__(self, patience, delta, snr_dB, hidden_layer):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.snr_dB = snr_dB
        self.hidden_layer = hidden_layer

    def __call__(self, val_loss, model, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_loss, model, model_path):
        os.makedirs(model_path, exist_ok=True)
        torch.save(model.state_dict(), f"{model_path}MLNN_model_hiddenlayer{self.hidden_layer}_BER{self.snr_dB}.pth")
        self.val_loss_min = val_loss