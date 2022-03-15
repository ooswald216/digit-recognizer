# -*- coding:utf-8 -*-
"""
作者：10309
日期：2022年03月07日
"""
import torch
import torch.nn as nn
from model import CNNModel
from data import test_loader
import numpy as np
import pandas as pd
# from data import val_loader
from torch.utils.data import Dataset, DataLoader

class DigitInferenceDataset(Dataset):
    def __init__(self, df, augmentations=None):  # for inference we only have the features dataframe
        self.features = df.values / 255  # scale (greyscale) features
        self.augmentations = augmentations

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        image = self.features[idx].reshape((1, 28, 28))
        return torch.FloatTensor(image)
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=1)[:, None]
val = pd.read_csv("data/test.csv",dtype = np.float32)
# # instantiate Inference Dataset class (create inference Dataset)
inference_dataset = DigitInferenceDataset(val, augmentations=None)
#
# # create Inference DataLoader object from Dataset class object
batch_size = 100
inference_dataloader = DataLoader(inference_dataset,
                                  batch_size=batch_size,
                                  shuffle=False)
model_path = "./model/CNNModel.pt"
save_info = torch.load(model_path)
model = CNNModel()
# criterion = nn.CrossEntropyLoss()
# optimizer.load_state_dict(save_info["optimizer"])
model.load_state_dict(save_info["model"])
model.eval()

list_predictions = list()
model.eval()
with torch.no_grad():
    for data in inference_dataloader:
        #         data = Variable(data)
        #         labels = Variable(labels)
        # Forward Pass
        # data = data.to(device)
        predictions = model(data)
        # Find the Loss
        y_pred = softmax(predictions.detach().cpu().numpy())
        predicted = np.argmax(predictions, axis = 1)
        list_predictions.append(predicted)
list_predictions_final = np.concatenate(list_predictions, axis=0)
submission = pd.read_csv('data/sample_submission.csv')
submission['Label'] = list_predictions_final
submission.head(10)
submission.to_csv('./submission.csv',index = False)
pd.read_csv('./submission.csv')