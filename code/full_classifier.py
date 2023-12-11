import os
import time
import torch
import wandb
import tensorflow_datasets as tfds
import clip
from PIL import Image
import yaml
import numpy as np
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import accuracy_score
import torch.nn as nn
from  torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision.models as models

# wandb.login()

# Configuration for the model
with open("./configs/full_clf_configs.yaml", "r") as file:
    config = yaml.safe_load(file)

#
# def KNN_classifer():
# run = wandb.init(
#     project="mv-assignment-knn-clf",
#     config = {
#         "clip-model": config["clip_model_parameters"]["ViT_model"], 
#         "embedding_size": config["clip_model_parameters"]["embedding_size"]
#     }

# )

# Device CPU/GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device} is used for model training")

# Load pretrained model and preprocess function
model, preprocess = clip.load(config["clip_model_parameters"]["ViT_model"], device=device, download_root=config["clip_model_dir"])

# model2 = models.vit_b_32(weights=models.ViT_B_32_Weights)
# print(model2)

# Fully connected layer
model.add_module("dense_final", nn.Linear(config["embedding_size"], config["output_size"]))

# model = model.encode_image

class OxfordIITPetDataset(Dataset):
    def __init__(self, mode="train"):
        if mode == "train":
            self.dataset = tfds.data_source("oxford_iiit_pet", data_dir=config["data_dir"],download=False)["train"]
        else:
            self.dataset = tfds.data_source("oxford_iiit_pet", data_dir=config["data_dir"],download=False)["test"]

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image = Image.fromarray(self.dataset[idx]["image"])
        label = self.dataset[idx]["label"]
        image = preprocess(image)
        return image, label

train_dataset = OxfordIITPetDataset(mode="train")
test_dataset = OxfordIITPetDataset(mode="test")

train_dataloader = DataLoader(dataset=train_dataset, batch_size=config["batch_size"], shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=config["batch_size"], shuffle=True)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

#training model
# for epoch in range(config["num_epochs"]):

#     correct_train = 0
#     total_train = 0
#     loss_accum = 0

#     for i, (images, labels) in enumerate(tqdm(train_dataloader)):
#         labels = labels.to(device)
#         images = images.to(device)
#         out = model.encode_image(images)
#         loss = loss_fn(out, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         out = torch.argmax(out, dim=1)
#         total_train += out.size(0)
#         correct_train += torch.sum(out==labels)
#         loss_accum += loss.item()

#     print(f"Training accuracy at epoch {epoch} is :{correct_train/total_train*100:.2f}%")
#     print(f"Training loss at epoch {epoch} is :{loss_accum/(total_train/config['batch_size']):.2f}")


#     correct_test = 0
#     total_test = 0
#     #testing model
#     for i, (images, labels) in enumerate(tqdm(test_dataloader)):
#         labels = labels.to(device)
#         images = images.to(device)
#         out = model.encode_image(images)
#         out = torch.argmax(out, dim=1)
#         total_test += out.size(0)
#         correct_test += torch.sum(out==labels)

#     print(f"Test accuracy at epoch {epoch} is :{correct_test/total_test*100:.2f}%")