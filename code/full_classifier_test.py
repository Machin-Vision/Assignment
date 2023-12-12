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
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision.models as models

from sklearn.model_selection import train_test_split

# Configuration for the model
with open("./configs/full_clf_configs.yaml", "r") as file:
    config = yaml.safe_load(file)


# Device CPU/GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device} is used for model training")

# Load pretrained model and preprocess function
model, preprocess = clip.load(
    config["clip_model_parameters"]["ViT_model"],
    device=device,
    download_root=config["clip_model_dir"],
)


class OxfordIITPetDataset(Dataset):
    def __init__(self, mode="train"):
        if mode == "train":
            self.dataset = tfds.data_source(
                "oxford_iiit_pet", data_dir=config["data_dir"], download=False
            )["train"]
        else:
            self.dataset = tfds.data_source(
                "oxford_iiit_pet", data_dir=config["data_dir"], download=False
            )["test"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = Image.fromarray(self.dataset[idx]["image"])
        label = self.dataset[idx]["label"]
        image = preprocess(image)
        return image, label


test_dataset = OxfordIITPetDataset(mode="test")

test_dataloader = DataLoader(
    dataset=test_dataset, batch_size=config["batch_size"], shuffle=True
)

loss_fn = nn.CrossEntropyLoss()


best_epoch = 13
model = torch.load(f"{config['save_model_dir']}/model_{best_epoch}.pt")
model = model.to(torch.float)

correct_test = 0
total_test = 0
loss_accum_test = 0

for i, (images, labels) in enumerate(tqdm(test_dataloader)):
    labels = labels.to(device)
    images = images.to(device)
    out = model(images)
    loss = loss_fn(out, labels)

    out = torch.argmax(out, dim=1)
    total_test += out.size(0)
    correct_test += torch.sum(out == labels)
    loss_accum_test += loss.item()

print(f"Test accuracy is :{correct_test/total_test*100:.2f}%")
print(f"Test loss is :{loss_accum_test/total_test:.2f}")
