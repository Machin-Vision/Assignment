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

wandb.login()

# Configuration for the model
with open("./configs/full_clf_configs.yaml", "r") as file:
    config = yaml.safe_load(file)

# def KNN_classifer():
run = wandb.init(
    project="mv-assignment-full-clf",
    config={
        "clip-model": config["clip_model_parameters"]["ViT_model"],
        "embedding-size": config["clip_model_parameters"]["embedding_size"],
        "batch-size": config["batch_size"],
        "learning-rate": config["lr"],
    },
)

# Device CPU/GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device} is used for model training")

# Load pretrained model and preprocess function
model, preprocess = clip.load(
    config["clip_model_parameters"]["ViT_model"],
    device=device,
    download_root=config["clip_model_dir"],
)

# Fully connected layer
final_model = nn.Sequential(
    model.visual, 
    nn.Linear(config["embedding_size"], config["output_size"])
)

model = final_model.to(device=device)
model = model.to(torch.float)

# def convert(tf_dataset, field):
#     b = len(tf_dataset)
#     shape = tf_dataset[0][field].shape
#     shape = (b,) + tuple(shape)

#     arr = np.zeros(shape)

#     print()
#     for i in range(len(tf_dataset)):
#         arr[i] = tf_dataset[i][field]

#     return arr


# train_dataset_images = convert(tfds.data_source("oxford_iiit_pet", data_dir=config["data_dir"], download=False)["train"], "image")
# print()
# # for i in range()
# train_dataset_images = tfds.data_source("oxford_iiit_pet", data_dir=config["data_dir"], download=False)["train"][i]["image"]
# train_dataset_labels = tfds.data_source("oxford_iiit_pet", data_dir=config["data_dir"], download=False)["train"][:]["label"]

# train_dataset_images = tfds.data_source("oxford_iiit_pet", data_dir=config["data_dir"], download=False)["train"][:]["image"]

train_dataset = tfds.data_source("oxford_iiit_pet", data_dir=config["data_dir"], download=False)["train"]
train_dataset_labels = []

for i in range(len(train_dataset)):
    train_dataset_labels.append(train_dataset[i]["label"])

# train_dataset_images = train_dataset_images.numpy()
train_dataset_labels = train_dataset_labels

train_dataset_idx = [i for i in range(len(train_dataset))]

# train_dataset_images = torch.from_numpy(train_dataset_images)
# valid_dataset_images = torch.from_numpy(valid_dataset_images)
# train_dataset_labels = torch.from_numpy(train_dataset_labels)
# valid_dataset_labels = torch.from_numpy(valid_dataset_labels)

# test_dataset_images = tfds.data_source("oxford_iiit_pet", data_dir=config["data_dir"], download=False)["test"][:]["image"]
# test_dataset_labels = tfds.data_source("oxford_iiit_pet", data_dir=config["data_dir"], download=False)["test"][:]["label"]

train_dataset_idx, valid_dataset_idx, train_dataset_labels, valid_dataset_labels = train_test_split(train_dataset_idx, train_dataset_labels, test_size=0.2, stratify=train_dataset_labels)

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


# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_dataset_idx)
valid_sampler = SubsetRandomSampler(valid_dataset_idx)

train_dataset = OxfordIITPetDataset(mode="train")
test_dataset = OxfordIITPetDataset(mode="test")

train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=config["batch_size"],  sampler=train_sampler
)

validation_dataloader = DataLoader(
    dataset=train_dataset, batch_size=config["batch_size"],  sampler=valid_sampler
)

test_dataloader = DataLoader(
    dataset=test_dataset, batch_size=config["batch_size"], shuffle=True
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

max_accuracy = 0
best_epoch = 0
for epoch in range(config["num_epochs"]):
    correct_train = 0
    total_train = 0
    loss_accum_train = 0

    for i, (images, labels) in enumerate(tqdm(train_dataloader)):
        labels = labels.to(device)
        images = images.to(device)
        out = model(images)
        loss = loss_fn(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        out = torch.argmax(out, dim=1)
        total_train += out.size(0)
        correct_train += torch.sum(out == labels)
        loss_accum_train += loss.item()

    print(
        f"Training accuracy at epoch {epoch} is :{correct_train/total_train*100:.2f}%"
    )
    print(
        f"Training loss at epoch {epoch} is :{loss_accum_train/total_train:.2f}"
    )

    correct_valid = 0
    total_valid = 0
    loss_accum_valid = 0
    # testing model
    for i, (images, labels) in enumerate(tqdm(validation_dataloader)):
        labels = labels.to(device)
        images = images.to(device)
        out = model(images)
        out = torch.argmax(out, dim=1)
        total_valid += out.size(0)
        correct_valid += torch.sum(out == labels)
        loss_accum_valid += loss.item()

    print(f"Validation accuracy at epoch {epoch} is :{correct_valid/total_valid*100:.2f}%")
    print(f"Validation loss at epoch {epoch} is :{loss_accum_valid/total_valid:.2f}")

    if (correct_valid/total_valid*100) > max_accuracy:
        max_accuracy = correct_valid/total_valid*100
        best_epoch = epoch

    wandb.log({"train_loss": loss_accum_train / total_train,
                "train_accuracy": correct_train / total_train * 100,
                "validation_loss": loss_accum_valid / total_valid,
                "validation_accuracy": correct_valid / total_valid * 100,
                "epoch": epoch})

    # save the model as binary object file
    torch.save(model, f"{config['save_model_dir']}/model_{epoch}.pt")

# evaluation testing accuracy
# clear previous model from gpu memory


# model = torch.load(f"{config['save_model_dir']}/model_{best_epoch}.pt")

# correct_test = 0
# total_test = 0
# loss_accum_test = 0

# for i, (images, labels) in enumerate(tqdm(test_dataloader)):
#     labels = labels.to(device)
#     images = images.to(device)
#     out = model(images)
#     loss = loss_fn(out, labels)

#     out = torch.argmax(out, dim=1)
#     total_test += out.size(0)
#     correct_test += torch.sum(out == labels)
#     loss_accum_test += loss.item()

# print(
#     f"Test accuracy at epoch {epoch} is :{correct_test/total_test*100:.2f}%"
# )
# print(
#     f"Test loss at epoch {epoch} is :{loss_accum_test/total_test:.2f}"
# )

print(f"Best epoch is {best_epoch} with accuracy {max_accuracy}")

# finish the run
wandb.finish()