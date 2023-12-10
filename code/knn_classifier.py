import os
import torch
import tensorflow_datasets as tfds
import clip
from PIL import Image
import yaml
import numpy as np


# Configuration for the model
with open("./configs/knn_clf_configs.yaml", "r") as file:
    config = yaml.safe_load(file)

# Dataset
dataset = tfds.data_source("oxford_iiit_pet", data_dir=config["data_dir"],download=False)

# Device CPU/GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device} is used for model training")

# Load pretrained model and preprocess function
model, preprocess = clip.load(config["clip_model_parameters"]["ViT_model"], device=device, download_root=config["clip_model_dir"])

# Train and Test dataset holders
train_size = len(dataset["train"])
train_embeddings= np.zeros((train_size, config["clip_model_parameters"]["embedding_size"]))
train_labels = np.zeros(train_size)

test_size = len(dataset["train"])
test_embeddings= np.zeros((test_size, config["clip_model_parameters"]["embedding_size"]))
test_labels = np.zeros(test_size)


# Generate embeddings for Train and Test datasets. 
with torch.no_grad():
    if config["save_train_test_embeddings"]:
        for i in range(len(dataset["train"])):
            image = Image.fromarray(dataset["train"][i]["image"])
            label = dataset["train"][i]["label"]
            image = preprocess(image).unsqueeze(0).to(device)     
            image_embedding = model.encode_image(image)
            train_embeddings[i] = image_embedding.numpy()
            train_labels[i] = label


        for i in range(len(dataset["test"])):
            image = Image.fromarray(dataset["test"][i]["image"])
            label = dataset["test"][i]["label"]
            image = preprocess(image).unsqueeze(0).to(device) 
            image_embedding = model.encode_image(image)
            test_embeddings[i] = image_embedding.numpy()
            test_labels[i] = label

        with open(f"{config['save_embeddings_dir']}/train/train_embeddings.npy", "wb") as file:
            np.save(file, train_embeddings)
        with open(f"{config['save_embeddings_dir']}/train/train_labels.npy", "wb") as file:
            np.save(file, train_labels)
        with open(f"{config['save_embeddings_dir']}/test/test_embeddings.npy", "wb") as file:
            np.save(file, test_embeddings)
        with open(f"{config['save_embeddings_dir']}/test/test_labels.npy", "wb") as file:
            np.save(file, test_labels)


# Load Train and Test embeddings
train_embeddings = np.load(f"{config['save_embeddings_dir']}/train/train_embeddings.npy")
train_labels = np.load(f"{config['save_embeddings_dir']}/train/train_labels.npy")
test_embeddings = np.load(f"{config['save_embeddings_dir']}/test/test_embeddings.npy")
test_labels = np.load(f"{config['save_embeddings_dir']}/test/test_labels.npy")

print()