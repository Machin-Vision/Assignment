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

wandb.login()

# Configuration for the model
with open("/home/avishka/biyon/MV/Assignment/configs/knn_clf_configs.yaml", "r") as file:
    config = yaml.safe_load(file)


run = wandb.init(
    project="mv-assignment-knn-clf",
    config = {
        "clip-model": config["clip_model_parameters"]["ViT_model"], 
        "embedding_size": config["clip_model_parameters"]["embedding_size"], 
        "knn_num_neighbhours_min": config["knn_parameters"]["num_neighbours_min"],
        "knn_num_neighbhours_max": config["knn_parameters"]["num_neighbours_max"],
        "knn_num_neighbhours_interval": config["knn_parameters"]["num_neighbours_interval"],
        "knn_algorithm": config["knn_parameters"]["algorithm"]
    }

)

#num_neighbours limits
num_neighbours_min = config["knn_parameters"]["num_neighbours_min"]
num_neighbours_max = config["knn_parameters"]["num_neighbours_max"]
num_neighbours_interval = config["knn_parameters"]["num_neighbours_interval"]

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

valid_embeddings = np.load(f"{config['save_embeddings_dir']}/train/valid_embeddings.npy")
valid_labels = np.load( f"{config['save_embeddings_dir']}/train/valid_labels.npy")

test_embeddings = np.load(f"{config['save_embeddings_dir']}/test/test_embeddings.npy")
test_labels = np.load(f"{config['save_embeddings_dir']}/test/test_labels.npy")

max_acc = 0
best_neighbors = 0

for n_neighbors in range(num_neighbours_min, num_neighbours_max+1, num_neighbours_interval):
    knn_clf = knn(n_neighbors=n_neighbors, algorithm=config["knn_parameters"]["algorithm"])
    knn_clf.fit(train_embeddings, train_labels)

    val_pred = knn_clf.predict(valid_embeddings)
    accuracy = accuracy_score(valid_labels, val_pred)

    now = time.time()
    now = datetime.fromtimestamp(now)

    wandb.log({"accuracy": accuracy*100, "num_neighbours": n_neighbors, "time": now})

    if max_acc<accuracy:
        max_acc = accuracy
        best_neighbors = n_neighbors


    print(f"Accuracy of KNN: {accuracy*100:.2f}")



knn_clf = knn(n_neighbors=best_neighbors, algorithm=config["knn_parameters"]["algorithm"])
knn_clf.fit(train_embeddings, train_labels)

test_pred = knn_clf.predict(test_embeddings)
accuracy = accuracy_score(test_labels, test_pred)

print("best n_neb : ",best_neighbors)
print("accuracy : ",accuracy)

wandb.finish()