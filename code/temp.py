import os
import numpy as np
from sklearn.model_selection import train_test_split

train_dataset = np.load("/home/dumindu/Desktop/mv-assignment/Assignment/embeddings/train/train_embeddings.npy")
train_labels = np.load("/home/dumindu/Desktop/mv-assignment/Assignment/embeddings/train/train_labels.npy")

train_dataset, valid_dataset, train_labels, valid_labels = train_test_split(train_dataset, train_labels, test_size=0.2, stratify=train_labels) 

with open("/home/dumindu/Desktop/mv-assignment/Assignment/embeddings/train/train_embeddings.npy", "wb") as file:
    np.save(file, train_dataset)

with open("/home/dumindu/Desktop/mv-assignment/Assignment/embeddings/train/train_labels.npy", "wb") as file:
    np.save(file, train_labels)

with open("/home/dumindu/Desktop/mv-assignment/Assignment/embeddings/train/valid_embeddings.npy", "wb") as file:
    np.save(file, valid_dataset)

with open("/home/dumindu/Desktop/mv-assignment/Assignment/embeddings/train/valid_labels.npy", "wb") as file:
    np.save(file, valid_labels)

# import tensorflow_datasets as tfds

# dataset = tfds.data_source("oxford_iiit_pet", data_dir="../data",download=False)
# print(dir(dataset["train"]))
# print(".....")
# print(dir(dataset["test"]))

# print(dir(dataset["train"][0]))
# print(type(dataset["train"][0]))
# print(dataset["train"][0]["label"]) #target_labet
# print(dataset["train"][0]["image"].shape) #image
# print(dataset["train"][0]["species"])

# import yaml
# # print(os.pwd())
# print(os.listdir("./"))
# with open("./configs/knn_clf_configs.yaml", "r") as file:
#     config = yaml.safe_load(file)

# print(config)