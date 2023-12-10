import os
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

import yaml
# print(os.pwd())
print(os.listdir("./"))
with open("./configs/knn_clf_configs.yaml", "r") as file:
    config = yaml.safe_load(file)

print(config)