import torch
import tensorflow_datasets as tfds
import clip
from PIL import Image
import yaml

# Configuration for the model
with open("./configs/knn_clf_configs.yaml", "r") as file:
    config = yaml.safe_load(file)

# Dataset
dataset = tfds.data_source("oxford_iiit_pet", data_dir=config["data_dir"],download=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(config["clip_model_parameters"]["ViT_model"], device=device, download_root=config["clip_model_dir"])

# print(model.named_parameters)
# image = preprocess(Image.open("./clip_test_images/CLIP.png")).unsqueeze(0).to(device)
# print(image.size())


image = Image.fromarray(dataset["train"][0]["image"])
image = preprocess(Image.open("./clip_test_images/CLIP.png")).unsqueeze(0).to(device)
image_embedding = model.encode_image(image)
print(image_embedding.size())