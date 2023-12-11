import torch
import torch.nn as nn
import clip
import numpy as np
import yaml
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader


# Configuration for the model
with open("./configs/linear_clf_configs.yaml", "r") as file:
    config = yaml.safe_load(file)

# Device CPU/GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device} is used for model training")

# Load Train and Test embeddings
train_embeddings_path = f"{config['save_embeddings_dir']}/train/train_embeddings.npy"
train_labels_path = f"{config['save_embeddings_dir']}/train/train_labels.npy"
test_embeddings_path = f"{config['save_embeddings_dir']}/test/test_embeddings.npy"
test_labels_path = f"{config['save_embeddings_dir']}/test/test_labels.npy"

# oneHotEncoder = OneHotEncoder()
# train_labels = oneHotEncoder.fit_transform(train_labels)
# train_labels = np.expand_dims(train_labels, axis=1)
# test_labels = oneHotEncoder.fit_transform(test_labels)




class OxfordIITPetDataset(Dataset):

    def __init__(self, embedding_file, label_file):
        self.embeddings = np.load(embedding_file)
        self.labels = np.load(label_file)
        self.embeddings = torch.from_numpy(self.embeddings)
        self.labels = torch.from_numpy(self.labels)

    def __len__(self):
        return self.embeddings.size(0)
    
    def __getitem__(self, idx):
        return  self.embeddings[idx], self.labels[idx]


train_dataset = OxfordIITPetDataset(train_embeddings_path, train_labels_path)
test_dataset = OxfordIITPetDataset(test_embeddings_path, test_labels_path)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=config["batch_size"], shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=config["batch_size"], shuffle=True)

print()
# model - 1 Linear layer

model = nn.Sequential()
model.add_module("dense_final", nn.Linear(config["embedding_size"], config["output_size"]))
# model.add_module("ReLU_final", nn.ReLU())
model = model.to(torch.float)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])



#training model
for epoch in range(config["num_epochs"]):

    correct = 0
    total = 0

    for i, (embeddings, labels) in enumerate(train_dataloader):
        embeddings = embeddings.to(torch.float)
        labels = labels.to(torch.long)
        out = model(embeddings)
        loss = loss_fn(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        out = torch.argmax(out, dim=1)
        total += out.size(0)
        correct += torch.sum(out==labels)


    print(f"Training accuracy at epoch {epoch} is :{correct/total*100:.2f}%")



correct = 0
total = 0

#testing model
for i, (embeddings, labels) in enumerate(test_dataloader):
    embeddings = embeddings.to(torch.float)
    labels = labels.to(torch.long)
    out = model(embeddings)
    out = torch.argmax(out, dim=1)
    total += out.size(0)
    correct += torch.sum(out==labels)


print(f"Test accuracy at epoch {epoch} is :{correct/total*100:.2f}%")