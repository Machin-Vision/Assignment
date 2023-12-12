import torch
import torch.nn as nn
import clip
import numpy as np
import yaml
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader
from sklearn import train_test_split
import wandb

wandb.login()

# Configuration for the model
with open("./configs/linear_clf_configs.yaml", "r") as file:
    config = yaml.safe_load(file)

run = wandb.init(
    project="mv-assignment-linear-clf",
    config = {
        "embedding_size": config["embedding_size"], 
        "output_size": config["output_size"], 
        "num_epochs": config["num_epochs"], 
        "batch_size": config["batch_size"], 
        "lr": config["lr"]
    }

)


# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y)
# np.unique(y_train, return_counts=True)
# np.unique(y_val, return_counts=True)

# train_dataset = Dataset(X_train, y_train, ...)
# train_loader = DataLoader(train_dataset, ...)

# Device CPU/GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device} is used for model training")

# Load Train and Test embeddings
train_embeddings_path = f"{config['save_embeddings_dir']}/train/train_embeddings.npy"
train_labels_path = f"{config['save_embeddings_dir']}/train/train_labels.npy"
test_embeddings_path = f"{config['save_embeddings_dir']}/test/test_embeddings.npy"
test_labels_path = f"{config['save_embeddings_dir']}/test/test_labels.npy"


class OxfordIITPetDataset(Dataset, is_valid=False):

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


model = nn.Sequential()
model.add_module("dense_final", nn.Linear(config["embedding_size"], config["output_size"]))
model = model.to(torch.float)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])



#training model
for epoch in range(config["num_epochs"]):

    torch.set_grad_enabled(True)

    correct = 0
    total = 0
    train_loss = 0

    for i, (embeddings, labels) in enumerate(train_dataloader):
        embeddings = embeddings.to(torch.float)
        labels = labels.to(torch.long)
        out = model(embeddings)
        loss = loss_fn(out, labels)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        out = torch.argmax(out, dim=1)
        total += out.size(0)
        correct += torch.sum(out==labels)

    train_accuracy = correct/total*100
    train_loss /= total

    print(f"Training accuracy at epoch {epoch} is :{correct/total*100:.2f}%")


    torch.save(model, f"{config['save_models_dir']}/{epoch}.pt")



    correct = 0
    total = 0
    test_loss = 0

    #testing model
    torch.set_grad_enabled(False)
    for i, (embeddings, labels) in enumerate(test_dataloader):
        embeddings = embeddings.to(torch.float)
        labels = labels.to(torch.long)
        out = model(embeddings)
        out = torch.argmax(out, dim=1)
        total += out.size(0)
        correct += torch.sum(out==labels)

    test_accuracy = correct/total*100
    test_loss /= correct

    print(f"Test accuracy at epoch {epoch} is :{correct/total*100:.2f}%")

    wandb.log({"training_accuracy": train_accuracy, "train_loss": train_loss, "test_accuracy": test_accuracy, "test_loss": test_loss})


wandb.finish()