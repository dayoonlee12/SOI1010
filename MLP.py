# -*- coding: utf-8 -*-

!pip install kaggle
from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c korean-sign-language-recognition-soi1010-2023
!ls

!unzip korean-sign-language-recognition-soi1010-2023.zip

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image
from torch.utils.data.dataset import random_split
from PIL import Image
import os
import pandas as pd
import numpy as np

class HandSignDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = f"{self.df.iloc[idx, 0]:03d}.jpg"
        #print(img_name)
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        label = int(self.df.iloc[idx, 1])
        return image, label

sampleset = HandSignDataset(csv_file="/content/train.csv",root_dir="/content/train",transform=ToTensor())

#Visualizing data
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

indices = [i for i in range(len(sampleset))]
print(indices)
random_indices = np.random.choice(indices,size=(20,),replace=False)
print(random_indices)
rows,cols = 4,5
fig, axes = plt.subplots(rows,cols,figsize=(20, 8))
for i in range(rows*cols):
  idx = random_indices[i]
  img,label = sampleset[idx]
  row = i//5
  col = i%5
  ax = axes[row][col]
  ax.imshow(img.permute(1,2,0).numpy())
  ax.set_title(f"{label}")
  ax.axis("off")
plt.tight_layout()
plt.show()

torch.manual_seed(17)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((32,32)),
    transforms.CenterCrop(28),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

other_transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

class Splitset(Dataset):
  def __init__(self, base, transform):
    super(Splitset,self).__init__()
    self.base = base
    self.transform = transform

  def __len__(self):
    return len(self.base)

  def __getitem__(self,idx):
    img, label = self.base[idx]
    return self.transform(img),label

trainingset = HandSignDataset(csv_file="/content/train.csv",root_dir="/content/train")

torch.manual_seed(17)
val_ratio = 0.2
num_train = len(trainingset)
num_val = int(num_train*val_ratio)
num_train -= num_val
train, val = random_split(trainingset,[num_train,num_val])
train_set,val_set = Splitset(train,train_transform),Splitset(val,other_transform)

torch.manual_seed(17)
train_set[0]
img = train_set[0][0]
plt.imshow(img.permute(1,2,0))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class MLPModel(nn.Module):
    def __init__(self, input_dim =3*28*28, hidden_dim = 512, num_classes = 11):
        super().__init__()
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.encoder(x)
        x = self.classifier(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        train_acc += (pred.argmax(1) == y).type(torch.float).sum().item()

    train_loss /= num_batches
    train_acc /= size
    print(f"Train Error: \n Accuracy: {(100*train_acc):>0.1f}%, Avg loss: {train_loss:>8f} \n")

    return train_loss, train_acc

def validation_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    validation_loss, validation_acc = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            validation_loss += loss_fn(pred, y).item()
            validation_acc += (pred.argmax(1) == y).type(torch.float).sum().item()

    validation_loss /= num_batches
    validation_acc /= size
    print(f"Validation Error: \n Accuracy: {(100*validation_acc):>0.1f}%, Avg loss: {validation_loss:>8f} \n")

    return validation_loss, validation_acc

# hyper-parameter
learning_rate = 1e-3
batch_size = 64
epochs = 15
hidden_dim = 512

#Data loader
train_dataloader = DataLoader(train_set, batch_size=batch_size,shuffle = True)
validation_dataloader = DataLoader(val_set, batch_size=batch_size,shuffle = False)

torch.manual_seed(17)
if device == "cuda":
    torch.cuda.manual_seed_all(17)
model = MLPModel(hidden_dim=hidden_dim).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#epochs = 10
best_validation_acc = 0.0
e = [i for i in range(1,epochs+1)]
t_loss = []
v_loss = []
fig, ax = plt.subplots()
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss, train_acc = train_loop(train_dataloader, model, loss_fn, optimizer)
    validation_loss, validation_acc = validation_loop(validation_dataloader, model, loss_fn)
    if (validation_acc > best_validation_acc):
        best_validation_acc = validation_acc
    t_loss.append(train_loss)
    v_loss.append(validation_loss)
    plt.plot()

    print("Done!")

ax.plot(e,t_loss,label="Training loss")
ax.plot(e,v_loss,label= "Validation loss")
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.set_title('Loss Plot')
ax.legend()

# Show the plot
plt.show()

class TestDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = f"{self.df.iloc[idx, 0]:03d}.jpg"
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image

test_transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = TestDataset(csv_file="/content/test.csv",root_dir="/content/test",transform=test_transform)

predictions = []
for i in range(len(testset)):
   x = testset[i].unsqueeze(0)
   x = x.to(device)
   model.eval()
   with torch.no_grad():
      pred = model(x)
      pred = pred.argmax(1).item()
   #pred = pred.argmax(1).item()
      predictions.append(pred)
testset.df['pred'] = predictions
testset.df.to_csv("/content/testresult.csv",index = False)