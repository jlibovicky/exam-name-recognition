#!/usr/bin/env python
# coding: utf-8

import os
import logging
import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch import optim
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision import models


logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


class CustomDataset(Dataset):
    def __init__(self, X, y, batch_size, transform):
        super().__init__()
        self.batch_size = batch_size
        self.y = y
        self.X = X
        self.transform = transform

    def __getitem__(self, idx):
        class_id = self.y[idx]
        img = self.X[idx].reshape(28, 28)
        img = Image.fromarray(np.uint8(img * 255)).convert('L')
        img = self.transform(img)
        return img, torch.tensor(int(class_id))

    def __len__(self):
        return len(self.X)


logging.info("Loading data.")

df = pd.read_csv(r"A_Z Handwritten Data.csv", dtype=np.float32)

logging.info("Splitting data, normalize pixel values.")
X = df.iloc[:, 1:].values / 255
X = np.concatenate([X, np.zeros((5000, 28 ** 2))])
y = df.iloc[:, 0].values
y = np.concatenate([y, np.full((5000,), 26)])

train_ratio = 0.90
validation_ratio = 0.05
test_ratio = 0.05

logging.info("Splitting Data into Train, Validation and Test Sets.")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 - train_ratio, stratify=y, random_state=0)

X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test,
    test_size=test_ratio / (test_ratio + validation_ratio),
    random_state=0)


logging.info(
    "Oversampling the Training Data to make the classes more balanced.")
X_train_new = []
y_train_new = []
for _ in range(3):
    for X, y in zip(X_train, y_train):
        if y in (5, 8, 21):
            X_train_new.append(X)
            y_train_new.append(y)
y_train = np.append(y_train, np.array(y_train_new))
X_train = np.append(X_train, np.array(X_train_new), axis=0)

transform = transforms.Compose([
    transforms.RandomRotation(10, fill=0),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
    transforms.ToTensor(),
    transforms.RandomAffine(degrees=0, translate=(0.025, 0.025), fill=256),
    transforms.Normalize([0.5], [0.5])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

dataset_stages = ['train', 'val', 'test']

batch_size = 320
image_datasets = {
    'train': CustomDataset(X_train, y_train, batch_size, transform),
    'val': CustomDataset(X_val, y_val, batch_size, test_transform),
    'test': CustomDataset(X_test, y_test, batch_size, test_transform)
}

dataloaders = {
    x: DataLoader(image_datasets[x],
                  batch_size=image_datasets[x].batch_size,
                  shuffle=True, num_workers=0)
    for x in dataset_stages
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer, scheduler,
                num_epochs=10, early_stop_value=0):
    since = time.time()
    for epoch in range(num_epochs):
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            num_batches = 0
            outputs = None
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                # Loading Bar
                if (phase == 'train'):
                    num_batches += 1
                    percentage_complete = (
                        (num_batches * batch_size) / (dataset_sizes[phase])) * 100
                    percentage_complete = np.clip(percentage_complete, 0, 100)
                    print("{:0.2f}".format(percentage_complete),
                          "% complete", end="\r")

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs.float(), labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

                predicted = torch.max(outputs.data, 1)[1]
                running_correct = (predicted == labels).sum()
                running_corrects += running_correct
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc.item()))

            # Early Stop
            if early_stop_value > 0:
                if phase == 'val':
                    val_accuracy = epoch_acc.item()
        if val_accuracy > early_stop_value:
            print("*** EARLY STOP ***")
            break
    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return model


shufflenet = models.shufflenet_v2_x1_0()
shufflenet.conv1[0] = nn.Conv2d(1, 24, kernel_size=(2, 2), stride=(1, 1))
shufflenet.fc = nn.Linear(in_features=1024, out_features=27, bias=True)
model_ft = shufflenet

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.01)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

shufflenet = train_model(
    shufflenet.to(device), criterion, optimizer_ft,
    exp_lr_scheduler, 15, early_stop_value=0.998)

accuracy_scores = []

running_corrects = 0
outputs = None
for inputs, labels in dataloaders['test']:
    model_ft.eval()

    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = model_ft(inputs)

    predicted = torch.max(outputs.data, 1)[1]
    running_correct = (predicted == labels).sum()
    running_corrects += running_correct

accuracy = running_corrects / dataset_sizes['test']
print("Accuracy: " + str(accuracy.item()))

# Save model
torch.save(model_ft, "shufflenet.pt")
