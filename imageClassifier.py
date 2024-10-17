# Imports here
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

#TODO: Load the datasets with ImageFolder

train_datasets = datasets.ImageFolder(train_dir,transform=train_transforms)
test_datasets = datasets.ImageFolder(test_dir,transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders

train_dataloaders = torch.utils.data.DataLoader(train_datasets,batch_size=64,shuffle=True)
test_dataloaders = torch.utils.data.DataLoader(test_datasets,batch_size=64,shuffle=True)

trainiter = iter(train_dataloaders)
images, labels = next(trainiter)
print(type(images))
print(images.shape)
img1 = images.view(images.shape[0],-1)
print(img1.shape)
print(labels.shape)


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
print(cat_to_name)

# TODO: Build and train your network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.vgg11(pretrained=True)
model.classifier = nn.Sequential(nn.Linear(150528, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(256, 2),
                                 nn.LogSoftmax(dim=1))
model
for param in model.parameters():
    param.requires_grad = False

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)


epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in train_dataloaders:
        #images = images.resize_(images.shape[0],-1)
        #print(images.shape)
        log_ps = model(images)      
        loss = criterion(log_ps, labels)   
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")
