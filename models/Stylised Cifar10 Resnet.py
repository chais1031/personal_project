#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from skimage import io
import numpy as np
import time
from PIL import Image


# In[2]:


IMG_SIZE = (128,128)

class StylisedCifarDataset(Dataset):
    def __init__(self, data_path, random_seed = 42, target_transform = None, num_classes = None):
        super(StylisedCifarDataset, self).__init__()
        self.data_path = data_path

        self.is_classes_limited = False

        if num_classes != None:
            self.is_classes_limited = True
            self.num_classes = num_classes

        self.classes = []
        class_idx = 0
        for class_name in os.listdir(data_path):
            if not os.path.isdir(os.path.join(data_path,class_name)):
                continue
            self.classes.append(
               dict(
                   class_idx = class_idx,
                   class_name = class_name,
               ))
            class_idx += 1

            if self.is_classes_limited:
                if class_idx == self.num_classes:
                    break

        if not self.is_classes_limited:
            self.num_classes = len(self.classes)

        self.image_list = []
        for cls in self.classes:
            class_path = os.path.join(data_path, cls['class_name'])
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                self.image_list.append(dict(
                    cls = cls,
                    image_path = image_path,
                    image_name = image_name,
                ))

        self.img_idxes = np.arange(0,len(self.image_list))

#         np.random.seed(random_seed)
#         np.random.shuffle(self.img_idxes)

#         last_train_sample = int(len(self.img_idxes) * train_split)
#         if is_train:
#             self.img_idxes = self.img_idxes[:last_train_sample]
#         else:
#             self.img_idxes = self.img_idxes[last_train_sample:]

    def __len__(self):
        return len(self.img_idxes)

    def __getitem__(self, index):

        img_idx = self.img_idxes[index]
        img_info = self.image_list[img_idx]

        img = Image.open(img_info['image_path'])

        tr = transforms.ToTensor()
        img = tr(img)

#         tr = transforms.RandomCrop(IMG_SIZE)
#         img = tr(img)

#         if (img.shape[0] != 3):
#             img = img[0:3]

        return dict(image = img, cls = img_info['cls']['class_idx'], class_name = img_info['cls']['class_name'])

    def get_number_of_classes(self):
        return self.num_classes

    def get_number_of_samples(self):
        return self.__len__()

    def get_class_names(self):
        return [cls['class_name'] for cls in self.classes]

    def get_class_name(self, class_idx):
        return self.classes[class_idx]['class_name']


# In[3]:


def get_stylised_cifar_datasets(data_path, num_classes = None):

#     random_seed = int(time.time())

    dataset = StylisedCifarDataset(data_path, num_classes = num_classes)

    return dataset


# In[5]:


data_path_train = "../../stylised_cifar/train/"
dataset_train = get_stylised_cifar_datasets(data_path_train)

data_path_val = "../../stylised_cifar/validation/"
dataset_val = get_stylised_cifar_datasets(data_path_val)

data_path_test = "../../stylised_cifar/test/"
dataset_test = get_stylised_cifar_datasets(data_path_test)

print(f"Number of train samples {dataset_train.__len__()}")
print("Class names are: " + str(dataset_train.get_class_names()))
print("Class 3rd class name is: " + dataset_train.get_class_name(2))

print(f"Number of val samples {dataset_val.__len__()}")
print("Class names are: " + str(dataset_val.get_class_names()))

print(f"Number of test samples {dataset_test.__len__()}")
print("Class names are: " + str(dataset_test.get_class_names()))

BATCH_SIZE = 12

data_loader_train = DataLoader(dataset_train, BATCH_SIZE, shuffle = True)
data_loader_val = DataLoader(dataset_val, BATCH_SIZE, shuffle = True)
data_loader_test = DataLoader(dataset_test, BATCH_SIZE, shuffle = True)


# In[6]:


import matplotlib.pyplot as plt

fig, axes = plt.subplots(BATCH_SIZE//3,3, figsize=(6,10))

for batch in data_loader_train:

    print(f"Shape of batch['image'] {batch['image'].shape}")
    print(f"Shape of batch['cls'] {batch['cls'].shape}")

    for i in range(BATCH_SIZE):

        col = i % 3
        row = i // 3

        img = batch['image'][i].numpy()

        axes[row,col].set_axis_off()
        axes[row,col].set_title(batch['class_name'][i])
        axes[row,col].imshow(np.transpose(img,(1,2,0)))

    plt.show()

    break


# In[7]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.resnet50().to(device)
# model = nn.DataParallel(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


min_val_loss = np.Inf
# Main Loop
for epoch in range(10):  # loop over the dataset multiple times
    val_loss = 0
    running_loss = 0
    
    # Training Loop
    for i, batch in enumerate(data_loader_train, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = batch['image'].to(device)
        labels = batch['cls'].to(device)
    
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    # Validation Loop
    for data in data_loader_val:
        # get the inputs; data is a list of [inputs, labels]
        inputs = data['image'].to(device)
        labels = data['cls'].to(device)

        # Update val loss
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss

    # Average val_loss
    val_loss = val_loss / len(data_loader_train)
    
    if val_loss < min_val_loss:
        print('saving model')
        torch.save(model.state_dict(), './unstylised_cifar.pth')
        min_val_loss = val_loss
    else:
        print('early stop')
        
correct = 0
total = 0
# Test Loop
model.load_state_dict(torch.load('./unstylised_cifar.pth'))
with torch.no_grad():
    for data in data_loader_test:
        # Get the inputs; data is a list of [inputs, labels]
        inputs = data['image'].to(device)
        labels = data['cls'].to(device)
        
        # Get output and update stats
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
print('Accuracy of the dataset is: %d %%' % (100 * correct / total))

