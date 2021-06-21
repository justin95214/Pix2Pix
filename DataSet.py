import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import numpy as np
from skimage import io, transform
from torch.autograd import Variable
from PIL import Image

if torch.cuda.is_available():
    print(torch.cuda.is_available())
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

print("using pytorch ver :",torch.__version__,"Device :",DEVICE)

anno_dir = "G:/CV/" #G:/CV/Set1_ground_truth_images/
gt_image_dir = "G:/CV/Set1_ground_truth_images/"
input_image_dir ="G:/CV/Set1_input_images/"
BATCH_SIZE = 32
EPOCHS = 50
learning_rate = 1e-3





class Custom_DataSet(torch.utils.data.Dataset):
    def __init__(self,csv_file,root_dir,transform= None):
        self.root_dir = root_dir
        self.transform = transform
        self.landmarks_frame = pd.read_csv(root_dir+csv_file)
        self.data_path =  self.landmarks_frame

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            idx = item.tolist()
        image = Image.open(self.data_path.iloc[item,1]).convert('RGB')
        landmarks = Image.open(self.data_path.iloc[item,2]).convert('RGB')
        #image = io.imread(self.data_path.iloc[item,1])
        #landmarks =  io.imread(self.data_path.iloc[item,2])

        # image = image.permute(0, 3, 1, 2)
        # data augmentation
        target_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
            transforms.Resize([256,256])

        ])

        #print("transform :")
        train_transform = target_transform

        image = target_transform(image)
        landmarks = target_transform(landmarks)


        sample = {'img': image, 'gt': landmarks}
        return sample

    def show_img(image, type):
        image = image[type].permute(1, 2, 0)
        plt.imshow(image)
        plt.title(type)
        plt.show()

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



csv_file = "test1.csv"
root = "G:/CV/"
#Dataset = Custom_DataSet(csv_file,root)

#show_img(Dataset[120],'gt')
#show_img(Dataset[120],'img')

"""
DataLoader = DataLoader(dataset=Dataset,batch_size=BATCH_SIZE,shuffle=True)


model = autoencoder().cuda()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)


l=len(DataLoader)
losslist=list()
epochloss=0
running_loss=0

with torch.cuda.device(0):
    for epoch in range(EPOCHS):
        print(len(DataLoader))
        for idx,data in enumerate(DataLoader):
            noise, gt = data['img'].view(-1,28*28), data['gt'].view(-1,28*28)
            noise = noise.type(torch.FloatTensor) / 255
            gt = gt.type(torch.FloatTensor) / 255
            noise= noise.cuda()
            gt = gt.cuda()
            # ===================forward=====================
            output = model(noise)
            loss = criterion(output, gt)
            MSE_loss = nn.MSELoss()(output, gt)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epochloss += loss.item()
            # ===================log========================
        losslist.append(running_loss / l)
        running_loss = 0
        print("======> epoch: {}/{}, Loss:{}".format(epoch, EPOCHS, loss.item()))

        torch.save(model.state_dict(), './sim_autoencoder.pth')
"""