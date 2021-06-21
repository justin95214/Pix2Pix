import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import time
import numpy as np
from skimage import io, transform
from torch.autograd import Variable
import numpy as np
import test as Custom_DataSet
import psutil
from torchvision.transforms.functional import to_pil_image
import cv2
import math


if torch.cuda.is_available():
    print(torch.cuda.is_available())
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


partitions = psutil.disk_partitions()
Disk = ""

def psnr(orgin,pred):
    mse =  np.mean((orgin-pred)**2)
    print("mse : ",mse)
    if mse ==0:
        return 100
    PIXEL_MAX =255.0
    return 20*math.log10(PIXEL_MAX/math.sqrt(mse))


for p in partitions:
    print(p.mountpoint, p.fstype,"\n")
    if p.fstype =="fuseblk":
        Disk = p.mountpoint.replace("\\","/")
        break
    #break

print(Disk)
# annotation 이 있는 폴더
anno_dir = "CV/" #G:/CV/Set1_ground_truth_images/
gt_image_dir = anno_dir+"Set1_ground_truth_images/"
input_image_dir =anno_dir+"Set1_input_images/"
print(input_image_dir)

print("using pytorch ver :",torch.__version__,"Device :",DEVICE)

#VAE


n_noise = 16
img_size = 28
n_in_out = img_size*img_size
n_mid = 16
eta = 0.01 #학습률

csv_file = "test3.csv"
root = "E:/CV/"

BATCH_SIZE = 64
EPOCHS = 100

from sklearn.model_selection import train_test_split

Dataset = Custom_DataSet.Custom_DataSet(csv_file,root)
valid_size =0.2
num_train = len(Dataset)
print(num_train)
indices = list(range(num_train))

np.random.shuffle(indices)

split = int(np.floor(valid_size * num_train))

train_idx, valid_idx = indices[split:], indices[:split]
print(train_idx)
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

print(train_sampler)
print("split the DataSet")
Train_DataLoader = DataLoader(Dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
                          drop_last=True, shuffle=False)

Test_DataLoader = DataLoader(Dataset, batch_size=BATCH_SIZE, sampler=valid_sampler,
                         drop_last=True, shuffle=False)

print("split completed")

class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]

        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels)),

        layers.append(nn.LeakyReLU(0.2))

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        x = self.down(x)
        return x

# check



class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()

        layers = [
            nn.ConvTranspose2d(in_channels, out_channels,4,2,1,bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU()
        ]

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.up = nn.Sequential(*layers)

    def forward(self,x,skip):
        x = self.up(x)
        x = torch.cat((x,skip),1)
        return x

# check

# generator: 가짜 이미지를 생성합니다.
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64,128)
        self.down3 = UNetDown(128,256)
        self.down4 = UNetDown(256,512,dropout=0.5)
        self.down5 = UNetDown(512,512,dropout=0.5)
        self.down6 = UNetDown(512,512,dropout=0.5)
        self.down7 = UNetDown(512,512,dropout=0.5)
        self.down8 = UNetDown(512,512,normalize=False,dropout=0.5)

        self.up1 = UNetUp(512,512,dropout=0.5)
        self.up2 = UNetUp(1024,512,dropout=0.5)
        self.up3 = UNetUp(1024,512,dropout=0.5)
        self.up4 = UNetUp(1024,512,dropout=0.5)
        self.up5 = UNetUp(1024,256)
        self.up6 = UNetUp(512,128)
        self.up7 = UNetUp(256,64)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(128,3,4,stride=2,padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8,d7)
        u2 = self.up2(u1,d6)
        u3 = self.up3(u2,d5)
        u4 = self.up4(u3,d4)
        u5 = self.up5(u4,d3)
        u6 = self.up6(u5,d2)
        u7 = self.up7(u6,d1)
        u8 = self.up8(u7)

        return u8

# check



class Dis_block(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x


# check


# Discriminator은 patch gan을 사용합니다.
# Patch Gan: 이미지를 16x16의 패치로 분할하여 각 패치가 진짜인지 가짜인지 식별합니다.
# high-frequency에서 정확도가 향상됩니다.

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.stage_1 = Dis_block(in_channels*2,64,normalize=False)
        self.stage_2 = Dis_block(64,128)
        self.stage_3 = Dis_block(128,256)
        self.stage_4 = Dis_block(256,512)

        self.patch = nn.Conv2d(512,1,3,padding=1) # 16x16 패치 생성

    def forward(self,a,b):
        x = torch.cat((a,b),1)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.patch(x)
        x = torch.sigmoid(x)
        return x
# check


# 가중치 초기화
def initialize_weights(model):
    class_name = model.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)


model_gen = GeneratorUNet().to(DEVICE)
model_dis = Discriminator().to(DEVICE)
# 가중치 초기화 적용
model_gen.apply(initialize_weights)
model_dis.apply(initialize_weights)

# 손실함수
loss_func_gan = nn.BCELoss()
loss_func_pix = nn.L1Loss()

# loss_func_pix 가중치
lambda_pixel = 100

# patch 수
patch = (1,256//2**4,256//2**4)

# 최적화 파라미터
from torch import optim
lr = 2e-4
beta1 = 0.5
beta2 = 0.999

opt_dis = optim.Adam(model_dis.parameters(),lr=lr,betas=(beta1,beta2))
opt_gen = optim.Adam(model_gen.parameters(),lr=lr,betas=(beta1,beta2))

# 학습
model_gen.train()
model_dis.train()

batch_count = 0
num_epochs = 100
start_time = time.time()

loss_hist = {'gen': [],
             'dis': []}

for epoch in range(num_epochs):
    print("success DataSet")
    for idx, data in enumerate(Train_DataLoader):
        noise, gt = data['img'], data['gt']
        #noise, gt = data['img'].view(-1, 256*256), data['gt'].view(-1,256*256)
        #noise = noise.type(torch.FloatTensor) / 255
        #gt = gt.type(torch.FloatTensor) / 255
        ba_si = noise.size(0)
        # real image
        #print("real image")
        real_a = noise.to(DEVICE)
        real_b = gt.to(DEVICE)

        # patch label
        #print("patch label")
        real_label = torch.ones(ba_si, *patch, requires_grad=False).to(DEVICE)
        fake_label = torch.zeros(ba_si, *patch, requires_grad=False).to(DEVICE)

        # generator
        #print("generator")
        model_gen.zero_grad()

        #print("model")
        fake_b = model_gen(real_a)  # 가짜 이미지 생성
        out_dis = model_dis(fake_b, real_b)  # 가짜 이미지 식별

        gen_loss = loss_func_gan(out_dis, real_label)
        pixel_loss = loss_func_pix(fake_b, real_b)

        g_loss = gen_loss + lambda_pixel * pixel_loss
        g_loss.backward()
        opt_gen.step()

        # discriminator
        #print("discriminator")
        model_dis.zero_grad()

        out_dis = model_dis(real_b, real_a)  # 진짜 이미지 식별
        real_loss = loss_func_gan(out_dis, real_label)

        out_dis = model_dis(fake_b.detach(), real_a)  # 가짜 이미지 식별
        fake_loss = loss_func_gan(out_dis, fake_label)

        d_loss = (real_loss + fake_loss) / 2.
        d_loss.backward()
        opt_dis.step()

        #print("loss_hist")
        loss_hist['gen'].append(g_loss.item())
        loss_hist['dis'].append(d_loss.item())

        batch_count += 1
        if batch_count % 50 == 0:
            print("step :[", idx,"/",len(Train_DataLoader),'] Epoch: %.0f, G_Loss: %.6f, D_Loss: %.6f, time: %.2f min' % (
            epoch, g_loss.item(), d_loss.item(), (time.time() - start_time) / 60))

    path2models = './models/'
    os.makedirs(path2models, exist_ok=True)
    path2weights_gen = os.path.join(path2models, str(epoch) + 'ep_weights_gen2.pt')
    print(path2weights_gen)
    path2weights_dis = os.path.join(path2models, str(epoch) + 'ep_weights_dis2.pt')

    torch.save(model_gen.state_dict(), path2weights_gen)
    torch.save(model_dis.state_dict(), path2weights_dis)

    # loss history
    plt.figure(figsize=(10,5))
    plt.title('Loss Progress')
    plt.plot(loss_hist['gen'], label='Gen. Loss')
    plt.plot(loss_hist['dis'], label='Dis. Loss')
    plt.xlabel('batch count')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./hist/'+str(epoch)+'_ep_weight_hist.png')
    print("save the hist png")
    # 가중치 불러오기
    weights = torch.load(path2weights_gen)
    model_gen.load_state_dict(weights)


    model_gen.eval()

    # 가짜 이미지 생성
    with torch.no_grad():
        for data in Test_DataLoader:
            noise, gt = data['img'], data['gt']
            # noise, gt = data['img'].view(-1, 256*256), data['gt'].view(-1,256*256)
            # noise = noise.type(torch.FloatTensor) / 255
            # gt = gt.type(torch.FloatTensor) / 255
            #ba_si = noise.size(0)
            # real image
            # print("real image")
            real_a = noise.to(DEVICE)
            fake_imgs = model_gen( real_a).detach().cpu()
            real_imgs = gt
            break

    # 가짜 이미지 시각화
    plt.figure(figsize=(10, 10))

    #d = psnr(real_imgs, fake_imgs)
    #print(d)
    #plt.title(str(d))
    for ii in range(0, 24, 3):
        plt.subplot(8, 3, ii + 1)
        plt.imshow(to_pil_image(0.5 * data['img'][ii] + 0.5))
        plt.axis('off')
        plt.subplot(8, 3, ii + 2)
        plt.imshow(to_pil_image(0.5 * real_imgs[ii] + 0.5))
        plt.axis('off')
        plt.subplot(8, 3, ii + 3)

        plt.imshow(to_pil_image(0.5 * fake_imgs[ii] + 0.5))
        plt.axis('off')
        plt.savefig('./result_fake/' + str(epoch) + '_ep_weight_result0.png')




