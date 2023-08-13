import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import math
import pandas as pd
import numpy as np 
import cv2
import networkss
import matplotlib.pyplot as plt 
import streamlit as st
torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)
import warnings 
warnings.filterwarnings('ignore')
data = pd.read_csv('data.csv')
st.set_option('deprecation.showPyplotGlobalUse', False)

class Network(nn.Module):
    
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        
        
        self.fc1 = nn.Linear(in_features=64*8*8, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=1)
        
    def forward(self, t):
        
        # Hidden Conv Layer 1
        t = self.conv1(t)
        t = F.leaky_relu(t)
        t = F.max_pool2d(t, kernel_size=  5, stride=2)
        
        # Hidden Conv Layer 2
        t = self.conv2(t)
        t = F.leaky_relu(t)
        t = F.max_pool2d(t, kernel_size = 5, stride=2)
        
        # Hidden Conv Layer 3
        t = self.conv3(t)
        t = F.leaky_relu(t)
        t = F.max_pool2d(t, kernel_size = 2, stride=2)
                
        # FC Layer
        t = t.reshape(-1,64*8*8)
        t = self.fc1(t)
        
        # Out Layer
        t = self.out(t)
        
        return t


def thresholding_func(img, thv):
    img2 = np.zeros(img.shape)
    
    for i in range(0,len(img[:,0])):
        for j in range(0,len(img[0])):
            if img[i][j] <= thv:
                img2[i][j] = 0
            elif img[i][j] > thv:
                img2[i][j] = 255
                
    return img2 

def data_generator(thv):
    
    img_data =  np.zeros([len(thv),448,448])
    
    for i in range(len(thv)):
        name = thv['Name'][i]
        img = cv2.imread('AlloyDatastet/' + name + '.tif',0)
        img_data[i] = img[0:448,0:448]
        
    print("Shape of 2D Data Generated: ",img_data.shape)
    
    return img_data

def data_generator3D(thv):
    
    img_data =  np.zeros([len(thv),448,448,3])
    
    for i in range(len(thv)):
        name = thv['Name'][i]
        img = cv2.imread('AlloyDatastet/' + name + '.tif')
        img_data[i] = img[0:448,0:448,:]
        
    print("Shape of 3D Data Generated: ",img_data.shape)
    
    return img_data

def size_reduction(tensor, size=(100,100)):
    
    image_data = np.zeros((len(tensor),100,100))
    
    for i in range(len(tensor)):
        image_data[i] = cv2.resize(tensor[i],size, interpolation = cv2.INTER_AREA)
    
    return image_data

tensor2d = data_generator(data)
tensor3d = data_generator3D(data)   

X = size_reduction(tensor2d)
Y = np.array(data['Value']).reshape(194,1)

network = Network()
network.load_state_dict(torch.load('Pytorch_model.h5'))
Xt = torch.Tensor(X)
Yt = torch.Tensor(Y)

def color_plot(i,color=[255,255,153]):
    orig_img = tensor3d[i]
    gray_img = tensor2d[i]
    #print(orig_img.shape)
    #print(gray_img.shape)
    pred = network(Xt[i].reshape((1,1,100,100)))
    pred_img = thresholding_func(gray_img, pred)
    #print(pred_img.shape)
    final_img = orig_img.copy()
    for i in range(448):
        for j in range(448):
            if pred_img[i,j] == 0:
                final_img[i,j,:] = np.array(color)
    fig=plt.figure(figsize=(20, 20))
    columns = 3
    rows = 1
    image = [orig_img, pred_img, final_img]
    title = ['Original Image', 'Thresholded Image', 'Final Image']
    for i in range(1, columns*rows +1):
        #img = np.random.randint(10, size=(h,w))
        fig.add_subplot(rows, columns, i)
        plt.title(title[i-1])
        plt.imshow(np.array(image[i-1],np.int32), cmap='gray')
    # plt.show()
    st.pyplot()
    


