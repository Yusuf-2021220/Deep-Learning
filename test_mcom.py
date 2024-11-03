import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import relu
from sklearn.metrics import r2_score
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

import os
import cv2
global locations_path
model_path = "model_secondU_normalization_b16_e10.pth"
locations_path = "locations.txt"
    

global scale_factor
scale_factor = 4

# helper functions for getting the data

def get_data():
    fobj = open(locations_path, "r")
    locations = []
    c=0
    for line in fobj:
        locations.append(line.split())
        locations[c] = [float(i) for i in locations[c]]
        c+=1

    fobj.close()
    # print(locations)
    x_min = np.min([ele[0] for ele in locations])
    y_min = np.min([ele[1] for ele in locations])
    locs = []
    for ele in locations:
        x = int(ele[0]-x_min)
        y = int(ele[1]-y_min)
        locs.append([x,y])
    return locs

# helper functions for the model

def convrelu(in_channels, out_channels, kernel, padding, pool):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        #In conv, the dimension of the output, if the input is H,W, is
        # H+2*padding-kernel +1
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pool, stride=pool, padding=0, dilation=1, return_indices=False, ceil_mode=False) 
        #pooling takes Height H and width W to (H-pool)/pool+1 = H/pool, and floor. Same for W.
        #altogether, the output size is (H+2*padding-kernel +1)/pool. 
    )

def convreluT(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=2, padding=padding),
        nn.ReLU(inplace=True)
        #input is H X W, output is   (H-1)*2 - 2*padding + kernel
    )


# model definition
    
class RadioWNet(nn.Module):

    def __init__(self,inputs=1,phase="firstU"):
        super().__init__()
        
        self.inputs=inputs
        self.phase=phase
        
        if inputs<=3:
            self.layer00 = convrelu(inputs, 6, 3, 1,1) 
            self.layer0 = convrelu(6, 40, 5, 2,2) 
        else:
            self.layer00 = convrelu(inputs, 10, 3, 1,1) 
            self.layer0 = convrelu(10, 40, 5, 2,2) 
         
        self.layer1 = convrelu(40, 50, 5, 2,2)  
        self.layer10 = convrelu(50, 60, 5, 2,1)  
        self.layer2 = convrelu(60, 100, 5, 2,2) 
        self.layer20 = convrelu(100, 100, 3, 1,1) 
        self.layer3 = convrelu(100, 150, 5, 2,2) 
        self.layer4 =convrelu(150, 300, 5, 2,2) 
        self.layer5 =convrelu(300, 500, 5, 2,2) 
        
        self.conv_up5 =convreluT(500, 300, 4, 1)  
        self.conv_up4 = convreluT(300+300, 150, 4, 1) 
        self.conv_up3 = convreluT(150 + 150, 100, 4, 1) 
        self.conv_up20 = convrelu(100 + 100, 100, 3, 1, 1) 
        self.conv_up2 = convreluT(100 + 100, 60, 6, 2) 
        self.conv_up10 = convrelu(60 + 60, 50, 5, 2, 1) 
        self.conv_up1 = convreluT(50 + 50, 40, 6, 2)
        self.conv_up0 = convreluT(40 + 40, 20, 6, 2) 
        if inputs<=3:
            self.conv_up00 = convrelu(20+6+inputs, 20, 5, 2,1)
           
        else:
            self.conv_up00 = convrelu(20+10+inputs, 20, 5, 2,1)
        
        self.conv_up000 = convrelu(20+inputs, 1, 5, 2,1)
        
        self.Wlayer00 = convrelu(inputs+1, 20, 3, 1,1) 
        self.Wlayer0 = convrelu(20, 30, 5, 2,2)  
        self.Wlayer1 = convrelu(30, 40, 5, 2,2)  
        self.Wlayer10 = convrelu(40, 50, 5, 2,1)  
        self.Wlayer2 = convrelu(50, 60, 5, 2,2) 
        self.Wlayer20 = convrelu(60, 70, 3, 1,1) 
        self.Wlayer3 = convrelu(70, 90, 5, 2,2) 
        self.Wlayer4 =convrelu(90, 110, 5, 2,2) 
        self.Wlayer5 =convrelu(110, 150, 5, 2,2) 
        
        self.Wconv_up5 =convreluT(150, 110, 4, 1)  
        self.Wconv_up4 = convreluT(110+110, 90, 4, 1) 
        self.Wconv_up3 = convreluT(90 + 90, 70, 4, 1) 
        self.Wconv_up20 = convrelu(70 + 70, 60, 3, 1, 1) 
        self.Wconv_up2 = convreluT(60 + 60, 50, 6, 2) 
        self.Wconv_up10 = convrelu(50 + 50, 40, 5, 2, 1) 
        self.Wconv_up1 = convreluT(40 + 40, 30, 6, 2)
        self.Wconv_up0 = convreluT(30 + 30, 20, 6, 2) 
        self.Wconv_up00 = convrelu(20+20+inputs+1, 20, 5, 2,1)
        self.Wconv_up000 = convrelu(20+inputs+1, 1, 5, 2,1)

    def forward(self, input):
        
        input0=input[:,0:self.inputs,:,:]
        # print(input0.shape)
        
        if self.phase=="firstU":
            layer00 = self.layer00(input0)
            # print("00",layer00.shape)
            layer0 = self.layer0(layer00)
            # print("0",layer0.shape)
            layer1 = self.layer1(layer0)
            # print("1",layer1.shape)
            layer10 = self.layer10(layer1)
            # print("10",layer10.shape)
            layer2 = self.layer2(layer10)
            layer20 = self.layer20(layer2)
            layer3 = self.layer3(layer20)
            layer4 = self.layer4(layer3)
            layer5 = self.layer5(layer4)
        
            layer4u = self.conv_up5(layer5)
            layer4u = torch.cat([layer4u, layer4], dim=1)
            layer3u = self.conv_up4(layer4u)
            layer3u = torch.cat([layer3u, layer3], dim=1)
            layer20u = self.conv_up3(layer3u)
            layer20u = torch.cat([layer20u, layer20], dim=1)
            layer2u = self.conv_up20(layer20u)
            layer2u = torch.cat([layer2u, layer2], dim=1)
            layer10u = self.conv_up2(layer2u)
            layer10u = torch.cat([layer10u, layer10], dim=1)
            layer1u = self.conv_up10(layer10u)
            layer1u = torch.cat([layer1u, layer1], dim=1)
            layer0u = self.conv_up1(layer1u)
            layer0u = torch.cat([layer0u, layer0], dim=1)
            layer00u = self.conv_up0(layer0u)
            layer00u = torch.cat([layer00u, layer00], dim=1)
            layer00u = torch.cat([layer00u,input0], dim=1)
            layer000u  = self.conv_up00(layer00u)
            layer000u = torch.cat([layer000u,input0], dim=1)
            output1  = self.conv_up000(layer000u)
        
            Winput=torch.cat([output1, input], dim=1).detach()
        
            Wlayer00 = self.Wlayer00(Winput).detach()
            Wlayer0 = self.Wlayer0(Wlayer00).detach()
            Wlayer1 = self.Wlayer1(Wlayer0).detach()
            Wlayer10 = self.Wlayer10(Wlayer1).detach()
            Wlayer2 = self.Wlayer2(Wlayer10).detach()
            Wlayer20 = self.Wlayer20(Wlayer2).detach()
            Wlayer3 = self.Wlayer3(Wlayer20).detach()
            Wlayer4 = self.Wlayer4(Wlayer3).detach()
            Wlayer5 = self.Wlayer5(Wlayer4).detach()
        
            Wlayer4u = self.Wconv_up5(Wlayer5).detach()
            Wlayer4u = torch.cat([Wlayer4u, Wlayer4], dim=1).detach()
            Wlayer3u = self.Wconv_up4(Wlayer4u).detach()
            Wlayer3u = torch.cat([Wlayer3u, Wlayer3], dim=1).detach()
            Wlayer20u = self.Wconv_up3(Wlayer3u).detach()
            Wlayer20u = torch.cat([Wlayer20u, Wlayer20], dim=1).detach()
            Wlayer2u = self.Wconv_up20(Wlayer20u).detach()
            Wlayer2u = torch.cat([Wlayer2u, Wlayer2], dim=1).detach()
            Wlayer10u = self.Wconv_up2(Wlayer2u).detach()
            Wlayer10u = torch.cat([Wlayer10u, Wlayer10], dim=1).detach()
            Wlayer1u = self.Wconv_up10(Wlayer10u).detach()
            Wlayer1u = torch.cat([Wlayer1u, Wlayer1], dim=1).detach()
            Wlayer0u = self.Wconv_up1(Wlayer1u).detach()
            Wlayer0u = torch.cat([Wlayer0u, Wlayer0], dim=1).detach()
            Wlayer00u = self.Wconv_up0(Wlayer0u).detach()
            Wlayer00u = torch.cat([Wlayer00u, Wlayer00], dim=1).detach()
            Wlayer00u = torch.cat([Wlayer00u,Winput], dim=1).detach()
            Wlayer000u  = self.Wconv_up00(Wlayer00u).detach()
            Wlayer000u = torch.cat([Wlayer000u,Winput], dim=1).detach()
            output2  = self.Wconv_up000(Wlayer000u).detach()
            
        else:
            layer00 = self.layer00(input0).detach()
            layer0 = self.layer0(layer00).detach()
            layer1 = self.layer1(layer0).detach()
            layer10 = self.layer10(layer1).detach()
            layer2 = self.layer2(layer10).detach()
            layer20 = self.layer20(layer2).detach()
            layer3 = self.layer3(layer20).detach()
            layer4 = self.layer4(layer3).detach()
            layer5 = self.layer5(layer4).detach()
        
            layer4u = self.conv_up5(layer5).detach()
            layer4u = torch.cat([layer4u, layer4], dim=1).detach()
            layer3u = self.conv_up4(layer4u).detach()
            layer3u = torch.cat([layer3u, layer3], dim=1).detach()
            layer20u = self.conv_up3(layer3u).detach()
            layer20u = torch.cat([layer20u, layer20], dim=1).detach()
            layer2u = self.conv_up20(layer20u).detach()
            layer2u = torch.cat([layer2u, layer2], dim=1).detach()
            layer10u = self.conv_up2(layer2u).detach()
            layer10u = torch.cat([layer10u, layer10], dim=1).detach()
            layer1u = self.conv_up10(layer10u).detach()
            layer1u = torch.cat([layer1u, layer1], dim=1).detach()
            layer0u = self.conv_up1(layer1u).detach()
            layer0u = torch.cat([layer0u, layer0], dim=1).detach()
            layer00u = self.conv_up0(layer0u).detach()
            layer00u = torch.cat([layer00u, layer00], dim=1).detach()
            layer00u = torch.cat([layer00u,input0], dim=1).detach()
            layer000u  = self.conv_up00(layer00u).detach()
            layer000u = torch.cat([layer000u,input0], dim=1).detach()
            output1  = self.conv_up000(layer000u).detach()
        
            Winput=torch.cat([output1, input], dim=1).detach()
        
            Wlayer00 = self.Wlayer00(Winput)
            Wlayer0 = self.Wlayer0(Wlayer00)
            Wlayer1 = self.Wlayer1(Wlayer0)
            Wlayer10 = self.Wlayer10(Wlayer1)
            Wlayer2 = self.Wlayer2(Wlayer10)
            Wlayer20 = self.Wlayer20(Wlayer2)
            Wlayer3 = self.Wlayer3(Wlayer20)
            Wlayer4 = self.Wlayer4(Wlayer3)
            Wlayer5 = self.Wlayer5(Wlayer4)
        
            Wlayer4u = self.Wconv_up5(Wlayer5)
            Wlayer4u = torch.cat([Wlayer4u, Wlayer4], dim=1)
            Wlayer3u = self.Wconv_up4(Wlayer4u)
            Wlayer3u = torch.cat([Wlayer3u, Wlayer3], dim=1)
            Wlayer20u = self.Wconv_up3(Wlayer3u)
            Wlayer20u = torch.cat([Wlayer20u, Wlayer20], dim=1)
            Wlayer2u = self.Wconv_up20(Wlayer20u)
            Wlayer2u = torch.cat([Wlayer2u, Wlayer2], dim=1)
            Wlayer10u = self.Wconv_up2(Wlayer2u)
            Wlayer10u = torch.cat([Wlayer10u, Wlayer10], dim=1)
            Wlayer1u = self.Wconv_up10(Wlayer10u)
            Wlayer1u = torch.cat([Wlayer1u, Wlayer1], dim=1)
            Wlayer0u = self.Wconv_up1(Wlayer1u)
            Wlayer0u = torch.cat([Wlayer0u, Wlayer0], dim=1)
            Wlayer00u = self.Wconv_up0(Wlayer0u)
            Wlayer00u = torch.cat([Wlayer00u, Wlayer00], dim=1)
            Wlayer00u = torch.cat([Wlayer00u,Winput], dim=1)
            Wlayer000u  = self.Wconv_up00(Wlayer00u)
            Wlayer000u = torch.cat([Wlayer000u,Winput], dim=1)
            output2  = self.Wconv_up000(Wlayer000u)
        
          
        # return [output1,output2]
        return output2
    
def average_pooling(val,tx,ty):
    # temp=val[0].cpu().detach().numpy()
    scale_factor = 4
    image_width = 13*scale_factor
    image_height = 14*scale_factor
    temp = val[0].copy()
    for i in range(0,image_height,scale_factor):
        for j in range(0,image_width,scale_factor):
            sum=0
            if(i==tx and j==ty):
                avg=1
            else:
                for i_s in range(i,i+scale_factor,1):
                    for j_s in range(j,j+scale_factor,1):
                        sum+=temp[i_s][j_s]
                avg=sum/16
            for i_s in range(i,i+scale_factor,1):
                for j_s in range(j,j+scale_factor,1):
                    temp[i_s][j_s]=avg
    
    return temp

def distance(x,y):   #Euclidean distance between two points
    dist = np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)
    return dist

    
def build_radiomap(tx, rx, tx_loc, rx_loc):
    # build radiomap
    scale_factor = 4
    image_width = 13*scale_factor
    image_height = 14*scale_factor
    radiomap = np.zeros((image_height, image_width))
    x_i = tx_loc[0]
    y_i = tx_loc[1]
    # print(x_i, y_i)
    for ele in rx_loc:
        x = ele[0]
        y = ele[1]
        # print(x, y)
        # print(radiomap.shape)
        radiomap[x*scale_factor][y*scale_factor] = 2
        for i in range(x*scale_factor, x*scale_factor+scale_factor):
            for j in range(y*scale_factor, y*scale_factor+scale_factor):
                radiomap[i][j] = 2
        radiomap[x_i*scale_factor][y_i*scale_factor] = 1
        for i in range(x_i*scale_factor, x_i*scale_factor+scale_factor):
            for j in range(y_i*scale_factor, y_i*scale_factor+scale_factor):
                radiomap[i][j] = 1
    radiomap = cv2.copyMakeBorder(radiomap, 0, 64-image_height, 0, 64-image_width, cv2.BORDER_CONSTANT, value=0)
    # plt.imshow(radiomap, cmap='gray')
    # plt.show()
    print(radiomap.shape)
    return radiomap

f=[]

# take inputs
for z in range(1):
    tx = int(input("Enter the Tx number (0-43): "))
    rx = input("Enter the Rx numbers (0-43) separated by commas: ")

    # convert rx to list
    rx = rx.split(",")
    rx = [int(i) for i in rx]
    if len(set(rx)) != len(rx):
        print("Duplicate Rx numbers are not allowed")
        exit()
    if len(rx) > 43:
        print("Number of Rx should be less than or equal to 43")
        exit()
    if tx in rx:
        print("Tx and Rx should not be the same")
        exit()
    if tx < 0 or tx > 43:
        print("Tx number should be between 0 and 43")
        exit()

    loctions = get_data()
    # make tx and rx locations
    tx_loc = loctions[tx]
    if len(rx) == 1 and rx[0]==-1:
        # all rx except tx
        rx = [i for i in range(44)]
        rx.remove(tx)
        rx_loc = loctions.copy()
        rx_loc.pop(tx)
    else:
        rx_loc = [loctions[i] for i in rx]

    # build radiomap

    radiomap = build_radiomap(tx, rx, tx_loc, rx_loc)

    # load the model
    model = RadioWNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # predict the radiomap
    radiomap = torch.tensor(radiomap).unsqueeze(0).unsqueeze(0).float()
    output = model(radiomap)
    output = output.detach().numpy()
    output = output[0][0]
    output = average_pooling([output], tx_loc[0]*4, tx_loc[1]*4)
   
    # perform inverse min max scaling
    min_rss = -70.735108623 
    max_rss = -32.7877407351

    # assign 0 to the rx keys
    dict_rx = {}
    for i in rx:
        dict_rx[i] = 0

    # assign the values to the rx keys
    print(len(rx_loc))
    print(len(rx))
    for i in range(len(rx_loc)):
        ele = rx_loc[i]
        x = ele[0]*scale_factor
        y = ele[1]*scale_factor
        
        temp = output[x][y]
        # doing inverse min max scaling
        dict_rx[rx[i]] = (temp*(max_rss-min_rss))+min_rss
        # dict_rx[rx[i]] = -temp

    print("UNet:")
    print(dict_rx)

    # do linear regression
    # calculate the distance between the tx and rxs

    fobj = open(locations_path, "r")
    locations = []
    c=0
    for line in fobj:
        locations.append(line.split())
        locations[c] = [float(i) for i in locations[c]]
        c+=1
    fobj.close()
    tx_loc = locations[tx]
    rx_loc = [locations[i] for i in rx]

    distances = [distance(tx_loc, rx_loc[i]) for i in range(len(rx_loc))]
    weights = [-40.45586332,   1.98171289]
    y = weights[0] + weights[1]*np.array(distances)
    print('-------------------------------------------------------------------------')
    print("Linear Regression: ")
    # make a dict of rx and y
    dict_rx = {}
    for i in range(len(rx)):
        dict_rx[rx[i]] = y[i]
    print(dict_rx)
   

