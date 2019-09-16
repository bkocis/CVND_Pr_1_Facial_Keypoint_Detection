## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
#        self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
    
        ## Define layers of a CNN
#        self.conv1 = nn.Conv2d(3,16,3, padding=1)
#        self.conv2 = nn.Conv2d(16,32,3, padding=1)
#        self.conv3 = nn.Conv2d(32,64,3, padding=1)
        self.conv1 = nn.Conv2d(1,16,3, padding=1)
        self.conv2 = nn.Conv2d(16,32,3, padding=1)
        self.conv3 = nn.Conv2d(32,64,3, padding=1)
        #self.conv4 = nn.Conv2d(64,128,3, padding=1)
        #self.conv5 = nn.Conv2d(128,256,3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        # 224x224 size images will go through 3 maxpooling layer of 2,2 => 224/2/2/2 = 28
        # final image size is 28x28.
        # the number of parameters will be 28*28*number of output features 64
        #self.fc1 = nn.Linear(7*7*256,500)
        self.fc1 = nn.Linear(28*28*64,500)
        self.fc2 = nn.Linear(500,136)
        self.dropout = nn.Dropout(0.2)
        
        self.batch_norm = nn.BatchNorm1d(num_features=500)
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        #x = self.pool(F.relu(self.conv4(x)))
        #x = self.pool(F.relu(self.conv5(x)))
        # flatten to a vector 
        #x = x.view(-1, 7*7*256)
        x = x.view(-1, 28*28*64)
        x = self.dropout(x)
        x = F.relu(self.batch_norm(self.fc1(x)))
        #x = F.relu(self.fc1(x))

        x = self.dropout(x)
        x = self.fc2(x)
          
        # a modified x, having gone through all the layers of your model, should be returned
        return x
