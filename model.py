import torch.nn as nn
import torch.nn.functional as F
import torch

def Conv(in_channels , out_channels , kernel_size , stride = 2 , padding = 1 , batch_norm = True):
     
        ''' Crates convolutional Layers with optional Batch normalization'''
        layers = []
        conv_layer = nn.Conv2d(in_channels = in_channels , out_channels = out_channels , kernel_size = kernel_size ,
                               stride = stride , padding = padding , bias = False)
        layers.append(conv_layer)
        
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)
    


def deConv(in_channels , out_channels , kernel_size , stride = 2 , padding = 1 , batch_norm = True):
     
    ''' Creates transpose convolutional Layers with optional Batch normalization'''
    layers = []
    conv_layer = nn.ConvTranspose2d(in_channels = in_channels , out_channels = out_channels , kernel_size = kernel_size ,
                                    stride = stride , padding = padding , bias = False)
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)



class residual_blocks(nn.Module):
    ''' This class defines a residual block'''
    
    def __init__(self , conv_dim):
        super(residual_blocks , self).__init__()
        
        self.conv1 = Conv(in_channels = conv_dim ,out_channels = conv_dim , kernel_size = 3 , stride = 1 , padding = 1 , batch_norm = True )
        self.conv2 = Conv(in_channels = conv_dim , out_channels = conv_dim , kernel_size = 3 , stride = 1 , padding = 1 , batch_norm = True)
        
    def forward(self , x):
        op1 = F.relu(self.conv1(x))
        op2 = x + self.conv2(op1)
        return op2


class Discriminator(nn.Module):
    
    def __init__(self , conv_dim = 64):
        super(Discriminator , self).__init__()
        self.conv1 = Conv(3 , conv_dim , 3 , batch_norm = False) 
        self.conv2 = Conv(conv_dim , conv_dim*2 , 3, padding = (0, 1)) 
        self.conv3 = Conv(conv_dim*2 , conv_dim*4 , 5) 
        self.conv4 = Conv(conv_dim*4 , conv_dim*8 , 5) 

        self.conv5 = Conv(conv_dim*8 , 1 , 11, stride = 1 , batch_norm = False)

    def forward(self , x):

        out = F.relu(self.conv1(x)) 
        out = F.relu(self.conv2(out)) 
        out = F.relu(self.conv3(out)) 
        out = F.relu(self.conv4(out)) 

        out = self.conv5(out)
        return out
    



class Generator(nn.Module):
    
    def __init__(self , conv_dim = 64 , res_blocks = 6):
        super(Generator , self).__init__()
        
        
        #defining the encoder part of the network
        self.conv1 = Conv(3 , conv_dim , 4)
        self.conv2 = Conv(conv_dim , conv_dim*2 , 4)
        self.conv3 = Conv(conv_dim*2 , conv_dim*4 , 4)
        
        
        #Adding residual layers
        res_layers = []
        for layers in range(res_blocks):
            res_layers.append(residual_blocks(conv_dim*4))
        
        self.residual_layers = nn.Sequential(*res_layers)
        
        #defining decoder part of the generator
        self.deconv1 = deConv(conv_dim*4 , conv_dim*2 , 4)
        self.deconv2 = deConv(conv_dim*2 , conv_dim , 4)
        self.deconv3 = deConv(conv_dim , 3 , 4 ,batch_norm = False)
        
        
    def forward(self , x):
        
        ''' Given an image x as input returns the transformed image'''
        op = F.relu(self.conv1(x))
        op = F.relu(self.conv2(op))
        op = F.relu(self.conv3(op))
        
        op = self.residual_layers(op)
        
        op = F.relu(self.deconv1(op))
        op = F.relu(self.deconv2(op))
        op = torch.tanh(self.deconv3(op))
        return op