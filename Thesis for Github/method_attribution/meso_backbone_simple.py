# MesoInception-4 backbone 
# inception module 4 parallel convolutional networks - capturing different features 

# import relevant modules 
import torch
import torch.nn as nn

# create the inception module for our mesoinception-4 block 
class MesoInceptionBlock(nn.Module):
# channels_in - RGB channels, output feature maps - 1,2,3,4
    def __init__(self, channels_in, fmap1, fmap2, fmap3, fmap4):
# inherit from nn.module
        super().__init__()
# setup the sequential layer for branch1 of the inception module 
        self.branch1 = nn.Sequential(
# 2d convolutional layer - height and width of a frame for the kernel, input channels_in, output channels fmap1, kernel size 1x1, fmap - how many kernels will be used 
            nn.Conv2d(channels_in, fmap1, kernel_size=1),
# add non linearity 
            nn.ReLU()
        )
# setup the sequential layer for branch2 of the inception module 
        self.branch2 = nn.Sequential(
# 2d convolutional layer - input channels_in, output fmap2, kernel size 1x1 - compress number of channels for 2nd convolutional layer 
            nn.Conv2d(channels_in, fmap2, kernel_size=1),
# add non linearity 
            nn.ReLU(),
# 2nd 2d layer, input channels same as output channels, 3x3 kernel with padding ensures output feature map is same as input feature map and outer edges of the feature map are filtered evenly 
            nn.Conv2d(fmap2, fmap2, kernel_size=3, padding=1),
# add non linearity 
            nn.ReLU()
        )
# setup the sequential layer for branch2 of the inception module 
        self.branch3 = nn.Sequential(
# 2d convolutional layer - input channels_in, output fmap3, kernel size 1x1 - compress number of channels for 2nd convolutional layer 
            nn.Conv2d(channels_in, fmap3, kernel_size=1),
# add non linearity 
            nn.ReLU(),
# 3rd 2d layer, input channels same as output channels, 3x3 kernel with padding ensures output feature map is same as input feature map and outer edges of the feature map are filtered evenly, dilation to increase the size of the kernel adding gaps in between each section of the kernel
# for wider structural patterns, padding increased to produce same size feature map as input. 
            nn.Conv2d(fmap3, fmap3, kernel_size=3, padding=2, dilation=2),
# # add non linearity 
            nn.ReLU()
        )

        self.branch4 = nn.Sequential(
# 2d convolutional layer - input channels_in, output fmap4, kernel size 1x1 - compress number of channels for 2nd convolutional layer 
            nn.Conv2d(channels_in, fmap4, kernel_size=1),
            nn.ReLU(),
# 3rd 2d layer, input channels same as output channels, 3x3 kernel with padding ensures output feature map is same as input feature map and outer edges of the feature map are filtered evenly, dilation to increase the size of the kernel adding gaps in between each section of the kernel
# even wider structural patterns, padding increased to produce same size feature map as input. 
            nn.Conv2d(fmap4, fmap4, kernel_size=3, padding=3, dilation=3),
            nn.ReLU()
        )
# channel depth determined by each convolutional network
        self.channels_out = fmap1 + fmap2 + fmap3 + fmap4
# the forward pass
    def forward(self, x):
        br1 = self.branch1(x)
        br2 = self.branch2(x)
        br3 = self.branch3(x)
        br4 = self.branch4(x)
# concatenating on the channel dimension 
        return torch.cat([br1, br2, br3, br4], dim=1)
