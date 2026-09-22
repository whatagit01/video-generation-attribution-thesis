"""
This is a version of the MesoInception-4 CNN that stops EARLIER than
usual, so it still outputs a small grid of features per frame (like a
mini image), instead of squashing everything down to one single list of
numbers. We need this grid because TimeSformer needs to look at
different PATCHES (small areas) of each frame, not just the whole frame
as a single point.
"""

import torch
import torch.nn as nn

from meso_backbone_simple import MesoInceptionBlock


class Patches(nn.Module):
# default grid size 8x8 inherit from nn.module
    def __init__(self, patch_grid_size=8):
        super().__init__()
# inception block input of 3 channels, set number of output channels/feature maps of each branch 
        self.block1 = MesoInceptionBlock(3, fmap1=1, fmap2=4, fmap3=4, fmap4=2)
# normalise the feature maps of the output of the inception block 
        self.batchn1 = nn.BatchNorm2d(self.block1.channels_out)
# reduce the feature maps/output size 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
# 2nd inception block input output of 1st layer, set number of output channels/feature maps of each branch 
        self.block2 = MesoInceptionBlock(self.block1.channels_out, fmap1=2, fmap2=4, fmap3=4, fmap4=2)
# normalise the feature maps of the output of the 2nd inception block 
        self.batchn2 = nn.BatchNorm2d(self.block2.channels_out)
# reduce the feature maps/output size - halving
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
# convolutional layer take the output of the second layer, set the number of output channels/feature maps of each branch, use kernel size 5x5 with padding to ensure the same resolution size, use 16 filters to create 16 feature maps from 12 
        self.conv3 = nn.Conv2d(self.block2.channels_out, 16, kernel_size=5, padding=2)
# non linearity 
        self.relu3 = nn.ReLU()
# normalise the 16 feature maps 
        self.batchn3 = nn.BatchNorm2d(16)
# reduce the feature maps/output size - halving
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

# adaptive pool to ensure that any incoming grid size will be resized to our desired grid size
        self.patch_grid = nn.AdaptiveAvgPool2d((patch_grid_size, patch_grid_size))
# define num_patches 
        self.num_patches = patch_grid_size * patch_grid_size
# set feature map depth 
        self.patch_feature_depth = 16  # matches conv3's output channel count

# define the forward pass 
    def forward(self, x):
# pass input frame through first block, normalise, pool
        x = self.pool1(self.batchn1(self.block1(x)))
# pass input frame through 2nd block, normalise, pool
        x = self.pool2(self.batchn2(self.block2(x)))
# pass input frame through 3rd block, normalise, pool
        x = self.pool3(self.batchn3(self.relu3(self.conv3(x))))
# reshape the feature map to our desired size
        x = self.patch_grid(x) 
        
# assign variable names to each dimension of the frame tensor
        batch_size, channels, grid_h, grid_w = x.shape
# flatten dimensions 2 and 3 for one dimension of all patches 
        x = x.flatten(start_dim=2)
# swap dimensions 1 and 2 for the transformer - batch number, number of patches, patch feature depth
        x = x.transpose(1, 2)          
        return x


if __name__ == "__main__":
    model = Patches(patch_grid_size=8)
# gnereate fake batch for testing
    fake_batch = torch.randn(2, 3, 256, 256)
# pass fake batch through the model/forward pass
    patches = model(fake_batch)
# sanity check
    print("Output shape:", patches.shape, "(should be [2, 64, 16])")
