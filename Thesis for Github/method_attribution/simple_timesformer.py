# define our hybrid model
# put each frame through patches to create patches. Transform a patch into a 256 dimensional feature along with a positional embedding. 
# pass through 4 attention blocks with Divided Space-Time Attention each with a residual connection and a linear layer to match the original timesformer implementation.
# average out the value of all tokens in a video resulting in a 256 dimensional tensor, reduces that 256 dimensional for classification. 
import torch
import torch.nn as nn
from patch_backbone import Patches

# define the classifications 
class_names = ["real", "gan", "diffusion"]


class AttentionBlock(nn.Module):
# festure dimensions and the attention heads 
    def __init__(self, dim, heads):
        super().__init__()
# normalise the patches feature value for every dimension
        self.norm_pretime = nn.LayerNorm(dim)
# create the temporal multi attention head, number of patch feature dimensions, number of heads, batch first 
        self.time_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
# linear layer for the residual connection 
        self.temporal_fc = nn.Linear(dim, dim)
# normalise the patches feature value for every dimension
        self.norm_prespace = nn.LayerNorm(dim)
# # create the spatial multi attention head, number of patch feature dimensions, number of heads, batch first 
        self.space_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
# normalise feature dimensions after attention steps
        self.norm_before_forward = nn.LayerNorm(dim)
# define the sequence after attention steps 
        self.forwardpass = nn.Sequential(
# expand feature dimensions by 4 to identify more complex relationships 
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
# reduce the expanded feature dimensions back to the original 
            nn.Linear(dim * 4, dim)
        )
# define forward pass
    def forward(self, video_tensor, frames, patches):
# batch size = size of batch times by frames 
        batch_size = video_tensor.shape[0]
# channels depth of feature maps 
        channels = video_tensor.shape[2]
# format video tensor into a tensor suitable for time attention
        video_tensor1 = video_tensor.view(batch_size, frames, patches, channels)
# rearrange the dimensions of the tensor - patches before frames - treat each patch across frames as a sequence 
        video_tensor1 = video_tensor1.permute(0, 2, 1, 3)
# by multiplying batch size and patches together we are comparing patches across time 
        video_tensor1 = video_tensor1.reshape(batch_size * patches, frames, channels)
# normalise the feature map dimensions using .norm_pretime
        normed = self.norm_pretime(video_tensor1)
# multihead Attention creates query, key and values
        tempattention_out, _ = self.time_attn(normed, normed, normed)
# keep the updated feature map that now has temporal context discard the attention weights matrix (how much attention every frame paid to another frame)
        tempattention_out = self.temporal_fc(tempattention_out)  # pass through the extra linear layer
# add the original input with the newly temporally enriched output - residual connection to ensure the original input is not forgotten 
        video_tensor1 = video_tensor1 + tempattention_out
# revert the temporally enriched 3d tensor used during temporal attention back to its orginal shape for spatial attention 
        video_tensor1 = video_tensor1.view(batch_size, patches, frames, channels)
# rearrange the dimensions of the tensor - patches before frames - treat each patch across frames as a sequence 
        video_tensor2 = video_tensor1.permute(0, 2, 1, 3)
# multiply batch_size with frames, this allows comparisons of patches within a frame 
        video_tensor2 = video_tensor2.reshape(batch_size*frames, patches, channels)

# normalise the feature map dimensions using .norm_prespace
        normed = self.norm_prespace(video_tensor2)
# multihead Attention creates query, key and values
        space_attention, _ = self.space_attn(normed, normed, normed)
# add the input with the newly temporally and spatially enriched output - residual connection to ensure the original input is not forgotten 
        video_tensor2 = video_tensor2 + space_attention
# restore original shape 
        video_tensor2 = video_tensor2.reshape(batch_size, frames * patches, channels)
# normalise feature dimensions 
        normed = self.norm_before_forward(video_tensor2)
# pass the normed feature map that has gone through temporal and spatial attention through two linear layers from self.forwardpass
        transformed_features = self.forwardpass(normed)
# returns temporally and spatially enriched tensor 
        return video_tensor2 + transformed_features


class MesoTimeSformer(nn.Module):
# define length of video clip for processing 
    def __init__(self, num_frames=75):
        super().__init__()
        self.num_frames = num_frames
# call patches module, extract patches from frame
        self.meso = Patches(patch_grid_size=8)
# hidden dimension (each patch represented in these dimensions) for the transformer 
        self.linear = nn.Linear(self.meso.patch_feature_depth, 256)
# create the tensor for the positional embedding, make it a trainable parameter (nn.Parameter), zero tensor ready for values to be entered, number of patches across clip = num_frames * patches, transformers hidden dimension
        self.space_positional_emb = nn.Parameter(torch.zeros(1, 1, self.meso.num_patches, 256))
        self.time_positional_emb = nn.Parameter(torch.zeros(1, num_frames, 1, 256))
# use trunc normal as standard of ViTs, every positional embedding gets unique values std = 0.02 to ensure the number stays small and does not over power other features of the patch 
        nn.init.trunc_normal_(self.space_positional_emb, std=0.02)
        nn.init.trunc_normal_(self.time_positional_emb, std=0.02)
# create 4 blocks with 8 attention heads with hidden dimension 256, ModuleList is used to ensure the attention block weights are managed by model training
        self.attention_blocks = nn.ModuleList([AttentionBlock(256, 8) for i in range(4)])
# classifcation head 
        self.head = nn.Sequential(
# take the 256 hidden dimensions reduce them down to 128 
            nn.Linear(256, 128),
# add non linearity 
            nn.ReLU(),
            nn.Dropout(0.3),
# reduce the 128 to 3 dimensions our classifications 
            nn.Linear(128, len(class_names))
        )

# define the forward pass 
    def forward(self, video):
# unpack the video shape 
        batch, frame, channels, height, width = video.shape
# flatten the video tensor to a 4d image tensor 
        video = video.view(batch * frame, channels, height, width)
# call the patch function on the input video - output of a patch = [batch*frame, num_patches, h*w]
        features = self.meso(video)
# represent each patch through a linear transformation expand the hidden dimension to 256 = [150 if batch = 2, 64,256- hidden dimension]
        features = self.linear(features)
# reshape the features tensor to a 3 dimensional tensor, restores the seperate batch dimension, multiply frame * num patches - whole sequence of the video, -1 keep hidden dimension size the same 
        features = features.reshape(batch, frame, self.meso.num_patches, -1)
        features = features + self.space_positional_emb + self.time_positional_emb
# we create our patch token for the transformer its the inception module output representing the content of the patch and the positional embeddding -the patch position
        patch_tokens = features.reshape(batch, frame * self.meso.num_patches, -1)
# go through each block one by one 
        for block in self.attention_blocks:
# pass patch_tokens, frame and num_patches through an attention block and reassign back to patch tokens - num patches gives information that each frame holds num_patches tokens 
# after 4 blocks it the finaly tensor is shaped by the temporal and spatial attention 
            patch_tokens = block(patch_tokens, frame, self.meso.num_patches)
# after 4 blocks get the average value of the total number of tokens, represent it in the dimensions of the hidden vector 256 - we now have a representation of 1 video 
        patch_tokens = patch_tokens.mean(dim=1)
# put i through the classificaiton head 
        return self.head(patch_tokens)

# model test 
if __name__ == "__main__":
    model = MesoTimeSformer(num_frames=16)
    dummy_data = torch.randn(2, 16, 3, 256, 256)
    output = model(dummy_data)
    print("Output shape:", output.shape)
    print("Total params:", sum(p.numel() for p in model.parameters()))
    loss = output.sum()
    loss.backward()
# ensuring gradients are being changed
    print("Gradients flow:", all(p.grad is not None for p in model.parameters()))