# Script made for metric calculations on  generated videos of MoStGAN-V
import os
import glob
import sys
import json
import torch
import torchvision
import torch.nn.functional as F
import gc
import torchvision.transforms.functional as TF
from pathlib import Path
# file path to import pre-made metrics 
sys.path.insert(0, "/home/jovyan/MsC_Project/thesis_video_project/common_metrics_on_video_quality")

from calculate_fvd import calculate_fvd
from calculate_lpips import calculate_lpips


# Control centre 
real_folder = "data/speech_segments_mp4"
gen_folder = "generated_samples/latte_speech_20k_full-seed-123"
num_frames = 75
frame_size = 256
max_videos = 100
save_to = "results_latte_speech_20k.json"

# load video functrion
def load_video(path, num_frames, frame_size):
# decompress video into raw pixels turn ticks into secs discard audio and metadata 
    frames, _, _ = torchvision.io.read_video(filename=path, pts_unit="sec")
# change video decoder dimensions to pytorch dimensions Time, Height, Width, Channels
    frames = frames.permute(0, 3, 1, 2) 
# get total number of frames from frames 
    total = frames.shape[0]

# if the video has more than 75 frames extract 75 frames evenly spaced for the length of the video through linspace
    if total >= num_frames:
        idx = torch.linspace(0, total - 1, num_frames).to(torch.int64)
    else:
# if the video has less than 75 frames pad the missing frames with copies of the last frame with torch.full. Arange the tensor sequentially and concatenate the padding to make 75 frames 
        diff = num_frames - total
        extra = torch.full((diff,), total - 1)
        idx = torch.cat([torch.arange(total), extra]).to(torch.int64)
# extract the frames for the video clip 
    clip = frames[idx]
# normalise the rgb integer for FVD and LPIPS calculations 
    clip = clip.float() / 255.0
# resize resolution to 256x256, include anti-alias due to resizing 
    clip = TF.resize(clip, size=[frame_size, frame_size], antialias=True)
    return clip

# process processed clips in batches 
def load_all_videos(folder, num_frames, frame_size, limit):
    """
    Loads every .mp4 file in a folder and stacks them into one big tensor
    of shape [number_of_videos, num_frames, 3, frame_size, frame_size].
    """
    from pathlib import Path
    
# create a Path object and add the users home directory with expanduser
    folder_path = Path(folder).expanduser()
    
# look for all .mp4 files in the folder path, convert files to sorted string list 
    videos = sorted([str(video) for video in folder_path.glob("*.mp4")])
    
# Cap the list at your maximum limit (e.g., 100)
    videos = videos[:limit]

# exception if no videos found
    if len(videos) == 0:
        raise Exception(f"No videos found in {folder_path}")
# sanity check
    print("Found", len(videos), "videos found:", str(folder_path))

# loop through the videos and create a list of video tensors using the load_video function
    clips = []
    for path in videos:
        clips.append(load_video(path, num_frames, frame_size))

# stack the tensors into a matrix to be processed 
    return torch.stack(clips)


def main():
# select device 
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

# check file counts first using pathlib 
    real_count = len(list(Path(real_folder).expanduser().glob("*.mp4")))
    gen_count = len(list(Path(gen_folder).expanduser().glob("*.mp4")))
    
# fvd and lpips are comparative metrics so we need to ensure there is a 1:1 video count
    total_to_load = min(real_count, gen_count, max_videos)

# call the load_all_videos function produce a 5d tensor of all the real videos e.g. [50, 75, 256, 256]
    print(f"\nLoading {total_to_load} real videos")
    real_videos = load_all_videos(real_folder, num_frames, frame_size, total_to_load)
# sanity check of the real_videos tensor 
    print("Real videos shape:", real_videos.shape)
    
# call the load_all_videos function produce a 5d tensor of all the generated videos e.g. [50, 75, 256, 256]
    print(f"\nLoading {total_to_load} generated videos")
    gen_videos = load_all_videos(gen_folder, num_frames, frame_size, total_to_load)
# sanity check of the gen_videos tensor 
    print("Generated videos shape:", gen_videos.shape)

    # sanity check of whats being compared 
    print(f"\nComparing {total_to_load} real videos against {total_to_load} generated videos")

# calculate the fvd score method = styleganv as MoStGAN-V uses this as its backbone
    print("\nCalculating FVD score")
    fvd_score = calculate_fvd(real_videos, gen_videos, device, method="styleganv", only_final=True)
    print("FVD score:", fvd_score)

# clean up RAM through gc.collect = removes anything currently in RAM so theres memory space for the LPIPS calculation 
    gc.collect()
# check if the gpu is currently being used if so empty any leftover process there 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# sanity check of batch size 
    print("\nCalculating LPIPS score in batches of 25")
    batch_size = 25
# empty list for lpips score
    lpips_scores = []

# Loop through the videos in groups of size batch_size 
    for i in range(0, total_to_load, batch_size):
# sanity check determining what batch is being processsed 
        print(f"Processing LPIPS batch {i} to {i + batch_size}")
# group the real videos and generated videos  
        real_batch = real_videos[i:i+batch_size]
        gen_batch = gen_videos[i:i+batch_size]

# calculate the lpips of the batch - how similar the real and gen batches are 
        batch_score = calculate_lpips(real_batch, gen_batch, device, only_final=True)
        
# if batch_score is in a dictionary extract the value  
        if isinstance(batch_score, dict) and 'value' in batch_score:
            val = batch_score['value'][0]
# if batch_tensor is returned as scalar tensor return the tensor value as float
        elif torch.is_tensor(batch_score):
            val = batch_score.item()
        else:
# otherwise return the float 
            val = float(batch_score)
# append it to the LPIPS list 
        lpips_scores.append(val)
        
# same as before
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# get the average LPIPS scores from all batches put it into a dictionary for presentation
    final_lpips = sum(lpips_scores) / len(lpips_scores)
    lpips_score_dict = {'value': [final_lpips]}
    print("Final LPIPS score:", lpips_score_dict)
# create a dictionary for the json file 
    results = {
        "real_videos_folder": real_folder,
        "generated_videos_folder": gen_folder,
        "number_of_videos_compared": total_to_load,
        "fvd_score": fvd_score,
        "lpips_score": lpips_score_dict,
    }
# create and save to json file 
    with open(save_to, "w") as f:
        json.dump(results, f, indent=1)

    print("\nResults saved:", save_to)


if __name__ == "__main__":
    main()