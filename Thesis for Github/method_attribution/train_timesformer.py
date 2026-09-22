# training script for the attribution model, includes attribution supervision
# loads 3 seconds - 75 frames at random 
import os
import glob
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from simple_timesformer import MesoTimeSformer as MesoTimeSformer, class_names


# control centre 
DATA_FOLDER = "/home/jovyan/MsC_Project/thesis_video_project/attribution_data"
# '''change to 15'''
NUM_FRAMES = 75          
IMAGE_SIZE = 256
BATCH_SIZE = 1
NUM_EPOCHS = 100
LEARNING_RATE = 0.0001
CHECKPOINT_FOLDER = "attribution_checkpoints"

# ClipDataset - find all files and classifications of those files 
# inherit from pytorch dataset to work with pytorch
class ClipDataset(Dataset):
# directory wehere the dataset is, how many frames per video, resolution
    def __init__(self, data_folder, num_frames, resolution):
        self.num_frames = num_frames
        self.resolution = resolution
# empty list for all files that have been used
        self.samples = []  

# loop through class names and enumerate 
        for class_number, class_name in enumerate(class_names):
# build path to folder with data_folder and class name 
            folder = os.path.join(data_folder, class_name)
            clip_paths = sorted([
# join folder path and file name for full path 
                os.path.join(folder, f)
# get all files within the folder 
                for f in os.listdir(folder) 
# only choose the mp4 files 
                if f.endswith(".mp4")
            ])
# loop through the clip paths, append to samples, the path along with its classification
            for path in clip_paths:
                self.samples.append((path, class_number))
                
# return the length of the samples - useful for dataloader 
    def __len__(self):
        return len(self.samples)

# define get item function 
    def __getitem__(self, index):
# look up sample with index number, extract the path and classification 
        path, class_number = self.samples[index]
# return only the number of frames, audio frame and fps are discarded, ensure units are seconds not ticks 
        frames, _, _ = torchvision.io.read_video(filename=path, pts_unit="sec")
# reshape the frame from frame height, width channel, to frames, channels, height, width 
        frames = frames.permute(0, 3, 1, 2) 
# check number of frames 
        total_frames = frames.shape[0]

# if frame has more or equal frames as desired number of frames 
        if total_frames >= self.num_frames:
# last possible index to satisfy the number of desired frames 
            last_frame_index = total_frames - self.num_frames
# randomly pick starting index return only 1 index 
            start = torch.randint(0, last_frame_index + 1, (1,)).item()
# start to finish frame index 
            frame_indices = torch.arange(start, start + self.num_frames)
        else:
# if frame is not at desired length then see how many are missing 
            missing = self.num_frames - total_frames
# create copies of the last frame to compensate for the missing frames, "missing," to specify its a tuple with one value
            extra = torch.full((missing,), total_frames - 1)
# concatenate the frames together in sequence starts from the first frame so we don't need to specify 
            frame_indices = torch.cat([torch.arange(total_frames), extra]).type(torch.int64)
# slice the frames of the clips, normalise the pixel values of the frames 
        clip = frames[frame_indices].float() / 255.0
# resize video using interpolate with mode bilinear to ensure smooth resizing (looks at the 4 closest pixel values) 
        clip = F.interpolate(clip, size=(self.resolution, self.resolution), mode="bilinear")
# zero centre the pixel values for stable training 
        clip = (clip - 0.5) / 0.5 
# return our clip tensor [frames, channels, height,width] along with its classification 
        return clip, class_number

def stratified_split(dataset, class_names, train_split=0.7, val_split=0.15, seed=123):
# random number generator with specified starting point/seed 
    generator = torch.Generator().manual_seed(seed)
    
# make the class groups - a dictionary where each classification is the key and the relevant files are the values 
    class_groups = {i: [] for i in range(len(class_names))}
# loop through each sample 
    for index, (path, class_number) in enumerate(dataset.samples):
# append the file index to the correct class groups 
        class_groups[class_number].append(index)

# make empty lists for training, validation and test sets 
    train_set, val_set, test_set = [], [], []
# loop through the class keys and values 
    for class_number, indices in class_groups.items():
# create a tensor of the indices 
        indices = torch.tensor(indices)
# reorder list of files within that class
        shuffled = indices[torch.randperm(len(indices), generator=generator)]
# length of shuffled files 
        total = len(shuffled)
# find total of clips for each set - remaining clips are given to the test set 
        total_train = int(train_split * total)
        total_val = int(val_split * total)
# slice the total list of indexes from the class and add it to the empty lists we made for each set 
        train_set.extend(shuffled[:total_train].tolist())
        val_set.extend(shuffled[total_train:total_train + total_val].tolist())
        test_set.extend(shuffled[total_train + total_val:].tolist())

# subset to avoid copying - lookups index of subset to main dataset list  
    train_data = torch.utils.data.Subset(dataset, train_set)
    val_data = torch.utils.data.Subset(dataset, val_set)
    test_data = torch.utils.data.Subset(dataset, test_set)
# return the tensors relevant to each set 
    return train_data, val_data, test_data


def main():
# sanity check which device is being used 
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)
# find all chosen files calling ClipDataset 
    dataset = ClipDataset(DATA_FOLDER, NUM_FRAMES, IMAGE_SIZE)
    print("Total clips found:", len(dataset))
    print("Class counts:", {name: sum(1 for _, label in dataset.samples if label == i) for i, name in enumerate(class_names)})
# training validation and test split of the files 
    train_data, val_data, test_data = stratified_split(dataset, class_names)
# sanity check - print length of each
    print("Training clips:", len(train_data))
    print("Validation clips:", len(val_data))
    print("Testing clips:", len(test_data))

# batch the sets - shuffle true for training, shuffle false for validation and test 
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    model = MesoTimeSformer(num_frames=NUM_FRAMES).to(device)

# cross entropy loss for classification 
    loss_function = nn.CrossEntropyLoss()
# adam optimiser for the models parameters - maybe add weight decay
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # optimiser = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0001)
# step decay scheduler as in the TimeSformer paper allows initial learning rate to be steep and later learning right to be more iterative, milestones 11,14 of the 15 epochs that have been set 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimiser, milestones=[11, 14], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimiser, milestones=[60, 80], gamma=0.1)
# create folder for model weights 
    os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)
# set validation accuracy tracking
    best_val_accuracy = 0.0
# start model training over the number of epochs 
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct_classifications = 0
        total_seen = 0

        for videos, labels in train_loader:
# transfer tensor and classifications to gpu for speed 
            videos = videos.to(device)
            labels = labels.to(device)
# clear all old gradients 
            optimiser.zero_grad()
# pass batch through the model 
            classifications = model(videos)
# cross entropy loss for wrong classifications 
            loss = loss_function(classifications, labels)
# back propagoation - calculate new gradients
            loss.backward()
# update model weights 
            optimiser.step()
# get total loss of the batch
            total_loss += loss.item() * videos.size(0)
# find highest probability scores for each video 
            predicted_classes = classifications.argmax(dim=1)
# how many match the true classification
            correct_classifications += (predicted_classes == labels).sum().item()
# the size of the batch
            total_seen += videos.size(0)

# the average loss per video  
        train_loss = total_loss / total_seen
# % of accuracy
        train_accuracy = correct_classifications / total_seen
# update the learning rate
        scheduler.step() 
# validation step
        model.eval()

        val_correct = 0
        val_seen = 0
# validation loop same as training loop - torch no grad so weights or gradients are changed 
        with torch.no_grad():
            for videos, labels in val_loader:
                videos = videos.to(device)
                labels = labels.to(device)
                classifications = model(videos)
                predicted_classes = classifications.argmax(dim=1)
                val_correct += (predicted_classes == labels).sum().item()
                val_seen += videos.size(0)

        val_accuracy = val_correct / val_seen

        print(f"Epoch {epoch}/{NUM_EPOCHS} - "
              f"train loss: {train_loss:.4f}, train accuracy: {train_accuracy:.2%}, "
              f"val accuracy: {val_accuracy:.2%}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_path = os.path.join(CHECKPOINT_FOLDER, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print("  New best model saved to", save_path)

    print("\nTraining finished. Best validation accuracy:", best_val_accuracy)


if __name__ == "__main__":
    main()