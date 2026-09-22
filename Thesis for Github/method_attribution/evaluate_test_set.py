import torch
from simple_timesformer import MesoTimeSformer, class_names
from train_timesformer import ClipDataset, stratified_split

DATA_FOLDER = "/home/jovyan/MsC_Project/thesis_video_project/attribution_data"
NUM_FRAMES = 75
IMAGE_SIZE = 256
CHECKPOINT_PATH = "attribution_checkpoints_full_30ep/best_model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dataset = ClipDataset(DATA_FOLDER, NUM_FRAMES, IMAGE_SIZE)
_, val_data, test_data = stratified_split(dataset, class_names)
print("Test clips:", len(test_data))

loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False)

model = MesoTimeSformer(num_frames=NUM_FRAMES).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

test_correct = 0
test_seen = 0
with torch.no_grad():
    for videos, labels in loader:
        videos = videos.to(device)
        labels = labels.to(device)
        predictions = model(videos)
        predicted_classes = predictions.argmax(dim=1)
        test_correct += (predicted_classes == labels).sum().item()
        test_seen += videos.size(0)

test_accuracy = test_correct / test_seen
print(f"Test accuracy: {test_accuracy:.2%}")
