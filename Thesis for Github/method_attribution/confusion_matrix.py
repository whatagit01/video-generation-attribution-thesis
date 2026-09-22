# confusion matrix script

import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simple_timesformer import MesoTimeSformer, class_names
from train_timesformer import ClipDataset, stratified_split

# control centre
DATA_FOLDER = "/home/jovyan/MsC_Project/thesis_video_project/attribution_data"
NUM_FRAMES = 75
IMAGE_SIZE = 256
CHECKPOINT_PATH = "attribution_checkpoints_full/best_model.pt"
SAVE_PLOT_TO = "confusion_matrix_full.png"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ClipDataset(DATA_FOLDER, NUM_FRAMES, IMAGE_SIZE)
    _, val_data, test_data = stratified_split(dataset, class_names)

    loader = torch.utils.data.DataLoader(val_data, batch_size=4, shuffle=False)

    model = MesoTimeSformer(num_frames=NUM_FRAMES).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    all_true_labels = []
    all_predicted_labels = []

    with torch.no_grad():
        for videos, labels in loader:
            videos = videos.to(device)
            predictions = model(videos)
            predicted_classes = predictions.argmax(dim=1).cpu()
            all_true_labels.extend(labels.tolist())
            all_predicted_labels.extend(predicted_classes.tolist())
            
    all_domains = []
    for idx in val_data.indices:
        path, _ = dataset.samples[idx]
        if "action" in path.lower():
            all_domains.append("action")
        elif "speech" in path.lower():
            all_domains.append("speech")
        else:
            all_domains.append("unknown")
            
    print("\nBreakdown by domain:")
    for domain in ["action", "speech"]:
        for i, name in enumerate(class_names):
            class_domain_total = sum(1 for t, d in zip(all_true_labels, all_domains) if t == i and d == domain)
            class_domain_correct = sum(1 for t, p, d in zip(all_true_labels, all_predicted_labels, all_domains)
                                        if t == i and d == domain and t == p)
            if class_domain_total > 0:
                print(f"  {name} ({domain}): {class_domain_correct}/{class_domain_total} = {class_domain_correct/class_domain_total:.1%}")

    cm = confusion_matrix(all_true_labels, all_predicted_labels, labels=list(range(len(class_names))))
    print("Confusion matrix (rows = true class, columns = predicted class):")
    print(cm)

    for i, name in enumerate(class_names):
        total = cm[i].sum()
        correct = cm[i, i]
        if total > 0:
            print(f"  {name}: {correct}/{total} = {correct/total:.1%}")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Attribution Classifier - Confusion Matrix")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(SAVE_PLOT_TO)
    print(f"\nSaved plot to {SAVE_PLOT_TO}")


if __name__ == "__main__":
    main()
