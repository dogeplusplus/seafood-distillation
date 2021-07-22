import os
import glob

from torch.utils.data import Dataset
from torchvision.io import read_image

class SeafoodDataset(Dataset):
    def __init__(self, dataset_dir):
        self.image_paths = glob.glob(f"{dataset_dir}/**/*.png")
        self.classes = {
            x: i for i, x in enumerate(os.listdir(dataset_dir))
            if os.path.isdir(os.path.join(dataset_dir, x))
        }

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = read_image(path)
        label = os.path.basename(os.path.dirname(path))
        class_idx = self.classes[label]
        image = image.float() / 255.

        return image, class_idx
