import numpy as np
from PIL import Image
import glob
def calculate_mean_and_std(dataset_path):
    mean = np.zeros(3)
    std = np.zeros(3)

    # Assuming images are in RGB format
    for image_path in dataset_path:
        img = np.array(Image.open(image_path).convert('RGB')) / 255.0
        mean += np.mean(img, axis=(0, 1))
        std += np.std(img, axis=(0, 1))

    mean /= len(dataset_path)
    std /= len(dataset_path)

    return mean, std

# Example usage
files = sorted(glob.glob("/mnt/data10t/bakuphome20210617/yaoxy/Diffusion-Model/datasets/HuxingDataset" + "/**/*.*", recursive=True)[:500])

mean, std = calculate_mean_and_std(files)
print("Mean:", mean)
print("Std:", std)
