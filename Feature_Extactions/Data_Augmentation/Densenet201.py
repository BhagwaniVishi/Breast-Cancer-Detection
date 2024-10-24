# Install necessary libraries
!pip install torch torchvision pandas
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import os

# Define the feature extractor model
model = models.densenet201(pretrained=True)
model.classifier = torch.nn.Identity()  # Remove the classification layer

# Set the model to evaluation mode
model.eval()

# Define the image transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define augmentation transformations
augmentations = [
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomVerticalFlip(p=1.0),
    transforms.RandomRotation(90),
    transforms.RandomRotation(180),
    transforms.RandomRotation(270),
    transforms.ColorJitter(brightness=0.5),
    transforms.ColorJitter(contrast=0.5),
    transforms.ColorJitter(saturation=0.5),
    transforms.ColorJitter(hue=0.5)
]

# Function to apply all augmentations and preprocess the image
def apply_augmentations(image):
    augmented_images = [preprocess(image)]
    for aug in augmentations:
        augmented_image = aug(image)
        augmented_images.append(preprocess(augmented_image))
    return augmented_images

def extract_features(image_tensor):
    with torch.no_grad():
        features = model(image_tensor.unsqueeze(0))
    return features.squeeze().numpy()

# Define the directory containing the images and the output file
image_dir = '/home/2321201101/BreakHis_Dataset/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/benign/SOB/adenosis/40x'
output_file = 'densenet201_40X_adenosis.csv'

if not os.path.exists(image_dir):
    raise FileNotFoundError(f"The directory {image_dir} does not exist.")

if not os.access(image_dir, os.R_OK):
    raise PermissionError(f"Read permission denied for directory {image_dir}.")

if os.path.exists(output_file) and not os.access(output_file, os.W_OK):
    raise PermissionError(f"Write permission denied for file {output_file}.")

# Extract features for each image and store them in a list
image_features = []
image_names = []

for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    if os.path.isfile(image_path) and os.access(image_path, os.R_OK):
        try:
            image = Image.open(image_path).convert('RGB')
            augmented_images = apply_augmentations(image)
            for i, aug_image in enumerate(augmented_images):
                features = extract_features(aug_image)
                image_features.append(features)
                image_names.append(f"{image_name}_aug_{i}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    else:
        print(f"Cannot read {image_path}")

# Convert the list of features to a DataFrame
features_df = pd.DataFrame(image_features)
features_df.insert(0, 'image_name', image_names)

try:
    # Save the DataFrame to a CSV file
    features_df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")
except PermissionError as e:
    print(f"Permission error: {e}")
except Exception as e:
    print(f"Error saving file: {e}")
