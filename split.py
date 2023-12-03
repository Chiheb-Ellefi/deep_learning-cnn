import os
import shutil
import random
from sklearn.model_selection import train_test_split

# Path to your dataset folder
dataset_path = './EuroSAT_data'

# List all classes in the dataset folder
classes = os.listdir(dataset_path)

# Create directories for training and testing sets
train_path = './data/train'
test_path = './data/test'

if not os.path.exists(train_path):
    os.makedirs(train_path)

if not os.path.exists(test_path):
    os.makedirs(test_path)

# Splitting the data into train and test (80% train, 20% test for each class)
for class_name in classes:
    print(class_name)
    class_images = os.listdir(os.path.join(dataset_path, class_name))
    train_images, test_images = train_test_split(class_images, test_size=0.2, random_state=42)

    # Create folders for each class in train and test sets
    class_train_path = os.path.join(train_path, class_name)
    class_test_path = os.path.join(test_path, class_name)

    if not os.path.exists(class_train_path):
        os.makedirs(class_train_path)

    if not os.path.exists(class_test_path):
        os.makedirs(class_test_path)

    # Copy images to the respective train and test folders
    for img in train_images:
        shutil.copy(os.path.join(dataset_path, class_name, img), os.path.join(class_train_path, img))

    for img in test_images:
        shutil.copy(os.path.join(dataset_path, class_name, img), os.path.join(class_test_path, img))
