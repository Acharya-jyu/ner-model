#load dataset into my drive
from datasets import load_dataset
import os
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Define the target folder in your Google Drive
drive_folder = "/content/drive/My Drive/4PubMedBert-BioBERTNER"
os.makedirs(drive_folder, exist_ok=True)

# Load the dataset
dataset = load_dataset("rjac/biobert-ner-diseases-dataset")

# Save the dataset directly to your Google Drive
dataset.save_to_disk(drive_folder)

# Verify the dataset structure and save status
print("Dataset Structure:")
print(dataset)
print("\nDataset saved to:", drive_folder)

# Explore the dataset
print("Dataset Structure:")
print(dataset)

# Check the first samples from train, validation, and test sets
train_sample = dataset['train'][0]  # Get a sample from the training set
#validation_sample = dataset['validation'][0]  # Get a sample from the validation set
test_sample = dataset['test'][0]  # Get a sample from the test set

print("\nSample from Training Set:")
print(train_sample)

#print("\nSample from Validation Set:")
#print(validation_sample)

print("\nSample from Test Set:")
print(test_sample)
