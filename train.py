# step 4 - train.
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from transformers import ViTForImageClassification
from tqdm import tqdm
import utils
import models

# Constants
NUM_CLASSES = 4
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing and augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_name="best_alzheimers_model.pth"):
    is_HF_model = isinstance(model, ViTForImageClassification)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)

            if is_HF_model:
                outputs = outputs.logits

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            train_bar.set_postfix({
                'Loss': running_loss/len(train_loader),
                'Acc': 100.*correct/total
            })

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)

                if is_HF_model:
                    outputs = outputs.logits

                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        print(f'Validation Accuracy: {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print("Saving model")
            torch.save(model.state_dict(), save_name)

def main():
    import zipfile
    import os
    from pathlib import Path

    # Get current working directory
    current_dir = ""

    # Define paths
    extraction_dir = os.path.join(current_dir, "extracted_data")

    # Set base_path to the 'Data' directory inside the extracted folder
    base_path = os.path.join(extraction_dir, "Data")
    print(f"Dataset extracted to: {base_path}")

    # Verify the extraction
    if not os.path.exists(base_path):
        raise Exception("Data directory not found after extraction!")

    # List the extracted directories
    print("\nFound directories:")
    for dir_name in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, dir_name)):
            print(f"- {dir_name}")

    # Create dataset DataFrame
    df = utils.create_dataset_df(base_path)

    # Create data loaders
    train_loader, val_loader = utils.create_data_loaders(
        df,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=BATCH_SIZE,
        val_size=0.2
    )

    model = models.Resnet().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Train model
    train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)




if __name__ == "__main__":
    main()