# # === train.py ===
# import torch
# import torch.nn as nn
# import torch.optim as optim

# def train_model(model, dataloader, device, num_epochs=5):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     model.to(device)
#     model.train()

#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         correct = 0
#         total = 0
#         print(f"Starting epoch {epoch+1}")

#         for images, labels in dataloader:
#             images, labels = images.to(device), labels.to(device)

#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#         epoch_accuracy = 100 * correct / total
#         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}, Accuracy: {epoch_accuracy:.2f}%")

#     # Save final accuracy to file
#     with open("accuracy.txt", "w") as f:
#         f.write(f"{epoch_accuracy:.2f}")

# === train.py ===
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# === Step 1: Config ===
DATA_DIR = "D:\Kranthi\damage_asses-main\AIDER"  # Path to your AIDER dataset with class-named folders
BATCH_SIZE = 32
NUM_EPOCHS = 5
MODEL_PATH = "model_aider.pth"

# === Step 2: Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Step 3: Dataset and Dataloader ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Step 4: Model Setup ===
from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

num_classes = len(dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# === Step 5: Train Function ===
def train_model(model, dataloader, device, num_epochs=NUM_EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        print(f"\nStarting epoch {epoch+1}")

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}, Accuracy: {epoch_accuracy:.2f}%")

    # Save accuracy to file
    with open("accuracy.txt", "w") as f:
        f.write(f"{epoch_accuracy:.2f}")

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH}")

# === Step 6: Run Training ===
train_model(model, dataloader, device)
