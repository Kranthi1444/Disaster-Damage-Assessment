# main.py (Updated without XBDDataset, shows damage class and cost)
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from train import train_model
from predict import predict_image
import os

def calculate_accuracy(model, test_dir, class_names, device, csv_path):
    correct_predictions = 0
    total_predictions = 0

    with open(csv_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        filename, true_label = line.strip().split(',')
        image_path = os.path.join(test_dir, filename)

        predicted_label, _ = predict_image(model, image_path, class_names, device)
        if predicted_label == true_label:
            correct_predictions += 1
        total_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    return accuracy

if __name__ == "__main__":
    csv_path = "mini_xbd/damage_labels.csv"
    img_path = "mini_xbd/images"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = datasets.ImageFolder(root="D:/Kranthi/damage_asses-main/AIDER", transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    class_names = ['collapsed_building', 'fire', 'flooded_areas', 'normal', 'traffic_incident']

    incident_to_damage = {
        'collapsed_building': 'destroyed',
        'fire': 'major-damage',
        'flooded_areas': 'minor-damage',
        'traffic_incident': 'minor-damage',
        'normal': 'no-damage'
    }

    cost_mapping = {
        'no-damage': 0,
        'minor-damage': 5000,
        'major-damage': 50000,
        'destroyed': 500000
    }

    from torchvision import models
    import torch.nn as nn

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_model(model, dataloader, device, num_epochs=5)

    torch.save(model.state_dict(), "model_aider.pth")
    print("Model saved as 'model_aider.pth' after training.")

    test_dir = 'mini_xbd/images'

    print("\n--- Testing the model ---")
    with open(csv_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        filename, _ = line.strip().split(',')
        image_path = os.path.join(test_dir, filename)
        predicted_class, = predict_image(model, image_path, class_names, device)
        damage_class = incident_to_damage.get(predicted_class, 'no-damage')
        repair_cost = cost_mapping.get(damage_class, 0)
        print(f"Image: {filename} → Damage: {damage_class}, Estimated Cost: ₹{repair_cost:,}")

    test_accuracy = calculate_accuracy(model, test_dir, class_names, device, csv_path)
    print(f"\nTest Accuracy: {test_accuracy:.2f}%")
