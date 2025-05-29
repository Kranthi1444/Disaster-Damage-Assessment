from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
from dataset import XBDDataset        
from model import DamageClassifier  
from train import train_model
from predict import predict_image
import os

def calculate_accuracy(model, test_dir, class_names, device, csv_path):
    correct_predictions = 0
    total_predictions = 0

    # Assuming you have a CSV file with the true labels
    with open(csv_path, 'r') as file:
        lines = file.readlines()

    # Loop through test images and calculate accuracy
    for line in lines:
        filename, true_label = line.strip().split(',')
        image_path = os.path.join(test_dir, filename)
        
        # Get predicted label
        predicted_label = predict_image(model, image_path, class_names, device)
        
        # Compare predicted label with true label
        if predicted_label == true_label:
            correct_predictions += 1
        total_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    return accuracy

if __name__ == "__main__":
    csv_path = "mini_xbd/damage_labels.csv"  # Ensure this CSV has image names and labels
    img_path = "mini_xbd/images"

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Load dataset
    dataset = XBDDataset(csv_path, img_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Initialize model
    model = DamageClassifier(num_classes=4)

    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train model
    train_model(model, dataloader, device, num_epochs=30)

    # Save the trained model after training
    torch.save(model.state_dict(), "model.pth")
    print("Model saved as 'model.pth' after training.")

    # Calculate accuracy on test images
    class_names = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']
    test_dir = 'mini_xbd/images'  

    print("\n--- Testing the model ---")
    test_accuracy = calculate_accuracy(model, test_dir, class_names, device, csv_path)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
